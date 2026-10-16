"""scripts/claims.py (PREREG confirmatory tests) and the graded metrics in scripts/stats_haystack.py,
on synthetic data with known answers only.

    uv run --with pytest --with pyyaml pytest tests/test_claims.py
"""

import itertools
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import claims  # noqa: E402
from stats_haystack import bpref, graded_metrics, holm, per_query  # noqa: E402

JUDGES = ["deepseek/deepseek-v4-flash", "google/gemini-3.1-flash-lite"]
LLM_DS = "llm-deepseek-v4-flash-top20-of-bm25__gene"
LLM_GM = "llm-gemini-3.1-flash-lite-top20-of-bm25__gene"
NQ, NDOC = 40, 30
GRADES = {"d0": 2, "d1": 2, "d2": 1} | {f"d{i}": 0 for i in range(3, NDOC)}  # every doc graded
IDCG = 2 + 2 / math.log2(3) + 1 / 2


def brute_sign_flip(d) -> float:
    obs = abs(sum(d))
    hits = [abs(sum(s * x for s, x in zip(signs, d))) >= obs - 1e-9
            for signs in itertools.product([-1, 1], repeat=len(d))]
    return sum(hits) / len(hits)


def ranked(qid: str, order: list[str]) -> list[tuple]:
    return [(qid, p, float(len(order) - i), i + 1) for i, p in enumerate(order)]


def save_run(hay: Path, stem: str, rows: list[tuple]) -> None:
    pd.DataFrame(rows, columns=["qid", "pmid", "score", "rank"]).to_parquet(hay / "runs" / f"{stem}.parquet")


# ---- synthetic world ---------------------------------------------------------------------------

@pytest.fixture
def world(tmp_path):
    """A haystack (NQ queries, gold d0 and d1, GRADES judged), a v2 small task, a 12-task suite."""
    rng = np.random.default_rng(7)
    hay = tmp_path / "haystack"
    (hay / "runs").mkdir(parents=True)
    qids = [f"q{i:02d}" for i in range(NQ)]
    docs = [f"d{i}" for i in range(NDOC)]
    pd.DataFrame([(q, p) for q in qids for p in ("d0", "d1")], columns=["qid", "pmid"]).to_parquet(hay / "qrels.parquet")

    def merged(name: str, d2: int) -> None:  # the files differ at d2, so tests can tell them apart
        g = GRADES | {"d2": d2}
        pd.DataFrame([(q, p, x) for q in qids for p, x in g.items()],
                     columns=["qid", "pmid", "grade"]).astype({"grade": "int8"}).to_parquet(hay / name)

    merged("judgments_merged.parquet", 1)
    merged("judgments_merged_no-deepseek.parquet", 2)
    merged("judgments_merged_no-gemini.parquet", 0)
    pd.DataFrame({"qid": "q00", "pmid": "d0", "judge": JUDGES, "grade": 2}).to_parquet(hay / "judgments.parquet")
    pd.DataFrame({"system": [LLM_DS, LLM_GM], "judge": JUDGES}).to_parquet(hay / "judgments_conflicts.parquet")

    zeros = docs[3:]
    for stem, make in {
        "good__gene": lambda: docs,  # d0, d1, d2 first, then the 0s
        "bad__gene": lambda: list(rng.permutation(docs)),
        LLM_DS: lambda: list(rng.permutation(docs)),
        LLM_GM: lambda: list(rng.permutation(docs)),
        # seeds: d0 at rank s, d1 and d2 below the top 10; off-the-shelf base: d0 at rank 4
        **{f"ft-s{s}__gene": (lambda s=s: zeros[:s - 1] + ["d0"] + zeros[s - 1:] + ["d1", "d2"])
           for s in (1, 2, 3)},
        "ft-base__gene": lambda: zeros[:3] + ["d0"] + zeros[3:] + ["d1", "d2"],
    }.items():
        save_run(hay, stem, [r for q in qids for r in ranked(q, make())])

    # v2: 10 pairs per batch; batch 0 (practice) is the reverse of batches 1-4, so using it would show
    b14 = np.arange(40)
    llm_hit, emb_hit = b14 % 4 != 0, b14 % 3 == 0  # 30 vs 14 right of 40
    v2 = []
    for b in range(5):
        for i in range(10):
            k = (b - 1) * 10 + i
            v2.append((b, f"b{b}q{i}", bool(llm_hit[k]) if b else False, bool(emb_hit[k]) if b else True))
    v2 = pd.DataFrame(v2, columns=["batch", "qid", "llm", "emb"])
    emb = pd.concat([v2.assign(model="emb-x", hit1=v2.emb), v2.assign(model="emb-y", hit1=~v2.emb)])
    emb.assign(rank_of_gold=1)[["model", "batch", "qid", "rank_of_gold", "hit1"]].to_parquet(tmp_path / "v2_perquery.parquet")
    for b, g in v2.groupby("batch"):
        (tmp_path / "llm_v2" / "llm-z").mkdir(parents=True, exist_ok=True)
        g.assign(task=f"v2_{b}", hit1=g.llm)[["task", "batch", "qid", "hit1"]].to_parquet(tmp_path / "llm_v2" / "llm-z" / f"{b}.parquet")

    # suite: 12 tasks, m-a - m-b known per task
    delta = np.array([0.05, 0.02, 0.03, -0.01, 0.04, 0.06, -0.02, 0.01, 0.03, 0.02, -0.03, 0.05])
    base = rng.uniform(0.3, 0.8, 12)
    tasks = [f"T{i}" for i in range(12)]
    pd.DataFrame({"model": ["m-a"] * 12 + ["m-b"] * 12, "model_type": "embedding", "task": tasks * 2,
                  "category": "Retrieval", "score": np.r_[base + delta, base], "source": "ours"}
                 ).to_csv(tmp_path / "bio_scores.csv", index=False)
    return SimpleNamespace(tmp=tmp_path, hay=hay, qids=qids, llm_hit=llm_hit, emb_hit=emb_hit, delta=delta,
                           sources={"haystack": str(hay), "v2_embedders": str(tmp_path / "v2_perquery.parquet"),
                                    "v2_llms": str(tmp_path / "llm_v2"), "suite": str(tmp_path / "bio_scores.csv")})


def hay_claim(a, b) -> dict:
    return {"data": "haystack", "test": "signflip", "a": a, "b": b}


def run(world, claim_map: dict, *flags: str, any_claims: bool = True) -> pd.DataFrame | None:
    """claims.py on a synthetic config; the results table, or None for a dry run."""
    cfg = world.tmp / "claims.yaml"
    cfg.write_text(yaml.safe_dump({"sources": world.sources, "claims": claim_map}))
    out = world.tmp / "out" / "claims"
    claims.main([str(cfg), "--out", str(out), *flags, *(["--any-claims"] if any_claims else [])])
    return None if "--dry-run" in flags else pd.read_csv(out.with_suffix(".csv")).set_index("claim")


# ---- tests and corrections ---------------------------------------------------------------------

def test_sign_flip_exact_matches_brute_force():
    d = np.random.default_rng(0).normal(0.05, 0.1, 10)
    d[3] = 0.0
    p, exact = claims.sign_flip_p(d)
    assert exact and p == pytest.approx(brute_sign_flip(d), abs=1e-12)
    assert claims.sign_flip_p(np.full(10, 0.1))[0] == pytest.approx(2 / 2 ** 10)


def test_sign_flip_monte_carlo_near_exact():
    d = np.random.default_rng(1).normal(0.03, 0.1, 16)  # 2^16 > 10,000: random sign vectors
    p, exact = claims.sign_flip_p(d)
    assert not exact and p == claims.sign_flip_p(d)[0]  # seeded
    signs = 1 - 2 * ((np.arange(2 ** 16)[:, None] >> np.arange(16)) & 1)
    p_exact = (np.abs(signs @ d) >= abs(d.sum()) - 1e-9).mean()
    assert abs(p - p_exact) < 4 * math.sqrt(p_exact * (1 - p_exact) / claims.N_PERM)


@pytest.mark.parametrize("only_a,only_b,both", [(12, 4, 20), (4, 12, 0), (7, 7, 3), (1, 0, 5), (30, 9, 50), (0, 0, 9)])
def test_mcnemar_matches_scipy(only_a, only_b, both):
    a = np.r_[np.ones(only_a), np.zeros(only_b), np.ones(both), np.zeros(5)].astype(bool)
    b = np.r_[np.zeros(only_a), np.ones(only_b), np.ones(both), np.zeros(5)].astype(bool)
    p, na, nb = claims.mcnemar_p(a, b)
    assert (na, nb) == (only_a, only_b)
    expect = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    assert p == pytest.approx(expect, rel=1e-9)


def test_holm_hand_example():
    # sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 = 0.02, 0.03, 0.06, 0.04 -> running max 0.06
    np.testing.assert_allclose(holm(np.array([0.01, 0.04, 0.03, 0.005])), [0.03, 0.06, 0.06, 0.02])


def test_bootstrap_ci_brackets_mean():
    d = np.random.default_rng(2).normal(0.1, 0.05, 50)
    lo, hi = claims.bootstrap_ci(d)
    assert lo < d.mean() < hi and claims.bootstrap_ci(d) == (lo, hi)


# ---- graded metrics ----------------------------------------------------------------------------

def test_graded_ndcg_precision_hand_example():
    grades = {"q": {"d1": 2, "d2": 1, "d3": 0, "d4": 2, "d5": 1}}
    run = pd.DataFrame(ranked("q", ["d3", "d1", "x", "d2", "d4"]), columns=["qid", "pmid", "score", "rank"])
    m = graded_metrics(run, grades).loc["q"]
    dcg = 2 / math.log2(3) + 1 / math.log2(5) + 2 / math.log2(6)     # ranks 2, 4, 5; d3 and x add 0
    idcg = 2 + 2 / math.log2(3) + 1 / math.log2(4) + 1 / math.log2(5)  # 2, 2, 1, 1, 0
    assert m["graded_ndcg@10"] == pytest.approx(dcg / idcg)
    assert dcg / idcg == pytest.approx(0.588246, abs=1e-6)  # 2.466242 / 4.192536
    assert m["p@10_ge1"] == pytest.approx(3 / 10) and m["p@10_ge2"] == pytest.approx(2 / 10)
    assert (m["judged@10"], m["n@10"]) == (4, 5)


def test_unjudged_docs_never_relevant():
    grades = {"q": {"d1": 2, "d2": 0}, "z": {"d1": 0}}  # z: no positive grade -> NaN, not 0 or 1
    run = pd.DataFrame(ranked("q", [f"u{i}" for i in range(12)] + ["d1"]) + ranked("z", ["d1"]),
                       columns=["qid", "pmid", "score", "rank"])
    m = graded_metrics(run, grades)
    assert m.loc["q", ["graded_ndcg@10", "p@10_ge1", "judged@10"]].tolist() == [0, 0, 0]
    assert m.loc["q", "bpref_ge1"] == 1.0  # the relevant doc is the first judged one
    assert m.loc["z", ["graded_ndcg@10", "p@10_ge1", "bpref_ge1"]].isna().all()


def test_bpref_hand_examples():
    g = {"r1": 2, "r2": 1, "r3": 2, "n1": 0, "n2": 0, "n3": 0}
    order = ["u", "r1", "n1", "r2", "n2", "n3"]  # u unjudged (skipped), r3 never retrieved
    # >= 1: R = 3. r1: 1 - 0/3; r2: 1 - 1/3; r3: 0  ->  (1 + 2/3) / 3
    assert bpref(order, g, 1) == pytest.approx(5 / 9)
    # >= 2: R = 2 (r1, r3), r2 is non-relevant. r1: 1 - 0/2  ->  1 / 2
    assert bpref(order, g, 2) == pytest.approx(1 / 2)
    # count of non-relevant above capped at R: R = 1, two above -> 1 - min(2, 1)/1 = 0
    assert bpref(["n1", "n2", "r1"], {"r1": 1, "n1": 0, "n2": 0, "n3": 0}, 1) == 0
    # N < R: trec_eval's denominator min(R, N) (the 2004 paper's R would give 2/3 and 2/3)
    assert bpref(["n1", "a", "b", "c"], {"a": 2, "b": 2, "c": 1, "n1": 0}, 1) == 0
    assert bpref(["a", "n1", "b", "n2", "c"], {"a": 1, "b": 1, "c": 1, "n1": 0, "n2": 0}, 1) == pytest.approx(1.5 / 3)
    # N = 0: no judged non-relevant doc, so each retrieved relevant doc scores 1
    assert bpref(["a", "u", "b"], {"a": 1, "b": 2, "c": 1}, 1) == pytest.approx(2 / 3)
    assert math.isnan(bpref(order, {"n1": 0}, 1))


# ---- claims end to end -------------------------------------------------------------------------

def test_known_effects(world):
    df = run(world, {
        "H": hay_claim(["good__gene"], ["bad__gene"]),
        "V": {"data": "v2", "test": "mcnemar", "a": "llm-z", "b": "emb-x"},
        "S": {"data": "suite", "test": "signflip_exact", "a": "m-a", "b": "m-b"},
    })
    h, v, s = df.loc["H"], df.loc["V"], df.loc["S"]
    assert h.n == NQ and h.mean_a == pytest.approx(1.0) and h.effect > 0.3 and not h.exact
    assert h.p_raw == pytest.approx(1 / (1 + claims.N_PERM)) and h.decision == "a better"
    assert h.judged10_a == pytest.approx(1.0) and h.judged10_b == pytest.approx(1.0)
    assert h.score == "graded nDCG@10 (judgments_merged.parquet)"
    only_a, only_b = int((world.llm_hit & ~world.emb_hit).sum()), int((~world.llm_hit & world.emb_hit).sum())
    assert v.n == 40 and v.effect == pytest.approx((world.llm_hit.sum() - world.emb_hit.sum()) / 40)
    assert v.p_raw == pytest.approx(binomtest(only_a, only_a + only_b, 0.5).pvalue)
    assert f"only a right {only_a}, only b right {only_b}" in v.note
    assert s.n == 12 and s.exact and s.effect == pytest.approx(world.delta.mean())
    assert s.p_raw == pytest.approx(brute_sign_flip(world.delta))
    assert df.p_holm.tolist() == pytest.approx(holm(df.p_raw.to_numpy()))
    md = (world.tmp / "out" / "claims.md").read_text()
    assert md.startswith("| claim |") and "95% CI, unadjusted" in md and "not Holm-adjusted" in md


def test_seed_mean_arm(world):
    df = run(world, {"C5": hay_claim([f"ft-s{s}__gene" for s in (1, 2, 3)], ["ft-base__gene"])})
    r = df.loc["C5"]
    seeds = np.array([2 / math.log2(s + 1) / IDCG for s in (1, 2, 3)])  # same on every query
    base = 2 / math.log2(5) / IDCG
    assert r.n == NQ and r.mean_a == pytest.approx(seeds.mean()) and r.mean_b == pytest.approx(base)
    assert r.effect == pytest.approx(seeds.mean() - base)
    assert r.seed_sd_a == pytest.approx(seeds.std(ddof=1)) and np.isnan(r.seed_sd_b)
    assert r.judged10_a == pytest.approx(1.0)
    assert r.arm_a == "ft-s1__gene, ft-s2__gene, ft-s3__gene"


def test_self_grading_file_choice(world):
    hay = world.hay
    assert claims.labels_for(["good__gene", "bad__gene"], hay) == hay / "judgments_merged.parquet"
    assert claims.labels_for([LLM_DS, "bad__gene"], hay) == hay / "judgments_merged_no-deepseek.parquet"
    assert claims.labels_for(["good__gene", LLM_GM], hay) == hay / "judgments_merged_no-gemini.parquet"
    assert claims.labels_for([LLM_DS, LLM_GM], hay) is None
    save_run(hay, "llm-gemini-new__gene", ranked("q00", ["d0"]))  # not in the conflicts file: stale
    with pytest.raises(SystemExit, match="rerun llm/judge_pool.py"):
        claims.labels_for(["llm-gemini-new__gene", "good__gene"], hay)

    df = run(world, {"DS": hay_claim([LLM_DS], ["good__gene"]), "BOTH": hay_claim([LLM_DS], [LLM_GM])})
    assert df.loc["DS", "score"] == "graded nDCG@10 (judgments_merged_no-deepseek.parquet)"
    # good__gene ranks d0, d1, d2 first, perfect under every grade file
    assert df.loc["DS", "mean_b"] == pytest.approx(1.0)
    assert df.loc["BOTH", "score"] == "curator-only nDCG@10" and np.isnan(df.loc["BOTH", "judged10_a"])
    gold = pd.read_parquet(hay / "qrels.parquet").groupby("qid").pmid.agg(set).to_dict()
    curator = per_query(pd.read_parquet(hay / "runs" / f"{LLM_DS}.parquet"), gold)["ndcg@10"].mean()
    assert df.loc["BOTH", "mean_a"] == pytest.approx(curator)


def test_dropped_left_out_of_holm(world):
    df = run(world, {
        "A": hay_claim(["good__gene"], ["bad__gene"]),
        "V": {"data": "v2", "test": "mcnemar", "a": "llm-z", "b": "emb-y"},
        "S": {"data": "suite", "test": "signflip_exact", "a": "m-a", "b": "m-b"},
        "C4": {"question": "memory", "dropped": "fewer than 15 remembered"},
    })
    assert df.loc["C4", "decision"] == "dropped" and df.loc["C4", "note"] == "fewer than 15 remembered"
    assert df.loc["C4", ["p_raw", "p_holm"]].isna().all()
    ran = df.loc[["A", "V", "S"]]
    assert ran.p_holm.tolist() == pytest.approx(holm(ran.p_raw.to_numpy()))  # a family of 3, not 4
    assert "over the 3 claims run" in (world.tmp / "out" / "claims.md").read_text()


def test_not_run_refuses_real_run(world, capsys):
    claim_map = {
        "A": hay_claim(["good__gene"], ["bad__gene"]),
        "MISSING": hay_claim(["ft-s1__gene", "ft-s9__gene"], ["ft-base__gene"]),  # one seed absent
        "VMISS": {"data": "v2", "test": "mcnemar", "a": "llm-none", "b": "emb-x"},
    }
    run(world, claim_map, "--dry-run")  # lists them, passes
    out = capsys.readouterr().out
    assert "runs/ft-s9__gene.parquet" in out and "llm-none" in out
    assert "the real run refuses until MISSING, VMISS can run" in out
    with pytest.raises(SystemExit, match="not run: MISSING, VMISS"):
        run(world, claim_map)  # the one look: no partial family
    assert not (world.tmp / "out").exists()


def test_registered_family_required(world):
    with pytest.raises(SystemExit, match="registered family C1, C2, C3a"):
        run(world, {"A": hay_claim(["good__gene"], ["bad__gene"])}, "--dry-run", any_claims=False)


def test_ungraded_top10_stops(world, monkeypatch, capsys):
    """A run never pooled and graded (its top 10 mostly ungraded) stops the dry run and the real one."""
    save_run(world.hay, "new__gene", [r for q in world.qids for r in ranked(q, ["d0"] + [f"x{i}" for i in range(20)])])
    for flags in (("--dry-run",), ()):
        with pytest.raises(SystemExit, match=r"360 of new__gene's 400 top-10 pairs have no grade in judgments_merged"):
            run(world, {"N": hay_claim(["new__gene"], ["good__gene"])}, *flags)
    # one ungraded pair (q00, rank 10) of 400: over the 0.1% allowed, within 1%
    rows = [r for q in world.qids for r in ranked(q, [f"d{i}" for i in range(9)] + (["x0"] if q == "q00" else ["d9"]))]
    save_run(world.hay, "few__gene", rows)
    with pytest.raises(SystemExit, match="1 of few__gene's 400"):
        run(world, {"F": hay_claim(["few__gene"], ["good__gene"])}, "--dry-run")
    monkeypatch.setattr(claims, "MAX_UNGRADED", 0.01)
    run(world, {"F": hay_claim(["few__gene"], ["good__gene"])}, "--dry-run")
    assert "ungraded top-10 pairs 1/400, 0/400]" in capsys.readouterr().out
    df = run(world, {"F": hay_claim(["few__gene"], ["good__gene"])})
    assert df.loc["F", "judged10_a"] == pytest.approx(399 / 400) and df.loc["F", "judged10_b"] == 1


def test_int_ids_read_as_strings(tmp_path):
    """A run with int qids and PMIDs scores like the same run with str ones (grades and qrels are str)."""
    (tmp_path / "runs").mkdir()
    pd.DataFrame({"qid": 7, "pmid": [101, 102, 103], "rank": [1, 2, 3]}).to_parquet(tmp_path / "runs" / "int.parquet")
    gold, grades = {"7": {"101", "102"}}, {"7": {"101": 2, "102": 2, "103": 0}}
    graded, judged = claims.haystack_arm(["int"], tmp_path, gold, grades)
    assert graded.loc["7", "int"] == pytest.approx(1.0) and judged == 1.0
    assert claims.haystack_arm(["int"], tmp_path, gold, None)[0].loc["7", "int"] == pytest.approx(1.0)
    assert claims.ungraded_top(claims.read_run(tmp_path, "int"), grades, list(gold)) == (0, 3)


def test_dry_run_computes_no_statistic(world, monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("dry run computed a statistic")
    for f in ("sign_flip_p", "mcnemar_p", "bootstrap_ci", "compare", "graded_metrics", "per_query"):
        monkeypatch.setattr(claims, f, boom)
    run(world, {"H": hay_claim([LLM_DS], ["good__gene"]), "M": hay_claim(["nope__gene"], ["good__gene"]),
                "V": {"data": "v2", "test": "mcnemar", "a": "llm-z", "b": "emb-x"}}, "--dry-run")
    out = capsys.readouterr().out
    assert "[judgments_merged_no-deepseek.parquet; ungraded top-10 pairs 0/400, 0/400]" in out
    assert "not run" in out and "Holm over 2 claims: H, V" in out and "no statistic computed" in out
    assert not (world.tmp / "out").exists()


def test_dry_run_reports_problems(world):
    pd.read_parquet(world.hay / "qrels.parquet").assign(pmid="zz").to_parquet(world.hay / "qrels.parquet")  # ungraded gold
    with pytest.raises(SystemExit, match="gold PMIDs of judged queries are not in judgments_merged.parquet"):
        run(world, {"H": hay_claim(["good__gene"], ["bad__gene"])}, "--dry-run")


@pytest.mark.parametrize("claim,msg", [
    ({"data": "v2", "test": "signflip", "a": "llm-z", "b": "emb-x"}, "use test 'mcnemar'"),
    ({"data": "v2", "test": "mcnemar", "a": ["llm-z", "emb-y"], "b": "emb-x"}, "only haystack arms"),
    ({"data": "haystack", "test": "signflip", "a": ["x"], "b": ["x"]}, "a and b share"),
    ({"data": "haystack", "test": "signflip", "a": ["x"]}, "b must be a name"),
    ({"data": "web", "test": "signflip", "a": "x", "b": "y"}, "data must be one of"),
    ({"data": "haystack", "test": "signflip", "a": ["TODO-best__gene"], "b": ["x"]}, "'TODO-best__gene' is a placeholder"),
    ({"question": "q", "dropped": None}, "dropped needs the reason"),
    ({"question": "q", "dropped": " "}, "dropped needs the reason"),
])
def test_config_errors(world, claim, msg):
    with pytest.raises(SystemExit, match=msg):
        run(world, {"X": claim}, "--dry-run")


def test_example_config_is_valid(tmp_path):
    """Valid but for its TODO- placeholders, which load_config rejects until filled."""
    example = ROOT / "docs" / "paper" / "claims_config.example.yaml"
    with pytest.raises(SystemExit) as e:
        claims.load_config(example)
    errors = str(e.value).splitlines()[1:]
    assert errors and all("is a placeholder" in x for x in errors)
    filled = tmp_path / "claims_config.yaml"
    filled.write_text(example.read_text().replace("TODO-", "picked-"))
    cfg = claims.load_config(filled)
    assert list(cfg["claims"]) == claims.REGISTERED
    assert "dropped" in cfg["claims"]["C4"]
