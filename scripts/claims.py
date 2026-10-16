#!/usr/bin/env python
"""Confirmatory tests of the PREREG claims (docs/PREREG.md, "The claims we test").

A YAML config (docs/paper/claims_config.example.yaml) gives each claim its question, data source,
test and two arms, a and b; effect = mean over units of a - b. The claim IDs must be the
registered family C1, C2, C3a, C3b, C4-C9 (--any-claims allows others, for tests). By data source:

  haystack  arms: lists of run file stems in <haystack>/runs/ (qid and pmid read as strings, like
            the grades and qrels). A list of several runs (a fine-tune's seeds) scores each
            query by their mean (PREREG, seeds in C5, C6, C8; seed_sd = SD of the seeds' mean
            scores). Unit = query, score = graded nDCG@10 (stats_haystack.graded_metrics) over
            llm/judge_pool.py's judgments_merged.parquet.
            No self-grading: if either arm holds a run that judgments_conflicts.parquet bars a
            judge from grading, both arms use judgments_merged_no-<judge>.parquet, or curator-only
            nDCG@10 (raw) if no judge is left. The bar goes by name, as judge_pool.py sets it: a
            run whose stem holds a judge's model family ('deepseek', 'gemini') is barred from that
            judge whether or not an LLM reranked it (wider than the PREREG wording; cautious).
            Each arm run's top 10 must be graded in the file used: at most MAX_UNGRADED (0.1%) of
            its top-10 (query, paper) pairs may lack a grade (the few a judge left out), else
            the script stops (pool and grade the run with judge_pool.py first).
            judged@10 = share of the arm's top-10 docs graded.
            Test 'signflip': two-sided paired sign-flip permutation, 10,000 random sign vectors
            (seed 0), p = (1 + #|sum| >= |obs|) / (1 + 10,000); exact over all 2^n vectors when
            2^n <= 10,000.
  v2        arms: an LLM (results/llm_v2/<name>/<batch>.parquet) or an embedder (a model in
            results/v2_perquery.parquet). Unit = pair, score = hit@1, batches 1-4 (0 is practice).
            Test 'mcnemar': exact McNemar, two-sided binomial test (p = 1/2) on discordant pairs.
  suite     arms: models in results/bio_scores.csv. Unit = task, score = the task score.
            Test 'signflip_exact': exact sign-flip over all 2^n sign vectors (4,096 for 12 tasks).

Arms must cover the same units. effect CI: 95% percentile paired bootstrap, 10,000 unit
resamples (seed 0), unadjusted (not Holm-corrected, so it can exclude 0 beside 'no difference
shown'). Holm-Bonferroni at alpha 0.05 over the claims run. A claim a PREREG rule drops (e.g. C4)
is written 'dropped: <reason>', reported, and left out of Holm. A claim with a missing arm (run
file, batch file or model absent) is 'not run', and the real run refuses while any claim is not
run: the run is the one look (a later rerun with more claims would be a second look with another
Holm bar). Arm names starting 'TODO-' are config errors. decision: 'a better' or 'b better' if
p_holm < 0.05, else 'no difference shown'.

--dry-run checks the config, the files, their columns, the units the arms cover, which grades
each haystack claim uses and how many of each arm run's top-10 pairs they leave ungraded, and
lists the claims not run; it computes no test statistic.

Run:  uv run python scripts/claims.py CONFIG [--dry-run] [--any-claims] [--out results/claims]
Output: <out>.csv (claim, question, data, test, arm_a, arm_b, score, n, mean_a, mean_b, effect,
        ci_low, ci_high, p_raw, exact, p_holm, decision, judged10_a, judged10_b, seed_sd_a,
        seed_sd_b, note) and <out>.md (the table)
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from stats_haystack import graded_metrics, holm, load_judgments, per_query

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {"haystack": "data/haystack/GeneDiseaseHaystack", "v2_embedders": "results/v2_perquery.parquet",
           "v2_llms": "results/llm_v2", "suite": "results/bio_scores.csv"}
TESTS = {"haystack": "signflip", "v2": "mcnemar", "suite": "signflip_exact"}  # PREREG, one per source
REGISTERED = ["C1", "C2", "C3a", "C3b", "C4", "C5", "C6", "C7", "C8", "C9"]  # PREREG, change log 2026-10-16
V2_BATCHES = [1, 2, 3, 4]
TOP_K = 10  # nDCG@10, and judge_pool.py pools each run's top 10
MAX_UNGRADED = 0.001  # share of an arm run's top-10 pairs allowed to lack a grade
N_PERM = N_BOOT = 10_000
SEED = 0
ALPHA = 0.05
CURATOR = "curator-only"


# ---- tests -------------------------------------------------------------------------------------

def sign_flip_p(d, n_perm: int = N_PERM, seed: int = SEED) -> tuple[float, bool]:
    """Two-sided paired sign-flip p for mean(d), and whether it is exact (all 2^n sign vectors,
    used when 2^n <= n_perm) rather than from n_perm seeded random ones."""
    d = np.asarray(d, dtype=float)
    obs = abs(d.sum()) - 1e-9  # tolerance as stats_haystack
    if 2 ** len(d) <= n_perm:
        signs = 1 - 2 * ((np.arange(2 ** len(d))[:, None] >> np.arange(len(d))) & 1)
        return float((np.abs(signs @ d) >= obs).mean()), True
    signs = np.random.default_rng(seed).choice([-1, 1], size=(n_perm, len(d))).astype(float)
    return float((1 + (np.abs(signs @ d) >= obs).sum()) / (1 + n_perm)), False


def mcnemar_p(a, b) -> tuple[float, int, int]:
    """Exact McNemar p (two-sided binomial, p = 1/2, on the discordant pairs) and the counts of
    pairs only a got right and only b got right."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail), only_a, only_b


def bootstrap_ci(d, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """95% percentile CI of mean(d), resampling units with replacement."""
    d = np.asarray(d, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


# ---- config ------------------------------------------------------------------------------------

def load_config(path: Path, any_claims: bool = False) -> dict:
    """The config, checked: arms become lists, sources absolute paths (relative ones from the repo).
    The claim IDs must be REGISTERED unless any_claims."""
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    errors = [f"unknown key {k!r}" for k in cfg if k not in ("sources", "merge_gold", "claims")]
    sources = SOURCES | (cfg.get("sources") or {})
    errors += [f"unknown source {k!r}" for k in sources if k not in SOURCES]
    claims = {}
    for cid, c in (cfg.get("claims") or {}).items():
        cid, c = str(cid), dict(c or {})
        errors += [f"{cid}: unknown key {k!r}" for k in c if k not in ("question", "data", "test", "a", "b", "dropped")]
        claims[cid] = c
        for arm in "ab":
            names = [c[arm]] if isinstance(c.get(arm), str) else c.get(arm) or []
            ok = isinstance(names, list) and all(isinstance(x, str) and x for x in names)
            if not ok or (not names and "dropped" not in c):
                errors.append(f"{cid}: {arm} must be a name or a list of names")
            c[arm] = names if ok else []
            errors += [f"{cid}: {arm} {x!r} is a placeholder; put the run picked on the practice set"
                       for x in c[arm] if x.startswith("TODO-")]
        if "dropped" in c:
            if not (isinstance(c["dropped"], str) and c["dropped"].strip()):
                errors.append(f"{cid}: dropped needs the reason (the PREREG rule that drops it)")
            continue
        if c.get("data") not in TESTS:
            errors.append(f"{cid}: data must be one of {', '.join(TESTS)}")
            continue
        if c.get("test") != TESTS[c["data"]]:
            errors.append(f"{cid}: {c['data']} claims use test {TESTS[c['data']]!r} (PREREG), not {c.get('test')!r}")
        if c["data"] != "haystack" and max(len(c["a"]), len(c["b"])) > 1:
            errors.append(f"{cid}: only haystack arms may list several runs (seeds)")
        if set(c["a"]) & set(c["b"]):
            errors.append(f"{cid}: a and b share {sorted(set(c['a']) & set(c['b']))}")
    if not claims:
        errors.append("no claims")
    elif not any_claims and set(claims) != set(REGISTERED):
        errors.append(f"claims must be the registered family {', '.join(REGISTERED)} (missing "
                      f"{sorted(set(REGISTERED) - set(claims))}, extra {sorted(set(claims) - set(REGISTERED))}); "
                      "--any-claims allows others")
    if errors:
        raise SystemExit("config errors:\n  " + "\n  ".join(errors))
    return {"sources": {k: ROOT / v for k, v in sources.items()},
            "merge_gold": bool(cfg.get("merge_gold", False)), "claims": claims}


# ---- data --------------------------------------------------------------------------------------

def columns_missing(path: Path, need: set[str]) -> list[str]:
    have = pq.read_schema(path).names if path.suffix == ".parquet" else pd.read_csv(path, nrows=0).columns
    return [f"{path.name} lacks columns {sorted(need - set(have))}"] if need - set(have) else []


def judge_family(model: str) -> str:
    """llm/judge_pool.py's conflict rule: a judge may not grade a run whose name holds the first word
    of its model slug (google/gemini-3.1-flash-lite -> gemini); judge_pool's short names for the
    judgments_merged_no-<judge> files are these words."""
    return model.split("/")[-1].split("-")[0]


def labels_for(stems: list[str], hay: Path) -> Path | None:
    """The merged grades a comparison of these runs uses: judgments_merged.parquet, or without each
    judge that judgments_conflicts.parquet bars from grading one of them; None (curator-only) if no
    judge is left."""
    conflicts = pd.read_parquet(hay / "judgments_conflicts.parquet")
    judges = set(pd.read_parquet(hay / "judgments.parquet", columns=["judge"]).judge)
    if not judges:
        raise SystemExit("judgments.parquet has no judges")
    for s in stems:
        named = {j for j in judges if judge_family(j) in s.lower()}
        listed = set(conflicts.judge[conflicts.system == s])
        if named != listed:
            raise SystemExit(f"judgments_conflicts.parquet lists {sorted(listed)} for {s}, but its name "
                             f"matches {sorted(named)}; rerun llm/judge_pool.py")
    barred = set(conflicts.judge[conflicts.system.isin(stems)])
    if barred >= judges:
        return None
    if len(barred) > 1:
        raise SystemExit(f"no merged grades without all of {sorted(barred)} (judge_pool.py drops one judge per file)")
    return hay / f"judgments_merged{''.join(f'_no-{judge_family(j)}' for j in barred)}.parquet"


def read_run(hay: Path, stem: str) -> pd.DataFrame:
    """A run's qid, pmid and rank; qid and pmid as str, like the grades and qrels (an int PMID
    would match no grade and score 0)."""
    run = pd.read_parquet(hay / "runs" / f"{stem}.parquet", columns=["qid", "pmid", "rank"])
    return run.astype({"qid": str, "pmid": str})


def ungraded_top(run: pd.DataFrame, grades: dict[str, dict[str, int]], qids: list[str]) -> tuple[int, int]:
    """How many of the run's top-TOP_K (qid, pmid) pairs over qids have no grade, out of how many."""
    top = run[run.qid.isin(qids)].sort_values("rank").groupby("qid").head(TOP_K)
    return sum(p not in grades.get(q, {}) for q, p in zip(top.qid, top.pmid)), len(top)


def haystack_arm(stems: list[str], hay: Path, gold: dict[str, set[str]],
                 grades: dict[str, dict[str, int]] | None) -> tuple[pd.DataFrame, float]:
    """Per-query nDCG@10 (graded, or curator-only if grades is None), one column per run, and the
    share of the runs' top-10 docs that are graded (over the queries scored)."""
    cols, judged, seen = {}, 0, 0
    for s in stems:
        run = read_run(hay, s)
        if missing := len(set(gold) - set(run.qid)):
            print(f"warning: {s} has no rows for {missing}/{len(gold)} queries (scored 0)")
        if grades is None:
            cols[s] = per_query(run, gold)["ndcg@10"]
            continue
        gm = graded_metrics(run, grades, thresholds=()).reindex(list(gold))
        cols[s] = gm["graded_ndcg@10"]
        kept = gm["graded_ndcg@10"].notna()
        judged, seen = judged + gm.loc[kept, "judged@10"].sum(), seen + gm.loc[kept, "n@10"].sum()
    return pd.DataFrame(cols), (judged / seen if seen else math.nan)


def v2_arm(name: str, src: dict[str, Path]) -> pd.Series:
    """hit@1 per pair over V2_BATCHES, for an LLM (results/llm_v2/<name>/) or an embedder."""
    llm = src["v2_llms"] / name
    if llm.is_dir():
        df = pd.concat([pd.read_parquet(llm / f"{b}.parquet", columns=["batch", "qid", "hit1"]) for b in V2_BATCHES])
    else:
        df = pd.read_parquet(src["v2_embedders"], columns=["model", "batch", "qid", "hit1"])
        df = df[df.model == name]
    df = df[df.batch.isin(V2_BATCHES)]
    if df.qid.duplicated().any():
        raise SystemExit(f"{name}: duplicate qids in batches {V2_BATCHES}")
    return df.set_index("qid").hit1.astype(bool).rename(name)


def suite_arm(name: str, path: Path) -> pd.Series:
    df = pd.read_csv(path)
    return df[df.model == name].set_index("task").score.rename(name)


def arm_missing(c: dict, src: dict[str, Path]) -> list[str]:
    """What is absent for this claim's arms ([] if both are there)."""
    names = c["a"] + c["b"]
    if c["data"] == "haystack":
        return [f"runs/{s}.parquet" for s in names if not (src["haystack"] / "runs" / f"{s}.parquet").exists()]
    if c["data"] == "suite":
        models = set(pd.read_csv(src["suite"], usecols=["model"]).model) if src["suite"].exists() else set()
        return [f"suite model {x}" for x in names if x not in models]
    emb = src["v2_embedders"]
    have = (pd.read_parquet(emb, columns=["model", "batch"]).groupby("model").batch.agg(set).to_dict()
            if emb.exists() else {})
    out = []
    for x in names:
        if (src["v2_llms"] / x).is_dir():
            out += [f"llm_v2/{x}/{b}.parquet" for b in V2_BATCHES if not (src["v2_llms"] / x / f"{b}.parquet").exists()]
        elif not set(V2_BATCHES) <= have.get(x, set()):
            out.append(f"v2 model {x} (batches {V2_BATCHES})")
    return out


def check(cfg: dict) -> tuple[dict[str, dict], list[str]]:
    """Per claim: status ('ready', 'not run', 'dropped'), note, and for haystack claims the grades
    file (None = curator-only) and, per arm run, (ungraded, all) top-10 pairs in it; plus the
    problems that stop a run. Computes no test statistic."""
    src, plan, problems = cfg["sources"], {}, []
    for cid, c in cfg["claims"].items():
        if "dropped" in c:
            plan[cid] = {"status": "dropped", "note": str(c["dropped"])}
            continue
        if missing := arm_missing(c, src):
            plan[cid] = {"status": "not run", "note": "missing: " + ", ".join(missing)}
            continue
        p = plan[cid] = {"status": "ready", "note": ""}
        try:
            if c["data"] == "haystack":
                hay = src["haystack"]
                bad = [f"{cid}: {e}" for s in c["a"] + c["b"]
                       for e in columns_missing(hay / "runs" / f"{s}.parquet", {"qid", "pmid", "rank"})]
                problems += bad
                p["labels"] = labels_for(c["a"] + c["b"], hay)
                if p["labels"] is not None and not p["labels"].exists():
                    problems.append(f"{cid}: {p['labels'].name} not found")
                elif p["labels"] is not None and not bad:
                    p["ungraded"] = {}  # filled below, once the grades are loaded
            elif c["data"] == "v2":
                a, b = v2_arm(c["a"][0], src), v2_arm(c["b"][0], src)
                if set(a.index) != set(b.index):
                    problems.append(f"{cid}: arms cover different pairs ({len(a)} vs {len(b)}, {len(set(a.index) & set(b.index))} shared)")
            else:
                problems += [f"{cid}: {e}" for e in columns_missing(src["suite"], {"model", "task", "score"})]
                a, b = suite_arm(c["a"][0], src["suite"]), suite_arm(c["b"][0], src["suite"])
                if set(a.index) != set(b.index):
                    problems.append(f"{cid}: arms cover different tasks ({len(a)} vs {len(b)})")
        except (SystemExit, OSError, KeyError, ValueError) as e:
            problems.append(f"{cid}: {e}")
    graded = [cid for cid in plan if "ungraded" in plan[cid]]
    if not graded:
        return plan, problems
    try:  # every haystack claim's grades cover the gold of the queries they judge
        gold = gold_sets(src["haystack"])
        grades = {path: load_judgments(path, gold, cfg["merge_gold"]) for path in {plan[cid]["labels"] for cid in graded}}
    except (SystemExit, OSError, AssertionError) as e:
        problems.append(f"haystack grades: {e}")
        return plan, problems
    for cid in graded:  # and each arm run's top 10: pooled and graded by llm/judge_pool.py
        p, c = plan[cid], cfg["claims"][cid]
        for s in c["a"] + c["b"]:
            try:
                u, n = p["ungraded"][s] = ungraded_top(read_run(src["haystack"], s), grades[p["labels"]], list(gold))
            except (OSError, KeyError, ValueError) as e:
                problems.append(f"{cid}: {s}: {e}")
                continue
            if u > MAX_UNGRADED * n:
                problems.append(f"{cid}: {u:,} of {s}'s {n:,} top-{TOP_K} pairs have no grade in {p['labels'].name} "
                                f"(at most {MAX_UNGRADED:.1%} may); pool and grade it with llm/judge_pool.py")
    return plan, problems


def gold_sets(hay: Path) -> dict[str, set[str]]:
    qrels = pd.read_parquet(hay / "qrels.parquet", columns=["qid", "pmid"]).astype({"qid": str, "pmid": str})
    return qrels.groupby("qid").pmid.agg(set).to_dict()


# ---- claims ------------------------------------------------------------------------------------

def compare(c: dict, cfg: dict, labels: Path | None) -> dict:
    src = cfg["sources"]
    out = {}
    if c["data"] == "haystack":
        gold = gold_sets(src["haystack"])
        grades = None if labels is None else load_judgments(labels, gold, cfg["merge_gold"])
        A, out["judged10_a"] = haystack_arm(c["a"], src["haystack"], gold, grades)
        B, out["judged10_b"] = haystack_arm(c["b"], src["haystack"], gold, grades)
        out["score"] = f"{CURATOR} nDCG@10" if labels is None else f"graded nDCG@10 ({labels.name})"
    elif c["data"] == "v2":
        A, B = (v2_arm(c[arm][0], src).astype(float).to_frame() for arm in "ab")
        out["score"] = f"hit@1, batches {V2_BATCHES[0]}-{V2_BATCHES[-1]}"
    else:
        A, B = (suite_arm(c[arm][0], src["suite"]).to_frame() for arm in "ab")
        out["score"] = "task score"
    if set(A.index) != set(B.index):
        raise SystemExit(f"arms cover different units ({len(A)} vs {len(B)})")
    X = pd.DataFrame({"a": A.mean(axis=1, skipna=False), "b": B.mean(axis=1, skipna=False).reindex(A.index)}).dropna()
    d = (X.a - X.b).to_numpy()
    if not len(d):
        raise SystemExit("no unit scored by both arms")
    if c["test"] == "mcnemar":
        p, only_a, only_b = mcnemar_p(X.a.astype(bool), X.b.astype(bool))
        exact, out["note"] = True, f"discordant: only a right {only_a}, only b right {only_b}"
    else:
        p, exact = sign_flip_p(d, n_perm=N_PERM if c["test"] == "signflip" else max(N_PERM, 2 ** len(d)))
    lo, hi = bootstrap_ci(d)
    for arm, M in (("a", A), ("b", B)):
        out[f"seed_sd_{arm}"] = M.loc[X.index].mean().std(ddof=1) if M.shape[1] > 1 else math.nan
    return out | {"n": len(d), "mean_a": X.a.mean(), "mean_b": X.b.mean(), "effect": d.mean(),
                  "ci_low": lo, "ci_high": hi, "p_raw": p, "exact": exact}


def run_claims(cfg: dict, plan: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for cid, c in cfg["claims"].items():
        p = plan[cid]
        row = {"claim": cid, "question": c.get("question", ""), "data": c.get("data"), "test": c.get("test"),
               "arm_a": ", ".join(c.get("a", [])), "arm_b": ", ".join(c.get("b", [])), "note": p["note"],
               "decision": p["status"]}
        if p["status"] == "ready":
            row |= compare(c, cfg, p.get("labels"))
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in ("n", "mean_a", "mean_b", "effect", "ci_low", "ci_high", "p_raw", "exact", "p_holm",
                "judged10_a", "judged10_b", "seed_sd_a", "seed_sd_b", "score"):
        if col not in df:
            df[col] = np.nan
    ran = df.decision == "ready"
    if ran.any():
        df.loc[ran, "p_holm"] = holm(df.loc[ran, "p_raw"].to_numpy(dtype=float))
        sig = ran & (df.p_holm < ALPHA)
        df.loc[ran, "decision"] = "no difference shown"
        df.loc[sig, "decision"] = np.where(df.loc[sig, "effect"] > 0, "a better", "b better")
    return df[["claim", "question", "data", "test", "arm_a", "arm_b", "score", "n", "mean_a", "mean_b",
               "effect", "ci_low", "ci_high", "p_raw", "exact", "p_holm", "decision", "judged10_a",
               "judged10_b", "seed_sd_a", "seed_sd_b", "note"]]


def markdown(df: pd.DataFrame) -> str:
    def fmt(r) -> list[str]:
        if pd.isna(r.p_raw):
            return [r.claim, r.question, r.arm_a, r.arm_b, "", "", "", "", "",
                    f"{r.decision} ({r.note})" if r.note else r.decision]
        return [r.claim, r.question, r.arm_a, r.arm_b, r.score, f"{int(r.n)}",
                f"{r.effect:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]", f"{r.p_raw:.4f}", f"{r.p_holm:.4f}",
                r.decision]
    head = ["claim", "question", "a", "b", "score", "n", "effect a - b [95% CI, unadjusted]", "p", "p (Holm)",
            "decision"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(x).replace("|", "/") for x in fmt(r)) + " |" for r in df.itertuples()]
    n_holm = int(df.p_holm.notna().sum())
    return "\n".join(lines) + (f"\n\nHolm-Bonferroni over the {n_holm} claims run, alpha {ALPHA}. Haystack: paired "
                               f"sign-flip ({N_PERM:,} sign vectors, exact if 2^n <= {N_PERM:,}); small task: exact "
                               "McNemar; suite: exact sign-flip over tasks. CI: paired bootstrap, "
                               f"{N_BOOT:,} resamples, not Holm-adjusted, so it can exclude 0 beside "
                               "'no difference shown'.\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="check config and files; compute no statistic")
    ap.add_argument("--any-claims", action="store_true",
                    help=f"allow claim IDs other than the registered family {', '.join(REGISTERED)} (tests)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "claims", help="writes <out>.csv and <out>.md")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.any_claims)
    plan, problems = check(cfg)
    ready = [cid for cid, p in plan.items() if p["status"] == "ready"]
    not_run = [cid for cid, p in plan.items() if p["status"] == "not run"]
    for cid, p in plan.items():
        c = cfg["claims"][cid]
        what = "" if p["status"] == "dropped" else f"{c['data']:8s} {' + '.join(c['a'])}  vs  {' + '.join(c['b'])}"
        grades = "" if "labels" not in p else f"  [{CURATOR if p['labels'] is None else p['labels'].name}"
        if p.get("ungraded"):
            grades += "; ungraded top-10 pairs " + ", ".join(f"{u:,}/{n:,}" for u, n in p["ungraded"].values())
        grades += "]" if grades else ""
        print(f"{cid:4s} {p['status']:8s} {what}{grades}{'  ' + p['note'] if p['note'] else ''}")
    print(f"Holm over {len(ready)} claims: {', '.join(ready)}")
    if problems:
        raise SystemExit("problems:\n  " + "\n  ".join(problems))
    if not_run and not args.dry_run:
        raise SystemExit(f"not run: {', '.join(not_run)}. The run is the one look, so every claim needs its runs; "
                         "write a claim a PREREG rule drops as 'dropped: <reason>'. Nothing computed.")
    if args.dry_run:
        print("dry run: config and files OK; "
              + (f"the real run refuses until {', '.join(not_run)} can run; " if not_run else "")
              + "no statistic computed")
        return

    df = run_claims(cfg, plan)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    md = markdown(df)
    args.out.with_suffix(".md").write_text(md)
    print(f"\n{md}\nwrote {args.out.with_suffix('.csv')} and .md")


if __name__ == "__main__":
    main()
