#!/usr/bin/env python
"""Confidence intervals and significance for GeneDiseaseHaystack runs.

Runs in data/haystack/<NAME>/runs/*.parquet (filtered by --runs) are scored
per query with score_haystack.ranked_metrics (one query at a time), raw and
condensed exactly as score_haystack.py does: recall@10, recall@100, nDCG@10
over the curated gold. With --judgments (qid, pmid, grade 0/1/2; parquet or
csv, e.g. llm/judge_pool.py's judgments_merged.parquet) the raw runs also get,
as variant 'pooled' (stats_haystack.graded_metrics):
  graded_ndcg@10   gains = grade, ideal DCG from all graded docs of the query
                   (queries with no positive grade dropped)
  p@10_ge<t>       share of the top 10 with grade >= t
  bpref_ge<t>      bpref as trec_eval computes it (denominator min(R, N),
                   not the R of Buckley & Voorhees 2004), grade >= t
                   relevant (queries with no doc at grade >= t dropped)
for t = --threshold (default 1) and t = 2. Unjudged docs are never relevant
(gain 0); each pooled row reports judged@10, the share of the run's top-10
docs that are judged, and unjudged@10, their count, over the queries kept.

Per (variant, metric), over queries:
  ci_*        percentile bootstrap 95% CI, 10k query resamples
  gene_ci_*   the same resampling genes (all of a gene's queries together)
  p_*         paired sign-flip permutation test over queries, --n-perm sign
              vectors shared by all pairs, p = (1 + #|perm| >= |obs|) / (1 + n),
              two-sided; p_mcse is its Monte Carlo standard error
  *_vs_best   tests of each system against the top system on that metric,
              Holm-corrected across that set; tied_with_best = p_holm >= 0.05;
              borderline = the call flips if every p moves by 2 MCSE
  letters_all_pairs
              compact letter display (insert-absorb, Piepho 2004) from all
              pairwise tests, Holm-corrected across all pairs; systems that
              share a letter are not significantly different. This is a
              different, bigger family than vs-best, so the two can disagree:
              put one or the other in a table, not both
All resampling is seeded (numpy default_rng, seed 0).

The comparison family is the set of runs selected, so Holm values move when
runs are added: fix it with --runs for any table. These are exploratory; the
confirmatory family is PREREG C1-C9.

Run:  uv run python scripts/stats_haystack.py [--haystack NAME] [--runs GLOB ...]
          [--n-perm N] [--judgments PATH [--merge-gold] [--threshold 1|2]] [--out DIR]
Output: results/haystack/stats.csv, stats_pairs.csv for the test haystack,
        results/haystack/<NAME>/ for any other
"""

import argparse
import fnmatch
import itertools
import math
import string
from pathlib import Path

import numpy as np
import pandas as pd

from score_haystack import ranked_metrics

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = "GeneDiseaseHaystack"
METRICS = ["recall@10", "recall@100", "ndcg@10"]
N_BOOT = 10_000
N_PERM = 10_000  # PREREG: 10,000 permutations for the confirmatory sign-flip tests
CHUNK = 10_000
SEED = 0
ALPHA = 0.05


def condense(run: pd.DataFrame, drop: set[tuple[str, str]]) -> pd.DataFrame:
    """score_haystack.py's condensed run: silver-not-gold PMIDs removed, re-ranked."""
    keep = [(q, p) not in drop for q, p in zip(run.qid, run.pmid)]
    out = run[keep].copy()
    out["rank"] = out.groupby("qid")["rank"].rank(method="first")
    return out


def per_query(run: pd.DataFrame, gold: dict[str, set[str]]) -> pd.DataFrame:
    groups = dict(tuple(run.groupby("qid")))
    rows = [ranked_metrics(groups.get(q, run.iloc[:0]), {q: g}) for q, g in gold.items()]
    return pd.DataFrame(rows, index=list(gold))[METRICS]


def load_judgments(path: Path, gold: dict[str, set[str]], merge_gold: bool) -> dict[str, dict[str, int]]:
    """qid -> pmid -> grade; every gold PMID of a judged query must be judged (or merged as 2)."""
    j = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype={"qid": str, "pmid": str})
    j = j.astype({"qid": str, "pmid": str, "grade": int})[["qid", "pmid", "grade"]]
    assert j.grade.isin([0, 1, 2]).all(), "grades must be 0/1/2"
    dup = j.duplicated(["qid", "pmid"]).sum()
    assert not dup, f"{dup} duplicate (qid, pmid) rows in {path.name}; resolve judges first"
    g = pd.DataFrame([(q, p) for q, ps in gold.items() for p in ps], columns=["qid", "pmid"])
    missing = (g[g.qid.isin(j.qid)].merge(j, how="left", indicator=True)
               .query("_merge == 'left_only'")[["qid", "pmid"]])
    if len(missing) and not merge_gold:
        raise SystemExit(f"{len(missing)} gold PMIDs of judged queries are not in {path.name}; "
                         "add them or pass --merge-gold to count them as grade 2")
    if len(missing):
        print(f"merged {len(missing)} unjudged gold PMIDs as grade 2")
        j = pd.concat([j, missing.assign(grade=2)])
    return {q: dict(zip(g.pmid, g.grade)) for q, g in j.groupby("qid")}


def bpref(ranked: list[str], grades: dict[str, int], threshold: int) -> float:
    """bpref as trec_eval computes it (so pytrec_eval and ir_measures agree), grade >= threshold
    relevant: 1/R times the sum, over retrieved relevant docs r, of 1 - min(n_r, M) / M, where
    n_r = judged non-relevant docs ranked above r, M = min(R, N), R and N = judged relevant and
    non-relevant docs; unjudged docs are skipped. NaN if R = 0. Buckley & Voorhees (2004) use R
    for M; the two agree whenever N >= R."""
    R = sum(x >= threshold for x in grades.values())
    if not R:
        return math.nan
    M = min(R, len(grades) - R)
    above, total = 0, 0.0
    for p in ranked:
        if p not in grades:
            continue
        if grades[p] >= threshold:
            total += 1 - min(above, M) / M if above else 1.0
        else:
            above += 1
    return total / R


def graded_metrics(run: pd.DataFrame, grades: dict[str, dict[str, int]], k: int = 10,
                   thresholds: tuple[int, ...] = (1, 2)) -> pd.DataFrame:
    """Per judged query: graded_ndcg@k (gain = grade, ideal DCG from all the query's graded docs;
    NaN if none is above 0), and per threshold t, p@k_ge<t> (top-k docs with grade >= t, over k)
    and bpref_ge<t> (whole run; both NaN if no doc has grade >= t). Unjudged docs are never
    relevant (gain 0); judged@k and n@k count the judged and all docs of the run's top k."""
    ranked = {q: g.sort_values("rank").pmid.tolist() for q, g in run.groupby("qid")}
    rows = {}
    for q, g in grades.items():
        full = ranked.get(q, [])
        top = full[:k]
        idcg = sum(x / math.log2(i + 2) for i, x in enumerate(sorted(g.values(), reverse=True)[:k]))
        dcg = sum(g.get(p, 0) / math.log2(i + 2) for i, p in enumerate(top))
        row = {f"graded_ndcg@{k}": dcg / idcg if idcg else math.nan,
               f"judged@{k}": sum(p in g for p in top), f"n@{k}": len(top)}
        for t in thresholds:
            has_rel = any(x >= t for x in g.values())
            row[f"p@{k}_ge{t}"] = sum(g.get(p, 0) >= t for p in top) / k if has_rel else math.nan
            row[f"bpref_ge{t}"] = bpref(full, g, t)
        rows[q] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.maximum.accumulate(np.minimum(1, (len(p) - np.arange(len(p))) * p[order]))
    out = np.empty_like(adj)
    out[order] = adj
    return out


def letter_display(systems: list[str], sig: list[tuple[str, str]]) -> dict[str, str]:
    """Compact letter display; systems are ordered best first, so 'a' marks the top group."""
    cols = [set(systems)]
    for a, b in sig:
        cols = [part for c in cols for part in ([c - {a}, c - {b}] if a in c and b in c else [c])]
        cols = [c for i, c in enumerate(cols)
                if not any(c < o or (c == o and j < i) for j, o in enumerate(cols) if j != i)]
    cols.sort(key=lambda c: min(systems.index(s) for s in c))
    names = string.ascii_lowercase + string.ascii_uppercase
    return {s: "".join(names[k] for k, c in enumerate(cols) if s in c) for s in systems}


def summarise(X: pd.DataFrame, Wq: np.ndarray, Wg: np.ndarray, signs: np.ndarray):
    """X: queries x runs. Returns per-run and per-pair frames, runs ordered best first."""
    valid = X.notna().all(axis=1).to_numpy()
    X = X.loc[valid, X.mean().sort_values(ascending=False).index]
    runs, V = list(X.columns), X.to_numpy()

    def boot(W):
        W = W[:, valid]
        return (W @ V) / W.sum(axis=1, keepdims=True)

    bq, bg = boot(Wq), boot(Wg)
    pairs = list(itertools.combinations(range(len(runs)), 2))
    D = np.stack([V[:, i] - V[:, j] for i, j in pairs], axis=1)
    obs = np.abs(D.sum(axis=0))
    exceed = sum((np.abs(signs[i:i + CHUNK, valid].astype(float) @ D) >= obs - 1e-9).sum(axis=0)
                 for i in range(0, len(signs), CHUNK))
    p = (1 + exceed) / (1 + len(signs))
    se = np.sqrt(p * (1 - p) / len(signs))
    vs_best = np.array([i == 0 for i, _ in pairs])

    def ci(b, q):
        return [np.percentile(b[:, i] - b[:, j], q) for i, j in pairs]

    pair_df = pd.DataFrame({
        "run_a": [runs[i] for i, _ in pairs], "run_b": [runs[j] for _, j in pairs],
        "mean_diff": D.mean(axis=0),
        "diff_ci_low": ci(bq, 2.5), "diff_ci_high": ci(bq, 97.5),
        "diff_gene_ci_low": ci(bg, 2.5), "diff_gene_ci_high": ci(bg, 97.5),
        "p_value": p, "p_mcse": se, "p_holm": holm(p), "vs_best": vs_best,
    })
    pb, sb = p[vs_best], se[vs_best]
    pair_df.loc[vs_best, "p_holm_vs_best"] = holm(pb)
    pair_df.loc[vs_best, "borderline"] = (holm(np.maximum(pb - 2 * sb, 0)) < ALPHA) != (holm(pb + 2 * sb) < ALPHA)

    sig = [(a, b) for a, b, q in zip(pair_df.run_a, pair_df.run_b, pair_df.p_holm) if q < ALPHA]
    letters = letter_display(runs, sig)
    best = pair_df[vs_best].set_index("run_b")
    rest = runs[1:]
    run_df = pd.DataFrame({
        "run": runs, "n_queries": int(valid.sum()), "mean": V.mean(axis=0),
        "ci_low": np.percentile(bq, 2.5, axis=0), "ci_high": np.percentile(bq, 97.5, axis=0),
        "gene_ci_low": np.percentile(bg, 2.5, axis=0), "gene_ci_high": np.percentile(bg, 97.5, axis=0),
        "letters_all_pairs": [letters[r] for r in runs],
        "diff_vs_best": [0.0] + [-best.mean_diff[r] for r in rest],
        "p_vs_best": [np.nan] + [best.p_value[r] for r in rest],
        "p_mcse_vs_best": [np.nan] + [best.p_mcse[r] for r in rest],
        "p_holm_vs_best": [np.nan] + [best.p_holm_vs_best[r] for r in rest],
        "borderline": [False] + [bool(best.borderline[r]) for r in rest],
    })
    run_df.insert(run_df.columns.get_loc("borderline"), "tied_with_best",
                  run_df.p_holm_vs_best.isna() | (run_df.p_holm_vs_best >= ALPHA))
    return run_df, pair_df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--haystack", default=DEFAULT)
    ap.add_argument("--runs", nargs="+", default=["*"], help="globs on run file stems; fixes the comparison family")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--judgments", type=Path, help="qid, pmid, grade (0/1/2) for graded nDCG@10")
    ap.add_argument("--merge-gold", action="store_true", help="count gold PMIDs missing from --judgments as grade 2")
    ap.add_argument("--threshold", type=int, default=1, choices=[1, 2],
                    help="grade >= this is relevant for p@10 and bpref (2 is reported as well)")
    ap.add_argument("--out", type=Path, help="output directory (default results/haystack[/<NAME>])")
    args = ap.parse_args()

    hay = ROOT / "data" / "haystack" / args.haystack
    out = args.out or ROOT / "results" / "haystack" / ("" if args.haystack == DEFAULT else args.haystack)
    qrels = pd.read_parquet(hay / "qrels.parquet")
    silver = pd.read_parquet(hay / "silver.parquet")
    queries = pd.read_parquet(hay / "queries.parquet")
    gold = qrels.groupby("qid").pmid.agg(set).to_dict()
    qids = list(gold)
    unlabelled = silver.merge(qrels, how="left", indicator=True).query("_merge == 'left_only'")
    drop = set(zip(unlabelled.qid, unlabelled.pmid))
    grades = load_judgments(args.judgments, gold, args.merge_gold) if args.judgments else None

    files = [f for f in sorted((hay / "runs").glob("*.parquet"))
             if any(fnmatch.fnmatchcase(f.stem, g) for g in args.runs)]
    thresholds = tuple(sorted({args.threshold, 2}))
    graded = ["graded_ndcg@10"] + [f"{m}_ge{t}" for t in thresholds for m in ("p@10", "bpref")]
    per: dict[tuple[str, str], dict[str, pd.Series]] = {}
    judged: dict[tuple[str, str], tuple[float, int]] = {}  # (metric, run) -> judged@10, unjudged@10
    for f in files:
        try:
            run = pd.read_parquet(f)
        except Exception as e:  # a run still being written
            print(f"skip {f.name}: {e}")
            continue
        missing = len(set(qids) - set(run.qid))
        if missing:
            print(f"warning: {f.stem} has no rows for {missing}/{len(qids)} queries (scored 0)")
        for variant, r in (("raw", run), ("condensed", condense(run, drop))):
            pq = per_query(r, gold)
            for m in METRICS:
                per.setdefault((variant, m), {})[f.stem] = pq[m]
        if grades is not None:
            gm = graded_metrics(run, grades, thresholds=thresholds)
            for m in graded:
                j, n = gm.loc[gm[m].notna(), ["judged@10", "n@10"]].sum()
                judged[m, f.stem] = (j / max(n, 1), int(n - j))
                per.setdefault(("pooled", m), {})[f.stem] = gm[m].reindex(qids)
    n_runs = len(per[("raw", METRICS[0])])
    print(f"{n_runs} runs in the family: {', '.join(per[('raw', METRICS[0])])}")
    if math.comb(n_runs, 2) / (args.n_perm + 1) >= ALPHA:
        print(f"warning: with {n_runs} runs and {args.n_perm} permutations no all-pairs Holm p can be < {ALPHA}")

    rng = np.random.default_rng(SEED)
    n = len(qids)
    gene_idx = pd.factorize(queries.set_index("qid").targetId.reindex(qids))[0]
    n_genes = gene_idx.max() + 1
    Wq = rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT).astype(float)
    Wg = rng.multinomial(n_genes, np.full(n_genes, 1 / n_genes), size=N_BOOT)[:, gene_idx].astype(float)
    signs = np.concatenate([rng.choice([-1, 1], size=(min(CHUNK, args.n_perm - i), n)).astype(np.int8)
                            for i in range(0, args.n_perm, CHUNK)])

    stats, pairs = [], []
    for (variant, metric), cols in per.items():
        run_df, pair_df = summarise(pd.DataFrame(cols), Wq, Wg, signs)
        if variant == "pooled":
            run_df["judged@10"], run_df["unjudged@10"] = zip(*(judged[metric, r] for r in run_df.run))
        stats.append(run_df.assign(variant=variant, metric=metric))
        pairs.append(pair_df.assign(variant=variant, metric=metric))

    def split(df: pd.DataFrame, col: str, sfx: str = "") -> pd.DataFrame:
        parts = df[col].str.rsplit("__", n=1, expand=True)
        return df.assign(**{f"system{sfx}": parts[0], f"track{sfx}": parts[1]}).drop(columns=col)

    stats_df = split(pd.concat(stats), "run").assign(haystack=args.haystack, n_perm=args.n_perm)
    pairs_df = split(split(pd.concat(pairs), "run_a", "_a"), "run_b", "_b").assign(haystack=args.haystack,
                                                                                    n_perm=args.n_perm)
    lead = ["haystack", "variant", "metric"]
    stats_df = stats_df[lead + ["system", "track"] + [c for c in stats_df if c not in lead + ["system", "track"]]]
    if grades is not None:  # last, so the existing columns keep their places
        stats_df["unjudged@10"] = stats_df.pop("unjudged@10")
    pairs_df = pairs_df[lead + ["system_a", "track_a", "system_b", "track_b"]
                        + [c for c in pairs_df if c not in lead + ["system_a", "track_a", "system_b", "track_b"]]]

    out.mkdir(parents=True, exist_ok=True)
    stats_df.round(4).to_csv(out / "stats.csv", index=False)
    pairs_df.round(4).to_csv(out / "stats_pairs.csv", index=False)

    for (variant, metric), g in stats_df.groupby(["variant", "metric"], sort=False):
        print(f"\n== {variant} {metric} (n={g.n_queries.iloc[0]}; 95% CI query / gene; letters; Holm p vs best)")
        for r in g.itertuples():
            call = "best" if np.isnan(r.p_holm_vs_best) else f"{r.p_holm_vs_best:.4f}"
            call += " tied" if r.tied_with_best and not np.isnan(r.p_holm_vs_best) else ""
            call += " borderline" if r.borderline else ""
            print(f"  {r.system:42s} {r.mean:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] "
                  f"[{r.gene_ci_low:.3f}, {r.gene_ci_high:.3f}]  {r.letters_all_pairs:6s} {call}")


if __name__ == "__main__":
    main()
