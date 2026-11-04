#!/usr/bin/env python
"""Read-out of the no-test-disease fine-tunes (PREREG change log 2026-11-04, check 2).

The chosen recipe retrained on data/finetune_gd/no_test_disease/ (scripts/gcp_jobs/hs-ntd.txt),
against the matching seed-42 model on the test haystack. Per arm pair, the effect is
mean(no-test-disease - seed 42) over queries with a paired percentile bootstrap 95% interval
(10,000 resamples, seed 0; scripts/claims.py's bootstrap_ci), on:
  graded nDCG@10      judgments_merged.parquet (both judges)
  DeepSeek-only       judgments_merged_no-gemini.parquet (cross-encoder only: the grades C8 uses
                      beside the Gemini rerank)
  curator-only        nDCG@10 over the curated gold
No p-values (registered). Rule: a drop (seed 42 - no-test-disease) of 0.02 or less on graded
nDCG@10 reads "the gain does not depend on test diseases"; a larger drop is a limitation of C5,
C6 and C8. Context rows (not registered, descriptive): each no-test-disease model against the
C5, C6 and C8 comparators, on the grades those claims use.

Each arm run's top 10 must be graded first (llm/judge_pool.py --haystack GeneDiseaseHaystack);
the script stops if more than claims.py's MAX_UNGRADED share is not.

Run:    uv run python scripts/ntd_readout.py
Output: results/ntd_readout.csv and .md
"""

from pathlib import Path

import pandas as pd

from claims import CURATOR, MAX_UNGRADED, SOURCES, bootstrap_ci, gold_sets, haystack_arm, read_run, ungraded_top
from stats_haystack import load_judgments

ROOT = Path(__file__).resolve().parent.parent
HAY = ROOT / SOURCES["haystack"]
FIRST = "rrf-bm25-symbol_disease-k0.9-b0.75+BAAI__bge-base-en-v1.5"
BI_NTD, BI_S42 = "bioembed__bge-base-gd-ntd-s42__gene", "bioembed__bge-base-gd-s42__gene"
CE_NTD = f"ce-bioembed__medcpt-ce-gd-ntd-s42-top100-of-{FIRST}__gene"
CE_S42 = f"ce-bioembed__medcpt-ce-gd-s42-top100-of-{FIRST}__gene"
GEMINI = f"llm-gemini-3.1-flash-lite-top20-of-{FIRST}__gene"
BOTH, NO_GEMINI = "judgments_merged.parquet", "judgments_merged_no-gemini.parquet"
RULE = 0.02

# (role, a, b, grades file or None for curator-only)
ROWS = [
    ("registered", BI_NTD, BI_S42, BOTH),
    ("registered", BI_NTD, BI_S42, None),
    ("registered", CE_NTD, CE_S42, BOTH),
    ("registered", CE_NTD, CE_S42, NO_GEMINI),
    ("registered", CE_NTD, CE_S42, None),
    ("context (C5 base)", BI_NTD, "BAAI__bge-base-en-v1.5__gene", BOTH),
    ("context (C6 BiCA-base)", BI_NTD, "bisectgroup__BiCA-base__gene", BOTH),
    ("context (C8 Gemini rerank)", CE_NTD, GEMINI, NO_GEMINI),
]


def main() -> None:
    gold = gold_sets(HAY)
    grades = {f: load_judgments(HAY / f, gold, merge_gold=False) for f in (BOTH, NO_GEMINI)}
    for f, g in grades.items():
        for s in (BI_NTD, CE_NTD):
            miss, n = ungraded_top(read_run(HAY, s), g, list(gold))
            if miss > MAX_UNGRADED * n:
                raise SystemExit(f"{s}: {miss}/{n} top-10 pairs ungraded in {f}; run llm/judge_pool.py first")
    rows = []
    for role, a, b, f in ROWS:
        g = None if f is None else grades[f]
        A, ja = haystack_arm([a], HAY, gold, g)
        B, jb = haystack_arm([b], HAY, gold, g)
        X = pd.DataFrame({"a": A[a], "b": B[b]}).dropna()
        d = (X.a - X.b).to_numpy()
        lo, hi = bootstrap_ci(d)
        rows.append({"role": role, "arm_a": a, "arm_b": b, "score": CURATOR if f is None else f,
                     "n": len(d), "mean_a": X.a.mean(), "mean_b": X.b.mean(), "effect": d.mean(),
                     "ci_low": lo, "ci_high": hi, "judged10_a": ja, "judged10_b": jb})
    df = pd.DataFrame(rows)
    reg = df[(df.role == "registered") & (df.score != CURATOR)]
    df["rule"] = ""
    df.loc[reg.index, "rule"] = ["within 0.02" if -e <= RULE else "drop > 0.02" for e in reg.effect]
    out = ROOT / "results" / "ntd_readout"
    df.to_csv(out.with_suffix(".csv"), index=False)
    md = ["| role | a | b | score | n | mean a | mean b | a - b [95% CI] | rule |", "|---" * 9 + "|"]
    md += [f"| {r.role} | {r.arm_a} | {r.arm_b} | {r.score} | {r.n} | {r.mean_a:.3f} | {r.mean_b:.3f} | "
           f"{r.effect:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {r.rule} |" for r in df.itertuples()]
    md.append("\nPaired bootstrap over queries, 10,000 resamples, seed 0; no p-values (PREREG 2026-11-04). "
              f"Rule: a drop of {RULE} or less on graded nDCG@10 (each set of grades) reads 'does not depend on test diseases'.")
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
