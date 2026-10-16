#!/usr/bin/env python
"""Reciprocal-rank fusion (RRF) of GeneDiseaseHaystack runs: $0, CPU only.

score(q, d) = sum over runs of 1 / (K + rank of d in that run), K = 60
(docs/PREREG.md); a doc missing from a run's top 1000 adds 0. The top 1000
per query are kept, ties broken by PMID.

--runs fuses the named systems into one run. --each-dense BASE fuses BASE
with every dense run in turn (the C1 first-pass candidates): a dense run is
an embedder's '<org>__<model>' run, not a fusion ('+') or a rerank ('-of-').

--exclude drops stale dense runs. The test fusions of 2026-10-15 excluded
ncbi__MedCPT (then the old mean-pool run); its CLS rerun keeps that name, so
don't exclude it once that exists.

Run:  uv run python scripts/rrf_haystack.py [--haystack GeneDiseaseHaystackDev] \
          --each-dense bm25-symbol_disease-k0.9-b0.75 [--exclude SYSTEM ...]
      uv run python scripts/rrf_haystack.py --runs A B [C ...]
Output: <haystack>/runs/rrf-<A>+<B>[+<C>...]__<track>.parquet (qid, pmid, score, rank)
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
K = 60
TOP_K = 1000


def rrf(files: list[Path], k: int = K, top_k: int = TOP_K) -> pd.DataFrame:
    return duckdb.execute(f"""
        WITH fused AS (
            SELECT qid, pmid, sum(1.0 / ({k} + rank)) AS score
            FROM read_parquet($files)
            GROUP BY ALL
        )
        SELECT qid, pmid, score,
               row_number() OVER (PARTITION BY qid ORDER BY score DESC, pmid) AS rank
        FROM fused
        QUALIFY rank <= {top_k}
        ORDER BY qid, rank
    """, {"files": [str(f) for f in files]}).df()


def is_dense(system: str) -> bool:
    return "__" in system and "+" not in system and "-of-" not in system


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--haystack", default="GeneDiseaseHaystack")
    ap.add_argument("--track", default="gene")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--runs", nargs="+", help="two or more systems to fuse into one run")
    mode.add_argument("--each-dense", metavar="BASE", help="fuse BASE with each dense run")
    ap.add_argument("--exclude", nargs="*", default=[], help="dense systems to leave out")
    args = ap.parse_args()
    runs = ROOT / "data" / "haystack" / args.haystack / "runs"

    if args.runs:
        assert len(args.runs) >= 2, "--runs needs two or more systems"
        combos = [args.runs]
    else:
        dense = sorted(s for f in runs.glob(f"*__{args.track}.parquet")
                       if is_dense(s := f.stem.rsplit("__", 1)[0]) and s not in args.exclude)
        combos = [[args.each_dense, d] for d in dense]
        print(f"{len(dense)} dense runs: {', '.join(dense) or 'none'}")

    for systems in combos:
        files = [runs / f"{s}__{args.track}.parquet" for s in systems]
        out = runs / f"rrf-{'+'.join(systems)}__{args.track}.parquet"
        fused = rrf(files)
        fused.to_parquet(out, index=False)
        print(f"-> {out.relative_to(ROOT)} ({fused.qid.nunique()} queries)", flush=True)


if __name__ == "__main__":
    main()
