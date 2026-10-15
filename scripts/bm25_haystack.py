#!/usr/bin/env python
"""BM25 baseline for GeneDiseaseHaystack (gene-filtered track): $0, CPU only.

Index = the 1.7M gene-tagged docs (docs_gene.parquet), English stopwords and
Snowball stemming (bm25s defaults otherwise). Each query's BM25 scores are
read over its own PubTator gene candidates only, as for the embedders.

Run:  uv run python scripts/bm25_haystack.py
Output: <haystack>/runs/bm25__gene.parquet (qid, pmid, score, rank)
"""

from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
import Stemmer

HAYSTACK = Path(__file__).resolve().parent.parent / "data" / "haystack" / "GeneDiseaseHaystack"
TOP_K = 1000


def main() -> None:
    docs = pd.read_parquet(HAYSTACK / "docs_gene.parquet")
    queries = pd.read_parquet(HAYSTACK / "queries.parquet")
    cands = pd.read_parquet(HAYSTACK / "gene_pmids.parquet")
    row = pd.Series(docs.index, index=docs.pmid)

    stemmer = Stemmer.Stemmer("english")
    index = bm25s.BM25()
    index.index(bm25s.tokenize(docs.text.tolist(), stopwords="en", stemmer=stemmer))
    qtok = bm25s.tokenize(queries.text.tolist(), stopwords="en", stemmer=stemmer,
                          return_ids=False)
    qtok = dict(zip(queries.qid, qtok))

    rows = []
    for qid, group in cands.groupby("qid", sort=True):
        idx = row[group.pmid].to_numpy()
        s = index.get_scores(qtok[qid])[idx]
        top = np.argsort(-s, kind="stable")[:TOP_K]
        rows += [(qid, docs.pmid.iat[idx[j]], float(s[j]), r + 1) for r, j in enumerate(top)]
    out = HAYSTACK / "runs" / "bm25__gene.parquet"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=["qid", "pmid", "score", "rank"]).to_parquet(out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
