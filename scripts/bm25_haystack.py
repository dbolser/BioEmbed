#!/usr/bin/env python
"""BM25 baseline for GeneDiseaseHaystack (gene-filtered track): $0, CPU only.

Index = the gene-tagged docs (docs_gene.parquet), English stopwords and
Snowball stemming (bm25s defaults otherwise). Each query's BM25 scores are
read over its own PubTator gene candidates only, as for the embedders.

Query formats (--query-format):
  full                '{SYMBOL} ({gene name}) in {disease}', the task's query text
  symbol_disease      '{SYMBOL} {disease}'
  symbol_disease_syn  SYMBOL, its HGNC alias and previous symbols, the disease
                      name and its OT exact synonyms (distinct strings)

Every combination of --query-format, --k1 and --b is run: docs are tokenised
once and indexed once per (k1, b). Pick the config on the dev haystack only
(docs/PREREG.md).

Run:  uv run python scripts/bm25_haystack.py --haystack GeneDiseaseHaystackDev \
          --query-format full symbol_disease symbol_disease_syn --k1 0.9 1.2 1.5 --b 0.4 0.75
Output: <haystack>/runs/bm25-<format>-k<k1>-b<b>__gene.parquet (qid, pmid, score,
rank). The v1.3 runs/bm25__gene.parquet (full, k1 1.5, b 0.75) is not rewritten.
"""

import argparse
import time
from pathlib import Path

import bm25s
import duckdb
import numpy as np
import pandas as pd
import Stemmer

ROOT = Path(__file__).resolve().parent.parent
IUK = "/outsee5/home/danbolser/Work/IUK-IPF/data"
OT = f"{IUK}/sources/opentargets/26.06"
HGNC = f"{IUK}/sources/hgnc/2026-10-14/hgnc_complete_set.txt"
FORMATS = ["full", "symbol_disease", "symbol_disease_syn"]
TOP_K = 1000


def query_texts(haystack: Path, fmt: str) -> dict[str, str]:
    q = duckdb.sql(f"""
        SELECT q.qid, q.text, t.approvedSymbol AS symbol, d.name AS disease,
               coalesce(d.exactSynonyms, []) AS disease_syn, g.gene_syn
        FROM '{haystack / "queries.parquet"}' q
        JOIN '{OT}/target/*.parquet' t ON t.id = q.targetId
        JOIN '{OT}/disease/*.parquet' d ON d.id = q.diseaseId
        LEFT JOIN (
            SELECT o.id AS targetId,
                   string_agg(concat_ws('|', h.alias_symbol, h.prev_symbol), '|') AS gene_syn
            FROM '{OT}/target/*.parquet' o, unnest(o.dbXrefs) x,
                 read_csv('{HGNC}', delim='\t', all_varchar=true) h
            WHERE x.unnest.source = 'HGNC' AND h.hgnc_id = 'HGNC:' || x.unnest.id
            GROUP BY 1
        ) g USING (targetId)
        ORDER BY q.qid
    """).df()
    if fmt == "full":
        return dict(zip(q.qid, q.text))
    if fmt == "symbol_disease":
        return {r.qid: f"{r.symbol} {r.disease}" for r in q.itertuples()}
    texts = {}
    for r in q.itertuples():
        terms = [r.symbol, *(r.gene_syn or "").split("|"), r.disease, *r.disease_syn]
        texts[r.qid] = " ".join(dict.fromkeys(t.lower() for t in terms if t))
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--haystack", default="GeneDiseaseHaystack")
    ap.add_argument("--query-format", nargs="+", choices=FORMATS, default=["full"])
    ap.add_argument("--k1", nargs="+", type=float, default=[1.5])
    ap.add_argument("--b", nargs="+", type=float, default=[0.75])
    args = ap.parse_args()
    haystack = ROOT / "data" / "haystack" / args.haystack

    docs = pd.read_parquet(haystack / "docs_gene.parquet")
    cands = pd.read_parquet(haystack / "gene_pmids.parquet")
    row = pd.Series(docs.index, index=docs.pmid)
    groups = [(qid, row[g.pmid].to_numpy()) for qid, g in cands.groupby("qid", sort=True)]

    stemmer = Stemmer.Stemmer("english")
    t0 = time.time()
    corpus = bm25s.tokenize(docs.text.tolist(), stopwords="en", stemmer=stemmer,
                            show_progress=False)
    print(f"tokenised {len(docs):,} docs ({time.time() - t0:.0f}s)", flush=True)
    qtok = {}
    for fmt in args.query_format:
        texts = query_texts(haystack, fmt)
        toks = bm25s.tokenize(list(texts.values()), stopwords="en", stemmer=stemmer,
                              return_ids=False, show_progress=False)
        qtok[fmt] = dict(zip(texts, toks))

    (haystack / "runs").mkdir(exist_ok=True)
    for k1 in args.k1:
        for b in args.b:
            t0 = time.time()
            index = bm25s.BM25(k1=k1, b=b)
            index.index((corpus.ids, dict(corpus.vocab)), show_progress=False)
            for fmt in args.query_format:
                rows = []
                for qid, idx in groups:
                    s = index.get_scores(qtok[fmt][qid])[idx]
                    top = np.argsort(-s, kind="stable")[:TOP_K]
                    rows += [(qid, docs.pmid.iat[idx[j]], float(s[j]), r + 1)
                             for r, j in enumerate(top)]
                out = haystack / "runs" / f"bm25-{fmt}-k{k1:g}-b{b:g}__gene.parquet"
                pd.DataFrame(rows, columns=["qid", "pmid", "score", "rank"]).to_parquet(out)
                print(f"-> {out.relative_to(ROOT)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
