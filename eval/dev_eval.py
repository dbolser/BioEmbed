"""Score a model or checkpoint on the practice sets, for choosing settings and
checkpoints before any test run (docs/PREREG.md). Bi-encoder or cross-encoder;
GPU (bf16) or CPU.

  haystack  GeneDiseaseHaystackDev. Bi-encoders (default --depth 0) rank every
            gene doc of each query (gene_pmids.parquet), as haystack.py will on
            the test set for C5/C6. Cross-encoders (default --depth 100, fixed
            by PREREG) re-rank each query's top --depth of the practice-selected
            BM25 run (runs/bm25-symbol_disease-k0.9-b0.75__gene); ranks below
            keep BM25 order. Curator-only nDCG@10 and recall@100, raw and
            condensed, by score_haystack.ranked_metrics.
  small     GeneDiseaseRetrieval_v2_0 (practice batch): hit@1 over all 300 docs.
            Docs are the text field, as mteb encodes them for
            results/v2_scores.csv (tasks_local drops the title, which the text
            already starts with); bge-small 0.80, bge-base 0.787, as there.

Bi-encoders load with run_embedders.get_model (hub names, bioembed/<x>) or, for
a local dir such as a finetune.py checkpoint, as a SentenceTransformer with the
prompts saved in it (fp32 on CPU, as get_model); both encode as haystack.py
does. Cross-encoders (--kind cross, or auto: a *ForSequenceClassification
config) score (query, doc) pairs with sentence-transformers CrossEncoder; the
raw logit, as rerank_haystack.py (a sigmoid saturates to ties at 1.0).

Usage (from eval/):
    uv run python dev_eval.py --model ../models/bioembed-bge-small-gd-ckpt/checkpoint-400
    uv run python dev_eval.py --model ncbi/MedCPT-Cross-Encoder --max-queries 10
Output: stdout; with --out CSV, one row appended per run.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.types import PromptType
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoConfig

from haystack import encode, to_bf16
from run_embedders import get_model

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from score_haystack import ranked_metrics  # noqa: E402

DEV = REPO / "data" / "haystack" / "GeneDiseaseHaystackDev"
FIRST_PASS = DEV / "runs" / "bm25-symbol_disease-k0.9-b0.75__gene.parquet"
GENE_DOCS = DEV / "gene_pmids.parquet"
CROSS_DEPTH = 100  # PREREG: cross-encoders re-sort the first pass's top 100
SMALL = REPO / "data" / "tasks" / "GeneDiseaseRetrieval_v2_0"


def is_cross_encoder(name: str) -> bool:
    try:
        archs = AutoConfig.from_pretrained(name).architectures or []
    except (OSError, ValueError):  # not a transformers repo/dir, e.g. ncbi/MedCPT
        return False
    return any(a.endswith("ForSequenceClassification") for a in archs)


class Scorer:
    """score(queries, docs, pairs) -> one score per (query row, doc row) pair."""

    def __init__(self, name: str, kind: str, batch_size: int, max_length: int):
        self.batch_size = batch_size
        self.cross = kind == "cross" or (kind == "auto" and is_cross_encoder(name))
        gpu = torch.cuda.is_available()
        if self.cross:
            self.model = CrossEncoder(name, max_length=max_length,
                                      model_kwargs={"dtype": torch.bfloat16} if gpu else None)
            return
        if Path(name).is_dir():
            model = SentenceTransformer(name)
            if not gpu:
                model.float()
        else:
            model = get_model(name)
        if isinstance(model, SentenceTransformer):
            model = SentenceTransformerEncoderWrapper(model)
        if gpu:
            to_bf16(model)
        self.model = model

    def score(self, queries: list[str], docs: list[str], pairs: np.ndarray) -> np.ndarray:
        if self.cross:
            return np.asarray(self.model.predict([(queries[i], docs[j]) for i, j in pairs],
                                                 batch_size=self.batch_size,
                                                 activation_fn=torch.nn.Identity()), dtype=np.float32)
        d = encode(self.model, docs, PromptType.document, self.batch_size)
        q = encode(self.model, queries, PromptType.query, self.batch_size)
        out = np.empty(len(pairs), dtype=np.float32)
        for i in np.unique(pairs[:, 0]):  # per query, as haystack.py
            m = pairs[:, 0] == i
            out[m] = np.asarray(self.model.similarity(q[i][None, :], d[pairs[m, 1]])).ravel()
        return out


def haystack(scorer: Scorer, depth: int, max_queries: int | None) -> dict[str, float]:
    """depth 0: every gene doc, ties in gene_pmids order (as haystack.py)."""
    queries = pd.read_parquet(DEV / "queries.parquet").sort_values("qid")
    if max_queries:
        queries = queries.head(max_queries)
    run = pd.read_parquet(FIRST_PASS if depth else GENE_DOCS, columns=["qid", "pmid", *(["rank"] if depth else [])])
    run = run[run.qid.isin(queries.qid)]
    if not depth:
        run["rank"] = run.groupby("qid").cumcount() + 1
    run = run.sort_values(["qid", "rank"])
    top = run[run["rank"] <= (depth or len(run))].reset_index(drop=True)
    docs = pd.read_parquet(DEV / "docs_gene.parquet")
    docs = docs[docs.pmid.isin(top.pmid)].reset_index(drop=True)
    qrow = {q: i for i, q in enumerate(queries.qid)}
    drow = {p: i for i, p in enumerate(docs.pmid)}
    pairs = np.array([(qrow[q], drow[p]) for q, p in zip(top.qid, top.pmid)])
    top["score"] = scorer.score(queries.text.tolist(), docs.text.tolist(), pairs)
    top = top.sort_values(["qid", "score", "rank"], ascending=[True, False, True], kind="stable")
    top["rank"] = top.groupby("qid").cumcount() + 1
    rest = run[run["rank"] > (depth or len(run))]
    reranked = pd.concat([top, rest]).sort_values(["qid", "rank"])

    qrels = pd.read_parquet(DEV / "qrels.parquet")
    gold = qrels[qrels.qid.isin(queries.qid)].groupby("qid").pmid.agg(set).to_dict()
    silver = pd.read_parquet(DEV / "silver.parquet")
    unlabelled = silver.merge(qrels, how="left", indicator=True).query("_merge == 'left_only'")
    drop = set(zip(unlabelled.qid, unlabelled.pmid))
    condensed = reranked[[(q, p) not in drop for q, p in zip(reranked.qid, reranked.pmid)]].copy()
    condensed["rank"] = condensed.groupby("qid")["rank"].rank(method="first")
    out = {}
    for variant, r in [("raw", reranked), ("condensed", condensed)]:
        m = ranked_metrics(r, gold)
        out |= {f"{variant}_ndcg@10": m["ndcg@10"], f"{variant}_recall@100": m["recall@100"]}
    return out


def corpus_texts(corpus) -> list[str]:
    """Doc inputs as mteb gets them from tasks_local (title dropped, stripped)."""
    return [x.strip() for x in corpus["text"]]


def small(scorer: Scorer, max_queries: int | None) -> dict[str, float]:
    dd = load_from_disk(str(SMALL))
    queries = dd["queries"].select(range(min(max_queries or len(dd["queries"]), len(dd["queries"]))))
    docs = dd["corpus"]["_id"]
    gold = {r["query-id"]: docs.index(r["corpus-id"]) for r in dd["qrels"]}
    n_q, n_d = len(queries), len(docs)
    pairs = np.array([(i, j) for i in range(n_q) for j in range(n_d)])
    s = scorer.score(queries["text"], corpus_texts(dd["corpus"]), pairs).reshape(n_q, n_d)
    return {"small_hit@1": float(np.mean([s[i].argmax() == gold[q] for i, q in enumerate(queries["_id"])]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="hub name or local checkpoint dir")
    ap.add_argument("--kind", choices=["auto", "bi", "cross"], default="auto")
    ap.add_argument("--depth", type=int, help="BM25 ranks re-scored per query; 0 = every gene doc. "
                    f"Default 0 for bi-encoders, {CROSS_DEPTH} for cross-encoders")
    ap.add_argument("--max-queries", type=int, help="first N queries of each set (smoke)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=512, help="cross-encoder tokens")
    ap.add_argument("--out", type=Path, help="CSV to append a result row to")
    args = ap.parse_args()

    t0 = time.time()
    scorer = Scorer(args.model, args.kind, args.batch_size, args.max_length)
    depth = args.depth if args.depth is not None else CROSS_DEPTH if scorer.cross else 0
    row = {"model": args.model, "kind": "cross" if scorer.cross else "bi", "depth": depth or "all",
           "max_queries": args.max_queries or 0}
    row |= haystack(scorer, depth, args.max_queries)
    row |= small(scorer, args.max_queries)
    row["seconds"] = round(time.time() - t0)
    print(pd.Series(row).to_string(), flush=True)
    if args.out:
        df = pd.DataFrame([row])
        df.to_csv(args.out, mode="a", header=not args.out.exists(), index=False)


if __name__ == "__main__":
    main()
