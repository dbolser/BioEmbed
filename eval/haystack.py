"""Rank GeneDiseaseHaystack with embedding models (gene-filtered track).

Each query's candidates are the PubMed abstracts PubTator 3 tags with the
query gene (gene_pmids.parquet; 1.7M distinct docs over 500 queries), so the
filter is the $0 database lookup and the model only orders it. Built by
scripts/build_tier_c_haystack.py.

Models are loaded exactly as for the 300-doc suite (run_embedders.get_model),
and encode with GeneDiseaseRetrieval's task metadata so instruction prompts
match. On GPU, weights run in bfloat16 (the measured L4 cost axis used
half precision too; fp32 is ~8x slower). Doc vectors are cached in float16 chunks so a preempted Spot VM resumes
where it stopped.

Usage (from eval/):
    uv run python haystack.py --models BAAI/bge-small-en-v1.5 ncbi/MedCPT

Output: <haystack>/runs/<model-slug>__gene.parquet (qid, pmid, score, rank),
top --top-k per query.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from mteb._create_dataloaders import create_dataloader
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.types import PromptType
from sentence_transformers import SentenceTransformer

from run_embedders import get_model
from tasks_local import GeneDiseaseRetrieval

REPO = Path(__file__).resolve().parent.parent
HAYSTACK = REPO / "data" / "haystack" / "GeneDiseaseHaystack"
CHUNK = 100_000


def slug(name: str) -> str:
    return name.replace("/", "__")


def to_bf16(model) -> None:
    """Cast the torch modules a model wrapper holds (MedCPT holds two)."""
    for v in vars(model).values():
        if isinstance(v, torch.nn.Module):
            v.to(torch.bfloat16)


def encode(model, texts: list[str], prompt_type: PromptType, batch_size: int) -> np.ndarray:
    meta = GeneDiseaseRetrieval.metadata
    ds = Dataset.from_dict({"id": [str(i) for i in range(len(texts))], "text": texts})
    loader = create_dataloader(ds, meta,
                               prompt_type=prompt_type, batch_size=batch_size)
    vecs = model.encode(loader, task_metadata=meta, hf_split="test", hf_subset="default",
                        prompt_type=prompt_type, batch_size=batch_size)
    if torch.is_tensor(vecs):
        vecs = vecs.float().cpu()
    return np.asarray(vecs, dtype=np.float32)


def doc_vectors(model, name: str, docs: pd.DataFrame, batch_size: int) -> np.ndarray:
    """Encode docs in CHUNK-sized pieces, cached as float16 .npy files."""
    cache = HAYSTACK / "vectors" / slug(name)
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    for i in range(0, len(docs), CHUNK):
        f = cache / f"{i // CHUNK:04d}.npy"
        try:
            parts.append(np.load(f))
            continue
        except (OSError, ValueError, EOFError):  # missing, or truncated by a preemption
            pass
        t0 = time.time()
        v = encode(model, docs.text.iloc[i:i + CHUNK].tolist(), PromptType.document, batch_size)
        tmp = f.with_suffix(".tmp.npy")
        with open(tmp, "wb") as fh:
            np.save(fh, v.astype(np.float16))
            fh.flush()
            os.fsync(fh.fileno())
        tmp.rename(f)
        print(f"  docs {i + len(v):,}/{len(docs):,} ({time.time() - t0:.0f}s)", flush=True)
        parts.append(np.load(f))
    return np.concatenate(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--max-queries", type=int, default=None, help="smoke test on the first N queries")
    args = ap.parse_args()

    queries = pd.read_parquet(HAYSTACK / "queries.parquet").sort_values("qid")
    cands = pd.read_parquet(HAYSTACK / "gene_pmids.parquet")
    docs = pd.read_parquet(HAYSTACK / "docs_gene.parquet")
    if args.max_queries:
        queries = queries.head(args.max_queries)
        cands = cands[cands.qid.isin(queries.qid)]
        docs = docs[docs.pmid.isin(cands.pmid)]
    docs = docs.reset_index(drop=True)
    row = pd.Series(docs.index, index=docs.pmid)
    out_dir = HAYSTACK / ("runs" if not args.max_queries else "runs_smoke")
    out_dir.mkdir(exist_ok=True)

    for name in args.models:
        out = out_dir / f"{slug(name)}__gene.parquet"
        if out.exists():
            print(f"=== {name}: done, skipping")
            continue
        print(f"=== {name} ===", flush=True)
        model = get_model(name)
        if isinstance(model, SentenceTransformer):
            model = SentenceTransformerEncoderWrapper(model)
        if torch.cuda.is_available():
            to_bf16(model)
        if args.max_queries:  # smoke runs never touch the real vector cache
            dvec = encode(model, docs.text.tolist(), PromptType.document, args.batch_size)
        else:
            dvec = doc_vectors(model, name, docs, args.batch_size)
        qvec = encode(model, queries.text.tolist(), PromptType.query, args.batch_size)

        qrow = dict(zip(queries.qid, range(len(queries))))
        rows = []
        for qid, group in cands.groupby("qid", sort=True):
            idx = row[group.pmid].to_numpy()
            q = qvec[qrow[qid]][None, :]
            s = np.asarray(model.similarity(q, dvec[idx].astype(np.float32))).ravel()
            top = np.argsort(-s, kind="stable")[: args.top_k]
            rows += [(qid, docs.pmid.iat[idx[j]], float(s[j]), r + 1) for r, j in enumerate(top)]
        pd.DataFrame(rows, columns=["qid", "pmid", "score", "rank"]).to_parquet(out)
        print(f"  -> {out.relative_to(REPO)}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
