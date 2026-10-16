"""Embed every abstract in genesearch/index/abstracts.parquet (28.3M) with one
model's document side, for genesearch's whole-PubMed semantic index.

Text is '{title}\\n\\n{abstract}' (the abstract alone if there is no title),
as genesearch/semantic.py doc_text. The model is loaded with
run_embedders.get_model and encoded exactly as haystack.py encodes documents
(mteb wrapper, document prompt, bf16 on GPU), so the vectors match the
haystack runs. Before the first shard, 3 corpus texts are checked against
get_model's own fp32 output (cosine >= MIN_COSINE).

Rows are read by parquet row group, in file order; shards of SHARD rows are
written as float16 .npy (tmp file, fsync, rename), so a rerun skips finished
shards and resumes at the first missing one. Each shard is encoded as
--threads contiguous parts on parallel threads, so one part's tokenisation
(CPU, ~900 docs/s per thread) overlaps another's GPU work. Vectors match one
encode call up to rounding (other batches, other padding; on CPU at most one
float16 step).

Run (from eval/, on a GPU):
    uv run python embed_corpus.py [--model abhinand/MedEmbed-small-v0.1] [--batch-size 1024]
    uv run python embed_corpus.py --limit 5000 --out /tmp/emb_smoke    # smoke test
Output: --out (default genesearch/index/emb_medembed_small/): pmids.npy
(int64, file order) and 0000.npy, 0001.npy, ... (float16, rows x dim).
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.types import PromptType
from sentence_transformers import SentenceTransformer

from haystack import encode, to_bf16
from run_embedders import get_model

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "genesearch" / "index" / "abstracts.parquet"
OUT = REPO / "genesearch" / "index" / "emb_medembed_small"
MODEL = "abhinand/MedEmbed-small-v0.1"
SHARD = 1_000_000
MAX_LENGTH = 512
MIN_COSINE = 0.995  # bf16 vs fp32 on the same text


def doc_text(title: str | None, abstract: str | None) -> str:
    return f"{title}\n\n{abstract or ''}" if title else abstract or ""


def load(name: str):
    model = get_model(name)
    if isinstance(model, SentenceTransformer):
        model = SentenceTransformerEncoderWrapper(model)
    st = getattr(model, "model", None)
    if isinstance(st, SentenceTransformer):
        st.max_seq_length = min(st.max_seq_length or MAX_LENGTH, MAX_LENGTH)
    return model


def rows(pf: pq.ParquetFile, start: int, stop: int) -> list[str]:
    """Texts of rows [start, stop) in file order, reading only the row groups needed."""
    texts, at = [], 0
    for g in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        if at + n > start and at < stop:
            t = pf.read_row_group(g, columns=["title", "abstract"]).slice(max(0, start - at), stop - max(at, start))
            texts += [doc_text(a, b) for a, b in zip(t.column("title").to_pylist(), t.column("abstract").to_pylist())]
        at += n
        if at >= stop:
            break
    return texts


def encode_parts(model, texts: list[str], batch_size: int, threads: int) -> np.ndarray:
    step = -(-len(texts) // threads)
    with ThreadPoolExecutor(threads) as pool:
        parts = pool.map(lambda i: encode(model, texts[i:i + step], PromptType.document, batch_size),
                         range(0, len(texts), step))
        return np.concatenate(list(parts))


def verify(model, name: str, texts: list[str], batch_size: int) -> None:
    """Corpus-path vectors vs get_model's own output (fp32, no bf16 cast)."""
    got = encode(model, texts, PromptType.document, batch_size)
    ref = load(name)
    for m in (ref, getattr(ref, "model", None)):
        if isinstance(m, torch.nn.Module):
            m.float()
    want = encode(ref, texts, PromptType.document, batch_size)
    cos = (got * want).sum(1) / np.linalg.norm(got, axis=1) / np.linalg.norm(want, axis=1)
    print(f"verify: cosine vs get_model fp32 {np.round(cos, 5).tolist()}, "
          f"norms {np.round(np.linalg.norm(got, axis=1), 4).tolist()}", flush=True)
    if cos.min() < MIN_COSINE:
        raise SystemExit(f"verify failed: min cosine {cos.min():.5f} < {MIN_COSINE}")
    del ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--shard", type=int, default=SHARD, help="rows per .npy shard")
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (smoke test)")
    ap.add_argument("--threads", type=int, default=2, help="parallel encode calls per shard")
    args = ap.parse_args()

    pf = pq.ParquetFile(args.corpus)
    n = min(pf.metadata.num_rows, args.limit or pf.metadata.num_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    if not (args.out / "pmids.npy").exists():
        pmids = pf.read(columns=["pmid"]).column("pmid").to_numpy()[:n].astype(np.int64)
        np.save(args.out / "pmids.npy", pmids)
    print(f"{args.corpus.name}: {n:,} rows -> {args.out} in shards of {args.shard:,}", flush=True)

    model = load(args.model)
    if torch.cuda.is_available():
        to_bf16(model)
    verify(model, args.model, rows(pf, 0, 3), args.batch_size)

    for s, start in enumerate(range(0, n, args.shard)):
        f = args.out / f"{s:04d}.npy"
        if f.exists():
            continue
        t0 = time.time()
        texts = rows(pf, start, min(start + args.shard, n))
        v = encode_parts(model, texts, args.batch_size, args.threads).astype(np.float16)
        tmp = f.with_suffix(".npy.tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, v)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.rename(f)
        print(f"  shard {s}: rows {start:,}-{start + len(v):,} of {n:,} ({time.time() - t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
