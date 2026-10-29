"""Whole-corpus version of genesearch/litmap.py on a GPU (RAPIDS cuML).

Input: a directory with X.npy (float16, n x d, L2-normalised document vectors) and
pmids.npy (int64). Steps, as litmap does per gene but for millions of papers:
  1. UMAP (cosine, k neighbours, min_dist 0) to --dims (12) -> u12.npy (float32)
  2. HDBSCAN on u12 -> labels.npy (int32, -1 = noise) and probabilities.npy
  3. UMAP (cosine, k neighbours, min_dist 0.1) to 2-D -> xy.npy (float32)
Each output is skipped if it already exists, so a preempted run resumes.
cuML's UMAP builds its own kNN graph (nn-descent, batched when the data is
bigger than GPU memory) so the two UMAPs do not share one graph as litmap's do.

Run (from the directory holding the env):
    uv run python geometry.py --data ~/medembed_small [--dims 12] [--k 15] [--min-cluster-size 200]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def save(path: Path, a: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, a)
    tmp.rename(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--dims", type=int, default=12)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--min-cluster-size", type=int, default=200)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (smoke test)")
    ap.add_argument("--out", type=Path, default=None, help="default: <data>/geom_<dims>d_k<k>")
    args = ap.parse_args()

    import cupy as cp
    from cuml.cluster import HDBSCAN
    from cuml.manifold import UMAP

    out = args.out or args.data / f"geom_{args.dims}d_k{args.k}"
    out.mkdir(parents=True, exist_ok=True)
    X = np.load(args.data / "X.npy", mmap_mode="r")
    n = min(len(X), args.limit or len(X))
    run = json.loads(json.dumps({**vars(args), "data": str(args.data), "out": str(out), "rows": n}, default=str))
    if (out / "run.json").exists() and (old := json.loads((out / "run.json").read_text())) != run:
        # The outputs below are reused if present, so they must come from the same settings.
        diff = {k: (old.get(k), v) for k, v in run.items() if old.get(k) != v}
        sys.exit(f"{out} holds a run with other settings {diff}; use another --out or delete it")
    (out / "run.json").write_text(json.dumps(run, indent=1))
    print(f"{n:,} x {X.shape[1]} vectors -> {out}", flush=True)

    def umap(dims: int, min_dist: float, name: str) -> np.ndarray:
        f = out / f"{name}.npy"
        if f.exists():
            print(f"{name}: exists, skipping", flush=True)
            return np.load(f)
        t0 = time.time()
        # float32 host array; with batched nn-descent cuML streams it to the GPU in parts.
        Xf = np.ascontiguousarray(X[:n], dtype=np.float32)
        # Partitioned kNN keeps about (overlap / clusters) of the rows on the GPU at once:
        # ~2M rows of 384-d float32 plus the nn-descent graph fits an L4's 23 GB.
        parts = -(-n // 2_000_000)
        kwds = {"knn_n_clusters": parts, "knn_overlap_factor": 2} if parts > 2 else {}
        um = UMAP(n_components=dims, n_neighbors=args.k, min_dist=min_dist, metric="cosine",
                  build_algo="nn_descent", build_kwds=kwds, random_state=None, verbose=True)
        emb = um.fit_transform(Xf)
        emb = cp.asnumpy(emb) if not isinstance(emb, np.ndarray) else emb
        save(f, emb.astype(np.float32))
        print(f"{name}: {emb.shape} in {time.time() - t0:.0f}s", flush=True)
        del um, Xf
        cp.get_default_memory_pool().free_all_blocks()
        return emb

    u = umap(args.dims, 0.0, f"u{args.dims}")

    if not (out / "labels.npy").exists():
        t0 = time.time()
        h = HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
                    cluster_selection_method="leaf", prediction_data=False, verbose=True)
        labels = cp.asnumpy(h.fit_predict(u))
        save(out / "labels.npy", labels.astype(np.int32))
        save(out / "probabilities.npy", cp.asnumpy(h.probabilities_).astype(np.float32))
        k = labels.max() + 1
        print(f"hdbscan: {k:,} clusters, {np.mean(labels < 0):.1%} noise, in {time.time() - t0:.0f}s", flush=True)
        del h
        cp.get_default_memory_pool().free_all_blocks()
    else:
        print("labels: exist, skipping", flush=True)

    umap(2, 0.1, "xy")
    print("done", flush=True)


if __name__ == "__main__":
    main()
