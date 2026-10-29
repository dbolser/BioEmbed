"""Score UMAP settings on the human-gene papers by how much structure they keep, before clustering.

Needs <data>/{X.npy, pmids.npy} and <data>/eval/ from geometry/truth.py (a fixed 20k-paper query
sample with its exact 100 nearest neighbours in the original space, journal fields and genes per paper).
For each setting: one UMAP of all papers (cuML, GPU), saved as <data>/sweep/<name>.npy (float16), then
  recall15, recall50   share of each query's true 15 / 50 nearest neighbours that are among its 15 / 50 nearest
                       in the map (the "kNN recall" of González-Márquez et al. 2024)
  graph_recall         share of the true n_neighbors neighbours present in UMAP's own neighbour graph (cuML
                       builds it approximately, in partitions, at this size: a bad graph caps everything else)
  gene15               share of the map's 15 nearest neighbours that share a human gene with the query
  field15              share of map neighbour pairs with the same journal field (both labelled)
  spearman             rank correlation of original vs map distances over 100k random paper pairs (global shape)
The same scores for the original vectors (recall = 1 by definition) and for PCA at each dimension are the
reference rows. Results go to <data>/sweep/results.csv, one row per setting, written as each finishes;
finished settings are skipped, so a preempted run resumes.

Run (from the env dir, on a GPU):  uv run python umap_sweep.py --data ~/medembed_small [--grid main]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

GRIDS = {
    # name: list of (n_neighbors, n_components, min_dist, build_kwds overrides)
    # (150 neighbours ran out of memory on an L4's 23 GB at 7.76M papers: 100 at the dimensions that matter)
    "main": [(k, d, 0.0, {}) for k in (15, 50) for d in (12, 2, 6, 24)] + [(100, d, 0.0, {}) for d in (12, 6)],
    # UMAP's neighbour graph: more partitions overlapping more, or more nn-descent work
    "graph": [(50, 12, 0.0, {"knn_n_clusters": 8, "knn_overlap_factor": 4}),
              (50, 12, 0.0, {"nnd_graph_degree": 128, "nnd_intermediate_graph_degree": 256, "nnd_max_iterations": 40})],
}
FIELDS = ["name", "method", "n_neighbors", "dims", "min_dist", "build", "secs", "recall15", "recall50",
          "graph_recall", "gene15", "field15", "spearman"]


def name_of(k, d, md, kw) -> str:
    extra = "".join(f"_{a.replace('nnd_', '').replace('knn_', '')}{b}" for a, b in sorted(kw.items()))
    return f"umap_k{k}_d{d}_md{md:g}{extra}"


class Scorer:
    def __init__(self, data: Path, X: np.ndarray):
        e = data / "eval"
        self.q = np.load(e / "queries.npy")
        self.knn = np.load(e / "knn.npy").astype(np.int64)
        self.field = np.load(e / "field.npy")
        self.ptr, self.genes = np.load(e / "genes_ptr.npy"), np.load(e / "genes.npy")
        rng = np.random.default_rng(1)
        n = len(X)
        self.pairs = rng.integers(0, n, (100_000, 2))
        a, b = self.pairs.T
        self.orig_d = 1 - np.einsum("ij,ij->i", np.asarray(X[a], np.float32), np.asarray(X[b], np.float32))
        self.qgenes = [set(self.genes[self.ptr[i]:self.ptr[i + 1]].tolist()) for i in self.q]

    def gene_share(self, nbrs: np.ndarray) -> float:
        hit = tot = 0
        for g, row in zip(self.qgenes, nbrs):
            for j in row:
                tot += 1
                hit += not g.isdisjoint(self.genes[self.ptr[j]:self.ptr[j + 1]].tolist())
        return hit / tot

    def field_share(self, nbrs: np.ndarray) -> float:
        fq = self.field[self.q][:, None]
        fn = self.field[nbrs]
        both = (fq > 0) & (fn > 0)
        return float((fq == fn)[both].mean())

    def score(self, E: np.ndarray, graph=None, k_graph: int = 15) -> dict:
        from cuml.neighbors import NearestNeighbors
        from scipy.stats import spearmanr
        nn = NearestNeighbors(n_neighbors=51, algorithm="brute").fit(E)
        _, idx = nn.kneighbors(E[self.q])
        idx = np.asarray(idx.get() if hasattr(idx, "get") else idx)
        nbrs = np.stack([r[r != qi][:50] for r, qi in zip(idx, self.q)])  # (drop the query itself)
        rec = lambda k: float(np.mean([len(np.intersect1d(a[:k], b[:k])) / k for a, b in zip(nbrs, self.knn)]))
        a, b = self.pairs.T
        low_d = np.linalg.norm(E[a].astype(np.float32) - E[b].astype(np.float32), axis=1)
        out = {"recall15": rec(15), "recall50": rec(50), "gene15": self.gene_share(nbrs[:, :15]),
               "field15": self.field_share(nbrs[:, :15]), "spearman": float(spearmanr(self.orig_d, low_d).statistic)}
        if graph is not None:
            g = graph.tocsr()
            out["graph_recall"] = float(np.mean([np.isin(t[:k_graph], g.indices[g.indptr[i]:g.indptr[i + 1]]).mean()
                                                 for i, t in zip(self.q, self.knn)]))
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--grid", default="main", choices=[*GRIDS, "all"])
    args = ap.parse_args()

    import cupy as cp
    from cuml.decomposition import PCA
    from cuml.manifold import UMAP

    out = args.data / "sweep"
    out.mkdir(exist_ok=True)
    res = out / "results.csv"
    done = {r["name"] for r in csv.DictReader(res.open())} if res.exists() else set()
    write = lambda row: (res.exists() or res.write_text(",".join(FIELDS) + "\n"),
                         res.open("a").write(",".join(str(row.get(f, "")) for f in FIELDS) + "\n"))

    X = np.load(args.data / "X.npy").astype(np.float32)
    n = len(X)
    S = Scorer(args.data, X)
    print(f"{n:,} papers; {len(S.q):,} queries", flush=True)

    if "original" not in done:  # the ceiling: the original space's own neighbours
        nb = S.knn[:, :15]
        a, b = S.pairs.T
        write({"name": "original", "method": "original", "dims": X.shape[1], "recall15": 1, "recall50": 1,
               "gene15": S.gene_share(nb), "field15": S.field_share(nb), "spearman": 1})
        print("original done", flush=True)

    if not {f"pca_d{d}" for d in (2, 6, 12, 24)} <= done:
        t0 = time.time()
        P = PCA(n_components=24).fit_transform(X)
        P = cp.asnumpy(P) if not isinstance(P, np.ndarray) else P
        secs = time.time() - t0
        for d in (2, 6, 12, 24):
            if f"pca_d{d}" in done:
                continue
            row = {"name": f"pca_d{d}", "method": "pca", "dims": d, "secs": round(secs), **S.score(np.ascontiguousarray(P[:, :d]))}
            write(row); print(row, flush=True)
        del P

    grid = [g for name, gs in GRIDS.items() if args.grid in (name, "all") for g in gs]
    for k, d, md, kw in grid:
        name = name_of(k, d, md, kw)
        if name in done:
            continue
        t0 = time.time()
        parts = -(-n // 2_000_000)
        build = {**({"knn_n_clusters": parts, "knn_overlap_factor": 2} if parts > 2 else {}), **kw}
        um = UMAP(n_components=d, n_neighbors=k, min_dist=md, metric="cosine", build_algo="nn_descent",
                  build_kwds=build, verbose=False)
        try:
            E = um.fit_transform(X)
        except MemoryError as e:  # (GPU out of memory: report it and go on; a rerun tries again)
            print(f"failed: {name}: {str(e)[:160]}", flush=True)
            del um
            cp.get_default_memory_pool().free_all_blocks()
            continue
        E = cp.asnumpy(E) if not isinstance(E, np.ndarray) else E
        secs = time.time() - t0
        np.save(out / f"{name}.npy", E.astype(np.float16))
        graph = getattr(um, "graph_", None)
        if graph is not None and hasattr(graph, "get"):
            graph = graph.get()
        row = {"name": name, "method": "umap", "n_neighbors": k, "dims": d, "min_dist": md,
               "build": " ".join(f"{a}={b}" for a, b in sorted(build.items())), "secs": round(secs),
               **S.score(E, graph, k_graph=min(k, 100))}
        write(row); print(row, flush=True)
        del um, E, graph
        cp.get_default_memory_pool().free_all_blocks()
    print("done", flush=True)


if __name__ == "__main__":
    main()
