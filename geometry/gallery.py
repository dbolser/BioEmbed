"""Does UMAP behave as density goes up? A gallery of 2-D maps (n_neighbors x min_dist, plus t-SNE at three
perplexities) on nested random subsets of the 7.76M human-gene papers, the same subsets as the intrinsic
dimension vs density check (rng seed 1: row is in subset f iff u < f).

Every map gets the umap_sweep.py scores, computed inside the subset on 5k queries: recall15 against the exact
384-D neighbours, gene15, field15, spearman. UMAP is given exact neighbours (brute force on the GPU), so only the
layout differs between its settings. t-SNE is openTSNE (cuML's t-SNE scored near chance on a 8k-paper test).
Writes gallery/{results.csv, gallery_<f>.png, <f>_<name>.npy} under data/geometry/medembed_small/.
  GPU (RAPIDS env): uv run python gallery.py --data ~/medembed_small --umap --fracs 0.01 0.1
  CPU:              uv run --with openTSNE --with annoy python geometry/gallery.py --data D --tsne --fracs ...
  CPU:              uv run python geometry/gallery.py --data D --draw --fracs ...
"""
import argparse, csv, time
from pathlib import Path
import numpy as np

KS = [2, 5, 10, 20, 50, 100, 200]
MDS = [0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.25, 0.5, 0.99]
PERPS = [10, 30, 100]
NQ = 5000
FIELDS = ["frac", "n", "name", "method", "n_neighbors", "min_dist", "perplexity", "secs",
          "recall15", "gene15", "field15", "spearman"]
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def gpu_knn(Xs, k):
    """Exact k nearest rows (cosine distance) of every row, the row itself first (UMAP wants that, and exact
    duplicates would otherwise sometimes come first)."""
    import cupy as cp
    Xg = cp.asarray(Xs); n = len(Xs)
    I = np.empty((n, k), np.int64); Dd = np.empty((n, k), np.float32)
    B = min(4096, int(1.5e9 // (4 * n)))  # (a block of similarities stays under 1.5 GB)
    for s in range(0, n, B):
        r = cp.arange(s, min(s + B, n))
        sim = Xg[r] @ Xg.T
        sim[cp.arange(len(r)), r] = -2  # (self goes in column 0 below)
        i = cp.argpartition(-sim, k - 1, axis=1)[:, :k - 1]
        v = cp.take_along_axis(sim, i, 1); o = cp.argsort(-v, axis=1)
        I[s:s + B, 0] = r.get(); Dd[s:s + B, 0] = 0
        I[s:s + B, 1:] = cp.take_along_axis(i, o, 1).get()
        Dd[s:s + B, 1:] = cp.clip(1 - cp.take_along_axis(v, o, 1), 0, None).get()
    del Xg; cp.get_default_memory_pool().free_all_blocks()
    return I, Dd


class Score:
    def __init__(self, Xs, rows, field, ptr, genes):
        rng = np.random.default_rng(2)
        self.q = np.sort(rng.choice(len(Xs), NQ, replace=False))
        self.truth = np.empty((NQ, 15), np.int64)
        for s in range(0, NQ, 250):
            q = self.q[s:s + 250]
            sim = Xs[q] @ Xs.T
            sim[np.arange(len(q)), q] = -2
            self.truth[s:s + 250] = np.argpartition(-sim, 15, axis=1)[:, :15]
        self.field, self.ptr, self.genes, self.rows = field[rows], ptr, genes, rows
        self.qg = [set(genes[ptr[r]:ptr[r + 1]].tolist()) for r in rows[self.q]]
        self.a, self.b = rng.integers(0, len(Xs), (100_000, 2)).T
        self.orig = 1 - np.einsum("ij,ij->i", Xs[self.a], Xs[self.b])

    def __call__(self, Y):
        from scipy.stats import spearmanr
        from sklearn.neighbors import NearestNeighbors
        _, idx = NearestNeighbors(n_neighbors=16).fit(Y).kneighbors(Y[self.q])
        nb = np.stack([r[r != qi][:15] for r, qi in zip(idx, self.q)])
        rec = float(np.mean([len(np.intersect1d(a, b)) / 15 for a, b in zip(nb, self.truth)]))
        hit = sum(not g.isdisjoint(self.genes[self.ptr[r]:self.ptr[r + 1]].tolist())
                  for g, row in zip(self.qg, self.rows[nb]) for r in row)
        fq, fn = self.field[self.q][:, None], self.field[nb]
        both = (fq > 0) & (fn > 0)
        low = np.linalg.norm(Y[self.a] - Y[self.b], axis=1)
        return {"recall15": rec, "gene15": hit / nb.size, "field15": float((fq == fn)[both].mean()),
                "spearman": float(spearmanr(self.orig, low).statistic)}


def draw(out, frac, rows, field):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    res = {r["name"]: r for r in csv.DictReader(open(out / "results.csv")) if float(r["frac"]) == frac}
    grid = [[f"umap_k{k}_md{md:g}" for md in MDS] for k in KS] + [[f"tsne_p{p}" for p in PERPS]]
    fig, axs = plt.subplots(len(grid), len(MDS), figsize=(3 * len(MDS), 3.3 * len(grid)))
    cmap = plt.get_cmap("tab20")
    n = len(rows); f = field[rows]
    sub = np.random.default_rng(0).permutation(n)[:60_000]
    lab = sub[f[sub] > 0]; unl = sub[f[sub] <= 0]
    for r, names in enumerate(grid):
        for c in range(len(MDS)):
            ax = axs[r, c]; ax.set_xticks([]); ax.set_yticks([])
            p = out / f"{frac:g}_{names[c]}.npy" if c < len(names) else None
            if p is None or not p.exists() or names[c] not in res:
                ax.set_axis_off(); continue
            Y = np.load(p); s = res[names[c]]
            ax.scatter(*Y[unl].T, s=0.3, c="#cccccc", lw=0, rasterized=True)
            ax.scatter(*Y[lab].T, s=0.5, c=cmap(f[lab] % 20), lw=0, rasterized=True)
            lo, hi = np.percentile(Y[sub], [0.2, 99.8], axis=0)  # (a few far fragments would squash the rest)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
            ax.set_title(f"{names[c]}\ngene {float(s['gene15']):.0%}  field {float(s['field15']):.0%}  "
                         f"kept {float(s['recall15']):.0%}", fontsize=8)
    fig.suptitle(f"{frac:g} of the human-gene papers (n = {n:,}). Rows: UMAP n_neighbors {KS}, then t-SNE "
                 f"perplexity {PERPS}. Columns: min_dist {MDS}. Colour: journal field (grey: none)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985)); fig.savefig(out / f"gallery_{frac:g}.png", dpi=100); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.01, 0.1])
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--umap", action="store_true"); m.add_argument("--tsne", action="store_true")
    m.add_argument("--draw", action="store_true")
    args = ap.parse_args()
    X = np.load(args.data / "X.npy", mmap_mode="r")
    e = args.data / "eval"
    field, ptr, genes = np.load(e / "field.npy"), np.load(e / "genes_ptr.npy"), np.load(e / "genes.npy")
    u = np.random.default_rng(1).random(len(X)).astype(np.float32)
    out = args.data / "gallery"; out.mkdir(exist_ok=True)
    res_path = out / "results.csv"
    if not res_path.exists():
        csv.DictWriter(open(res_path, "w", newline=""), FIELDS).writeheader()
    for frac in args.fracs:
        rows = np.flatnonzero(u < frac); n = len(rows)
        if args.draw:
            draw(out, frac, rows, field); log(f"gallery {frac} drawn"); continue
        done = {r["name"] for r in csv.DictReader(open(res_path)) if float(r["frac"]) == frac}
        Xs = np.asarray(X[rows], np.float32)
        log(f"frac {frac}: {n:,} papers")
        score = Score(Xs, rows, field, ptr, genes)
        if args.umap:
            from cuml.manifold import UMAP
            I, Dd = gpu_knn(Xs, max(KS))
            log("exact neighbours done")
            jobs = [(f"umap_k{k}_md{md:g}", k, md, None) for k in KS for md in MDS]
        else:
            from openTSNE import TSNE
            jobs = [(f"tsne_p{p}", None, None, p) for p in PERPS]
        for name, k, md, p in jobs:
            if name in done:
                continue
            t1 = time.time()
            try:
                if args.umap:
                    Y = UMAP(n_components=2, n_neighbors=k, min_dist=md, metric="cosine", random_state=0,
                             precomputed_knn=(I[:, :k], Dd[:, :k])).fit_transform(Xs)
                    Y = Y.get() if hasattr(Y, "get") else Y
                else:
                    Y = TSNE(perplexity=p, initialization="pca", neighbors="annoy", metric="cosine", n_jobs=96,
                             random_state=0).fit(Xs)
                Y = np.asarray(Y, np.float32)
            except Exception as ex:  # (e.g. out of memory: skip the setting, keep going)
                log(f"failed {name}: {ex!r}"[:300]); continue
            np.save(out / f"{frac:g}_{name}.npy", Y)
            row = {"frac": frac, "n": n, "name": name, "method": "umap" if args.umap else "tsne",
                   "n_neighbors": k or "", "min_dist": "" if md is None else md, "perplexity": p or "",
                   "secs": round(time.time() - t1), **score(Y)}
            csv.DictWriter(open(res_path, "a", newline=""), FIELDS).writerow(row)
            log({k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in row.items()})
    print("exit 0", flush=True)


if __name__ == "__main__":
    main()
