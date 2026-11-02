"""t-SNE of all 7.76M human-gene papers (MedEmbed-small), for comparison with the UMAP maps of umap_sweep.py.

openTSNE defaults (perplexity 30, PCA initialisation, learning rate n/12, early exaggeration 12 then 1), Annoy
neighbours. Then the umap_sweep.py scores on the 20k truth.py queries: recall15/50 against the exact 384-D
neighbours, gene15, field15, spearman. Writes sweep/tsne_p30_d2.npy and appends to sweep/results.csv.
CPU (hours): uv run --with openTSNE --with annoy python geometry/tsne_full.py
"""
import csv, time
from pathlib import Path
import numpy as np
from openTSNE import TSNE
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parent.parent
D = REPO / "data/geometry/medembed_small"
E, S = D / "eval", D / "sweep"
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

X = np.load(D / "X.npy")
out = S / "tsne_p30_d2.npy"
if out.exists():
    Y = np.load(out)
else:
    Y = np.asarray(TSNE(perplexity=30, initialization="pca", neighbors="annoy", metric="cosine", n_jobs=96,
                        random_state=0, verbose=True).fit(X), np.float32)
    np.save(out, Y)
secs = time.time() - t0
log("embedded")

q, knn = np.load(E / "queries.npy"), np.load(E / "knn.npy").astype(np.int64)
field, ptr, genes = np.load(E / "field.npy"), np.load(E / "genes_ptr.npy"), np.load(E / "genes.npy")
_, idx = NearestNeighbors(n_neighbors=51, n_jobs=96).fit(Y).kneighbors(Y[q])
nbrs = np.stack([r[r != qi][:50] for r, qi in zip(idx, q)])
rec = lambda k: float(np.mean([len(np.intersect1d(a[:k], b[:k])) / k for a, b in zip(nbrs, knn)]))
hit = tot = 0
for qi, row in zip(q, nbrs[:, :15]):
    g = set(genes[ptr[qi]:ptr[qi + 1]].tolist())
    for j in row:
        tot += 1; hit += not g.isdisjoint(genes[ptr[j]:ptr[j + 1]].tolist())
fq, fn = field[q][:, None], field[nbrs[:, :15]]
both = (fq > 0) & (fn > 0)
a, b = np.random.default_rng(1).integers(0, len(X), (100_000, 2)).T
orig = 1 - np.einsum("ij,ij->i", X[a], X[b])
row = {"name": "tsne_p30_d2", "method": "tsne", "n_neighbors": 90, "dims": 2, "min_dist": "", "build": "annoy",
       "secs": round(secs), "recall15": rec(15), "recall50": rec(50), "graph_recall": "", "gene15": hit / tot,
       "field15": float((fq == fn)[both].mean()),
       "spearman": float(spearmanr(orig, np.linalg.norm(Y[a] - Y[b], axis=1)).statistic)}
log(row)
with open(S / "results.csv", "a", newline="") as f:
    csv.DictWriter(f, fieldnames=list(row)).writerow(row)
