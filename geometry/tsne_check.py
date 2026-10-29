"""Is t-SNE worth trying? Compare the published PubMedBERT t-SNE map (González-Márquez et al. 2024) with our
UMAP maps on the same papers and the same scores.

Pool: the human-gene papers that are on the published map (6.26M). Queries: the 20k fixed queries of
truth.py that are in the pool. Neighbours are searched within the pool in each space:
  medembed_384   our MedEmbed-small vectors (cosine)          the source of our UMAP maps
  pubmedbert_768 their PubMedBERT vectors (cosine)            the source of their t-SNE map
  tsne_2         their published 2-D t-SNE positions
  umap_k15_2     our 2-D UMAP (15 neighbours)
  umap_k15_12    our 12-D UMAP (15 neighbours)
Scores: gene15 and field15 as in umap_sweep.py, and recall15 of each map against its own source space
(t-SNE vs PubMedBERT, UMAP vs MedEmbed): the share of the source's 15 nearest neighbours kept.
Writes data/geometry/medembed_small/sweep/tsne_check.csv. CPU only (~10 min): uv run python geometry/tsne_check.py
"""
import csv, time
from pathlib import Path
import duckdb, numpy as np, torch

REPO = Path(__file__).resolve().parent.parent
D = REPO / "data/geometry/medembed_small"
E = D / "eval"
LAND = "/data/public/PubMedLandscape/2024_v2"
K = 15
torch.set_num_threads(96)
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)

pm = np.load(D / "pmids.npy")
field = np.load(E / "field.npy")
ptr, genes = np.load(E / "genes_ptr.npy"), np.load(E / "genes.npy")
pool = np.flatnonzero(field >= 0)                    # rows (in X order) on the published map
q_all = np.load(E / "queries.npy")
q = q_all[field[q_all] >= 0]                          # queries in the pool
qpos = np.searchsorted(pool, q)                       # their positions within the pool
log(f"pool {len(pool):,} papers, {len(q):,} queries")

# Row of each pool paper in their table (= row of its PubMedBERT vector, and its t-SNE x, y).
tab = duckdb.sql(f"""SELECT PMID, min(rn) AS rn, arg_min(x, rn) AS x, arg_min(y, rn) AS y
    FROM (SELECT PMID, x, y, row_number() OVER () - 1 AS rn FROM '{LAND}/pubmed_landscape_data_2024_v2.parquet')
    GROUP BY PMID ORDER BY PMID""").fetchnumpy()
j = np.searchsorted(tab["PMID"], pm[pool]); assert (tab["PMID"][j] == pm[pool]).all()
brow = tab["rn"][j].astype(np.int64)
tsne = np.stack([tab["x"][j], tab["y"][j]], 1).astype(np.float32)
log("table rows matched")


def knn(V: np.ndarray, cosine: bool, k: int = K) -> np.ndarray:
    """k nearest pool positions of each query (itself excluded), brute force in chunks."""
    Vq = torch.from_numpy(np.asarray(V[qpos], np.float32))
    if cosine:
        Vq = torch.nn.functional.normalize(Vq, dim=1)
    best_s = torch.full((len(q), k + 1), -np.inf); best_i = torch.zeros((len(q), k + 1), dtype=torch.int64)
    for s in range(0, len(V), 500_000):
        B = torch.from_numpy(np.asarray(V[s:s + 500_000], np.float32))
        sc = Vq @ torch.nn.functional.normalize(B, dim=1).T if cosine else 2 * Vq @ B.T - (B * B).sum(1)[None]
        v, i = sc.topk(k + 1, dim=1)
        cs, ci = torch.cat([best_s, v], 1), torch.cat([best_i, i + s], 1)
        best_s, o = cs.topk(k + 1, dim=1); best_i = ci.gather(1, o)
    I = best_i.numpy()
    return np.stack([r[r != p][:k] for r, p in zip(I, qpos)])


qgenes = [set(genes[ptr[i]:ptr[i + 1]].tolist()) for i in q]
def gene15(nb):  # nb: pool positions
    rows = pool[nb]
    return float(np.mean([not g.isdisjoint(genes[ptr[r]:ptr[r + 1]].tolist()) for g, rr in zip(qgenes, rows) for r in rr]))
def field15(nb):
    fq, fn = field[q][:, None], field[pool[nb]]
    both = (fq > 0) & (fn > 0)
    return float((fq == fn)[both].mean())
recall = lambda a, b: float(np.mean([len(np.intersect1d(x, y)) / K for x, y in zip(a, b)]))

X = np.load(D / "X.npy", mmap_mode="r")
nb = {"medembed_384": knn(np.asarray(X[pool]), True)}; log("medembed done")
P = np.load(f"{LAND}/PubMedBERT_embeddings_float16_2024.npy", mmap_mode="r")
order = np.argsort(brow)
Pp = np.empty((len(pool), P.shape[1]), np.float16); Pp[order] = P[brow[order]]
nb["pubmedbert_768"] = knn(Pp, True); del Pp; log("pubmedbert done")
nb["tsne_2"] = knn(tsne, False); log("tsne done")
for name, f in [("umap_k15_2", "umap_k15_d2_md0"), ("umap_k15_12", "umap_k15_d12_md0")]:
    U = np.load(D / "sweep" / f"{f}.npy", mmap_mode="r")
    nb[name] = knn(np.asarray(U[pool], np.float32), False); log(f"{name} done")

src = {"tsne_2": "pubmedbert_768", "umap_k15_2": "medembed_384", "umap_k15_12": "medembed_384"}
rows = [{"space": s, "recall15_vs_source": round(recall(v, nb[src[s]]), 4) if s in src else "",
         "gene15": round(gene15(v), 4), "field15": round(field15(v), 4)} for s, v in nb.items()]
out = D / "sweep" / "tsne_check.csv"
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
