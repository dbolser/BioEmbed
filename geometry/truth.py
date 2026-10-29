"""Ground truth for scoring low-dimensional maps of the human-gene papers (geometry/umap_sweep.py).

From data/geometry/medembed_small/{X.npy, pmids.npy} (7.76M L2-normalised MedEmbed-small vectors):
  eval/queries.npy     int64 (Q,)        a fixed random sample of rows (seed 0)
  eval/knn.npy         int32 (Q, K)      each query's exact K nearest rows by cosine (itself excluded)
  eval/knn_sim.npy     float16 (Q, K)    their cosine similarities
  eval/field.npy       int8 (n,)         journal field per row from the PubMedBERT landscape (0 = none, -1 = not on it)
  eval/genes_ptr.npy, eval/genes.npy     CSR of each row's HGNC genes (NCBI ids), for "neighbours share a gene"
  eval/intrinsic.json  intrinsic dimension: TwoNN (Facco et al. 2017) and Levina-Bickel MLE at k=10, 20, 50
"""
import json, sys, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch

REPO = Path(__file__).resolve().parent.parent
D = REPO / "data/geometry/medembed_small"
OUT = D / "eval"; OUT.mkdir(exist_ok=True)
Q, K = 20_000, 100
torch.set_num_threads(96)

X = np.load(D / "X.npy", mmap_mode="r"); pm = np.load(D / "pmids.npy"); n = len(pm)
q = np.sort(np.random.default_rng(0).choice(n, Q, replace=False)); np.save(OUT / "queries.npy", q)
t0 = time.time()
Xq = torch.from_numpy(np.asarray(X[q], np.float32))
best_s = torch.full((Q, K + 1), -2.0); best_i = torch.zeros((Q, K + 1), dtype=torch.int64)
C = 250_000
for s in range(0, n, C):
    blk = torch.from_numpy(np.asarray(X[s:s + C], np.float32))
    sim = Xq @ blk.T
    v, i = sim.topk(K + 1, dim=1)
    cs, ci = torch.cat([best_s, v], 1), torch.cat([best_i, i + s], 1)
    best_s, o = cs.topk(K + 1, dim=1); best_i = ci.gather(1, o)
    print(f"  knn {min(s + C, n):,}/{n:,} ({time.time() - t0:.0f}s)", flush=True)
# drop the query itself (or, for exact duplicates, any one copy at similarity ~1 in its place)
self_mask = best_i == torch.from_numpy(q)[:, None]
keep = ~self_mask
keep[~self_mask.any(1), -1] = False  # (self not found: drop the last instead, keeping K)
best_i = best_i[keep].view(Q, K); best_s = best_s[keep].view(Q, K)
np.save(OUT / "knn.npy", best_i.numpy().astype(np.int32)); np.save(OUT / "knn_sim.npy", best_s.numpy().astype(np.float16))

# Intrinsic dimension from the neighbour distances (Euclidean on the unit sphere: sqrt(2 - 2 cos)).
d = np.sqrt(np.clip(2 - 2 * best_s.numpy().astype(np.float64), 1e-12, None))
ok = d[:, 0] > 1e-6  # (exact duplicates break both estimators)
mu = d[ok, 1] / d[ok, 0]
mu = np.sort(mu)[: int(0.9 * ok.sum())]  # TwoNN: discard the top 10% of ratios, fit through the origin
F = np.arange(1, len(mu) + 1) / ok.sum()
twonn = float(np.sum(np.log(mu) * -np.log(1 - F)) / np.sum(np.log(mu) ** 2))
mle = {}
for k in (10, 20, 50):
    dk = d[ok, :k]
    m = (k - 2) / np.log(dk[:, -1:] / dk[:, :-1]).sum(1)  # Levina-Bickel with the MacKay-Ghahramani correction
    mle[k] = float(1 / np.mean(1 / m))
res = {"twonn": twonn, "mle": mle, "queries": Q, "n": n, "duplicates_skipped": int((~ok).sum())}
(OUT / "intrinsic.json").write_text(json.dumps(res, indent=1)); print(res, flush=True)

# Labels per row: journal field (landscape) and HGNC genes.
L = REPO / "genesearch/index/landscape"
lp, lf = np.load(L / "pmid.npy"), np.load(L / "field.npy")
i = np.searchsorted(lp, pm).clip(0, len(lp) - 1); on = lp[i] == pm
field = np.full(n, -1, np.int8); field[on] = lf[i[on]]; np.save(OUT / "field.npy", field)
g = pq.read_table(REPO / "data/geometry/human_gene_pmids.parquet", columns=["pmid", "ncbi_genes"])
gp, gl = g.column("pmid").to_numpy(), g.column("ncbi_genes").combine_chunks()
j = np.searchsorted(gp, pm); assert (gp[j] == pm).all()
offs = gl.offsets.to_numpy(); vals = gl.values.to_numpy(zero_copy_only=False).astype(np.int64)
lens = (offs[1:] - offs[:-1])[j]
ptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
idx = np.repeat(offs[:-1][j] - ptr[:-1], lens) + np.arange(ptr[-1])
genes = vals[idx]
np.save(OUT / "genes_ptr.npy", ptr); np.save(OUT / "genes.npy", genes.astype(np.int32))
print(f"fields on landscape {on.mean():.1%}, labelled {(field > 0).mean():.1%}; gene links {len(genes):,}; {time.time() - t0:.0f}s")
