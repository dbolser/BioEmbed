"""Themes from between-topic vectors: cluster the directions that lead from one topic centroid to its neighbours.

1. Topic positions: centroids (centroids.py) minus the mean of all papers (every MedEmbed paper shares one big
   direction: the mean vector has length 0.79), optionally kept to the top --pcs principal components.
2. Edges: each topic to its --k nearest topics (cosine of the centred centroids), undirected.
3. Between-topic vector of an edge (i, j): r_j - r_i, unit length. An edge has no natural direction, so v and -v
   are the same axis: distance = 1 - |cos| (sign-invariant), as a precomputed matrix.
4. HDBSCAN (sklearn, metric="precomputed") on that matrix. A theme = one cluster of edges; its axis = the top
   eigenvector of sum(v v^T) over its edges (the Watson / axial mean), sign fixed so the bigger topic mass is at +.
5. Per theme: edges, distinct topics touched, coherence (mean |cos| of member edges to the axis), spread (how many
   different regions of the map its edges come from: participation ratio of edge midpoints' coarse regions),
   the axis's overlap with the top principal components, and the topics at each pole.
Null: the same on centroids drawn from a Gaussian with the real covariance (no structure beyond the PCA spectrum).

CPU, root env: uv run python geometry/manifold/directions.py eom_mcs200_ms3 [--k 10] [--pcs 0] [--whiten]
Writes data/geometry/medembed_small/manifold/<name>_themes_<tag>.{npz,csv}
"""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import HDBSCAN, KMeans

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/geometry/medembed_small"
M = D / "manifold"
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--k", type=int, default=10); ap.add_argument("--pcs", type=int, default=0)
ap.add_argument("--whiten", action="store_true"); ap.add_argument("--mcs", type=int, default=15)
ap.add_argument("--ms", type=int, default=5); ap.add_argument("--null", action="store_true")
ap.add_argument("--min-size", type=int, default=0, help="drop topics smaller than this")
a = ap.parse_args()
tag = f"k{a.k}_pc{a.pcs}{'_w' if a.whiten else ''}{'_null' if a.null else ''}"

S, n = np.load(M / f"{a.name}_sum.npy").astype(np.float64), np.load(M / f"{a.name}_size.npy")
mu = np.load(M / "mean.npy").astype(np.float64)
keep = np.flatnonzero(n >= max(a.min_size, 1))
R = S[keep] / n[keep, None] - mu
summ = pd.read_csv(D / f"clusters/{a.name}_summary.csv").set_index("cluster")
names = np.array([f"{summ.genes.get(c, '')} | {summ.field.get(c, '')}".strip(" |") for c in keep], dtype=object)

Rc = R - R.mean(0)
U, Sv, Vt = np.linalg.svd(Rc, full_matrices=False)
if a.null:  # Gaussian centroids with the same covariance
    rng = np.random.default_rng(0)
    R = R.mean(0) + (rng.standard_normal((len(R), len(Sv))) * Sv / np.sqrt(len(R) - 1)) @ Vt
    Rc = R - R.mean(0)
P = Rc @ Vt.T  # coordinates on the principal axes
if a.pcs: P = P[:, :a.pcs]
if a.whiten: P = P / P.std(0)
k_top = Vt[: P.shape[1]]

# topic kNN graph on cosine of centred positions
Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
sim = Pn @ Pn.T; np.fill_diagonal(sim, -9)
nb = np.argsort(-sim, 1)[:, : a.k]
E = np.unique(np.sort(np.stack([np.repeat(np.arange(len(P)), a.k), nb.ravel()], 1), 1), axis=0)
V = P[E[:, 1]] - P[E[:, 0]]; V /= np.linalg.norm(V, axis=1, keepdims=True)
log(f"{a.name} [{tag}]: {len(P)} topics in {P.shape[1]}-D, {len(E)} edges")

Dm = 1 - np.abs(V @ V.T); np.clip(Dm, 0, None, out=Dm); np.fill_diagonal(Dm, 0)
lab = HDBSCAN(metric="precomputed", min_cluster_size=a.mcs, min_samples=a.ms).fit_predict(Dm)
nt = lab.max() + 1
log(f"themes {nt}, edges in a theme {np.mean(lab >= 0):.1%}")

# coarse regions of the topic map for "spread": k-means of topic positions into 20 regions
region = KMeans(20, n_init=4, random_state=0).fit_predict(Pn)
rows, axes = [], []
for t in range(nt):
    m = lab == t; Vm = V[m]
    w, ev = np.linalg.eigh(Vm.T @ Vm); ax = ev[:, -1]
    proj = P @ ax
    if (n[keep] * proj).sum() < 0: ax, proj = -ax, -proj
    coh = float(np.abs(Vm @ ax).mean())
    tops = np.unique(E[m]); reg = np.bincount(region[tops], minlength=20) / len(tops)
    spread = float(1 / np.sum(reg ** 2))
    # overlap with principal axes (in the working space, axis coordinate j = PC j when not whitened)
    pc_share = ax ** 2
    o = np.argsort(proj)
    rows.append(dict(theme=t, edges=int(m.sum()), topics=len(tops), coherence=round(coh, 3), spread=round(spread, 1),
                     top_pc=int(np.argmax(pc_share)), top_pc_share=round(float(pc_share.max()), 2),
                     plus=" ; ".join(names[o[-4:][::-1]]), minus=" ; ".join(names[o[:4]])))
    axes.append(ax)
df = pd.DataFrame(rows).sort_values("edges", ascending=False)
pd.set_option("display.width", 250, "display.max_colwidth", 90, "display.max_rows", 200)
print(df.drop(columns=["plus", "minus"]).to_string(index=False))
for r in df.head(25).itertuples():
    print(f"\n#{r.theme} ({r.edges} edges, coh {r.coherence}, spread {r.spread}, PC{r.top_pc} {r.top_pc_share})\n"
          f"  + {r.plus}\n  - {r.minus}")
df.to_csv(M / f"{a.name}_themes_{tag}.csv", index=False)
np.savez(M / f"{a.name}_themes_{tag}.npz", edges=keep[E], V=V, lab=lab, axes=np.array(axes), keep=keep,
         basis=k_top, scale=(P.std(0) if a.whiten else np.ones(P.shape[1])))
log("done")
