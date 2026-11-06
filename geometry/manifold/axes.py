"""Common within-topic axes ("themes behind the clusters") and per-topic time vectors.

For every topic g (cluster): its papers' scatter around the centroid, C_g = cov(x - c_g), and its time vector,
t_g = the least-squares slope of the paper vectors on publication year (384-D per year, within the topic).
  W = mean over topics of C_g / trace(C_g)   (each topic weighs the same; unit trace so no topic dominates)
  B = covariance of the centred centroids     (where topics sit relative to each other)
Eigenvectors of W are directions that papers vary along INSIDE topics, shared across topics: the candidate
cross-cutting themes (e.g. study type, organism, time). Eigenvectors of B are the region axes.
Null for "shared": each topic's own top-m subspace overlaps its neighbours' by chance only ~m/384; we report,
per W axis, the share of topics for which the axis is in their own top-20 local subspace (projection energy).

Writes data/geometry/medembed_small/manifold/<name>_axes.npz:
  W_vals, W_vecs (384, 384, ascending order flipped to descending), B_vals, B_vecs,
  tvec (k, 384) per-topic time slope, tn (k,) papers with a year, ybar (k,), share_top20 (384,) per W axis.
CPU, root env: uv run python geometry/manifold/axes.py eom_mcs200_ms3
"""
import sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/geometry/medembed_small"
M = D / "manifold"
torch.set_num_threads(16)
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
name = sys.argv[1]

X = np.load(D / "X.npy", mmap_mode="r")
lab = np.load(D / f"clusters/{name}_all.npy").astype(np.int64)
year = np.load(M / "year.npy").astype(np.float64)
k = lab.max() + 1
order = np.argsort(lab, kind="stable"); bounds = np.searchsorted(lab[order], np.arange(k + 1))
Cg = np.zeros((k, 384, 384), np.float64); cen = np.zeros((k, 384)); tvec = np.zeros((k, 384))
tn = np.zeros(k, np.int64); ybar = np.zeros(k)
for g in range(k):
    idx = np.sort(order[bounds[g]:bounds[g + 1]])
    xg = torch.from_numpy(np.asarray(X[idx], np.float32)).double()
    c = xg.mean(0); r = xg - c
    Cg[g] = (r.T @ r / len(idx)).numpy(); cen[g] = c.numpy()
    y = year[idx]; ok = y > 0
    if ok.sum() > 10:
        yc = torch.from_numpy(y[ok] - y[ok].mean())
        tvec[g] = ((yc[:, None] * r[torch.from_numpy(ok)]).sum(0) / (yc ** 2).sum()).numpy()
        tn[g] = ok.sum(); ybar[g] = y[ok].mean()
    if g % 100 == 0: log(f"topic {g}/{k}")
tr = np.trace(Cg, axis1=1, axis2=2)
W = (Cg / tr[:, None, None]).mean(0)
wv, wV = np.linalg.eigh(W); wv, wV = wv[::-1], wV[:, ::-1]
mu = np.load(M / "mean.npy").astype(np.float64)
Rc = cen - cen.mean(0)
bv, bV = np.linalg.eigh(Rc.T @ Rc / k); bv, bV = bv[::-1], bV[:, ::-1]
# per W axis: share of topics whose own top-20 subspace holds most of it (energy >= 0.5)
energy = np.zeros((k, 384))
for g in range(k):
    v, Vg = np.linalg.eigh(Cg[g]); top = Vg[:, -20:]
    energy[g] = ((top.T @ wV) ** 2).sum(0)
share = (energy >= 0.5).mean(0)
# leave one topic out: W rebuilt without topic g, its axes matched to the full W's by |cos|; is each still mostly
# inside g's own top-20 directions? (W averages the same topics' covariances, so the plain share is partly circular)
loo = np.zeros((k, 6))
for g in range(k):
    Wg = (k * W - Cg[g] / tr[g]) / (k - 1)
    v, Vg = np.linalg.eigh(Wg); Vg = Vg[:, ::-1][:, :12]
    match = Vg[:, np.abs(wV[:, :6].T @ Vg).argmax(1)]  # (384, 6): the LOO axis nearest each full-W axis
    top = np.linalg.eigh(Cg[g])[1][:, -20:]
    loo[g] = ((top.T @ match) ** 2).sum(0) >= 0.5
share_loo = loo.mean(0)
np.savez(M / f"{name}_axes.npz", W_vals=wv, W_vecs=wV, B_vals=bv, B_vecs=bV, tvec=tvec, tn=tn, ybar=ybar,
         share_top20=share, share_top20_loo=share_loo, trace=tr, cen=cen)
log(f"leave-one-topic-out shares for W0-W5: {np.round(share_loo, 2)}")
ev = wv / wv.sum(); evb = bv / bv.sum()
log(f"W: top10 share {np.round(ev[:10], 3)}; participation ratio {1 / np.sum(ev ** 2):.1f}")
log(f"B: top10 share {np.round(evb[:10], 3)}; participation ratio {1 / np.sum(evb ** 2):.1f}")
log(f"W axes in >=50% of topics' own top-20: {[(i, round(float(share[i]), 2)) for i in range(15)]}")
# overlap of the top-20 W subspace with the top-20 B subspace (1 = same, 20/384 ~ 0.05 = chance)
ov = np.sum((wV[:, :20].T @ bV[:, :20]) ** 2) / 20
log(f"top-20 subspace overlap W vs B: {ov:.2f} (chance {20 / 384:.2f})")
tu = tvec / np.maximum(np.linalg.norm(tvec, axis=1, keepdims=True), 1e-12)
ok = tn > 200
gt = tu[ok].mean(0); log(f"time vectors: mean pairwise cos {(np.linalg.norm(gt) ** 2 * ok.sum() - 1) / (ok.sum() - 1):.3f}"
                        f" over {ok.sum()} topics; length of mean unit vector {np.linalg.norm(gt):.3f}")
gt /= np.linalg.norm(gt)
log(f"global time direction: energy in top-5 W axes {np.round((wV[:, :5].T @ gt) ** 2, 3)}, "
    f"cos with mean vector {gt @ mu / np.linalg.norm(mu):.3f}")
