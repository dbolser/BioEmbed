"""Analogy themes, calibrated against a null: groups of between-topic vectors more parallel than chance allows.

relations.py showed that between-topic vectors (topic -> neighbouring topic) have parallel partners elsewhere on
the map far more often than in a null (Gaussian centroids with the real covariance): 10.9% against 0.0% at
|cos| > 0.6. HDBSCAN on that sparse signal finds about as many themes in the null as in the real data, so this
script uses the null directly:
  1. Edges: mutual k-nearest topics (no hubs). Vector = unit difference of the centred centroids.
  2. Two edges may vouch for each other only if they share no topic and no endpoint region (k-means regions).
  3. Threshold = the largest |cos| between any allowed pair of edges in --nulls null draws (a family-wise bound:
     in the null, no pair at all passes it).
  4. Themes = connected components of the "more parallel than the null ever is" graph, with >= 3 edges.
A theme's axis is the axial mean of its edges, signed so its edges point mostly from the - to the + end.
Writes data/geometry/medembed_small/manifold/<name>_analogies.{json,npz}
CPU, root env: uv run python geometry/manifold/analogies.py eom_mcs200_ms3 [--k 10] [--regions 20] [--nulls 5]
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/geometry/medembed_small"
M = D / "manifold"
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def graph(P, k, regions, seed=0):
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    sim = Pn @ Pn.T; np.fill_diagonal(sim, -9)
    nb = np.argsort(-sim, 1)[:, :k]
    E = np.stack([np.repeat(np.arange(len(P)), k), nb.ravel()], 1)
    isnb = np.zeros((len(P), len(P)), bool); isnb[E[:, 0], E[:, 1]] = True
    E = np.unique(np.sort(E[isnb[E[:, 1], E[:, 0]]], 1), axis=0)
    V = P[E[:, 1]] - P[E[:, 0]]; V /= np.linalg.norm(V, axis=1, keepdims=True)
    reg = KMeans(regions, n_init=4, random_state=seed).fit_predict(Pn)
    A = np.abs(V @ V.T)
    ends = [E[:, 0], E[:, 1]]; rends = [reg[E[:, 0]], reg[E[:, 1]]]
    for x in ends:
        for y in ends: A[x[:, None] == y[None, :]] = 0
    for x in rends:
        for y in rends: A[x[:, None] == y[None, :]] = 0
    np.fill_diagonal(A, 0)
    return E, V, A, reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--k", type=int, default=10); ap.add_argument("--regions", type=int, default=20)
    ap.add_argument("--nulls", type=int, default=5); ap.add_argument("--min-edges", type=int, default=3)
    a = ap.parse_args()
    S, n = np.load(M / f"{a.name}_sum.npy").astype(np.float64), np.load(M / f"{a.name}_size.npy")
    mu = np.load(M / "mean.npy").astype(np.float64)
    R = S / n[:, None] - mu; Rc = R - R.mean(0)
    U, Sv, Vt = np.linalg.svd(Rc, full_matrices=False)
    null_max = []
    for s in range(a.nulls):
        Pn_ = np.random.default_rng(100 + s).standard_normal((len(Rc), len(Sv))) * Sv / np.sqrt(len(Rc) - 1)
        _, _, An, _ = graph(Pn_, a.k, a.regions, seed=s)
        null_max.append(float(An.max())); log(f"null {s}: largest allowed |cos| {null_max[-1]:.3f}")
    thr = max(null_max)
    P = Rc @ Vt.T
    E, V, A, reg = graph(P, a.k, a.regions)
    adj = A > thr
    log(f"real: {len(E)} edges; pairs above the null's maximum {thr:.3f}: {adj.sum() // 2}; "
        f"edges with such a partner {np.mean(adj.any(1)):.1%}")
    def coherence(m):
        Vm = V[m]; ax = np.linalg.eigh(Vm.T @ Vm)[1][:, -1]
        return float(np.abs(Vm @ ax).mean())

    def split(m, t):
        """Components chain different relations together (single linkage): while a big group is loose
        (coherence < 0.6), raise the bar inside it and take the components again."""
        if len(m) <= 10 or coherence(m) >= 0.6 or t > 0.95: return [m]
        nc_, c_ = connected_components(csr_matrix(A[np.ix_(m, m)] > t + 0.03), directed=False)
        return [p for c in range(nc_) for p in split(m[c_ == c], t + 0.03)]

    nc, comp = connected_components(csr_matrix(adj), directed=False)
    groups = [p for c in range(nc) for p in split(np.flatnonzero(comp == c), thr)]
    comp = np.full(len(E), -1)
    lf = M / f"{a.name}_labels.csv"
    names = pd.read_csv(lf).set_index("cluster").label.to_dict() if lf.exists() else {}
    nm = lambda c: names.get(int(c), str(c))
    themes = []
    for m in groups:
        if len(m) < a.min_edges: continue
        comp[m] = len(themes)
        Vm = V[m]; ax = np.linalg.eigh(Vm.T @ Vm)[1][:, -1]
        s = np.sign(Vm @ ax)
        fr = np.where(s > 0, E[m, 0], E[m, 1]); to = np.where(s > 0, E[m, 1], E[m, 0])
        regs = np.unique(np.concatenate([reg[E[m, 0]], reg[E[m, 1]]]))
        star = float(np.bincount(E[m].ravel()).max() / len(m))
        themes.append(dict(edges=len(m), regions=len(regs), coherence=round(float(np.abs(Vm @ ax).mean()), 3),
                           star=round(star, 2), pairs=[(nm(f), nm(t)) for f, t in zip(fr, to)],
                           axis=(ax @ Vt).tolist()))
    # Analogy completion, leave one edge out: start at the edge's "from" topic, add the theme's mean vector built
    # from its OTHER edges (scaled to their median length), and rank the true "to" topic among all topics by
    # distance to that point (the start topic excluded). Placebo: the same with another theme's vector.
    def completion(fr, to, others_vec, scale):
        q = P[fr] + scale * others_vec
        d = np.linalg.norm(P - q, axis=1); d[fr] = np.inf
        return int((d < d[to]).sum())
    rng = np.random.default_rng(1)
    for ti, t in enumerate(themes):
        m = np.flatnonzero(comp == ti); ranks, placebo, base = [], [], []
        ax = np.asarray(t["axis"]) @ Vt.T  # back to PC coordinates
        o = lambda x: (E[x, 0], E[x, 1]) if (P[E[x, 1]] - P[E[x, 0]]) @ ax >= 0 else (E[x, 1], E[x, 0])
        for j, e in enumerate(m):
            rest = np.delete(m, j)
            f_, t_ = o(e)  # every edge oriented along the theme's axis
            vecs = np.array([P[o(x)[1]] - P[o(x)[0]] for x in rest])
            scale = np.median(np.linalg.norm(vecs, axis=1)); mv = vecs.mean(0); mv /= np.linalg.norm(mv)
            ranks.append(completion(f_, t_, mv, scale))
            other = themes[rng.choice([i for i in range(len(themes)) if i != ti])]
            ov = np.asarray(other["axis"]) @ Vt.T; ov /= np.linalg.norm(ov)
            placebo.append(min(completion(f_, t_, ov, scale), completion(f_, t_, -ov, scale)))
            base.append(completion(f_, t_, np.zeros_like(mv), 0))  # just the start topic's own neighbours
        t["completion"] = dict(top1=float(np.mean(np.array(ranks) == 0)), top5=float(np.mean(np.array(ranks) < 5)),
                               nearest_top1=float(np.mean(np.array(base) == 0)), nearest_top5=float(np.mean(np.array(base) < 5)),
                               placebo_top5=float(np.mean(np.array(placebo) < 5)))
    for i, t in enumerate(themes): t["cid"] = i
    themes.sort(key=lambda t: -t["edges"])
    remap = np.full(len(themes) + 1, -1); remap[[t["cid"] for t in themes]] = np.arange(len(themes))
    comp = np.where(comp >= 0, remap[comp], -1)  # theme ids now follow the sorted order (A0 = largest)
    allc =[(t["edges"], t["completion"]) for t in themes]
    w = np.array([e for e, _ in allc], float)
    for key in ("top1", "top5", "nearest_top1", "nearest_top5", "placebo_top5"):
        log(f"analogy completion {key}: {np.average([c[key] for _, c in allc], weights=w):.1%} (edge-weighted)")
    for i, t in enumerate(themes):
        print(f"\nA{i}: {t['edges']} edges, {t['regions']} regions, coherence {t['coherence']}, star {t['star']}")
        for f, to_ in t["pairs"][:8]: print(f"    {f}  ->  {to_}")
    log(f"{len(themes)} themes with >= {a.min_edges} edges; null threshold {thr:.3f} (max over {a.nulls} null draws)")
    (M / f"{a.name}_analogies.json").write_text(json.dumps(dict(threshold=thr, null_max=null_max,
        themes=[{k: v for k, v in t.items() if k != "axis"} for t in themes]), indent=1))
    np.savez(M / f"{a.name}_analogies.npz", axes=np.array([t["axis"] for t in themes]), edges=E, comp=comp)


if __name__ == "__main__":
    main()
