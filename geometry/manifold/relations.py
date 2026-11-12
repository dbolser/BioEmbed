"""Relation themes: between-topic vectors that recur in different parts of the map (analogies between topics).

Like directions.py, but with a custom distance so that a theme has to recur across the map:
  d(e, f) = 1 - |cos(v_e, v_f)|   if edges e and f share no topic and no endpoint region
          = 1                     otherwise (same region or a shared topic cannot vouch for each other)
Regions: k-means (--regions, default 20) of the topics' centred directions. HDBSCAN (sklearn, precomputed) on
that matrix. The same run on Gaussian centroids with the real covariance (--null) calibrates how many themes
chance alone gives. A theme's axis is the axial mean (top eigenvector of sum v v^T) of its edges.

Before the edge vectors are taken, topic positions can be expressed in a whitened top-PC space (--pcs, --whiten)
so the strongest axis does not take every edge (directions.py's first run: one loose theme of 3,107 edges, mostly the
centroids' second principal axis, B1).
Writes data/geometry/medembed_small/manifold/<name>_relations_<tag>.{csv,npz}
CPU, root env: uv run python geometry/manifold/relations.py eom_mcs200_ms3 [--k 10] [--null]
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


def topic_names(name, keep):
    summ = pd.read_csv(D / f"clusters/{name}_summary.csv").set_index("cluster")
    lf = M / f"{name}_labels.csv"  # LLM names, when label_topics has run
    llm = pd.read_csv(lf).set_index("cluster").label if lf.exists() else {}
    return np.array([llm.get(c) if c in llm else f"{summ.genes.get(c, '')} | {summ.field.get(c, '')}" for c in keep], dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--k", type=int, default=10); ap.add_argument("--pcs", type=int, default=0)
    ap.add_argument("--whiten", action="store_true"); ap.add_argument("--regions", type=int, default=20)
    ap.add_argument("--mcs", type=int, default=8); ap.add_argument("--ms", type=int, default=3)
    ap.add_argument("--null", action="store_true"); ap.add_argument("--show", type=int, default=30)
    ap.add_argument("--mutual", action="store_true")
    a = ap.parse_args()
    tag = f"k{a.k}_pc{a.pcs}{'_w' if a.whiten else ''}_r{a.regions}{'_m' if a.mutual else ''}{'_null' if a.null else ''}"
    S, n = np.load(M / f"{a.name}_sum.npy").astype(np.float64), np.load(M / f"{a.name}_size.npy")
    mu = np.load(M / "mean.npy").astype(np.float64)
    keep = np.arange(len(n)); R = S / n[:, None] - mu
    Rc = R - R.mean(0); U, Sv, Vt = np.linalg.svd(Rc, full_matrices=False)
    if a.null:
        Rc = (np.random.default_rng(0).standard_normal((len(Rc), len(Sv))) * Sv / np.sqrt(len(Rc) - 1)) @ Vt
    P = Rc @ Vt.T
    if a.pcs: P = P[:, :a.pcs]
    scale = P.std(0) if a.whiten else np.ones(P.shape[1])
    P = P / scale
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    sim = Pn @ Pn.T; np.fill_diagonal(sim, -9)
    nb = np.argsort(-sim, 1)[:, : a.k]
    E = np.stack([np.repeat(np.arange(len(P)), a.k), nb.ravel()], 1)
    if a.mutual:  # keep i-j only if each is among the other's k nearest: no hubs collecting one-way edges
        isnb = np.zeros((len(P), len(P)), bool); isnb[E[:, 0], E[:, 1]] = True
        E = E[isnb[E[:, 1], E[:, 0]]]
    E = np.unique(np.sort(E, 1), axis=0)
    V = P[E[:, 1]] - P[E[:, 0]]; V /= np.linalg.norm(V, axis=1, keepdims=True)
    reg = KMeans(a.regions, n_init=4, random_state=0).fit_predict(Pn)
    r0, r1 = reg[E[:, 0]], reg[E[:, 1]]
    Dm = 1 - np.abs(V @ V.T)
    blocked = np.zeros_like(Dm, bool)
    for x in (E[:, 0], E[:, 1]):
        for y in (E[:, 0], E[:, 1]):
            blocked |= x[:, None] == y[None, :]
    for x in (r0, r1):
        for y in (r0, r1):
            blocked |= x[:, None] == y[None, :]
    Dm[blocked] = 1.0; np.fill_diagonal(Dm, 1); np.clip(Dm, 0, 1, out=Dm)
    best = 1 - Dm.min(1)  # each edge's most parallel partner elsewhere on the map
    log(f"edges with a parallel partner elsewhere: |cos| > 0.5 {np.mean(best > 0.5):.1%}, > 0.6 {np.mean(best > 0.6):.1%}, "
        f"> 0.7 {np.mean(best > 0.7):.1%}")
    np.fill_diagonal(Dm, 0)
    lab = HDBSCAN(metric="precomputed", min_cluster_size=a.mcs, min_samples=a.ms, copy=True).fit_predict(Dm)
    nt = lab.max() + 1
    log(f"{a.name} [{tag}]: {len(P)} topics, {len(E)} edges, {nt} themes, {np.mean(lab >= 0):.1%} of edges in one")
    names = topic_names(a.name, keep)
    rows, axes = [], []
    for t in range(nt):
        m = lab == t; Vm = V[m]
        ax = np.linalg.eigh(Vm.T @ Vm)[1][:, -1]
        coh = float(np.abs(Vm @ ax).mean())
        regs = np.unique(np.concatenate([r0[m], r1[m]]))
        # orient each edge along the axis: (from, to) with the 'to' end at +
        s = np.sign(Vm @ ax); fr = np.where(s > 0, E[m, 0], E[m, 1]); to = np.where(s > 0, E[m, 1], E[m, 0])
        ex = " || ".join(f"{names[f]} -> {names[t_]}" for f, t_ in list(zip(fr, to))[:6])
        # star: share of the theme's edges that touch its most common topic (1 = one hub to many, not an analogy)
        star = float(np.bincount(E[m].ravel()).max() / m.sum())
        rows.append(dict(theme=t, edges=int(m.sum()), regions=len(regs), coherence=round(coh, 3),
                         star=round(star, 2), examples=ex))
        axes.append(ax * scale if a.whiten else ax)  # back to (unwhitened) PC coordinates
    df = pd.DataFrame(rows)
    df["analogy"] = (df.star <= 0.34) & (df.regions >= 3) & (df.coherence >= 0.5)
    df = df.sort_values(["analogy", "regions", "edges"], ascending=False)
    log(f"analogy themes (star <= 0.34, >= 3 regions, coherence >= 0.5): {int(df.analogy.sum())}")
    pd.set_option("display.width", 250, "display.max_colwidth", 400)
    print(df.drop(columns="examples").to_string(index=False))
    for r in df.head(a.show).itertuples():
        print(f"\n#{r.theme} edges {r.edges}, regions {r.regions}, coh {r.coherence}\n  {r.examples}")
    df.to_csv(M / f"{a.name}_relations_{tag}.csv", index=False)
    ax384 = (np.array(axes) @ Vt[: P.shape[1]]) if axes else np.zeros((0, 384))
    ax384 = ax384 / np.maximum(np.linalg.norm(ax384, axis=1, keepdims=True), 1e-12)
    np.savez(M / f"{a.name}_relations_{tag}.npz", edges=E, V=V, lab=lab, axes384=ax384, region=reg)


if __name__ == "__main__":
    main()
