"""Contrastive axes: directions along which this subject's papers vary MORE than human-gene papers in general.

Subject (Cs) and background (Cb, al.background(): 50k random human-gene papers) covariances, each centred on its own
mean. Three variants:
  contrastive_cpca     top-2 eigenvectors of Cs/tr(Cs) - 4 Cb/tr(Cb) (contrastive PCA, both scaled to unit total
                       variance, alpha 4: an axis only earns the subject variance beyond 4x its share in general).
                       alpha <= 1 is plain PCA again (2-D overlap 0.93-0.99); alpha >= 12 the 2nd axis drops into a
                       direction with ~0 variance in both sets. 4 sits in the useful range and names well.
  contrastive_gev      top-2 generalised eigenvectors of Cs v = l (0.8 Cb + 0.2 tr(Cb)/384 I) v: the largest
                       subject/background variance ratio, background shrunk 20% toward isotropy so that near-empty
                       background directions do not win.
  contrastive_gevcore  the same, after removing the subject's own direction (its centroid minus al.MU) from both
                       matrices: the top contrast is often "about the subject <-> off-target or passing mentions", this
                       variant asks what varies among the papers beyond that.
Each axis is signed so its + end is the newer papers. Stability: refit on 5 random halves of the subject's papers
(background fixed) and on 3 random halves of the background; |cos| of each half-fit axis with the full-fit one (the up
axis as the explorer draws it, perpendicular to across). Run (root env, from this directory):
  nice -n 10 ../../.venv/bin/python m_contrastive.py
"""

import os

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "8"

import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.linalg as sl  # noqa: E402

import axislab as al  # noqa: E402

ALPHA, GAMMA, D = 4.0, 0.2, 384
BG = al.background().astype(np.float64)


def cov(X) -> np.ndarray:
    X = np.asarray(X, np.float64)
    Xc = X - X.mean(0)
    return Xc.T @ Xc / (len(X) - 1)


def shrunk(Cb) -> np.ndarray:
    return (1 - GAMMA) * Cb + GAMMA * np.trace(Cb) / D * np.eye(D)


def fit_cpca(X, Cb) -> np.ndarray:
    Cs = cov(X)
    w, V = np.linalg.eigh(Cs / np.trace(Cs) - ALPHA * Cb / np.trace(Cb))
    return V[:, ::-1][:, :2]


def fit_gev(X, Cb, own=False) -> np.ndarray:
    Cs, R = cov(X), shrunk(Cb)
    if own:  # work in the complement of the subject's own direction (keeps the RHS positive definite)
        d = al.unit(np.asarray(X, np.float64).mean(0) - al.MU)
        P = np.eye(D) - np.outer(d, d)
        Cs, R = P @ Cs @ P, P @ R @ P + np.outer(d, d)
    w, V = sl.eigh(Cs, R)
    V = V[:, ::-1][:, :2]
    return V / np.linalg.norm(V, axis=0)


VARIANTS = {
    "contrastive_cpca": (fit_cpca, "contrastive PCA axis {i}: subject share of variance minus 4x the background share"),
    "contrastive_gev": (fit_gev, "generalised eigenvector {i}: subject / background variance ratio (background shrunk 20%)"),
    "contrastive_gevcore": (lambda X, Cb: fit_gev(X, Cb, own=True),
                            "generalised eigenvector {i}, beyond the subject's own direction (centroid minus corpus mean)"),
}


def drawn(V) -> tuple[np.ndarray, np.ndarray]:
    """The two axes as the explorer draws them: across, and up made perpendicular to it."""
    return al.unit(V[:, 0]), al.perp(V[:, 1], V[:, 0])


def oriented(S, V) -> list[np.ndarray]:
    yr = S["year"] > 0
    out = []
    for v in V.T:
        v = al.unit(v)
        out.append(v if np.corrcoef(S["X"][yr] @ v, S["year"][yr])[0, 1] >= 0 else -v)
    return out


def stability(X, Cb, fit, V, halves=5, bg_halves=3) -> tuple[np.ndarray, np.ndarray]:
    a, b = drawn(V)
    sub, bgs = [], []
    for s in range(halves):
        idx = np.random.default_rng(s).permutation(len(X))[: len(X) // 2]
        ah, bh = drawn(fit(X[idx], Cb))
        sub.append((abs(a @ ah), abs(b @ bh)))
    for s in range(bg_halves):
        idx = np.random.default_rng(100 + s).permutation(len(BG))[: len(BG) // 2]
        ah, bh = drawn(fit(X, cov(BG[idx])))
        bgs.append((abs(a @ ah), abs(b @ bh)))
    return np.median(sub, 0), np.median(bgs, 0)


def main():
    Cb = cov(BG)
    A, _ = al.named_axes()
    QA, _ = np.linalg.qr(A.T.astype(np.float64))  # span of the 67 named axes (a random direction has 67/384 = 0.17 in it)
    summary = {m: [] for m in VARIANTS}
    for name in al.SUBJECTS:
        S = al.load(name)
        X = S["X"]
        Cs = cov(X)
        _, Vp = np.linalg.eigh(Cs)
        pca = drawn(Vp[:, ::-1][:, :2])
        for method, (fit, why) in VARIANTS.items():
            t0 = time.perf_counter()
            V = fit(X, Cb)
            secs = time.perf_counter() - t0
            st_sub, st_bg = stability(X, Cb, fit, V)
            a, b = drawn(V)
            ratio = [float((v @ Cs @ v / np.trace(Cs)) / (v @ Cb @ v / np.trace(Cb))) for v in (a, b)]
            named = [float(np.sum((QA.T @ v) ** 2)) for v in (a, b)]
            vs_pca = [float(abs(a @ pca[0])), float(abs(b @ pca[1]))]
            note = (f"fit {secs * 1000:.0f} ms from the papers' vectors and a precomputed 384x384 background covariance. "
                    f"Contrast (subject share of variance / background share): x {ratio[0]:.1f}, y {ratio[1]:.1f}. "
                    f"Half-fit |cos| (5 subject halves): x {st_sub[0]:.2f}, y {st_sub[1]:.2f}; "
                    f"(3 background halves): x {st_bg[0]:.2f}, y {st_bg[1]:.2f}. "
                    f"Share of axis inside the 67 named axes' span (random 0.17): x {named[0]:.2f}, y {named[1]:.2f}. "
                    f"|cos| with the subject's PC1/PC2: {vs_pca[0]:.2f}, {vs_pca[1]:.2f}. + end = newer papers.")
            res = al.record(method, S, oriented(S, V), why=[why.format(i=i) for i in (1, 2)], note=note)
            m = res["metrics"]
            summary[method].append({"subject": name, "eta2": m["eta2"], "var_x": m["var_x"], "var_y": m["var_y"],
                                    "stab": st_sub, "stab_bg": st_bg, "ratio": ratio, "named": named, "pca": vs_pca,
                                    "secs": secs, "names": [ax["name"]["axis"] for ax in res["axes"]]})
            print(f"{method:20s} {name:12s} eta2 {m['eta2']:.3f}  var {m['var_x']:.3f}/{m['var_y']:.3f}  "
                  f"ratio {ratio[0]:.1f}/{ratio[1]:.1f}  half |cos| {st_sub[0]:.2f}/{st_sub[1]:.2f}  "
                  f"bg-half {st_bg[0]:.2f}/{st_bg[1]:.2f}  named {named[0]:.2f}/{named[1]:.2f}  {secs * 1000:.0f} ms\n"
                  f"    x: {res['axes'][0]['name']['axis']}\n    y: {res['axes'][1]['name']['axis']}", flush=True)
    print("\nmedian over subjects: half-fit |cos| x / y (subject halves; background halves), eta2")
    for method, rows in summary.items():
        st = np.median([r["stab"] for r in rows], 0)
        sb = np.median([r["stab_bg"] for r in rows], 0)
        print(f"  {method:20s} {st[0]:.2f} / {st[1]:.2f}   ({sb[0]:.2f} / {sb[1]:.2f})   eta2 "
              f"{np.median([r['eta2'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
