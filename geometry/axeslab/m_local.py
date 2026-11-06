"""local: the subject's OWN main directions of variation as Map Explorer axes (the baseline family).

Variants, each recorded for all 5 subjects as local_<variant>:
  pca      PC1 and PC2 of the subject's papers, centred on the subject's mean.
  varimax  The top-3 PCs rotated to 'simple structure': varimax on the papers' row-normalised (Kaiser) PC scores, so
           each paper sits mostly on one axis. The 2 rotated axes with the most variance; they are orthogonal, so the
           explorer's 'up made perpendicular' rule changes nothing.
  ica      FastICA (logcosh) on the whitened top-15 PCs. Keep the components that come back in 6 refits on random
           halves (median best-match |cos| >= 0.9), show the 2 most non-Gaussian of those (negentropy), the one with
           more variance across. In practice: the subject's two sharpest niches (+ end = the niche, - end = the rest).
All directions are signed so the longer tail is the + end.

Stability: each variant is refitted on 3 random halves of the papers; |cos| between the half-fit and full-fit axes as
drawn (across, and up made perpendicular to across), median of the 3. 'plane' = the smaller cosine of the principal
angles between the two 2-D planes (1 = same plane, even if the axes rotated inside it).

Run: cd geometry/axeslab && nice -n 10 ../../.venv/bin/python m_local.py [variant ...]
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "8"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import kurtosis, skew  # noqa: E402
from sklearn.decomposition import FastICA  # noqa: E402

import axislab as al  # noqa: E402

warnings.filterwarnings("ignore", module="sklearn")
EG_LOGCOSH = 0.3745672  # E[log cosh z] for z ~ N(0, 1)


def top_pcs(X: np.ndarray, k: int):
    """(k x 384 principal directions, their variances, the centred papers)."""
    Xc = X - X.mean(0)
    ev, V = np.linalg.eigh(Xc.T @ Xc / len(X))
    o = np.argsort(ev)[::-1][:k]
    return V[:, o].T, ev[o], Xc


def orient(D: np.ndarray, Xc: np.ndarray) -> np.ndarray:
    """Sign each direction so its longer tail is the + end (deterministic; for ICA that is the niche end)."""
    return D * np.where(skew(Xc @ D.T, axis=0) < 0, -1.0, 1.0)[:, None]


def negentropy(s: np.ndarray) -> np.ndarray:
    z = (s - s.mean(0)) / s.std(0)
    return (np.log(np.cosh(z)).mean(0) - EG_LOGCOSH) ** 2


def varimax(L: np.ndarray, it: int = 500, tol: float = 1e-8) -> np.ndarray:
    """Orthogonal rotation R maximising the variance of the squared entries of L @ R (Kaiser's varimax)."""
    R, d = np.eye(L.shape[1]), 0.0
    for _ in range(it):
        Lr = L @ R
        u, s, vt = np.linalg.svd(L.T @ (Lr ** 3 - Lr * (Lr ** 2).mean(0)))
        R = u @ vt
        if s.sum() < d * (1 + tol):
            break
        d = s.sum()
    return R


def fit_pca(X):
    V, ev, Xc = top_pcs(X, 2)
    sh = ev / Xc.var(0).sum()
    return orient(V, Xc), [f"PC1 of the subject's papers ({sh[0]:.1%} of their variance)",
                           f"PC2 of the subject's papers ({sh[1]:.1%})"], {}


def fit_varimax(X, k: int = 3):
    V, ev, Xc = top_pcs(X, k)
    Y = Xc @ V.T
    D = orient(varimax(Y / np.linalg.norm(Y, axis=1, keepdims=True)).T @ V, Xc)
    var = (Xc @ D.T).var(0)
    j = np.argsort(-var)[:2]
    sh = var[j] / Xc.var(0).sum()
    return D[j], [f"Varimax-rotated top-{k} PCs: the rotated axis with most variance ({sh[0]:.1%})",
                  f"Varimax-rotated top-{k} PCs: second ({sh[1]:.1%})"], {"k": k}


def ica_filters(X, k: int, seed: int):
    """FastICA on the whitened top-k PCs; returns unit 384-d filters (paper . filter = the component) and Xc."""
    V, ev, Xc = top_pcs(X, k)
    W = FastICA(whiten=False, fun="logcosh", random_state=seed, max_iter=1000, tol=1e-4).fit(Xc @ V.T / np.sqrt(ev)).components_
    F = (W / np.sqrt(ev)) @ V
    return orient(F / np.linalg.norm(F, axis=1, keepdims=True), Xc), Xc


def fit_ica(X, k: int = 15, R: int = 6, thr: float = 0.9, seed: int = 0):
    F, Xc = ica_filters(X, k, seed)
    rng = np.random.default_rng(seed)
    rep = np.median([np.abs(F @ ica_filters(X[rng.permutation(len(X))[:len(X) // 2]], k, seed + 1 + r)[0].T).max(1)
                     for r in range(R)], axis=0)
    s = Xc @ F.T
    ng, var, ku = negentropy(s), s.var(0), kurtosis(s, axis=0)
    ok = rep >= thr
    if ok.sum() < 2:  # small subjects: nothing reproducible enough; fall back to the two most reproducible
        ok = rep >= np.sort(rep)[-2]
    j = np.argsort(np.where(ok, -ng, np.inf))[:2]
    j = j[np.argsort(-var[j])]
    z = (s[:, j] - s[:, j].mean(0)) / s[:, j].std(0)
    tail = (z > 2.5).sum(0)
    why = [f"ICA on top-{k} PCs: most non-Gaussian reproducible component (kurtosis {ku[j[0]]:.1f}, "
           f"{tail[0]} papers beyond +2.5 sd, reproducibility {rep[j[0]]:.2f})",
           f"ICA: next (kurtosis {ku[j[1]]:.1f}, {tail[1]} papers beyond +2.5 sd, reproducibility {rep[j[1]]:.2f})"]
    return F[j], why, {"k": k, "n_reproducible": int((rep >= thr).sum()), "reproducibility": np.round(rep[j], 3).tolist(),
                       "kurtosis": np.round(ku[j], 2).tolist(), "tail_2.5sd": tail.tolist()}


VARIANTS = {"pca": fit_pca, "varimax": fit_varimax, "ica": fit_ica}


def drawn(D):
    return np.array([al.unit(D[0]), al.perp(D[1], D[0])])


def stability(fit, X, reps: int = 3, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    full, out = drawn(fit(X)[0]), []
    for _ in range(reps):
        h = drawn(fit(X[rng.permutation(len(X))[:len(X) // 2]])[0])
        out.append([abs(full[0] @ h[0]), abs(full[1] @ h[1]), np.linalg.svd(full @ h.T, compute_uv=False).min()])
    m = np.median(out, axis=0)
    return {"across": round(float(m[0]), 3), "up": round(float(m[1]), 3), "plane": round(float(m[2]), 3)}


def main(variants):
    summary = {}
    for v in variants:
        method, fit = f"local_{v}", VARIANTS[v]
        for name in al.SUBJECTS:
            S = al.load(name)
            X = S["X"].astype(np.float64)
            t0 = time.time()
            D, why, info = fit(X)
            sec = time.time() - t0
            st = stability(fit, X)
            note = (f"local family, variant {v}; fit {sec:.2f} s on {len(X):,} papers (8 threads, shared box); "
                    f"half-fit stability |cos| across {st['across']}, up {st['up']}, plane {st['plane']}; {json.dumps(info)}")
            res = al.record(method, S, [D[0], D[1]], why=why, note=note)
            res.update(stability_half_fit=st, fit_seconds=round(sec, 3), info=info)  # extra keys, same file
            (al.RESULTS / method / f"{name}.json").write_text(json.dumps(res, indent=1))
            m = res["metrics"]
            summary.setdefault(method, {})[name] = {"eta2": m["eta2"], "var_x": m["var_x"], "var_y": m["var_y"],
                                                    "r_xy": m["r_xy"], "stability": st, "fit_s": round(sec, 2),
                                                    "names": [a["name"]["axis"] for a in res["axes"]],
                                                    "year_rho": [a["year_rho"] for a in res["axes"]]}
            print(f"{method:14s} {name:12s} eta2 {m['eta2']:.3f} var {m['var_x']:.3f}/{m['var_y']:.3f} r {m['r_xy']:+.2f} "
                  f"stab {st['across']:.2f}/{st['up']:.2f}/{st['plane']:.2f} fit {sec:.2f}s | "
                  + " || ".join(a["name"]["axis"] for a in res["axes"]), flush=True)
        med = {k: round(float(np.median([summary[method][n]["stability"][k] for n in al.SUBJECTS])), 3)
               for k in ("across", "up", "plane")}
        summary[method]["median_stability"] = med
        print(f"{method}: median half-fit |cos| over subjects: {med}", flush=True)
    f = al.RESULTS / "local_summary.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    f.write_text(json.dumps({**old, **summary}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or list(VARIANTS))
