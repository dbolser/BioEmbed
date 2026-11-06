"""Supervised / metadata directions for the Map Explorer, learned from the subject's own papers and their labels.

Three directions, each a 384-d vector (all fits centre the vectors first; the space is a cone):
  time : ridge regression of the year's rank (older -> newer) on the papers' vectors. Ridge strength by 5-fold CV
         of Spearman(predicted, year); we take the strongest ridge within 0.02 of the best, which keeps the
         direction stable and shows more of the papers' spread for almost no loss.
  lda  : Fisher LDA on the map's HDBSCAN topics (noise papers left out). The within-topic scatter is shrunk
         toward (trace/d) I; the shrinkage by 5-fold CV of the held-out 2-D eta2 (same 0.02 rule). LDA1, LDA2 are the
         first two generalised eigenvectors (their scores are uncorrelated within topics).
  cur  : Open Targets curated papers vs the rest, shrunk Fisher (gamma 0.9, close to a difference of means). From
         the subject's own curated papers if it has at least 10; otherwise borrowed from the other subjects
         (pooled, each subject centred on its own mean). The curated papers are genetics papers, so this transfers.

Variants recorded (method names): supervised_time_lda (x time, y LDA1), supervised_lda (x LDA1, y LDA2),
supervised_curated_time (x curated-like, y time). The harness makes y perpendicular to x.
Stability: everything is refit on a random half of the papers; we report |cos| of half- vs full-fit axes (after the
explorer's perpendicular rule) and the half fit's scores on the other half (held-out eta2, year rho, curated AUC).

Run:  cd geometry/axeslab && nice -n 10 ../../.venv/bin/python m_supervised.py
"""

import os

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "8"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.linalg import eigh  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import axislab as al  # noqa: E402

RIDGE_GRID = (0.1, 0.3, 1, 3, 10, 30, 100)  # x mean eigenvalue of the centred scatter
GAMMA_GRID = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
TOL = 0.02  # take the most regularised setting within this of the best CV score
MIN_OWN_CURATED = 10
CUR_GAMMA = 0.9


def folds(n, k=5, seed=0):
    return np.random.default_rng(seed).integers(0, k, n)


# ---------- time ----------

def _ridge(X, t, alphas):
    """Ridge coefficients for each alpha (columns), via the eigendecomposition of the centred scatter."""
    Xc = X - X.mean(0)
    e, V = np.linalg.eigh(Xc.T @ Xc)
    e = np.clip(e, 0, None)
    r = V.T @ (Xc.T @ (t - t.mean()))
    return np.stack([V @ (r / (e + a * e.mean())) for a in alphas], 1)


def time_dir(X, year):
    """Unit direction older -> newer, the chosen alpha and its CV Spearman."""
    ok = year > 0
    X, year = X[ok].astype(np.float64), year[ok]
    t = rankdata(year) / len(year)
    f = folds(len(t))
    pred = np.empty((len(t), len(RIDGE_GRID)))
    for i in range(f.max() + 1):
        tr = f != i
        pred[~tr] = X[~tr] @ _ridge(X[tr], t[tr], RIDGE_GRID)
    cv = np.array([spearmanr(pred[:, j], year)[0] for j in range(len(RIDGE_GRID))])
    j = max(j for j in range(len(RIDGE_GRID)) if cv[j] >= cv.max() - TOL)
    w = _ridge(X, t, [RIDGE_GRID[j]])[:, 0]
    return al.unit(w), RIDGE_GRID[j], float(cv[j])


# ---------- topics (LDA) ----------

def _scatter(X, c):
    """Within- and between-topic scatter (d x d, per paper), topics c >= 0 only."""
    m = c >= 0
    X, c = X[m].astype(np.float64), c[m]
    ks, ci = np.unique(c, return_inverse=True)
    M = np.stack([X[ci == i].mean(0) for i in range(len(ks))])
    nk = np.bincount(ci)
    R = X - M[ci]
    D = M - X.mean(0)
    return R.T @ R / len(X), (D.T * nk) @ D / len(X)


def _lda(Sw, Sb, gamma, k=2):
    d = len(Sw)
    ev, V = eigh(Sb, (1 - gamma) * Sw + gamma * np.trace(Sw) / d * np.eye(d))
    W = V[:, ::-1][:, :k]
    return W / np.linalg.norm(W, axis=0)


def _eta2(X, c, a, b):
    """The harness's 2-D topic eta2 for the explorer's view (a, b made perpendicular to a)."""
    return al.metrics({"X": X, "c": c}, X @ al.unit(a), X @ al.perp(b, a))["eta2"]


def lda_dirs(X, c):
    """LDA1, LDA2 (unit columns), the chosen shrinkage and its CV (held-out) eta2."""
    f = folds(len(c))
    cv = np.zeros(len(GAMMA_GRID))
    for i in range(f.max() + 1):
        tr = f != i
        Sw, Sb = _scatter(X[tr], c[tr])
        for j, g in enumerate(GAMMA_GRID):
            W = _lda(Sw, Sb, g)
            cv[j] += _eta2(X[~tr], c[~tr], W[:, 0], W[:, 1]) / (f.max() + 1)
    j = max(j for j in range(len(GAMMA_GRID)) if cv[j] >= cv.max() - TOL)
    W = _lda(*_scatter(X, c), GAMMA_GRID[j])
    # fix signs: + end toward the largest topic, so runs are comparable
    big = np.bincount(c[c >= 0]).argmax()
    s = np.sign((X[c == big].mean(0) - X.mean(0)) @ W)
    return W * np.where(s == 0, 1, s), GAMMA_GRID[j], float(cv[j])


# ---------- curated ----------

def fisher2(X, y, gamma=CUR_GAMMA):
    """Shrunk two-class Fisher direction, + = curated."""
    X = X.astype(np.float64)
    m1, m0 = X[y].mean(0), X[~y].mean(0)
    R = X - np.where(y[:, None], m1, m0)
    Sw = R.T @ R / len(X)
    d = len(Sw)
    return al.unit(np.linalg.solve((1 - gamma) * Sw + gamma * np.trace(Sw) / d * np.eye(d), m1 - m0))


def cur_dir(name, pool, keep=None):
    """pool: {subject: (X, curated)}; keep: optional {subject: bool mask} (for half fits)."""
    sel = lambda n: keep[n] if keep else slice(None)  # noqa: E731
    X, y = pool[name][0][sel(name)], pool[name][1][sel(name)].astype(bool)
    if pool[name][1].sum() >= MIN_OWN_CURATED:
        return fisher2(X, y), f"own {int(y.sum())} curated"
    others = [n for n in pool if n != name]
    Xo = np.vstack([pool[n][0][sel(n)] - pool[n][0][sel(n)].mean(0) for n in others])
    yo = np.concatenate([pool[n][1][sel(n)].astype(bool) for n in others])
    return fisher2(Xo, yo), f"borrowed: {int(yo.sum())} curated papers of {', '.join(others)}"


# ---------- driver ----------

def fit(S, X, c, year, pool, keep=None):
    """All three directions for one subject (on X, c, year, which may be a half)."""
    t0 = time.time()
    tw, alpha, tcv = time_dir(X, year)
    t1 = time.time()
    W, gamma, ecv = lda_dirs(X, c)
    t2 = time.time()
    cw, src = cur_dir(S["name"], pool, keep)
    return {"time": tw, "lda1": W[:, 0], "lda2": W[:, 1], "cur": cw, "alpha": alpha, "time_cv": tcv,
            "gamma": gamma, "lda_cv": ecv, "cur_src": src, "sec": {"time": t1 - t0, "lda": t2 - t1}}


VARIANTS = {"supervised_time_lda": ("time", "lda1"), "supervised_lda": ("lda1", "lda2"),
            "supervised_curated_time": ("cur", "time")}


def why(F, k):
    return {"time": f"time: ridge of year rank on centred vectors (alpha {F['alpha']}, CV Spearman {F['time_cv']:.2f})",
            "lda1": f"LDA1 of the map's topics (shrinkage {F['gamma']}, CV 2-D eta2 {F['lda_cv']:.2f})",
            "lda2": f"LDA2 of the map's topics (shrinkage {F['gamma']})",
            "cur": f"curated vs rest, shrunk Fisher ({F['cur_src']})"}[k]


def main(only=None):
    pool = {n: (al.load(n)["X"], al.load(n)["curated"]) for n in al.SUBJECTS}
    summary = {}
    for name in al.SUBJECTS:
        S = al.load(name)
        X, c, year, cur = S["X"], S["c"], S["year"], S["curated"].astype(bool)
        F = fit(S, X, c, year, pool)
        rng = np.random.default_rng(12345)
        keep = {n: rng.random(len(pool[n][1])) < 0.5 for n in al.SUBJECTS}
        h = keep[name]
        H = fit(S, X[h], c[h], year[h], pool, keep)
        Xo, co, yo, uo = X[~h], c[~h], year[~h], cur[~h]
        summary[name] = {"full": {k: F[k] for k in ("alpha", "time_cv", "gamma", "lda_cv", "cur_src", "sec")},
                         "half": {k: H[k] for k in ("alpha", "time_cv", "gamma", "lda_cv", "cur_src")}}
        for method, (ka, kb) in VARIANTS.items():
            if only and method not in only:
                continue
            a, b = F[ka], F[kb]
            ah, bh = H[ka], H[kb]
            stab = {"cos_x": round(abs(float(a @ ah)), 3), "cos_y": round(abs(float(al.perp(b, a) @ al.perp(bh, ah))), 3)}
            ho = {"eta2": _eta2(Xo, co, ah, bh)}
            for ax, v in (("x", al.unit(ah)), ("y", al.perp(bh, ah))):
                s = Xo @ v
                ho[f"year_rho_{ax}"] = round(float(spearmanr(s[yo > 0], yo[yo > 0])[0]), 3)
                if 0 < uo.sum() < len(uo):
                    ho[f"cur_auc_{ax}"] = round(float(roc_auc_score(uo, s)), 3)
            note = (f"half-fit vs full |cos| x {stab['cos_x']}, y {stab['cos_y']}; half-fit on the other half: "
                    + ", ".join(f"{k} {v}" for k, v in ho.items()))
            res = al.record(method, S, [a, b], why=[why(F, ka), why(F, kb)], note=note)
            res["stability"], res["heldout"] = stab, ho
            (al.RESULTS / method / f"{name}.json").write_text(json.dumps(res, indent=1))
            m = res["metrics"]
            summary[name][method] = {"metrics": m, "stability": stab, "heldout": ho,
                                     "x": res["axes"][0]["name"], "y": res["axes"][1]["name"],
                                     "year_rho": [ax["year_rho"] for ax in res["axes"]],
                                     "cur_shift": [ax["curated_shift_sd"] for ax in res["axes"]]}
            print(f"{name:12s} {method:24s} eta2 {m['eta2']:.3f} (held-out {ho['eta2']:.3f}) r_xy {m['r_xy']:+.2f} "
                  f"var {m['var_x']:.3f}/{m['var_y']:.3f} |cos| {stab['cos_x']}/{stab['cos_y']}\n"
                  f"{'':12s}   x: {res['axes'][0]['name']['axis']}\n{'':12s}   y: {res['axes'][1]['name']['axis']}",
                  flush=True)
        print(f"{name}: alpha {F['alpha']} (cv rho {F['time_cv']:.3f}), gamma {F['gamma']} (cv eta2 {F['lda_cv']:.3f}), "
              f"cur {F['cur_src']}; fit s {F['sec']}", flush=True)
    (al.RESULTS / "supervised_summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
