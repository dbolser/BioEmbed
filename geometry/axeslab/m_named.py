"""Named family: draw each subject's map on 2 of Dan's 67 named global axes (al.named_axes()), with the bank's names.

Variants, recorded as 'named_<variant>' for every subject:
  spread  x = the axis along which the subject's papers spread most relative to 50k random human-gene papers
          (variance ratio); y = the next one, among axes nearly orthogonal to x (|cos| <= 0.3) and uncorrelated with
          it on the subject's papers (|r| <= 0.3, after the explorer's perpendicular rule).
  topics  the pair that best separates the map's own topics (the harness eta2, computed in closed form for all pairs),
          same constraints; x = the more global axis of the two (more background variance), so the order never
          depends on the subject's papers.
  fixed   one frame for every subject: the spread rule applied to the background itself (the two named axes holding
          most of a random paper's variance); nothing is fitted to the subject, so all maps share it.
Low-confidence axes are never used; medium-confidence ones are scored x0.9. Stability: 10 random halves of each
subject's papers, |cos| between the half-fit and full-fit displayed axes (x, and y made perpendicular to x).

Run: cd geometry/axeslab && nice -n 10 ../../.venv/bin/python m_named.py
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "8"

import time  # noqa: E402
from functools import cache  # noqa: E402

import numpy as np  # noqa: E402

import axislab as al  # noqa: E402

A, META = al.named_axes()
A32, A = A.astype(np.float32), A.astype(np.float64)
G = A @ A.T                                  # cosines between bank axes
W = np.array([{"high": 1.0, "medium": 0.9, "low": 0.0}[m["confidence"]] for m in META])
MAX_COS, MAX_R, HALVES = 0.3, 0.3, 10
EYE = np.eye(len(A), dtype=bool)


@cache
def bg_var() -> np.ndarray:
    return (al.background().astype(np.float64) @ A.T).var(0)


def stats(X: np.ndarray, c: np.ndarray):
    """The subject's papers in bank coordinates: total covariance C (all papers), and over topic papers the
    scatter T and between-topic scatter B (as the harness's eta2 counts them)."""
    P = (X @ A32.T).astype(np.float64)
    Pc = P - P.mean(0)
    k = c >= 0
    Pk = P[k] - P[k].mean(0)
    H = np.eye(len(np.unique(c[k])))[np.unique(c[k], return_inverse=True)[1]]  # one-hot topics
    n = H.sum(0)
    M = H.T @ Pk / n[:, None]
    return Pc.T @ Pc / len(P), Pk.T @ Pk, (M * n[:, None]).T @ M


def pairs(C, T, B):
    """eta2 and r for every ordered pair (row i = x axis A_i, column j = y axis A_j made perpendicular to A_i), as
    al.metrics would score them. y's direction is A_j - g A_i, g = cos(A_i, A_j): quadratic forms expand in g."""
    def q(M):  # (A_j - g A_i)' M (A_j - g A_i)
        d = np.diag(M)
        return d[None, :] - 2 * G * M + G ** 2 * d[:, None]
    cx, cy = np.diag(C)[:, None], q(C)
    with np.errstate(divide="ignore", invalid="ignore"):  # the diagonal (an axis against itself) is 0/0; never used
        eta = (np.diag(B)[:, None] / cx + q(B) / cy) / (np.diag(T)[:, None] / cx + q(T) / cy)
        r = (C - G * np.diag(C)[:, None]) / np.sqrt(cx * cy)
    return eta, r


def allowed(r):
    usable = W > 0
    return (np.abs(G) <= MAX_COS) & (np.abs(r) <= MAX_R) & ~EYE & usable[:, None] & usable[None, :]


def pick_spread(C, T, B, var=None):
    """Greedy: the highest weighted variance ratio (or plain variance, var given), then the best allowed partner."""
    score = (np.diag(C) / bg_var() if var is None else var) * W
    _, r = pairs(C, T, B)
    i = int(np.argmax(score))
    j = int(np.argmax(np.where(allowed(r)[i], score, -np.inf)))
    return i, j, {"score": score}


def pick_topics(C, T, B):
    eta, r = pairs(C, T, B)
    eta1 = np.diag(B) / np.diag(T)
    vb = bg_var()
    s = np.where(allowed(r) & (vb[:, None] > vb[None, :]), eta * W[:, None] * W[None, :], -np.inf)
    i, j = np.unravel_index(np.argmax(s), s.shape)
    return int(i), int(j), {"eta": eta, "eta1": eta1}


@cache
def pick_fixed():
    bg = al.background().astype(np.float64)
    C = np.cov(bg @ A.T, rowvar=False, bias=True)
    i, j, _ = pick_spread(C, C, C, var=np.diag(C))  # T, B unused for the choice
    return i, j


PICK = {"spread": pick_spread, "topics": pick_topics, "fixed": lambda C, T, B: (*pick_fixed(), {})}


def shown(i, j):
    return al.unit(A[i]), al.perp(A[j], A[i])


def stability(S, variant, full):
    """Per half: |cos| of x and of displayed y with the full fit, and whether the same ordered pair came back."""
    n = len(S["X"])
    a0, b0 = shown(*full)
    out = []
    for seed in range(HALVES):
        h = np.random.default_rng(seed).choice(n, n // 2, replace=False)
        i, j, _ = PICK[variant](*stats(S["X"][h], S["c"][h]))
        a, b = shown(i, j)
        out.append((abs(a @ a0), abs(b @ b0), (i, j) == tuple(full)))
    o = np.array(out, float)
    return {"cos_x": float(np.median(o[:, 0])), "cos_y": float(np.median(o[:, 1])), "mean_cos_x": float(o[:, 0].mean()),
            "mean_cos_y": float(o[:, 1].mean()), "same_pair": float(o[:, 2].mean())}


def why(variant, k, i, info, eta2):
    m = META[i]
    if variant == "spread":
        sc = info["score"]
        return (f"{m['id']}: the subject's papers have {np.diag(info['C'])[i] / bg_var()[i]:.2f}x the variance of random "
                f"human-gene papers along it (rank {int((sc > sc[i]).sum()) + 1} of {int((W > 0).sum())} usable named axes)")
    if variant == "topics":
        return (f"{m['id']}: best topic-separating pair of named axes (pair eta2 {eta2:.3f}); "
                f"on its own it separates the topics with eta2 {info['eta1'][i]:.3f}")
    return f"{m['id']}: fixed frame for every map: the {'first' if k == 0 else 'second'} named axis by background variance"


def main():
    t0 = time.time()
    bg_var()
    print(f"background variance per named axis: {time.time() - t0:.2f} s (once; 67 numbers)")
    fi, fj = pick_fixed()
    print(f"fixed frame: {META[fi]['id']} x {META[fj]['id']}")
    summary = {v: [] for v in PICK}
    for name in al.SUBJECTS:
        S = al.load(name)
        t0 = time.time()
        st = stats(S["X"], S["c"])
        picks = {v: PICK[v](*st) for v in PICK}
        dt = time.time() - t0
        eta, _ = pairs(*st)
        for v, (i, j, info) in picks.items():
            info["C"] = st[0]
            stab = stability(S, v, (i, j))
            names = [{"minus": META[k]["minus_pole"], "plus": META[k]["plus_pole"], "axis": f"{META[k]['id']}: {META[k]['name']}"}
                     for k in (i, j)]
            note = (f"named axes {META[i]['id']} x {META[j]['id']} (bank cos {G[i, j]:.2f}); all picks for this subject took "
                    f"{dt * 1000:.0f} ms; stability over {HALVES} random halves: |cos| half vs full, median (mean) x "
                    f"{stab['cos_x']:.2f} ({stab['mean_cos_x']:.2f}), y {stab['cos_y']:.2f} ({stab['mean_cos_y']:.2f}); "
                    f"same pair {stab['same_pair']:.0%}")
            res = al.record(f"named_{v}", S, [A[i], A[j]], why=[why(v, 0, i, info, eta[i, j]), why(v, 1, j, info, eta[i, j])],
                            names=names, note=note)
            summary[v].append((name, META[i]["id"], META[j]["id"], res["metrics"], stab))
            print(f"{v:7s} {name:12s} {META[i]['id']:>4s} x {META[j]['id']:<4s} {res['metrics']}  stab {stab}", flush=True)
    for v, rows in summary.items():
        cx, cy, mx, my = (np.median([r[4][k] for r in rows]) for k in ("cos_x", "cos_y", "mean_cos_x", "mean_cos_y"))
        ax = sorted({a for r in rows for a in r[1:3]})
        print(f"\n{v}: median eta2 {np.median([r[3]['eta2'] for r in rows]):.3f}; stability median over subjects of |cos| median (mean) over halves: "
              f"x {cx:.2f} ({mx:.2f}), y {cy:.2f} ({my:.2f}); {len(ax)} distinct axes over {len(rows)} subjects: {', '.join(ax)}")


if __name__ == "__main__":
    main()
