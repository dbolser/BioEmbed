"""Independent check of analogy themes: apply each theme's vector to topics that did NOT define it.

For every topic g outside the theme: step from g along the theme axis by the theme's median edge length, land on
the nearest other topic, and record how the landing topic's MeSH branch shares differ from g's (branches.py).
The same with random unit directions of the same length gives the placebo spread, so each branch gets a z-score.
If "X in cancer -> X" means what its edges suggest, stepping backwards from any topic should raise C04 Neoplasms.
CPU, root env: uv run python geometry/manifold/analogy_check.py eom_mcs200_ms3 [--themes 10]
"""
import argparse, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/geometry/medembed_small"
M = D / "manifold"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--themes", type=int, default=10)
    ap.add_argument("--random", type=int, default=200)
    a = ap.parse_args()
    S, n = np.load(M / f"{a.name}_sum.npy").astype(np.float64), np.load(M / f"{a.name}_size.npy")
    mu = np.load(M / "mean.npy").astype(np.float64)
    R = S / n[:, None] - mu; P = R - R.mean(0)
    th = json.load(open(M / f"{a.name}_analogies.json"))["themes"]
    z = np.load(M / f"{a.name}_analogies.npz"); axes, E, comp = z["axes"], z["edges"], z["comp"]
    br = np.load(M / f"{a.name}_branches.npz", allow_pickle=True); share, names = br["share"], br["names"]
    rng = np.random.default_rng(0)
    out = []
    for ti in range(min(a.themes, len(th))):
        m = np.flatnonzero(comp == ti)  # analogies.py numbers themes by size: A0 = largest
        ax = axes[ti] / np.linalg.norm(axes[ti])
        vec = P[E[m, 1]] - P[E[m, 0]]; vec *= np.sign(vec @ ax)[:, None]
        step = np.median(np.linalg.norm(vec, axis=1))
        inside = np.unique(E[m])
        start = np.setdiff1d(np.arange(len(P)), inside)

        pp = (P ** 2).sum(1)

        def land(dirn):
            Q = P[start] + step * dirn
            d = pp[None, :] - 2 * Q @ P.T  # + |Q|^2, constant per row
            d[np.arange(len(start)), start] = np.inf
            return d.argmin(1)

        res = {}
        for sign, lab_ in ((1, "forward"), (-1, "backward")):
            delta = (share[land(sign * ax)] - share[start]).mean(0)
            rnd = []
            for _ in range(a.random):
                r = rng.standard_normal(P.shape[1]); r -= (r @ ax) * ax; r /= np.linalg.norm(r)
                rnd.append((share[land(r)] - share[start]).mean(0))
            rnd = np.array(rnd); zz = (delta - rnd.mean(0)) / np.maximum(rnd.std(0), 1e-6)
            o = np.argsort(-zz)
            res[lab_] = dict(up=[(names[i], round(float(delta[i]), 3), round(float(zz[i]), 1)) for i in o[:6]],
                             down=[(names[i], round(float(delta[i]), 3), round(float(zz[i]), 1)) for i in o[::-1][:6]])
        pairs = th[ti]["pairs"][:3]
        print(f"\nA{ti} ({th[ti]['edges']} edges) e.g. {pairs[0][0]} -> {pairs[0][1]}; start topics {len(start)}")
        for lab_ in ("forward", "backward"):
            print(f"  {lab_:8s} up:   " + "; ".join(f"{x} {d:+.3f} (z {s})" for x, d, s in res[lab_]["up"][:4]))
            print(f"  {'':8s} down: " + "; ".join(f"{x} {d:+.3f} (z {s})" for x, d, s in res[lab_]["down"][:4]))
        out.append(dict(theme=ti, **res))
    (M / f"{a.name}_analogy_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
