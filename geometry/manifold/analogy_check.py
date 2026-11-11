"""Independent check of analogy themes: apply each theme's vector to topics that did NOT define it.

For every topic g outside the theme: step from g along the theme axis by the theme's median edge length, land on
the nearest other topic, and record how the landing topic's MeSH branch shares differ from g's (branches.py).
The placebo: the same step along real between-topic directions (other edges of the topic graph, random sign), so
each branch gets a z-score against moves that also land somewhere real, and the share of those placebo steps the
theme beats (z is unstable for branches that placebo steps barely move; the share is not). (Random isotropic
directions are kept as a second, much weaker reference, z_iso: in 384-D most of them land back on the start topic's
own neighbour.) Each record: [branch, change in share, z, z_iso, share of placebo steps beaten].
If "X in cancer -> X" means what its edges suggest, stepping backwards from any topic should raise C04 Neoplasms.
CPU, root env: uv run python geometry/manifold/analogy_check.py eom_mcs200_ms3 [--themes 10]
  (finer topics, all themes: eom_mcs50_ms3 --themes 182 --first 20 --no-iso keeps the first 20 already checked)
"""
import argparse, json
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/geometry/medembed_small"
M = D / "manifold"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--themes", type=int, default=10)
    ap.add_argument("--random", type=int, default=200)
    ap.add_argument("--first", type=int, default=0, help="start at this theme, keeping earlier ones already in the file")
    ap.add_argument("--no-iso", action="store_true", help="skip the (weak) isotropic reference: half the time")
    ap.add_argument("--seed", type=int, default=0, help="placebo draws (to check how stable the z-scores are)")
    ap.add_argument("--out", default=None, help="output file name (default <name>_analogy_check.json)")
    a = ap.parse_args()
    torch.set_num_threads(16)
    S, n = np.load(M / f"{a.name}_sum.npy").astype(np.float64), np.load(M / f"{a.name}_size.npy")
    mu = np.load(M / "mean.npy").astype(np.float64)
    R = S / n[:, None] - mu; P = R - R.mean(0)
    th = json.load(open(M / f"{a.name}_analogies.json"))["themes"]
    z = np.load(M / f"{a.name}_analogies.npz"); axes, E, comp = z["axes"], z["edges"], z["comp"]
    br = np.load(M / f"{a.name}_branches.npz", allow_pickle=True); share, names = br["share"], br["names"]
    rng = np.random.default_rng(a.first + a.seed)
    f_out = M / (a.out or f"{a.name}_analogy_check.json")
    out = [r for r in json.loads(f_out.read_text()) if r["theme"] < a.first] if a.first and f_out.exists() else []
    pp = (P ** 2).sum(1); GP = P @ P.T  # landing distances: |P_j|^2 - 2 (P_start + step d) . P_j, split so the
    # start-to-topic part is made once per theme and each direction only adds a per-topic offset
    for ti in range(a.first, min(a.themes, len(th))):
        m = np.flatnonzero(comp == ti)  # analogies.py numbers themes by size: A0 = largest
        ax = axes[ti] / np.linalg.norm(axes[ti])
        vec = P[E[m, 1]] - P[E[m, 0]]; vec *= np.sign(vec @ ax)[:, None]
        step = np.median(np.linalg.norm(vec, axis=1))
        inside = np.unique(E[m])
        start = np.setdiff1d(np.arange(len(P)), inside)

        baseT = torch.from_numpy(pp[None, :] - 2 * GP[start])  # (+ |Q|^2, constant per row)
        baseT[np.arange(len(start)), start] = np.inf
        sh0 = share[start].mean(0)

        def shifts(dirs):
            """Mean change in branch shares, start topic -> landing topic, for a step along each row of dirs."""
            res_ = np.empty((len(dirs), share.shape[1]))
            for b in range(0, len(dirs), 16):
                off = torch.from_numpy((2 * step) * (dirs[b:b + 16] @ P.T))
                res_[b:b + 16] = share[(baseT[None] - off[:, None, :]).argmin(2).numpy()].mean(1) - sh0
            return res_

        others = np.setdiff1d(np.arange(len(E)), m)
        pool = P[E[others, 1]] - P[E[others, 0]]; pool /= np.linalg.norm(pool, axis=1, keepdims=True)
        res = {}
        for sign, lab_ in ((1, "forward"), (-1, "backward")):
            delta = shifts((sign * ax)[None])[0]
            dr, di = [], []
            for _ in range(a.random):
                dr.append(pool[rng.integers(len(pool))] * rng.choice([-1, 1]))
                if a.no_iso: continue
                q = rng.standard_normal(P.shape[1]); q -= (q @ ax) * ax; q /= np.linalg.norm(q); di.append(q)
            rnd = shifts(np.array(dr)); iso = shifts(np.array(di)) if di else None
            zz = (delta - rnd.mean(0)) / np.maximum(rnd.std(0), 1e-6)
            zi = (delta - iso.mean(0)) / np.maximum(iso.std(0), 1e-6) if iso is not None else np.full(len(delta), np.nan)
            # z is unstable where placebo steps barely move a branch (tiny sd): also keep the share of placebo steps
            # that the theme beats (moves the branch less far in the same direction)
            below, above = (rnd < delta).mean(0), (rnd > delta).mean(0)
            o = np.argsort(-zz)
            rec = lambda i, b: (names[i], round(float(delta[i]), 3), round(float(zz[i]), 1),
                                None if np.isnan(zi[i]) else round(float(zi[i]), 1), round(float(b[i]), 3))
            res[lab_] = dict(up=[rec(i, below) for i in o[:6]], down=[rec(i, above) for i in o[::-1][:6]])
        pairs = th[ti]["pairs"][:3]
        print(f"\nA{ti} ({th[ti]['edges']} edges) e.g. {pairs[0][0]} -> {pairs[0][1]}; start topics {len(start)}")
        for lab_ in ("forward", "backward"):
            for k_ in ("up", "down"):
                print(f"  {lab_ if k_ == 'up' else '':8s} {k_ + ':':5s} " + "; ".join(
                    f"{x} {d:+.3f} (z {s}; beats {b:.0%}; iso {zi_})" for x, d, s, zi_, b in res[lab_][k_][:4]))
        out.append(dict(theme=ti, start_topics=int(len(start)), **res))
        if ti % 10 == 9: f_out.write_text(json.dumps(out, indent=1))  # (progress survives a kill)
    f_out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
