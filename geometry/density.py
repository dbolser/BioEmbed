"""Intrinsic dimension vs density: does the data look simpler or harder as papers are added?

40k query papers; their nearest neighbours are searched in nested random subsets of the 7.76M human-gene papers
(1%, 3%, 10%, 30%, all; rng seed 1: a row is in subset f iff u < f, the subsets gallery.py uses). The queries
are in every subset. For each subset: TwoNN (with a bootstrap spread over queries) and Levina-Bickel at
k = 10, 20, 50 as in truth.py, and the median distance to the 1st and 50th neighbour.
Writes data/geometry/medembed_small/eval/density.json. CPU (~20 min on 96 threads; peak ~30 GB): uv run python geometry/density.py
"""
import json, time
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parent.parent
D = REPO / "data/geometry/medembed_small"
Q, K = 40_000, 50
FRACS = [0.01, 0.03, 0.1, 0.3, 1.0]
torch.set_num_threads(96)


def twonn(d: np.ndarray) -> float:
    mu = np.sort(d[:, 1] / d[:, 0])[: int(0.9 * len(d))]  # (discard the top 10% of ratios, fit through the origin)
    F = np.arange(1, len(mu) + 1) / len(d)
    return float(np.sum(np.log(mu) * -np.log(1 - F)) / np.sum(np.log(mu) ** 2))


def estimate(sim: np.ndarray) -> dict:
    d = np.sqrt(np.clip(2 - 2 * sim.astype(np.float64), 1e-12, None))
    d = d[d[:, 0] > 1e-6]  # (exact duplicates break both estimators)
    mle = {k: float(1 / np.mean(1 / ((k - 2) / np.log(d[:, k - 1:k] / d[:, :k - 1]).sum(1)))) for k in (10, 20, 50)}
    boots = [twonn(d[np.random.default_rng(b).integers(0, len(d), len(d))]) for b in range(20)]
    return {"twonn": twonn(d), "twonn_sd": float(np.std(boots)), "mle": mle,
            "median_d1": float(np.median(d[:, 0])), "median_d50": float(np.median(d[:, -1]))}


def main() -> None:
    X = np.load(D / "X.npy", mmap_mode="r"); n = len(X)
    rng = np.random.default_rng(1)
    u = rng.random(n).astype(np.float32)
    q = np.sort(rng.choice(np.flatnonzero(u < FRACS[0]), Q, replace=False))
    Xq = torch.from_numpy(np.asarray(X[q], np.float32)); qt = torch.from_numpy(q)
    best = {f: (torch.full((Q, K + 1), -2.0), torch.zeros((Q, K + 1), dtype=torch.int64)) for f in FRACS}
    C, t0 = 50_000, time.time()  # (40k x 50k similarities: 8 GB a block)
    for s in range(0, n, C):
        sim = Xq @ torch.from_numpy(np.asarray(X[s:s + C], np.float32)).T
        ub = torch.from_numpy(u[s:s + C])
        for f in FRACS:
            sm = sim if f == 1.0 else sim.masked_fill(~(ub < f)[None, :], -2.0)
            v, i = sm.topk(K + 1, dim=1)
            bs, bi = best[f]
            cs, ci = torch.cat([bs, v], 1), torch.cat([bi, i + s], 1)
            bs, o = cs.topk(K + 1, dim=1); best[f] = (bs, ci.gather(1, o))
        print(f"  {min(s + C, n):,}/{n:,} ({time.time() - t0:.0f}s)", flush=True)
    res = {}
    for f in FRACS:
        bs, bi = best[f]
        keep = bi != qt[:, None]; keep[~(~keep).any(1), -1] = False  # (drop the query itself, or the last)
        res[f] = {"n_ref": int((u < f).sum()), **estimate(bs[keep].view(Q, K).numpy())}
        print(f, res[f], flush=True)
    (D / "eval/density.json").write_text(json.dumps({"queries": Q, "results": res}, indent=1))


if __name__ == "__main__":
    main()
