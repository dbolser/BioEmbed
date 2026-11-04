"""ForeSCITE-style test (Chang et al. 2026, arXiv 2610.38719): shift each topic's recent papers along its own time
vector and ask whether the shifted cloud lands closer to the papers that were actually published later.

Cutoff T = 2015; nothing after T is used to build anything.
  topics      the 780 topics; their pre-T centroids from centroids.py yearly sums (years <= T)
  time vector per topic: least-squares slope of its papers' centred vectors on year, papers <= T only (axes.py
              did this on all years; here it is refitted)
  recent      up to 2,000 of the topic's papers from T-4..T (the "existing findings" to shift)
  future      papers from 2019-2023, given to topics by their nearest pre-T centroid (cosine of centred vectors),
              NOT by the clustering, which saw all years; up to 2,000 per topic
  shift       x' = unit(x + alpha * step * u), u = the unit time vector, step = the topic's slope length x 6 years
              (from the middle of the recent window to the middle of the future one), alpha in ALPHAS
Coverage: each future paper's best cosine to the (shifted) recent cloud; reported as the mean gain over alpha = 0
and the share of future papers whose best match improves. Placebos: alpha < 0 (backwards), another topic's time
vector, a random direction, all with the same step.
Decoding: which MeSH headings does each topic's time vector point toward (cos(u, concept vector))? Compared with
how each heading's share in the topic actually changed (future vs recent; Spearman per topic, over the headings
on >= 1% of its recent papers), and with simple trend extrapolation (the change from T-9..T-5 to T-4..T).
CPU, root env: uv run python geometry/manifold/foresee.py [--name eom_mcs200_ms3]
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, torch
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/geometry/medembed_small"
M = D / "manifold"
YEARS = np.arange(1975, 2027)
T = 2015
ALPHAS = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0]
torch.set_num_threads(16)
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", default="eom_mcs200_ms3"); a = ap.parse_args()
    rng = np.random.default_rng(0)
    X = np.load(D / "X.npy", mmap_mode="r"); n = len(X)
    mu = np.load(M / "mean.npy").astype(np.float32)
    year = np.load(M / "year.npy"); lab = np.load(D / f"clusters/{a.name}_all.npy").astype(np.int64)
    k = lab.max() + 1
    ys, yn = np.load(M / f"{a.name}_ysum.npy"), np.load(M / f"{a.name}_ysize.npy")
    hi = T - YEARS[0]
    cen = ys[:, :hi + 1].sum(1) / np.maximum(yn[:, :hi + 1].sum(1), 1)[:, None] - mu
    cen_u = cen / np.linalg.norm(cen, axis=1, keepdims=True)
    # future papers -> nearest pre-T topic
    fut = np.flatnonzero((year >= 2019) & (year <= 2023))
    ftop = np.empty(len(fut), np.int64); Cu = torch.from_numpy(cen_u.astype(np.float32))
    for s in range(0, len(fut), 500_000):
        xb = torch.from_numpy(np.asarray(X[fut[s:s + 500_000]], np.float32) - mu)
        ftop[s:s + 500_000] = (xb @ Cu.T).argmax(1).numpy()
    agree = np.mean(ftop == lab[fut])
    log(f"future papers {len(fut):,}: nearest pre-T centroid = their cluster for {agree:.1%}")
    # per-topic time vectors from papers <= T
    order = np.argsort(lab, kind="stable"); bounds = np.searchsorted(lab[order], np.arange(k + 1))
    U = np.zeros((k, 384), np.float32); slope_len = np.zeros(k); recent_rows = {}
    for g in range(k):
        idx = np.sort(order[bounds[g]:bounds[g + 1]]); idx = idx[(year[idx] > 0) & (year[idx] <= T)]
        if len(idx) < 200: continue
        xg = np.asarray(X[idx], np.float32) - mu; y = year[idx].astype(np.float64)
        yc = y - y.mean(); sl = (yc[:, None] * (xg - xg.mean(0))).sum(0) / (yc ** 2).sum()
        U[g] = sl / np.linalg.norm(sl); slope_len[g] = np.linalg.norm(sl)
        rec = idx[year[idx] >= T - 4]
        if len(rec) >= 100: recent_rows[g] = rng.choice(rec, min(2000, len(rec)), replace=False)
    topics = [g for g in recent_rows if (ftop == g).sum() >= 100]
    log(f"topics with >= 100 recent and >= 100 future papers: {len(topics)}")
    unit = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True)
    gains = {f"own a={al}": [] for al in ALPHAS if al != 0}
    gains |= {"other topic a=1": [], "random a=1": []}
    improved = {kk: [] for kk in gains}
    for g in topics:
        xr = np.asarray(X[recent_rows[g]], np.float32) - mu
        fr = fut[ftop == g]; fr = rng.choice(fr, min(2000, len(fr)), replace=False)
        xf = unit(np.asarray(X[fr], np.float32) - mu)
        step = slope_len[g] * 6
        def best(dirn, al):
            S = unit(xr + al * step * dirn[None]) @ xf.T
            return S.max(0)
        b0 = best(U[g], 0.0)
        for al in ALPHAS:
            if al == 0: continue
            b = best(U[g], al); gains[f"own a={al}"].append(float((b - b0).mean())); improved[f"own a={al}"].append(float((b > b0).mean()))
        o = rng.choice([h for h in topics if h != g]); b = best(U[o], 1.0)
        gains["other topic a=1"].append(float((b - b0).mean())); improved["other topic a=1"].append(float((b > b0).mean()))
        r = rng.standard_normal(384).astype(np.float32); r /= np.linalg.norm(r); b = best(r, 1.0)
        gains["random a=1"].append(float((b - b0).mean())); improved["random a=1"].append(float((b > b0).mean()))
    out = {"topics": len(topics), "future_assignment_agrees_with_cluster": float(agree)}
    for kk in gains:
        gm, im = np.array(gains[kk]), np.array(improved[kk])
        out[kk] = dict(mean_gain=float(gm.mean()), topics_better=float(np.mean(gm > 0)), share_future_improved=float(im.mean()))
        log(f"{kk:16s}: best-match cosine gain {gm.mean():+.4f}, topics better {np.mean(gm > 0):.1%}, "
            f"future papers with a closer match {im.mean():.1%}")
    # decoding: MeSH headings each time vector points to, against what happened
    z = np.load(M / "features.npz", allow_pickle=True)
    F = sp.csr_matrix((np.ones(len(z["col"]), np.float32), z["col"], z["ptr"]), shape=(n, len(z["names"])))
    fnames = list(z["names"])
    cdf = pd.read_parquet(M / "concepts.parquet"); C = np.load(M / "concept_def.npy").astype(np.float32)
    C = C - C.mean(0)
    byname = {nm: i for i, nm in enumerate(cdf.name) if cdf.source[i] == "mesh"}
    cols = [j for j, nm in enumerate(fnames) if nm in byname]
    Cv = unit(C[[byname[fnames[j]] for j in cols]])
    Fm = F[:, cols].tocsr()
    r_time, r_trend, r_both = [], [], []
    for g in topics:
        rows_g = order[bounds[g]:bounds[g + 1]]
        early = rows_g[(year[rows_g] >= T - 9) & (year[rows_g] <= T - 5)]
        recent = rows_g[(year[rows_g] >= T - 4) & (year[rows_g] <= T)]
        future = fut[ftop == g]
        if min(len(early), len(recent)) < 200: continue
        fe, fr_, ff = (np.asarray(Fm[r].mean(0)).ravel() for r in (early, recent, future))
        keep = fr_ >= 0.01
        if keep.sum() < 20: continue
        actual = np.log((ff[keep] + 1e-3) / (fr_[keep] + 1e-3))
        trend = np.log((fr_[keep] + 1e-3) / (fe[keep] + 1e-3))
        point = Cv[keep] @ U[g]
        r_time.append(spearmanr(point, actual).statistic); r_trend.append(spearmanr(trend, actual).statistic)
        # does the time vector add to the trend? Spearman of the residual of actual ~ trend
        A_ = np.column_stack([np.ones(keep.sum()), trend]); res = actual - A_ @ np.linalg.lstsq(A_, actual, rcond=None)[0]
        r_both.append(spearmanr(point, res).statistic)
    for kk, v in (("time vector", r_time), ("past trend", r_trend), ("time vector, beyond trend", r_both)):
        v = np.array(v); out[f"decode {kk}"] = dict(median_spearman=float(np.median(v)), topics_positive=float(np.mean(v > 0)), topics=len(v))
        log(f"decode, {kk:26s}: median Spearman with actual change {np.median(v):+.3f}, positive in {np.mean(v > 0):.1%} of {len(v)} topics")
    (M / f"{a.name}_foresee.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
