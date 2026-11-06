"""ForeSCITE-style test (Chang et al. 2026, arXiv 2610.38719): shift each topic's recent papers along its own time
vector and ask whether the shifted cloud lands closer to the papers that were actually published later.

Cutoff T = 2015. Built from papers up to T only, except that the topic seeds are the all-years clustering:
  centring    the mean of papers up to T
  topics      the 780 topics' centroids from their papers up to T; then EVERY paper (1975-2023) is given to the
              nearest such centroid (cosine of centred vectors), one rule for past, recent and future papers
  time vector per topic: least-squares slope of its assigned papers' centred vectors on year, papers up to T
  recent      up to 2,000 of the topic's papers from T-4..T (the "existing findings" to shift); centre year 2013
  future      up to 2,000 of its papers from 2019-2023; centre year 2021
  shift       x' = unit(x + alpha * step * u), u = the unit time vector, step = slope length x 8 years (2013 -> 2021),
              so alpha = 1 is the topic's own past rate carried over the real gap
Coverage: each future paper's best cosine to the (shifted) recent cloud: mean gain over alpha = 0, share of topics
whose mean improves. Controls at the same alphas: backwards (-alpha), another topic's time vector, a random direction.
Decoding: which MeSH headings does each topic's time vector point toward (cos(u, concept vector))? Against how each
heading's share changed in the topic (2019-23 vs 2011-15; Spearman over headings on >= 1% of its recent papers), and
against simple trend extrapolation (the change from 2006-10 to 2011-15).
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
ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.5]
torch.set_num_threads(16)
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
unit = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", default="eom_mcs200_ms3"); a = ap.parse_args()
    rng = np.random.default_rng(0)
    X = np.load(D / "X.npy", mmap_mode="r"); n = len(X)
    year = np.load(M / "year.npy"); lab = np.load(D / f"clusters/{a.name}_all.npy").astype(np.int64)
    ys, yn = np.load(M / f"{a.name}_ysum.npy"), np.load(M / f"{a.name}_ysize.npy")
    k = len(yn); hi = T - YEARS[0]
    mu = (ys[:, :hi + 1].sum((0, 1)) / yn[:, :hi + 1].sum()).astype(np.float32)  # mean of papers up to T
    cen = ys[:, :hi + 1].sum(1) / np.maximum(yn[:, :hi + 1].sum(1), 1)[:, None] - mu
    Cu = torch.from_numpy(unit(cen).astype(np.float32))
    rows = np.flatnonzero((year > 0) & (year <= 2023))
    top = np.empty(len(rows), np.int64)
    for s in range(0, len(rows), 500_000):
        xb = torch.from_numpy(np.asarray(X[rows[s:s + 500_000]], np.float32) - mu)
        top[s:s + 500_000] = (xb @ Cu.T).argmax(1).numpy()
    assign = np.full(n, -1); assign[rows] = top
    fut_mask = (year >= 2019) & (year <= 2023)
    log(f"assigned {len(rows):,} papers; agreement with the clustering: up to T {np.mean(assign[year <= T] == lab[year <= T]):.1%}, "
        f"2019-23 {np.mean(assign[fut_mask] == lab[fut_mask]):.1%}")
    order = np.argsort(assign, kind="stable"); bounds = np.searchsorted(assign[order], np.arange(-1, k + 1))[1:]
    members = lambda g: np.sort(order[bounds[g]:bounds[g + 1]])
    U = np.zeros((k, 384), np.float32); slope_len = np.zeros(k); recent_rows, future_rows = {}, {}
    for g in range(k):
        idx = members(g); pre = idx[year[idx] <= T]
        if len(pre) < 200: continue
        xg = np.asarray(X[pre], np.float32) - mu; y = year[pre].astype(np.float64)
        yc = y - y.mean(); sl = (yc[:, None] * (xg - xg.mean(0))).sum(0) / (yc ** 2).sum()
        U[g] = sl / np.linalg.norm(sl); slope_len[g] = np.linalg.norm(sl)
        rec = pre[year[pre] >= T - 4]; fu = idx[fut_mask[idx]]
        if len(rec) >= 100 and len(fu) >= 100:
            recent_rows[g] = rng.choice(rec, min(2000, len(rec)), replace=False)
            future_rows[g] = rng.choice(fu, min(2000, len(fu)), replace=False)
    topics = sorted(recent_rows)
    log(f"topics with >= 100 recent and >= 100 future papers: {len(topics)}")
    conds = ["own", "backwards", "other topic", "random"]
    gains = {(c, al): [] for c in conds for al in ALPHAS}
    for g in topics:
        xr = np.asarray(X[recent_rows[g]], np.float32) - mu
        xf = unit(np.asarray(X[future_rows[g]], np.float32) - mu)
        step = slope_len[g] * 8
        best = lambda d_, al: (unit(xr + al * step * d_[None]) @ xf.T).max(0).mean()
        b0 = best(U[g], 0.0)
        o = U[rng.choice([h for h in topics if h != g])]
        r = rng.standard_normal(384).astype(np.float32); r /= np.linalg.norm(r)
        for al in ALPHAS:
            gains[("own", al)].append(best(U[g], al) - b0); gains[("backwards", al)].append(best(U[g], -al) - b0)
            gains[("other topic", al)].append(best(o, al) - b0); gains[("random", al)].append(best(r, al) - b0)
    out = {"topics": len(topics), "step_years": 8}
    for al in ALPHAS:
        line = []
        for c in conds:
            gm = np.array(gains[(c, al)]); out[f"{c} a={al}"] = dict(mean_gain=float(gm.mean()), topics_better=float(np.mean(gm > 0)))
            line.append(f"{c} {np.mean(gm > 0):.0%} ({gm.mean():+.4f})")
        log(f"alpha {al}: topics better / mean best-match gain:  " + ";  ".join(line))
    # decoding: MeSH headings each time vector points to, against what happened (same assignment rule throughout)
    z = np.load(M / "features.npz", allow_pickle=True)
    F = sp.csr_matrix((np.ones(len(z["col"]), np.float32), z["col"], z["ptr"]), shape=(n, len(z["names"])))
    fnames = list(z["names"])
    cdf = pd.read_parquet(M / "concepts.parquet"); C = np.load(M / "concept_def.npy").astype(np.float32); C = C - C.mean(0)
    byname = {nm: i for i, nm in enumerate(cdf.name) if cdf.source[i] == "mesh"}
    cols = [j for j, nm in enumerate(fnames) if nm in byname]
    Cv = unit(C[[byname[fnames[j]] for j in cols]]); Fm = F[:, cols].tocsr()
    r_time, r_trend, r_both = [], [], []
    for g in topics:
        idx = members(g)
        early, recent, future = (idx[(year[idx] >= a_) & (year[idx] <= b_)] for a_, b_ in ((T - 9, T - 5), (T - 4, T), (2019, 2023)))
        if min(len(early), len(recent)) < 200: continue
        fe, fr_, ff = (np.asarray(Fm[r_].mean(0)).ravel() for r_ in (early, recent, future))
        keep = fr_ >= 0.01
        if keep.sum() < 20: continue
        actual = np.log((ff[keep] + 1e-3) / (fr_[keep] + 1e-3)); trend = np.log((fr_[keep] + 1e-3) / (fe[keep] + 1e-3))
        point = Cv[keep] @ U[g]
        r_time.append(spearmanr(point, actual).statistic); r_trend.append(spearmanr(trend, actual).statistic)
        A_ = np.column_stack([np.ones(keep.sum()), trend]); res = actual - A_ @ np.linalg.lstsq(A_, actual, rcond=None)[0]
        r_both.append(spearmanr(point, res).statistic)
    for kk, v in (("time vector", r_time), ("past trend", r_trend), ("time vector, beyond trend", r_both)):
        v = np.array(v); out[f"decode {kk}"] = dict(median_spearman=float(np.median(v)), topics_positive=float(np.mean(v > 0)), topics=len(v))
        log(f"decode, {kk:26s}: median Spearman with actual change {np.median(v):+.3f}, positive in {np.mean(v > 0):.1%} of {len(v)} topics")
    (M / f"{a.name}_foresee.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
