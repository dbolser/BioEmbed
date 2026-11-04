"""Time vectors: can the direction topics move in predict where the literature goes next?

All positions are centred (paper vector minus the corpus mean, centroids.py). Three falsifiable tests, each with a
"nothing moves" or popularity baseline, fitted only on papers up to a cutoff year:

drift   Topic centroids per year (centroids.py). Train 2009-2018, predict the pooled 2021-2023 centroid:
          persist   the 2016-2018 centroid
          own       persist + the topic's own fitted slope x years ahead (least squares on yearly centroids)
          shared    persist + the mean slope over all topics (one global time vector) x years ahead
          mix       persist + (own + shared) / 2 x years ahead
        Scores: squared error relative to persist (< 1 = better), share of topics improved, and cos between the
        predicted and the actual move (0 = no idea of the direction).
growth  Does a topic's place along the shared time vector (in 2018) predict its growth 2019-2023 beyond its own
        growth 2009-2018? Spearman correlations and a two-variable least-squares fit.
links   Latent knowledge (Tshitoyan et al. 2019, Nature 571:95): gene-disease pairs never co-mentioned up to the
        cutoff (PubTator gene, MeSH disease heading, same paper); positive if >= 3 co-mentioning papers in the
        horizon. Scores: cosine of the gene's and disease's centroids (papers up to the cutoff), the same after
        moving both along their own time vectors (slope x years ahead: "time vectors predict the future"),
        popularity (log papers of gene + disease) and common neighbours (genes co-mentioned with the gene that
        are linked to the disease). AUC overall and within popularity strata; a logistic fit on the 2010 cutoff
        tested on 2015 says whether geometry adds to the baselines.
CPU, root env: uv run python geometry/manifold/timevec.py drift|growth|links [--name eom_mcs200_ms3]
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, scipy.sparse as sp
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/geometry/medembed_small"
M = D / "manifold"
YEARS = np.arange(1975, 2027)
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
yi = lambda y: int(y - YEARS[0])


def yearly(name):
    ys, yn = np.load(M / f"{name}_ysum.npy").astype(np.float64), np.load(M / f"{name}_ysize.npy")
    mu = np.load(M / "mean.npy").astype(np.float64)
    return ys, yn, mu


def pooled(ys, yn, mu, a, b):
    s, c = ys[:, yi(a):yi(b) + 1].sum(1), yn[:, yi(a):yi(b) + 1].sum(1)
    return s / np.maximum(c, 1)[:, None] - mu, c


def slopes(ys, yn, mu, a, b):
    """Per-topic least-squares slope of yearly centroids on year, weighted by papers per year."""
    yrs = np.arange(a, b + 1); C = ys[:, yi(a):yi(b) + 1] / np.maximum(yn[:, yi(a):yi(b) + 1], 1)[..., None] - mu
    w = yn[:, yi(a):yi(b) + 1].astype(np.float64)
    yb = (w * yrs).sum(1) / w.sum(1)
    dy = yrs[None] - yb[:, None]
    cb = (w[..., None] * C).sum(1) / w.sum(1)[:, None]
    return ((w * dy)[..., None] * (C - cb[:, None])).sum(1) / (w * dy ** 2).sum(1)[:, None]


def drift(name):
    ys, yn, mu = yearly(name)
    ok = (yn[:, yi(2009):yi(2018) + 1].min(1) >= 30) & (yn[:, yi(2021):yi(2023) + 1].sum(1) >= 100)
    persist, _ = pooled(ys, yn, mu, 2016, 2018); actual, _ = pooled(ys, yn, mu, 2021, 2023)
    sl = slopes(ys, yn, mu, 2009, 2018)
    shared = np.average(sl[ok], axis=0, weights=yn[ok, yi(2009):yi(2018) + 1].sum(1))
    ahead = 2022 - 2017
    preds = {"persist": persist, "own": persist + ahead * sl, "shared": persist + ahead * shared,
             "mix": persist + ahead * (sl + shared) / 2, "own_half": persist + ahead * sl / 2}
    e0 = ((actual - persist) ** 2).sum(1)
    move = actual - persist
    out = {"topics": int(ok.sum())}
    for k, p in preds.items():
        e = ((actual - p) ** 2).sum(1); dm = p - persist
        cosm = (dm * move).sum(1) / np.maximum(np.linalg.norm(dm, axis=1) * np.linalg.norm(move, axis=1), 1e-12)
        out[k] = dict(err_ratio=round(float(np.median(e[ok] / e0[ok])), 3), improved=round(float(np.mean(e[ok] < e0[ok])), 3),
                      move_cos=round(float(np.median(cosm[ok])), 3) if k != "persist" else None)
        log(f"{k:9s} median error vs persist {out[k]['err_ratio']}, topics improved {out[k]['improved']:.1%}, "
            f"median cos(predicted move, actual move) {out[k]['move_cos']}")
    # does the past direction of movement repeat? cos(slope 2009-2018, actual move)
    c = (sl * move).sum(1) / np.linalg.norm(sl, axis=1) / np.linalg.norm(move, axis=1)
    cs = move @ shared / np.linalg.norm(move, axis=1) / np.linalg.norm(shared)
    log(f"cos(own past slope, actual move): median {np.median(c[ok]):.3f}; cos(shared, actual move): median {np.median(cs[ok]):.3f}")
    # placebo: other topics' slopes
    perm = np.random.default_rng(0).permutation(np.flatnonzero(ok))
    cp = (sl[perm] * move[np.flatnonzero(ok)]).sum(1) / np.linalg.norm(sl[perm], axis=1) / np.linalg.norm(move[np.flatnonzero(ok)], axis=1)
    log(f"placebo cos(another topic's slope, actual move): median {np.median(cp):.3f}")
    out["cos_own_slope_move"] = float(np.median(c[ok])); out["cos_shared_move"] = float(np.median(cs[ok]))
    out["placebo"] = float(np.median(cp))
    # Pre-registered shrinkage: choose lambda (pred = persist + ahead * (lambda * own + (1 - lambda) * shared)) and a
    # step multiplier on an EARLIER era (fit 1999-2008, predict 2011-2013 from 2006-2008), then apply both unchanged.
    def era(fit_a, fit_b, p_a, p_b, t_a, t_b, lam, mult):
        okk = (yn[:, yi(fit_a):yi(fit_b) + 1].min(1) >= 30) & (yn[:, yi(t_a):yi(t_b) + 1].sum(1) >= 100)
        per, _ = pooled(ys, yn, mu, p_a, p_b); act, _ = pooled(ys, yn, mu, t_a, t_b)
        s_ = slopes(ys, yn, mu, fit_a, fit_b)
        sh = np.average(s_[okk], axis=0, weights=yn[okk, yi(fit_a):yi(fit_b) + 1].sum(1))
        ah = (t_a + t_b) / 2 - (p_a + p_b) / 2
        pred = per + mult * ah * (lam * s_ + (1 - lam) * sh)
        e0_, e_ = ((act - per) ** 2).sum(1), ((act - pred) ** 2).sum(1)
        return float(np.median(e_[okk] / e0_[okk])), float(np.mean(e_[okk] < e0_[okk])), int(okk.sum())
    grid = [(l_, m_) for l_ in np.linspace(0, 1, 11) for m_ in (0.25, 0.5, 0.75, 1.0, 1.25)]
    tr = {g_: era(1999, 2008, 2006, 2008, 2011, 2013, *g_)[0] for g_ in grid}
    lam, mult = min(tr, key=tr.get)
    te = era(2009, 2018, 2016, 2018, 2021, 2023, lam, mult)
    log(f"chosen on 1999-2008 -> 2011-13: lambda (own share) {lam:.1f}, step x{mult} (train error ratio {tr[(lam, mult)]:.3f}); "
        f"held-out 2009-2018 -> 2021-23: error ratio {te[0]:.3f}, topics improved {te[1]:.1%} of {te[2]}")
    out["preregistered"] = dict(lam=float(lam), mult=float(mult), train_err_ratio=tr[(lam, mult)], test_err_ratio=te[0],
                                test_improved=te[1], topics=te[2])
    (M / f"{name}_drift.json").write_text(json.dumps(out, indent=1))
    np.save(M / f"{name}_shared_time_2009_2018.npy", shared.astype(np.float32))


def drift_reassigned(name):
    """The pre-registered drift test with the target rebuilt without the clustering: 2021-2023 papers go to the topic
    whose 2016-2018 centroid is nearest (cosine of centred vectors), so the 'actual future centroid' does not depend
    on a clustering that saw those papers."""
    import torch
    torch.set_num_threads(16)
    ys, yn, mu = yearly(name); d = json.load(open(M / f"{name}_drift.json"))["preregistered"]
    X = np.load(D / "X.npy", mmap_mode="r"); year = np.load(M / "year.npy")
    persist, pc = pooled(ys, yn, mu, 2016, 2018)
    Pu = torch.from_numpy((persist / np.linalg.norm(persist, axis=1, keepdims=True)).astype(np.float32))
    fut = np.flatnonzero((year >= 2021) & (year <= 2023)); k = len(persist)
    S = np.zeros((k, 384)); cnt = np.zeros(k)
    for s in range(0, len(fut), 500_000):
        xb = np.asarray(X[fut[s:s + 500_000]], np.float32) - mu.astype(np.float32)
        a_ = (torch.from_numpy(xb) @ Pu.T).argmax(1).numpy()
        np.add.at(S, a_, xb); cnt += np.bincount(a_, minlength=k)
    actual = S / np.maximum(cnt, 1)[:, None]
    ok = (yn[:, yi(2009):yi(2018) + 1].min(1) >= 30) & (cnt >= 100)
    sl = slopes(ys, yn, mu, 2009, 2018)
    shared = np.average(sl[ok], axis=0, weights=yn[ok, yi(2009):yi(2018) + 1].sum(1))
    pred = persist + d["mult"] * 5 * (d["lam"] * sl + (1 - d["lam"]) * shared)
    e0, e = ((actual - persist) ** 2).sum(1), ((actual - pred) ** 2).sum(1)
    r = dict(err_ratio=float(np.median(e[ok] / e0[ok])), improved=float(np.mean(e[ok] < e0[ok])), topics=int(ok.sum()))
    log(f"reassigned target: error ratio {r['err_ratio']:.3f}, topics improved {r['improved']:.1%} of {r['topics']}")
    out = json.load(open(M / f"{name}_drift.json")); out["preregistered_reassigned"] = r
    (M / f"{name}_drift.json").write_text(json.dumps(out, indent=1))


def growth(name):
    ys, yn, mu = yearly(name)
    tot = yn.sum(0)
    share = lambda a, b: yn[:, yi(a):yi(b) + 1].sum(1) / tot[yi(a):yi(b) + 1].sum()
    s1, s2, s3 = share(2009, 2013), share(2014, 2018), share(2019, 2023)
    ok = (yn[:, yi(2009):yi(2013) + 1].sum(1) >= 200)
    past, fut = np.log(s2 / s1), np.log(s3 / s2)
    sl = slopes(ys, yn, mu, 2009, 2018)
    shared = np.average(sl[ok], axis=0, weights=yn[ok, yi(2009):yi(2018) + 1].sum(1)); shared /= np.linalg.norm(shared)
    pos, _ = pooled(ys, yn, mu, 2014, 2018); pos = pos @ shared
    speed = sl @ shared
    out = {}
    for k, v in (("past growth", past), ("position on time vector", pos), ("speed along time vector", speed)):
        r = spearmanr(v[ok], fut[ok]).statistic; out[k] = round(float(r), 3)
        log(f"Spearman(future growth, {k}) = {r:+.3f} over {ok.sum()} topics")
    A = np.column_stack([np.ones(ok.sum()), past[ok], pos[ok]])
    for cols, lab in (([0, 1], "past"), ([0, 1, 2], "past + position")):
        beta, res, *_ = np.linalg.lstsq(A[:, cols], fut[ok], rcond=None)
        r2 = 1 - np.sum((fut[ok] - A[:, cols] @ beta) ** 2) / np.sum((fut[ok] - fut[ok].mean()) ** 2)
        out[f"R2 {lab}"] = round(float(r2), 3); log(f"R^2 future growth ~ {lab}: {r2:.3f}")
    (M / f"{name}_growth.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------ links
def entity_matrices():
    """Paper x gene and paper x disease (MeSH C tree) incidence, CSR, rows = X rows."""
    E = D / "eval"; ptr, genes = np.load(E / "genes_ptr.npy"), np.load(E / "genes.npy")
    n = len(ptr) - 1
    gu, gi = np.unique(genes, return_inverse=True)
    G = sp.csr_matrix((np.ones(len(genes), np.float32), gi, ptr), shape=(n, len(gu))); G.sum_duplicates(); G.data[:] = 1
    desc = pq.read_table(M / "mesh_desc.parquet").to_pandas()
    dis = desc[desc.trees.map(lambda t: any(x.startswith("C") for x in t))]
    # drop the broad headings (any descriptor whose tree numbers are all at depth <= 2, e.g. "Neoplasms")
    dis = dis[dis.trees.map(lambda t: max(x.count(".") for x in t) >= 1)]
    col = {int(u): i for i, u in enumerate(dis.ui)}
    mp = pq.read_table(M / "mesh_papers.parquet", columns=["row", "mesh"])
    rows = mp.column("row").to_numpy(); mesh = mp.column("mesh").combine_chunks()
    mo, mv = mesh.offsets.to_numpy(), mesh.values.to_numpy()
    c = np.array([col.get(int(u), -1) for u in mv.tolist()]); r = np.repeat(rows, np.diff(mo)); k = c >= 0
    Dz = sp.csr_matrix((np.ones(k.sum(), np.float32), (r[k], c[k])), shape=(n, len(dis))); Dz.sum_duplicates(); Dz.data[:] = 1
    return G, gu, Dz, dis.reset_index(drop=True)


def centroids_and_slopes(Inc, X, mu, year, rows_mask):
    """Centroid (centred) and time slope of each column entity over the papers in rows_mask."""
    I = sp.csr_matrix(Inc.multiply(rows_mask[:, None]).T)  # (entities, n)
    cnt = np.asarray(I.sum(1)).ravel()
    y = year.astype(np.float64)
    S = np.zeros((I.shape[0], 384)); SY = np.zeros((I.shape[0], 384))
    for s in range(0, X.shape[0], 1_000_000):
        xb = np.asarray(X[s:s + 1_000_000], np.float32) - mu.astype(np.float32)
        Ib = I[:, s:s + 1_000_000]
        S += Ib @ xb; SY += Ib @ (xb * y[s:s + 1_000_000, None].astype(np.float32))
    sy = I @ y; syy = I @ (y ** 2)
    nn = np.maximum(cnt, 1); cen = S / nn[:, None]; yb = sy / nn
    var = syy / nn - yb ** 2
    slope = (SY / nn[:, None] - yb[:, None] * cen) / np.maximum(var, 1e-9)[:, None]
    slope[var < 4] = 0  # fewer than ~2 years of spread: no reliable direction
    return cen, slope, cnt, yb


def auc(score, y):
    from scipy.stats import rankdata
    r = rankdata(score); npos = y.sum(); nneg = len(y) - npos
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def links(name, min_gene=10, min_dis=100, pos_min=3):
    X = np.load(D / "X.npy", mmap_mode="r"); mu = np.load(M / "mean.npy"); year = np.load(M / "year.npy")
    G, gu, Dz, dis = entity_matrices(); log(f"genes {G.shape[1]:,}, disease headings {Dz.shape[1]:,}")
    res, feats = {}, {}
    for T, H in ((2010, 5), (2015, 8)):
        before = (year > 0) & (year <= T); after = (year > T) & (year <= T + H)
        gc, gsl, gn, gyb = centroids_and_slopes(G, X, mu, year, before)
        dc, dsl, dn, dyb = centroids_and_slopes(Dz, X, mu, year, before)
        gk, dk = np.flatnonzero(gn >= min_gene), np.flatnonzero(dn >= min_dis)
        Gb, Ga = sp.csr_matrix(G.multiply(before[:, None])), sp.csr_matrix(G.multiply(after[:, None]))
        Db, Da = sp.csr_matrix(Dz.multiply(before[:, None])), sp.csr_matrix(Dz.multiply(after[:, None]))
        cob = (Gb[:, gk].T @ Db[:, dk]).toarray(); coa = (Ga[:, gk].T @ Da[:, dk]).toarray()
        GG = (Gb[:, gk].T @ Gb[:, gk]); GG.setdiag(0); GG.data[:] = 1
        cn = (GG @ sp.csr_matrix((cob > 0).astype(np.float32))).toarray() if hasattr(GG, "toarray") else None
        cand = cob == 0; ypos = (coa >= pos_min)[cand]
        unit = lambda A: A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
        ahead = H / 2 + (T - 0)  # project from each entity's mean year to the middle of the horizon
        g_now, d_now = unit(gc[gk]), unit(dc[dk])
        g_fut = unit(gc[gk] + (T + H / 2 - gyb[gk])[:, None] * gsl[gk])
        d_fut = unit(dc[dk] + (T + H / 2 - dyb[dk])[:, None] * dsl[dk])
        f = {"cos": (g_now @ d_now.T)[cand], "cos_time": (g_fut @ d_fut.T)[cand],
             "cos_gene_time": (g_fut @ d_now.T)[cand],
             "popularity": (np.log(gn[gk])[:, None] + np.log(dn[dk])[None, :])[cand],
             "common_nbrs": np.log1p(cn)[cand]}
        pop = f["popularity"]; strata = np.digitize(pop, np.quantile(pop, np.linspace(0, 1, 11)[1:-1]))
        r = {"genes": len(gk), "diseases": len(dk), "candidates": int(cand.sum()), "positives": int(ypos.sum()),
             "pos_rate": round(float(ypos.mean()), 4)}
        for k, v in f.items():
            within = [auc(v[strata == s], ypos[strata == s]) for s in range(10) if 0 < ypos[strata == s].sum() < (strata == s).sum()]
            r[k] = dict(auc=round(auc(v, ypos), 3), auc_within_popularity=round(float(np.mean(within)), 3))
            log(f"T={T} +{H}y {k:14s} AUC {r[k]['auc']:.3f}, within popularity deciles {r[k]['auc_within_popularity']:.3f}")
        log(f"T={T}: {r['candidates']:,} unlinked pairs, {r['positives']:,} new links ({r['pos_rate']:.2%})")
        res[T] = r; feats[T] = (f, ypos)
    # logistic fit on 2010, test on 2015
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    sets = {"baseline": ["popularity", "common_nbrs"], "+cos": ["popularity", "common_nbrs", "cos"],
            "+cos_time": ["popularity", "common_nbrs", "cos_time"], "+both": ["popularity", "common_nbrs", "cos", "cos_time"]}
    rng = np.random.default_rng(0)
    for lab, cols in sets.items():
        f10, y10 = feats[2010]; f15, y15 = feats[2015]
        sub = rng.random(len(y10)) < min(1, 3e6 / len(y10))
        A10 = np.column_stack([f10[c][sub] for c in cols]); A15 = np.column_stack([f15[c] for c in cols])
        mu_, sd_ = A10.mean(0), A10.std(0)
        lr = LogisticRegression(max_iter=500).fit((A10 - mu_) / sd_, y10[sub])
        p = lr.decision_function((A15 - mu_) / sd_)
        a_, ap_ = roc_auc_score(y15, p), average_precision_score(y15, p)
        top = np.argsort(-p)[:1000]
        res[f"fit2010_test2015 {lab}"] = dict(auc=round(a_, 3), avg_precision=round(ap_, 4), prec_at_1000=round(float(y15[top].mean()), 3))
        log(f"fit 2010 -> test 2015 {lab:10s}: AUC {a_:.3f}, average precision {ap_:.4f}, precision@1000 {y15[top].mean():.3f}")
    (M / f"links.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["drift", "drift_reassigned", "growth", "links"])
    ap.add_argument("--name", default="eom_mcs200_ms3"); a = ap.parse_args()
    {"drift": lambda: drift(a.name), "drift_reassigned": lambda: drift_reassigned(a.name),
     "growth": lambda: growth(a.name), "links": lambda: links(a.name)}[a.what]()
