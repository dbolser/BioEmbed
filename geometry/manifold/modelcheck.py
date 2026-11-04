"""Is the space of ideas a MedEmbed quirk? The same measurements on the same 78k papers (100 per topic,
subset_texts.py) in MedEmbed-small (384-D) and a stronger model (Qwen3-Embedding-8B, 4096-D, gcp_embed.sh).

Per model, on the subset:
  cone        length of the mean paper vector
  agreement   linear CKA between the two models' centred paper vectors; overlap of 10 nearest neighbours
  W axes      pooled within-topic covariance (each topic unit trace); share of topics with each top axis in their
              own top-20 local directions; cross-model match of axes by correlating papers' within-topic scores
  time        per-topic slopes on year: mean pairwise cos; correlation of the shared direction with year
  analogies   the analogy themes' topic pairs (analogies.py, found in MedEmbed): are their between-topic vectors
              still parallel in the other model? coherence vs random topic pairs of the same count (z)
  held out    A0 ("X in cancer -> X") built in the model from its own edges, applied backwards to the other topics:
              change in the landing topics' share of Neoplasms headings, against random directions
  concepts    MeSH "name: definition" texts against their papers' centroids in the subset (headings on >= 20
              subset papers): own centroid first / in top 10, concept-mean centred
CPU, root env: uv run python geometry/manifold/modelcheck.py [--model Qwen3-Embedding-8B]
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, scipy.sparse as sp, torch
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/geometry/medembed_small"
M = D / "manifold"
CK = REPO / "data/geometry/manifold_check"
NAME = "eom_mcs200_ms3"
torch.set_num_threads(16)
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
unit = lambda A: A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)


def interleave(parts):
    n = sum(len(p) for p in parts); out = np.zeros((n, parts[0].shape[1]), np.float32)
    for i, p in enumerate(parts): out[i::len(parts)] = p
    return out


def cka(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    hsic = lambda X, Y: np.linalg.norm(X.T @ Y) ** 2
    return float(hsic(A, B) / np.sqrt(hsic(A, A) * hsic(B, B)))


def measure(X, lab, year, k, mesh_F, br_share, edges_by_theme, C_txt, mesh_cols_idx, label):
    r = {}
    mu = X.mean(0); r["cone"] = float(np.linalg.norm(mu)); Xc = X - mu
    cen = np.stack([Xc[lab == g].mean(0) for g in range(k)])
    d = X.shape[1]
    # within-topic axes
    W = np.zeros((d, d)); slopes = np.zeros((k, d)); locals_ = []
    for g in range(k):
        R = Xc[lab == g] - cen[g]; Cg = R.T @ R / len(R); W += Cg / np.trace(Cg)
        y = year[lab == g].astype(np.float64); ok = y > 0; yc = y[ok] - y[ok].mean()
        slopes[g] = (yc[:, None] * R[ok]).sum(0) / max((yc ** 2).sum(), 1e-9)
        # local top-20 directions via the small (n x n) problem: R is 100 x d
        U, S, Vt = np.linalg.svd(R, full_matrices=False); locals_.append(Vt[:20])
    w, V = np.linalg.eigh(W / k); V = V[:, ::-1][:, :12]
    share = [float(np.mean([np.sum((L @ V[:, j]) ** 2) >= 0.5 for L in locals_])) for j in range(6)]
    r["W_share_top20"] = [round(s, 2) for s in share]; r["chance_top20"] = round(20 / d, 4)
    within = Xc - cen[lab]
    r["W_scores"] = within @ V  # (n, 12) per-paper within-topic scores, for cross-model matching
    tu = unit(slopes); g_ = tu.mean(0); r["time_pairwise_cos"] = float((np.linalg.norm(g_) ** 2 * k - 1) / (k - 1))
    gt = g_ / np.linalg.norm(g_); ok = year > 0
    r["time_year_r"] = float(np.corrcoef(within[ok] @ gt, year[ok])[0, 1])
    # analogy themes: coherence of the MedEmbed-found topic pairs in this model, vs random topic pairs
    rng = np.random.default_rng(0); r["themes"] = []
    def coh(E):
        Vv = unit(cen[E[:, 1]] - cen[E[:, 0]]); ax = np.linalg.eigh(Vv.T @ Vv)[1][:, -1]
        return float(np.abs(Vv @ ax).mean()), ax
    for ti, E in enumerate(edges_by_theme):
        c, _ = coh(E)
        null = [coh(rng.choice(k, size=(len(E), 2)))[0] for _ in range(200)]
        r["themes"].append(dict(theme=f"A{ti}", edges=len(E), coherence=round(c, 3), null_mean=round(float(np.mean(null)), 3),
                                z=round(float((c - np.mean(null)) / np.std(null)), 1)))
    # held out: A0 backwards from the other topics -> Neoplasms share of the landing topic
    E0 = edges_by_theme[0]; _, ax = coh(E0)
    vv = cen[E0[:, 1]] - cen[E0[:, 0]]; ax = ax * np.sign((vv @ ax).sum()); step = np.median(np.abs(vv @ ax))
    start = np.setdiff1d(np.arange(k), np.unique(E0)); cc = (cen ** 2).sum(1)
    def land(dirn):
        Q = cen[start] + step * dirn; dd = cc[None] - 2 * Q @ cen.T; dd[np.arange(len(start)), start] = np.inf
        return dd.argmin(1)
    neo = br_share
    delta = float((neo[land(-ax)] - neo[start]).mean())
    rnd = []
    for _ in range(200):
        q = rng.standard_normal(d); q -= (q @ ax) * ax; q /= np.linalg.norm(q); rnd.append((neo[land(q)] - neo[start]).mean())
    r["A0_backward_neoplasms"] = dict(delta=round(delta, 3), z=round(float((delta - np.mean(rnd)) / np.std(rnd)), 1))
    # concepts: MeSH texts vs their subset paper centroids
    cnt = np.asarray(mesh_F.sum(0)).ravel(); keep = np.flatnonzero(cnt >= 20)
    Pc = unit(np.asarray((mesh_F[:, keep].T @ Xc)) / cnt[keep][:, None])
    T = C_txt[mesh_cols_idx[keep]]; T = unit(T - C_txt.mean(0))
    S = T @ Pc.T; own = S[np.arange(len(keep)), np.arange(len(keep))]
    rank = (S > own[:, None]).sum(1)
    r["concepts"] = dict(n=int(len(keep)), top1=round(float(np.mean(rank == 0)), 3), top10=round(float(np.mean(rank < 10)), 3))
    log(f"{label}: cone {r['cone']:.3f}; W share {r['W_share_top20']} (chance {r['chance_top20']}); time cos "
        f"{r['time_pairwise_cos']:.3f}, year r {r['time_year_r']:.3f}; A0 backward Neoplasms {r['A0_backward_neoplasms']}; "
        f"concepts {r['concepts']}")
    for t in r["themes"][:8]: log(f"   {t}")
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen3-Embedding-8B"); a = ap.parse_args()
    rows = np.load(CK / "subset_rows.npy")
    lab = np.load(D / f"clusters/{NAME}_all.npy")[rows].astype(np.int64); k = lab.max() + 1
    year = np.load(M / "year.npy")[rows]
    Xm = np.asarray(np.load(D / "X.npy", mmap_mode="r")[rows], np.float32)
    Xq = interleave([np.load(CK / a.model / f"papers_{i}.npy").astype(np.float32) for i in range(2)])
    Cq = interleave([np.load(CK / a.model / f"concepts_{i}.npy").astype(np.float32) for i in range(2)])
    Cm = np.load(M / "concept_def.npy").astype(np.float32)
    log(f"papers {Xm.shape} / {Xq.shape}; concepts {Cm.shape} / {Cq.shape}")
    # MeSH incidence on the subset, and concept rows for each MeSH ui
    mp = pq.read_table(M / "mesh_papers.parquet", columns=["row", "mesh"]).to_pandas().set_index("row")
    cdf = pd.read_parquet(M / "concepts.parquet")
    ui_idx = {int(s[1:]): i for i, s in enumerate(cdf.id) if s.startswith("D")}
    r_, c_ = [], []
    for i, row in enumerate(rows):
        if row in mp.index:
            for u in mp.at[row, "mesh"]:
                if int(u) in ui_idx: r_.append(i); c_.append(int(u))
    uis = np.unique(c_); col = {u: j for j, u in enumerate(uis)}
    F = sp.csr_matrix((np.ones(len(r_), np.float32), (r_, [col[u] for u in c_])), shape=(len(rows), len(uis))); F.data[:] = 1
    cidx = np.array([ui_idx[u] for u in uis])
    # Neoplasms (C04) share per topic, from the branch table built on all papers
    br = np.load(M / f"{NAME}_branches.npz", allow_pickle=True)
    neo = br["share"][:, list(br["codes"]).index("C04")]
    an = np.load(M / f"{NAME}_analogies.npz"); E, comp = an["edges"], an["comp"]
    edges_by_theme = [E[comp == t] for t in range(min(12, comp.max() + 1))]
    out = {}
    out["MedEmbed-small"] = measure(Xm, lab, year, k, F, neo, edges_by_theme, Cm, cidx, "MedEmbed-small")
    out[a.model] = measure(Xq, lab, year, k, F, neo, edges_by_theme, Cq, cidx, a.model)
    # cross-model agreement
    rng = np.random.default_rng(1); s = rng.choice(len(rows), 20000, replace=False)
    out["cka"] = cka(Xm[s], Xq[s])
    q = rng.choice(len(rows), 3000, replace=False)
    nn = lambda X: np.argsort(-(unit(X - X.mean(0))[q] @ unit(X - X.mean(0)).T), 1)[:, 1:11]
    a_, b_ = nn(Xm), nn(Xq)
    out["knn10_overlap"] = float(np.mean([len(set(x) & set(y)) / 10 for x, y in zip(a_, b_)]))
    Sm, Sq = out["MedEmbed-small"].pop("W_scores"), out[a.model].pop("W_scores")
    Rr = np.abs(np.corrcoef(Sm.T, Sq.T)[:12, 12:])
    i_, j_ = linear_sum_assignment(-Rr)
    out["W_match"] = [dict(medembed=f"W{i}", other=f"W{j}", abs_r=round(float(Rr[i, j]), 2)) for i, j in zip(i_, j_)][:8]
    log(f"CKA {out['cka']:.3f}; 10-NN overlap {out['knn10_overlap']:.3f}; W matches {out['W_match'][:6]}")
    (CK / f"modelcheck_{a.model}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
