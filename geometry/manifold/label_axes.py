"""Name axes by what papers at each end are about: MeSH headings, publication types and year.

For an axis a (unit 384-D), each paper gets a score: within-topic (x - c_topic) . a for the common within-topic
axes of axes.py (W), plain x . a for between-topic axes (B) and the global time direction. For every MeSH
descriptor on >= MIN papers and every publication type, effect = (mean score of papers with it - mean of all) / sd,
so +0.5 means papers with that heading sit half a standard deviation toward the + end. Also: correlation of the
score with publication year.
Writes data/geometry/medembed_small/manifold/<name>_axis_labels.json and prints the top headings per axis.
CPU, root env: uv run python geometry/manifold/label_axes.py eom_mcs200_ms3 [--n 20]
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, scipy.sparse as sp, torch

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/geometry/medembed_small"
M = D / "manifold"
MIN = 3000
torch.set_num_threads(96)
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def features(n):
    """Sparse paper x feature matrix (MeSH descriptors and publication types on >= MIN papers) and names."""
    cache = M / "features.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return sp.csr_matrix((np.ones(len(z["col"]), np.float32), z["col"], z["ptr"]), shape=(n, len(z["names"]))), list(z["names"])
    t = pq.read_table(M / "mesh_papers.parquet", columns=["row", "mesh", "ptypes"])
    rows = t.column("row").to_numpy()
    mesh = t.column("mesh").combine_chunks(); pt = t.column("ptypes").combine_chunks()
    mo, mv = mesh.offsets.to_numpy(), mesh.values.to_numpy()
    po = pt.offsets.to_numpy(); pv = np.array(pt.values.to_pylist(), dtype=object)
    desc = pq.read_table(M / "mesh_desc.parquet", columns=["ui", "name"]).to_pandas().set_index("ui").name
    mu, mc = np.unique(mv, return_counts=True); keepm = mu[mc >= MIN]
    pu, pc = np.unique(pv, return_counts=True); keepp = pu[pc >= MIN // 3]
    mi = {u: i for i, u in enumerate(keepm)}; pi = {u: i + len(keepm) for i, u in enumerate(keepp)}
    r_m = np.repeat(rows, np.diff(mo)); c_m = np.array([mi.get(u, -1) for u in mv.tolist()])
    r_p = np.repeat(rows, np.diff(po)); c_p = np.array([pi.get(u, -1) for u in pv.tolist()])
    r = np.concatenate([r_m, r_p]); c = np.concatenate([c_m, c_p]); ok = c >= 0
    F = sp.csr_matrix((np.ones(ok.sum(), np.float32), (r[ok], c[ok])), shape=(n, len(keepm) + len(keepp)))
    F.sum_duplicates(); F.data[:] = 1
    names = [desc.get(u, str(u)) for u in keepm] + [f"[{p}]" for p in keepp]
    np.savez(cache, col=F.indices, ptr=F.indptr, names=np.array(names, dtype=object))
    return F, names


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    X = np.load(D / "X.npy", mmap_mode="r"); n = len(X)
    lab = np.load(D / f"clusters/{a.name}_all.npy").astype(np.int64)
    year = np.load(M / "year.npy").astype(np.float32)
    z = np.load(M / f"{a.name}_axes.npz")
    F, names = features(n); log(f"features {F.shape[1]:,}, links {F.nnz:,}")
    cen = z["cen"].astype(np.float32)
    tu = z["tvec"] / np.maximum(np.linalg.norm(z["tvec"], axis=1, keepdims=True), 1e-12)
    gt = tu[z["tn"] > 200].mean(0); gt /= np.linalg.norm(gt)
    axes = {f"W{i}": ("within", z["W_vecs"][:, i]) for i in range(a.n)}
    axes |= {f"B{i}": ("between", z["B_vecs"][:, i]) for i in range(a.n)}
    axes["time"] = ("within", gt)
    keys = list(axes); A = torch.from_numpy(np.stack([axes[k][1] for k in keys], 1).astype(np.float32))
    within = torch.tensor([axes[k][0] == "within" for k in keys])
    S = np.zeros((n, len(keys)), np.float32)
    C = 1_000_000
    for s in range(0, n, C):
        xb = torch.from_numpy(np.asarray(X[s:s + C], np.float32))
        sw = (xb - torch.from_numpy(cen[lab[s:s + C]])) @ A; sb = xb @ A
        S[s:s + C] = torch.where(within, sw, sb).numpy()
    log("scores done")
    mu_s, sd_s = S.mean(0), S.std(0)
    cnt = np.asarray(F.sum(0)).ravel()
    eff = ((F.T @ S) / cnt[:, None] - mu_s) / sd_s  # (features, axes)
    ok = year > 0
    ycor = [float(np.corrcoef(S[ok, j], year[ok])[0, 1]) for j in range(len(keys))]
    out = {}
    for j, kk in enumerate(keys):
        o = np.argsort(eff[:, j])
        out[kk] = dict(kind=axes[kk][0], year_corr=round(ycor[j], 3),
                       plus=[(names[i], round(float(eff[i, j]), 2), int(cnt[i])) for i in o[::-1][:15]],
                       minus=[(names[i], round(float(eff[i, j]), 2), int(cnt[i])) for i in o[:15]])
        p = ", ".join(f"{x[0]} {x[1]:+.2f}" for x in out[kk]["plus"][:8])
        m = ", ".join(f"{x[0]} {x[1]:+.2f}" for x in out[kk]["minus"][:8])
        print(f"\n{kk} ({axes[kk][0]}, year r={ycor[j]:+.2f})\n  + {p}\n  - {m}")
    (M / f"{a.name}_axis_labels.json").write_text(json.dumps(out, indent=1))
    np.save(M / f"{a.name}_axis_scores.npy", S.astype(np.float16)); (M / f"{a.name}_axis_keys.json").write_text(json.dumps(keys))
    log("done")


if __name__ == "__main__":
    main()
