"""HDBSCAN straight on the 384-D MedEmbed vectors (no UMAP), with cosine done right.

cuML's HDBSCAN is Euclidean only. On unit vectors Euclidean^2 = 2 - 2 cos, so every pair is in the same order and
the HDBSCAN tree is the cosine tree (only the stability-based cluster choice can differ a little). Two spaces:
  raw       the unit vectors as they are: plain cosine
  centred   minus the corpus mean, then unit length again: cosine around the corpus centre (the space is a narrow
            cone, mean vector length 0.79, and centring changes ~20% of nearest neighbours, manifold/README.md s1)
Each setting is fitted on cluster_sweep.py's 1M-paper sample (seed 3, holding truth.py's 20k queries) and scored
as there (clusters, noise, together15, gene_lift, field_lift, field_purity), plus
  together15_c   together15 against centred-cosine truth (each query's 15 nearest by centred cosine, made here),
                 since truth.py's neighbours are raw cosine and would favour the raw space
  ari_umap       adjusted Rand index with the 12-D UMAP topics (clusters/eom_mcs200_ms3_all.npy), on the papers
                 this fit clusters
Appends to <data>/clusters/hdbscan384.csv; labels to <data>/clusters/h384_<space>_<setting>.npy (sample rows in
h384_rows.npy). A preempted run resumes: finished (space, setting) rows are skipped.
GPU (RAPIDS env, next to cluster_sweep.py): uv run python hdbscan_384.py --data ~/medembed_small [--sample N]
"""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
from cluster_sweep import score

GRID = [("eom", 200, 3), ("eom", 1000, 3), ("eom", 50, 3), ("leaf", 200, 3), ("eom", 200, 10), ("leaf", 50, 3), ("leaf", 1000, 10)]
FIELDS = ["space", "name", "n", "secs", "clusters", "noise", "largest", "together15", "together15_c", "together15_all",
          "gene_lift", "field_lift", "field_purity", "ari_umap"]
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def ari(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index from the contingency table."""
    _, ai = np.unique(a, return_inverse=True); _, bi = np.unique(b, return_inverse=True)
    _, nij = np.unique(ai.astype(np.int64) * (bi.max() + 1) + bi, return_counts=True)
    c2 = lambda x: float((x.astype(np.float64) * (x - 1) / 2).sum())
    s, sa, sb, n2 = c2(nij), c2(np.bincount(ai)), c2(np.bincount(bi)), len(a) * (len(a) - 1) / 2
    e = sa * sb / n2
    return (s - e) / ((sa + sb) / 2 - e)


def unit_rows(X, mu, rows, cp):
    """float32 unit vectors for rows, optionally centred by mu, built on the GPU in chunks."""
    out = np.empty((len(rows), X.shape[1]), np.float32)
    for s in range(0, len(rows), 500_000):
        b = cp.asarray(np.asarray(X[rows[s:s + 500_000]], np.float32))
        if mu is not None: b -= cp.asarray(mu)
        b /= cp.linalg.norm(b, axis=1, keepdims=True)
        out[s:s + 500_000] = cp.asnumpy(b)
    return out


def centred_truth(X, mu, q, k, cp) -> np.ndarray:
    """Each query's exact k nearest rows of all X by centred cosine (itself excluded)."""
    n = len(X); Q = unit_rows(X, mu, q, cp).astype(np.float16)
    best_s = np.full((len(q), k + 1), -9, np.float32); best_i = np.zeros((len(q), k + 1), np.int64)
    for s in range(0, n, 1_000_000):
        B = cp.asarray(unit_rows(X, mu, np.arange(s, min(s + 1_000_000, n)), cp).astype(np.float16))
        for a in range(0, len(q), 200):  # 200 x 1M scores (+ int64 indices) stay well inside an L4
            S = (cp.asarray(Q[a:a + 200]) @ B.T).astype(cp.float32)
            i = cp.argpartition(S, -(k + 1), axis=1)[:, -(k + 1):]; v = cp.take_along_axis(S, i, 1)
            del S
            cs = np.concatenate([best_s[a:a + 200], cp.asnumpy(v)], 1)
            ci = np.concatenate([best_i[a:a + 200], cp.asnumpy(i) + s], 1)
            o = np.argsort(-cs, 1)[:, :k + 1]
            best_s[a:a + 200] = np.take_along_axis(cs, o, 1); best_i[a:a + 200] = np.take_along_axis(ci, o, 1)
            cp.get_default_memory_pool().free_all_blocks()
        del B
        log(f"  centred truth: {min(s + 1_000_000, n):,}/{n:,} rows")
    return np.array([r[r != qq][:k] for r, qq in zip(best_i, q)])


def main() -> None:
    import cupy as cp
    from cuml.cluster import HDBSCAN
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=1_000_000)
    a = ap.parse_args()
    X = np.load(a.data / "X.npy", mmap_mode="r"); n = len(X)
    e = a.data / "eval"; cl = a.data / "clusters"
    q, knn = np.load(e / "queries.npy"), np.load(e / "knn.npy")[:, :15].astype(np.int64)
    field, ptr, genes = np.load(e / "field.npy"), np.load(e / "genes_ptr.npy"), np.load(e / "genes.npy")
    umap_lab = np.load(cl / "eom_mcs200_ms3_all.npy")
    mu_f = a.data / "mean384.npy"
    if not mu_f.exists():
        mu = np.zeros(X.shape[1], np.float64)
        for s in range(0, n, 1_000_000): mu += np.asarray(X[s:s + 1_000_000], np.float64).sum(0)
        np.save(mu_f, (mu / n).astype(np.float32))
    mu = np.load(mu_f); log(f"corpus mean length {np.linalg.norm(mu):.3f}")
    kc_f = e / "knn_centred.npy"
    if not kc_f.exists(): np.save(kc_f, centred_truth(X, mu, q, 15, cp).astype(np.int32))
    knn_c = np.load(kc_f).astype(np.int64)
    log(f"15-NN shared by raw and centred truth: {np.mean([len(set(x) & set(y)) / 15 for x, y in zip(knn, knn_c)]):.3f}")
    # the same sample as cluster_sweep.py (its queries plus random rows, seed 3)
    rest = np.setdiff1d(np.arange(n), q)
    rows = np.sort(np.concatenate([q, np.random.default_rng(3).choice(rest, a.sample - len(q), replace=False)]))
    np.save(cl / "h384_rows.npy", rows)
    pos = np.full(n, -1, np.int64); pos[rows] = np.arange(len(rows))
    q_local = pos[q]
    res = cl / "hdbscan384.csv"
    if not res.exists(): csv.DictWriter(open(res, "w", newline=""), FIELDS).writeheader()
    done = {(r["space"], r["name"], r["n"]) for r in csv.DictReader(open(res))}
    for space in ("centred", "raw"):
        U = unit_rows(X, mu if space == "centred" else None, rows, cp)
        for sel, mcs, ms in GRID:
            name = f"{sel}_mcs{mcs}_ms{ms}"
            if (space, name, str(len(rows))) in done: continue
            t1 = time.time()
            lab = HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_method=sel).fit_predict(U)
            lab = np.asarray(lab.get() if hasattr(lab, "get") else lab, np.int64)
            secs = round(time.time() - t1)
            np.save(cl / f"h384_{space}_{name}.npy", lab.astype(np.int32))
            ok = lab >= 0
            if ok.sum() < 2 or lab.max() < 1:  # (score() needs at least two clusters)
                row = {"space": space, "name": name, "n": len(rows), "secs": secs, "clusters": int(lab.max() + 1),
                       "noise": float(1 - ok.mean())}
                csv.DictWriter(open(res, "a", newline=""), FIELDS).writerow(row); log(row); continue
            row = {"space": space, "name": name, "n": len(rows), "secs": secs,
                   "largest": float(np.bincount(lab[ok]).max() / len(lab)) if ok.any() else 0.0,
                   **score(lab, rows, q_local, pos[knn], field, ptr, genes)}
            # together15 against centred truth (score() already gives it against raw truth)
            tog = []
            for qi, t in zip(q_local, pos[knn_c]):
                t = t[t >= 0]; t = t[lab[t] >= 0]
                if lab[qi] >= 0 and len(t): tog.append(np.mean(lab[t] == lab[qi]))
            row["together15_c"] = float(np.mean(tog)) if tog else 0.0
            row["ari_umap"] = ari(lab[ok], umap_lab[rows][ok]) if ok.sum() > 1 else 0.0
            csv.DictWriter(open(res, "a", newline=""), FIELDS).writerow(row)
            log(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}))
        del U
    print("exit 0", flush=True)


if __name__ == "__main__":
    main()
