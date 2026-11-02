"""Tune HDBSCAN on the 12-D UMAP (map_gpu.py's geom_12d_k15/u12.npy) before clustering all 7.76M papers.

Each setting is fitted on a 1M-paper sample (rng seed 3, always holding truth.py's 20k queries), or on everything
with --full. Scores:
  clusters, noise      number of clusters, share of papers left unclustered
  together15           of each clustered query's exact 384-D 15 nearest neighbours that are in the fit and
                       clustered, the share in the query's own cluster (queries with none are skipped)
  gene_lift            P(two papers share a gene | same cluster) / P(share a gene | any two clustered papers);
                       same-cluster pairs: a random clustered paper, then a random other paper of its cluster
                       (so a cluster of n papers weighs about n, not n^2)
  field_lift           the same for journal field (papers with a field only)
  field_purity         over clustered papers that have a journal field, the share in their cluster's most
                       common field
  together15_all       of every query's 15 nearest neighbours that are in the fit, the share in the query's own
                       cluster, noise counting as a miss (one number for the trade between purity and coverage)
Appends to data/geometry/medembed_small/clusters/results.csv; --full also writes clusters/<name>.npy (labels).

--assign NAME: fit NAME on the 1M sample, then give every one of the 7.76M papers the majority cluster of its 15
nearest clustered sample papers in the 12-D UMAP (HDBSCAN leaves 40 to 60% of papers as noise at every setting
tried). Writes clusters/<name>_all.npy (int32 labels for all papers) and clusters/<name>_core.npy (bool: HDBSCAN
itself clustered the paper, rather than the vote), and scores the full labelling as fit "assigned".
GPU (RAPIDS env): uv run python cluster_sweep.py --data ~/medembed_small [--grid samples] [--full | --assign NAME]
"""
import argparse, csv, time
from pathlib import Path
import numpy as np

GRIDS = {
    "main": [(sel, mcs, 10) for sel in ("eom", "leaf") for mcs in (50, 200, 1000, 5000)],
    # half the papers stayed noise at min_samples 10 whatever the rest: fewer samples, less noise?
    "samples": [("eom", mcs, ms) for ms in (1, 3) for mcs in (50, 200, 1000)],
}
FIELDS = ["name", "fit", "n", "selection", "min_cluster_size", "min_samples", "secs", "clusters", "noise",
          "together15", "together15_all", "gene_lift", "field_lift", "field_purity"]
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
name_of = lambda sel, mcs, ms: f"{sel}_mcs{mcs}_ms{ms}"


def pairs_same(lab: np.ndarray, m: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """m random pairs of distinct clustered rows from the same cluster (a cluster is hit in proportion to size)."""
    idx = np.flatnonzero(lab >= 0); o = idx[np.argsort(lab[idx], kind="stable")]
    sl = lab[o]; start = np.searchsorted(sl, sl, "left"); size = np.searchsorted(sl, sl, "right") - start
    i = rng.integers(0, len(o), m)
    j = start[i] + rng.integers(0, size[i])
    ok = o[i] != o[j]
    return o[i][ok], o[j][ok]


def score(lab, rows, q_local, truth_local, field, ptr, genes) -> dict:
    rng = np.random.default_rng(4)
    share_gene = lambda a, b: float(np.mean([not set(genes[ptr[x]:ptr[x + 1]].tolist()).isdisjoint(
        genes[ptr[y]:ptr[y + 1]].tolist()) for x, y in zip(rows[a], rows[b])]))
    cl = np.flatnonzero(lab >= 0)
    a, b = pairs_same(lab, 40_000, rng)
    ra, rb = rng.choice(cl, 40_000), rng.choice(cl, 40_000)
    f = field[rows]
    fa, fb = pairs_same(np.where(f > 0, lab, -1), 200_000, rng)
    fl = cl[f[cl] > 0]; ga, gb = rng.choice(fl, 200_000), rng.choice(fl, 200_000)
    tog, tog_all = [], []
    for qi, t in zip(q_local, truth_local):
        t = t[t >= 0]
        if not len(t): continue
        tog_all.append(np.mean(lab[t] == lab[qi]) if lab[qi] >= 0 else 0.0)
        t = t[lab[t] >= 0]
        if lab[qi] >= 0 and len(t): tog.append(np.mean(lab[t] == lab[qi]))
    lf = lab[fl]; ff = f[fl]
    key = lf.astype(np.int64) * 256 + ff
    uk, cnt = np.unique(key, return_counts=True)
    top = np.zeros(lab.max() + 1); np.maximum.at(top, uk // 256, cnt)
    tot = np.bincount(lf, minlength=lab.max() + 1)
    return {"clusters": int(lab.max() + 1), "noise": float((lab < 0).mean()), "together15": float(np.mean(tog)),
            "together15_all": float(np.mean(tog_all)),
            "gene_lift": share_gene(a, b) / share_gene(ra, rb),
            "field_lift": float(np.mean(f[fa] == f[fb]) / np.mean(f[ga] == f[gb])),
            "field_purity": float(top[tot > 0].sum() / tot.sum())}


def assign(args, U, q, knn, field, ptr, genes) -> None:
    from cuml.cluster import HDBSCAN
    from cuml.neighbors import NearestNeighbors
    n = len(U)
    rest = np.setdiff1d(np.arange(n), q)
    rows = np.sort(np.concatenate([q, np.random.default_rng(3).choice(rest, args.sample - len(q), replace=False)]))
    sel, mcs, ms = args.assign.split("_")
    t1 = time.time()
    lab = HDBSCAN(min_cluster_size=int(mcs[3:]), min_samples=int(ms[2:]), cluster_selection_method=sel).fit_predict(U[rows])
    lab = np.asarray(lab.get() if hasattr(lab, "get") else lab, np.int64)
    log(f"fitted: {lab.max() + 1} clusters, {(lab < 0).mean():.1%} noise")
    core_rows, core_lab = rows[lab >= 0], lab[lab >= 0]
    nn = NearestNeighbors(n_neighbors=15).fit(U[core_rows])
    out_lab = np.empty(n, np.int64)
    for s in range(0, n, 500_000):
        _, idx = nn.kneighbors(U[s:s + 500_000])
        v = core_lab[np.asarray(idx.get() if hasattr(idx, "get") else idx)]
        # majority vote per row: sort each row's labels, take the longest run
        v.sort(axis=1)
        run = np.ones_like(v)
        for j in range(1, v.shape[1]):
            run[:, j] = np.where(v[:, j] == v[:, j - 1], run[:, j - 1] + 1, 1)
        out_lab[s:s + 500_000] = v[np.arange(len(v)), run.argmax(1)]
        log(f"  voted {min(s + 500_000, n):,}/{n:,}")
    core = np.zeros(n, bool); core[core_rows] = True
    out_lab[core_rows] = core_lab  # (HDBSCAN's own label wins where it gave one)
    out = args.data / "clusters"; out.mkdir(exist_ok=True)
    if not (out / "results.csv").exists():
        csv.DictWriter(open(out / "results.csv", "w", newline=""), FIELDS).writeheader()
    np.save(out / f"{args.assign}_all.npy", out_lab.astype(np.int32)); np.save(out / f"{args.assign}_core.npy", core)
    row = {"name": args.assign, "fit": "assigned", "n": n, "selection": sel, "min_cluster_size": int(mcs[3:]),
           "min_samples": int(ms[2:]), "secs": round(time.time() - t1),
           **score(out_lab, np.arange(n), q, knn, field, ptr, genes)}
    csv.DictWriter(open(out / "results.csv", "a", newline=""), FIELDS).writerow(row)
    log({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    print("exit 0", flush=True)


def main() -> None:
    from cuml.cluster import HDBSCAN
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=1_000_000)
    ap.add_argument("--grid", default="main", choices=list(GRIDS))
    ap.add_argument("--assign", help="fit this setting on the sample, then vote every paper into a cluster")
    ap.add_argument("--full", help="fit this one setting (a results name, e.g. eom_mcs1000_ms10) on every paper")
    args = ap.parse_args()
    U = np.load(args.data / "geom_12d_k15/u12.npy"); n = len(U)
    e = args.data / "eval"
    q, knn = np.load(e / "queries.npy"), np.load(e / "knn.npy")[:, :15].astype(np.int64)
    field, ptr, genes = np.load(e / "field.npy"), np.load(e / "genes_ptr.npy"), np.load(e / "genes.npy")
    if args.full:
        rows = np.arange(n)
        sel, mcs, ms = args.full.split("_"); grid = [(sel, int(mcs[3:]), int(ms[2:]))]
    elif args.assign:
        return assign(args, U, q, knn, field, ptr, genes)
    else:
        rest = np.setdiff1d(np.arange(n), q)
        rows = np.sort(np.concatenate([q, np.random.default_rng(3).choice(rest, args.sample - len(q), replace=False)]))
        grid = GRIDS[args.grid]
    pos = np.full(n, -1, np.int64); pos[rows] = np.arange(len(rows))
    q_local, truth_local = pos[q], pos[knn]
    out = args.data / "clusters"; out.mkdir(exist_ok=True)
    res = out / "results.csv"
    if not res.exists():
        csv.DictWriter(open(res, "w", newline=""), FIELDS).writeheader()
    fit = "full" if args.full else f"sample{len(rows)}"
    done = {(r["fit"], r["name"]) for r in csv.DictReader(open(res))}
    X = U[rows]
    for sel, mcs, ms in grid:
        name = name_of(sel, mcs, ms)
        if (fit, name) in done: continue
        t1 = time.time()
        lab = HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_method=sel).fit_predict(X)
        lab = np.asarray(lab.get() if hasattr(lab, "get") else lab, np.int64)
        secs = round(time.time() - t1)
        if args.full: np.save(out / f"{name}.npy", lab.astype(np.int32))
        row = {"name": name, "fit": fit, "n": len(rows), "selection": sel, "min_cluster_size": mcs,
               "min_samples": ms, "secs": secs, **score(lab, rows, q_local, truth_local, field, ptr, genes)}
        csv.DictWriter(open(res, "a", newline=""), FIELDS).writerow(row)
        log({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    print("exit 0", flush=True)


if __name__ == "__main__":
    main()
