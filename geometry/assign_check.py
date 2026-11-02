"""Score `cluster_sweep.py --assign` labellings on held-out papers.

The 20k truth.py queries are forced into the 1M-paper HDBSCAN sample, so scoring the full labelling on them
flatters it: most papers get a voted label, the queries mostly keep HDBSCAN's own. Here: 20k papers outside the
sample (rng seed 5), their exact 384-D 15 nearest neighbours among all 7.76M papers (as truth.py), and
cluster_sweep.py's scores of each clusters/<name>_all.npy on them. Appends rows with fit "assigned_heldout" to
clusters/results.csv. CPU (~4 min): uv run python geometry/assign_check.py eom_mcs200_ms3 eom_mcs50_ms3
"""
import csv, sys, time
from pathlib import Path
import numpy as np, torch
from cluster_sweep import FIELDS, score

REPO = Path(__file__).resolve().parent.parent
D = REPO / "data/geometry/medembed_small"
E, C = D / "eval", D / "clusters"
Q, K, SAMPLE = 20_000, 15, 1_000_000
torch.set_num_threads(96)
t0 = time.time()

X = np.load(D / "X.npy", mmap_mode="r"); n = len(X)
q0 = np.load(E / "queries.npy")
rest = np.setdiff1d(np.arange(n), q0)
rows = np.concatenate([q0, np.random.default_rng(3).choice(rest, SAMPLE - len(q0), replace=False)])  # (as cluster_sweep)
out = np.ones(n, bool); out[rows] = False
q = np.sort(np.random.default_rng(5).choice(np.flatnonzero(out), Q, replace=False))

Xq = torch.from_numpy(np.asarray(X[q], np.float32))
best_s = torch.full((Q, K + 1), -2.0); best_i = torch.zeros((Q, K + 1), dtype=torch.int64)
for s in range(0, n, 250_000):
    v, i = (Xq @ torch.from_numpy(np.asarray(X[s:s + 250_000], np.float32)).T).topk(K + 1, dim=1)
    best_s, o = torch.cat([best_s, v], 1).topk(K + 1, dim=1); best_i = torch.cat([best_i, i + s], 1).gather(1, o)
knn = best_i.numpy()
keep = knn != q[:, None]; keep[~(~keep).any(1), -1] = False  # (drop the query itself, or the last)
knn = knn[keep].reshape(Q, K)
print(f"held-out neighbours in {time.time() - t0:.0f}s", flush=True)

field, ptr, genes = np.load(E / "field.npy"), np.load(E / "genes_ptr.npy"), np.load(E / "genes.npy")
for name in sys.argv[1:]:
    lab = np.load(C / f"{name}_all.npy").astype(np.int64)
    sel, mcs, ms = name.split("_")
    row = {"name": name, "fit": "assigned_heldout", "n": n, "selection": sel, "min_cluster_size": int(mcs[3:]),
           "min_samples": int(ms[2:]), "secs": "", **score(lab, np.arange(n), q, knn, field, ptr, genes)}
    csv.DictWriter(open(C / "results.csv", "a", newline=""), FIELDS).writerow(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
