"""Share of each topic's papers under each MeSH tree branch (e.g. C04 Neoplasms, C10 Nervous System Diseases,
A08 Nervous System, E02 Therapeutics, G05 Genetic Phenomena), for checking what a direction means.
Branch = the first 3 characters of a tree number (C04, A08, ...), plus the 2-level nodes (C04.557, ...) with >= 1%
of papers. A paper counts once per branch.
Writes data/geometry/medembed_small/manifold/<name>_branches.npz (share (k, b), names (b,), codes (b,)).
CPU, root env: uv run python geometry/manifold/branches.py eom_mcs200_ms3
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/geometry/medembed_small"
M = D / "manifold"
TOP = {"A": "Anatomy", "B": "Organisms", "C": "Diseases", "D": "Chemicals and Drugs", "E": "Techniques",
       "F": "Psychiatry and Psychology", "G": "Phenomena and Processes", "H": "Disciplines", "I": "Social",
       "J": "Technology", "K": "Humanities", "L": "Information Science", "M": "Named Groups", "N": "Health Care",
       "V": "Publication Characteristics", "Z": "Geographicals"}


def main(name):
    lab = np.load(D / f"clusters/{name}_all.npy").astype(np.int64); n = len(lab); k = lab.max() + 1
    desc = pq.read_table(M / "mesh_desc.parquet").to_pandas()
    tree_name = {}
    for nm, trees in zip(desc.name, desc.trees):
        for t in trees: tree_name[t] = nm
    codes = sorted({t[:3] for ts in desc.trees for t in ts} | {t[:7] for ts in desc.trees for t in ts if len(t) >= 7})
    ci = {c: i for i, c in enumerate(codes)}
    ui_br = {}
    for u, trees in zip(desc.ui, desc.trees):
        ui_br[int(u)] = sorted({ci[t[:3]] for t in trees} | {ci[t[:7]] for t in trees if len(t) >= 7})
    mp = pq.read_table(M / "mesh_papers.parquet", columns=["row", "mesh"])
    rows = mp.column("row").to_numpy(); mesh = mp.column("mesh").combine_chunks()
    mo, mv = mesh.offsets.to_numpy(), mesh.values.to_numpy()
    # descriptor -> branches as a sparse matrix, then papers x branches = papers x descriptors @ that
    nd = int(mv.max()) + 1
    r_, c_ = zip(*[(u, b) for u, bs in ui_br.items() if u < nd for b in bs])
    DB = sp.csr_matrix((np.ones(len(r_), np.float32), (r_, c_)), shape=(nd, len(codes)))
    PD = sp.csr_matrix((np.ones(len(mv), np.float32), (np.repeat(rows, np.diff(mo)), mv)), shape=(n, nd))
    PB = PD @ DB; PB.data[:] = 1
    has = np.zeros(n, bool); has[rows[np.diff(mo) > 0]] = True
    L = sp.csr_matrix((np.ones(n, np.float32), (lab, np.arange(n))), shape=(k, n))
    share = (L @ PB).toarray() / np.maximum(np.bincount(lab[has], minlength=k), 1)[:, None]
    overall = np.asarray(PB.sum(0)).ravel() / has.sum()
    keep = np.array([len(c) == 3 or overall[i] >= 0.01 for i, c in enumerate(codes)])
    names = [TOP.get(c[0], c[0]) + ": " + tree_name.get(c, c) if len(c) == 3 else tree_name.get(c, c) for c in codes]
    np.savez(M / f"{name}_branches.npz", share=share[:, keep].astype(np.float32), codes=np.array(codes)[keep],
             names=np.array(names, dtype=object)[keep], overall=overall[keep])
    print(f"{keep.sum()} branches over {k} topics")


if __name__ == "__main__":
    main(sys.argv[1])
