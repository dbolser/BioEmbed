"""Does centring give better nearest neighbours, or only different ones?

For geometry/truth.py's 20,000 query papers: their exact 15 nearest among all 7.76M papers by raw cosine
(eval/knn.npy) and by centred cosine (eval/knn_centred.npy, made by geometry/hdbscan_384.py). Each neighbour is
scored by what the vectors never saw:
  mesh    shares an informative MeSH heading with the query (a heading on at most 1% of the papers that have
          MeSH, and on 2+; so check tags such as Humans drop out); queries with such a heading only
  gene    shares an HGNC gene (PubTator) with the query
  field   same journal field (PubMedBERT landscape), papers with a field only
Each is reported for all 15 neighbours, and for only the neighbours that differ between the two (raw-only vs
centred-only), with the paired difference and its standard error over queries.
CPU, root env: uv run python geometry/manifold/centre_nn.py
"""
from collections import Counter
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, pyarrow.compute as pc

D = Path(__file__).resolve().parents[2] / "data/geometry/medembed_small"
E = D / "eval"
q = np.load(E / "queries.npy"); nr = np.load(E / "knn.npy")[:, :15].astype(np.int64)
nc = np.load(E / "knn_centred.npy")[:, :15].astype(np.int64)
print(f"{len(q):,} queries; 15-NN shared by raw and centred: {np.mean([len(set(a) & set(b)) / 15 for a, b in zip(nr, nc)]):.3f}")

t = pq.read_table(D / "manifold/mesh_papers.parquet", columns=["row", "mesh"])
cnt = Counter(pc.list_flatten(t["mesh"]).to_numpy().tolist()); n_mesh = t.num_rows
keep = {h for h, c in cnt.items() if 2 <= c <= 0.01 * n_mesh}
need = np.unique(np.concatenate([q, nr.ravel(), nc.ravel()]))
sub = t.filter(pc.is_in(t["row"], value_set=__import__("pyarrow").array(need)))
H = {int(r): set(m) & keep for r, m in zip(sub["row"].to_numpy(), sub["mesh"].to_pylist())}
ptr, genes = np.load(E / "genes_ptr.npy"), np.load(E / "genes.npy")
G = lambda r: set(genes[ptr[r]:ptr[r + 1]].tolist())
field = np.load(E / "field.npy")

tests = {
    "mesh": (lambda r: bool(H.get(int(r))), lambda a, b: bool(H.get(int(a), set()) & H.get(int(b), set()))),
    "gene": (lambda r: bool(G(r)), lambda a, b: bool(G(a) & G(b))),
    "field": (lambda r: field[r] > 0, lambda a, b: field[a] == field[b]),
}
for name, (has, share) in tests.items():
    allr, allc, onlyr, onlyc = [], [], [], []
    for qq, a, b in zip(q, nr, nc):
        if not has(qq): continue
        allr.append(np.mean([share(qq, j) for j in a])); allc.append(np.mean([share(qq, j) for j in b]))
        ra, cb = [j for j in a if j not in set(b)], [j for j in b if j not in set(a)]
        if ra and cb:
            onlyr.append(np.mean([share(qq, j) for j in ra])); onlyc.append(np.mean([share(qq, j) for j in cb]))
    d1, d2 = np.array(allc) - np.array(allr), np.array(onlyc) - np.array(onlyr)
    print(f"{name:5s} all 15: raw {np.mean(allr):.4f}, centred {np.mean(allc):.4f} (diff {d1.mean():+.4f} ± {d1.std() / np.sqrt(len(d1)):.4f}, "
          f"{len(d1):,} queries) | only the neighbours that differ: raw {np.mean(onlyr):.4f}, centred {np.mean(onlyc):.4f} "
          f"(diff {d2.mean():+.4f} ± {d2.std() / np.sqrt(len(d2)):.4f}, {len(d2):,} queries)")
