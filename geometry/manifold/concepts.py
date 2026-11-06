"""Concepts next to the papers: MeSH descriptors and GO terms embedded with the papers' own model (MedEmbed-small).

Each concept is embedded twice: its name alone, and "name: definition" (MeSH scope note, GO def). MeSH concepts
also get a data position: the centroid of the human-gene papers indexed with that heading (>= 50 papers), the
"ground truth" of where the idea lives among the papers. Comparing the two says how far a short text lands from
the papers about it (the gap between label and literature), and a linear map fitted on half the descriptors
(ridge, text -> paper centroid) is tested on the other half.
Writes data/geometry/medembed_small/manifold/:
  concepts.parquet   source (mesh|go), id, name, definition, trees/namespace, slim (GO slim generic), npapers
  concept_name.npy, concept_def.npy   float32 (c, 384) unit vectors
  mesh_centroid.npy  float32 (c, 384) paper centroid per concept (NaN rows where < 50 papers or not MeSH)
CPU, root env: uv run python geometry/manifold/concepts.py [--embed] [--score]
"""
import argparse, re, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/geometry/medembed_small"
M = D / "manifold"
GO = Path("/data/public/GeneOntology/gene_ontology/go-basic.obo")
THREADS = 16
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def go_terms():
    rows, cur = [], None
    for line in GO.read_text().splitlines():
        if line == "[Term]": cur = {"subsets": []}; rows.append(cur); continue
        if line.startswith("[") or cur is None: cur = None if line.startswith("[") else cur; continue
        k, _, v = line.partition(": ")
        if k == "id": cur["id"] = v
        elif k == "name": cur["name"] = v
        elif k == "namespace": cur["ns"] = v
        elif k == "def": cur["def"] = re.sub(r'^"(.*)" \[.*$', r"\1", v)
        elif k == "is_obsolete": cur["obs"] = True
        elif k == "subset": cur["subsets"].append(v)
    df = pd.DataFrame([r for r in rows if "id" in r and not r.get("obs")])
    return pd.DataFrame({"source": "go", "id": df.id, "name": df.name, "definition": df["def"].fillna(""),
                         "tag": df.ns, "slim": df.subsets.map(lambda s: "goslim_generic" in s)})


def build_table():
    md = pq.read_table(M / "mesh_desc.parquet").to_pandas()
    mesh = pd.DataFrame({"source": "mesh", "id": md.uid, "name": md.name,  # the NLM id string (6 or 9 digits)
                         "definition": md.scope.fillna(""), "tag": md.trees.map(lambda t: "|".join(t)), "slim": False})
    return pd.concat([mesh, go_terms()], ignore_index=True)


def embed(texts):
    import sys; sys.path.insert(0, str(REPO / "genesearch"))
    from semantic import load_model
    m = load_model(THREADS)  # the box is shared: oversubscribing torch threads is far slower than fewer
    order = np.argsort([len(t) for t in texts])  # similar lengths per batch: less padding
    out = np.zeros((len(texts), 384), np.float32)
    for s in range(0, len(texts), 4096):
        idx = order[s:s + 4096]
        out[idx] = m.encode([texts[i] for i in idx], batch_size=128, convert_to_numpy=True, normalize_embeddings=True,
                            show_progress_bar=False)
        log(f"  embedded {min(s + 4096, len(texts)):,}/{len(texts):,}")
    return out


def mesh_centroids(cdf):
    """Centroid of the papers with each MeSH heading (rows aligned with cdf; NaN unless MeSH with >= 50 papers)."""
    X = np.load(D / "X.npy", mmap_mode="r"); n = len(X)
    t = pq.read_table(M / "mesh_papers.parquet", columns=["row", "mesh"])
    rows = t.column("row").to_numpy(); mesh = t.column("mesh").combine_chunks()
    mo, mv = mesh.offsets.to_numpy(), mesh.values.to_numpy()
    uid = {int(s[1:]): i for i, s in enumerate(cdf.id) if s.startswith("D")}
    col = np.array([uid.get(int(u), -1) for u in mv.tolist()]); r = np.repeat(rows, np.diff(mo)); ok = col >= 0
    F = sp.csr_matrix((np.ones(ok.sum(), np.float32), (col[ok], r[ok])), shape=(len(cdf), n)); F.sum_duplicates(); F.data[:] = 1
    cnt = np.asarray(F.sum(1)).ravel()
    S = np.zeros((len(cdf), 384), np.float64)
    for s in range(0, n, 1_000_000):
        S += F[:, s:s + 1_000_000] @ np.asarray(X[s:s + 1_000_000], np.float32)
    cen = (S / np.maximum(cnt, 1)[:, None]).astype(np.float32); cen[cnt < 50] = np.nan
    return cen, cnt


def score():
    cdf = pd.read_parquet(M / "concepts.parquet")
    cen = np.load(M / "mesh_centroid.npy"); mu = np.load(M / "mean.npy")
    has = ~np.isnan(cen[:, 0]); idx = np.flatnonzero(has)
    C = cen[idx] - mu; C /= np.linalg.norm(C, axis=1, keepdims=True)
    rng = np.random.default_rng(0); test = rng.random(len(idx)) < 0.5
    for kind in ("name", "def"):
        T = np.load(M / f"concept_{kind}.npy")[idx]
        for label, Tq in (("raw", T), ("centred", T - T.mean(0))):
            Q = Tq / np.linalg.norm(Tq, axis=1, keepdims=True)
            sim = Q[test] @ C.T  # find each test concept's own paper centroid among all centroids
            rank = (sim > sim[np.arange(test.sum()), np.flatnonzero(test)][:, None]).sum(1)
            log(f"{kind:4s} {label:7s}: own centroid top-1 {np.mean(rank == 0):.1%}, top-10 {np.mean(rank < 10):.1%}, "
                f"median rank {np.median(rank):.0f} of {len(idx):,}; mean cos to own (raw space) "
                f"{np.mean(np.sum(T[test] * cen[idx][test], 1) / np.linalg.norm(cen[idx][test], axis=1)):.3f}")
        # ridge map text -> centred paper centroid, fitted on the training half
        A, Bt = T[~test], (cen[idx] - mu)[~test]
        Wm = np.linalg.solve(A.T @ A + 1.0 * np.eye(384), A.T @ Bt)
        P = T[test] @ Wm; P /= np.linalg.norm(P, axis=1, keepdims=True)
        sim = P @ C.T; rank = (sim > sim[np.arange(test.sum()), np.flatnonzero(test)][:, None]).sum(1)
        log(f"{kind:4s} ridge  : own centroid top-1 {np.mean(rank == 0):.1%}, top-10 {np.mean(rank < 10):.1%}, median rank {np.median(rank):.0f}")
        np.save(M / f"text2paper_{kind}.npy", np.linalg.solve(T.T @ T + 1.0 * np.eye(384), T.T @ (cen[idx] - mu)).astype(np.float32))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--embed", action="store_true"); ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.embed:
        if (M / "concepts.parquet").exists() and (M / "mesh_centroid.npy").exists():
            cdf = pd.read_parquet(M / "concepts.parquet")
        else:
            cdf = build_table(); log(f"concepts: {cdf.source.value_counts().to_dict()}")
            cen, cnt = mesh_centroids(cdf); cdf["npapers"] = cnt.astype(np.int64)
            np.save(M / "mesh_centroid.npy", cen); log(f"MeSH centroids: {np.sum(~np.isnan(cen[:, 0])):,} with >= 50 papers")
            cdf.to_parquet(M / "concepts.parquet")
        np.save(M / "concept_name.npy", embed(cdf.name.tolist())); log("names embedded")
        defs = [f"{n}: {d}" if d else n for n, d in zip(cdf.name, cdf.definition)]
        np.save(M / "concept_def.npy", embed(defs)); log("definitions embedded")
    if a.score: score()
