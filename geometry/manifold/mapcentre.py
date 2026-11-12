"""Should the genesearch Map view centre the vectors before its cosine UMAP? A test on real gene maps.

The Map view (genesearch/litmap.py) takes a gene's papers (here only the PubTator-tagged ones; the Map view also
adds Open Targets curated and Europe PMC text-mined papers), finds 15 neighbours by cosine, UMAPs to 12-D, picks
up to 8 HDBSCAN topics there (litmap.topics), and UMAPs to 2-D for the picture. Here the same steps run on four
versions of the same vectors:
  raw        as the Map view does now (unit MedEmbed vectors)
  raw2       raw again: UMAP is not seeded, so this is the run-to-run noise floor
  corpus     minus the mean of all 7.76M human-gene papers (manifold/mean.npy)
  map        minus the mean of this map's own papers (removes what they all share: mostly the gene itself)
Scores use MeSH headings (manifold/parse_mesh.py), which the vectors never saw. A heading is informative for a
map if 2+ of its papers carry it and at most 10% do (so the gene's own heading and check tags drop out).
  mesh_knn_2d / _12d / _384   of each paper's 15 nearest neighbours (on the 2-D picture, in 12-D, in the 384-D
                              vectors themselves), the share that share an informative heading with it
  field_knn_2d                the same for journal field (geometry/truth.py), papers with a field only
  topic_lift                  P(two papers share an informative heading | same topic) / P(the same | any two)
  topics, grey                number of topics (at most 8) and the share of papers left grey
Root env can't run UMAP; use the genesearch env:
  GENESEARCH_MAP_JOBS=16 uv run --project genesearch python geometry/manifold/mapcentre.py [SYMBOL ...]
Writes data/geometry/medembed_small/manifold/mapcentre.csv (appends; finished gene x version rows are skipped).
"""
import csv, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/geometry/medembed_small"
M = D / "manifold"
sys.path.insert(0, str(REPO / "genesearch"))
GENES = ["MUC5B", "SFTPC", "HFE", "TREM2", "LRRK2", "MECP2", "NOD2", "HTT", "PCSK9", "TERT", "CFTR",
         "SNCA", "ACE2", "APOE", "BRCA1"]
VERSIONS = ["raw", "raw2", "corpus", "map"]
FIELDS = ["gene", "version", "n", "secs", "topics", "grey", "mesh_knn_2d", "mesh_knn_12d", "mesh_knn_384",
          "field_knn_2d", "topic_lift", "mesh_random"]
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def knn_share(Y, ok, share, k=15):
    """Mean over papers in ok of the share of their k nearest (Euclidean in Y; among ok papers) for which
    share(i, j) holds."""
    from sklearn.neighbors import NearestNeighbors
    idx = np.flatnonzero(ok)
    nb = NearestNeighbors(n_neighbors=k + 1).fit(Y[idx]).kneighbors(Y[idx], return_distance=False)[:, 1:]
    return float(np.mean([np.mean([share(idx[i], idx[j]) for j in row]) for i, row in enumerate(nb)]))


def main():
    import litmap
    from umap import UMAP
    from umap.umap_ import nearest_neighbors
    jobs = int(os.environ.get("GENESEARCH_MAP_JOBS", "16"))
    genes = sys.argv[1:] or GENES
    X = np.load(D / "X.npy", mmap_mode="r"); pm = np.load(D / "pmids.npy"); mu = np.load(M / "mean.npy")
    row_of = pd.Series(np.arange(len(pm)), index=pm)
    field = np.load(D / "eval/field.npy")
    sym = pd.read_parquet(REPO / "genesearch/index/genes.parquet", columns=["symbol", "entrez_id"]).dropna()
    gid = {s: str(int(e)) for s, e in zip(sym.symbol, sym.entrez_id)}
    hg = pq.read_table(REPO / "data/geometry/human_gene_pmids.parquet", columns=["pmid", "ncbi_genes"])
    mp = pq.read_table(M / "mesh_papers.parquet", columns=["row", "mesh"])
    out = M / "mapcentre.csv"
    if not out.exists(): csv.DictWriter(open(out, "w", newline=""), FIELDS).writeheader()
    done = {(r["gene"], r["version"]) for r in csv.DictReader(open(out))}
    flat = pc.list_flatten(hg["ncbi_genes"]); par = np.asarray(pc.list_parent_indices(hg["ncbi_genes"]))
    for g in genes:
        pmids = np.asarray(hg["pmid"])[np.unique(par[np.asarray(pc.equal(flat, gid[g]))])]
        rows = np.sort(row_of.reindex(pmids).dropna().astype(np.int64).values)
        n = len(rows)
        # MeSH sets of the map's papers, and the informative headings
        sub = mp.filter(pc.is_in(mp["row"], pa_array(rows)))
        mesh = dict(zip(np.asarray(sub["row"]).tolist(), sub["mesh"].to_pylist()))
        cnt = pd.Series([h for r in rows for h in mesh.get(int(r), [])]).value_counts()
        keep = set(cnt[(cnt >= 2) & (cnt <= 0.1 * n)].index)
        H = [set(mesh.get(int(r), [])) & keep for r in rows]
        hasH = np.array([len(h) > 0 for h in H])
        share = lambda i, j: not H[i].isdisjoint(H[j])
        f = field[rows]; okf = f > 0
        fshare = lambda i, j: f[i] == f[j]
        rng = np.random.default_rng(0)
        ri, rj = rng.choice(np.flatnonzero(hasH), 20000), rng.choice(np.flatnonzero(hasH), 20000)
        base = float(np.mean([share(i, j) for i, j in zip(ri, rj) if i != j]))
        V0 = np.asarray(X[rows], np.float32)
        log(f"{g}: {n:,} papers, {hasH.mean():.0%} with an informative heading ({len(keep):,} headings), random pairs share {base:.3f}")
        for ver in VERSIONS:
            if (g, ver) in done: continue
            t1 = time.time()
            V = V0 if ver in ("raw", "raw2") else V0 - (mu if ver == "corpus" else V0.mean(0))
            V = (V / np.linalg.norm(V, axis=1, keepdims=True)).astype(np.float32)
            knn = nearest_neighbors(V, 15, "cosine", {}, False, None, n_jobs=jobs)
            u12 = UMAP(n_components=12, n_neighbors=15, min_dist=0.0, metric="cosine", precomputed_knn=knn,
                       n_jobs=jobs).fit_transform(V)
            lab = litmap.topics(u12)
            xy = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", precomputed_knn=knn,
                      n_jobs=jobs).fit_transform(V)
            secs = round(time.time() - t1)
            tl = lab >= 0
            a, b = pairs_same(lab, rng, 20000)
            same = np.mean([share(i, j) for i, j in zip(a, b) if hasH[i] and hasH[j]])
            r = dict(gene=g, version=ver, n=n, secs=secs, topics=int(lab.max() + 1), grey=round(float(1 - tl.mean()), 3),
                     mesh_knn_2d=knn_share(xy, hasH, share), mesh_knn_12d=knn_share(u12, hasH, share),
                     mesh_knn_384=knn_share(V, hasH, share), field_knn_2d=knn_share(xy, okf, fshare),
                     topic_lift=float(same / base), mesh_random=base)
            csv.DictWriter(open(out, "a", newline=""), FIELDS).writerow(r)
            log({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


def pa_array(rows):
    import pyarrow as pa
    return pa.array(rows.astype(np.int64))


def pairs_same(lab, rng, m):
    """m random pairs of different papers from the same topic (a topic is hit in proportion to its size)."""
    idx = np.flatnonzero(lab >= 0); o = idx[np.argsort(lab[idx], kind="stable")]; sl = lab[o]
    start = np.searchsorted(sl, sl, "left"); size = np.searchsorted(sl, sl, "right") - start
    i = rng.integers(0, len(o), m); j = start[i] + rng.integers(0, size[i]); ok = o[i] != o[j]
    return o[i][ok], o[j][ok]


if __name__ == "__main__":
    main()
