"""Pull the human-gene papers' MedEmbed-small vectors out of the whole-PubMed index.

Rows of data/geometry/human_gene_pmids.parquet (PubTator3 gene tags that are HGNC
genes) that have a vector in genesearch/index/emb_medembed_small/ ->
data/geometry/medembed_small/{pmids.npy (int64), X.npy (float16, n x 384)}, in
pmids.npy order of the source index (file order of the corpus).
"""
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "genesearch/index/emb_medembed_small"
OUT = REPO / "data/geometry/medembed_small"
OUT.mkdir(parents=True, exist_ok=True)

want = pq.read_table(REPO / "data/geometry/human_gene_pmids.parquet", columns=["pmid"]).column("pmid").to_numpy()
pmids = np.load(SRC / "pmids.npy")
keep = np.isin(pmids, want)
print(f"human-gene pmids {len(want):,}; with a vector {keep.sum():,} of {len(pmids):,}", flush=True)
np.save(OUT / "pmids.npy", pmids[keep])
shards = sorted(SRC.glob("[0-9]*.npy"))
X = np.lib.format.open_memmap(OUT / "X.npy.tmp", mode="w+", dtype=np.float16, shape=(int(keep.sum()), 384))
at = row = 0
for f in shards:
    v = np.load(f, mmap_mode="r")
    m = keep[row:row + len(v)]
    X[at:at + m.sum()] = v[m]
    at += m.sum(); row += len(v)
    print(f"  {f.name}: {at:,}", flush=True)
X.flush(); del X
(OUT / "X.npy.tmp").rename(OUT / "X.npy")
print("done", at)
