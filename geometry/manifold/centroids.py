"""Topic centroids in the full 384-D MedEmbed space, overall and per publication year.

Inputs: data/geometry/medembed_small/{X.npy, pmids.npy}, clusters/<name>_all.npy (cluster_sweep.py --assign),
genesearch/index/abstracts.parquet (year).
Writes data/geometry/medembed_small/manifold/:
  year.npy                    int16 (n,)  publication year per row (0 = unknown)
  mean.npy                    float32 (384,)  mean of all paper vectors (the shared "anisotropy" direction)
  <name>_sum.npy              float32 (k, 384) sum of each cluster's vectors
  <name>_size.npy             int64 (k,)
  <name>_ysum.npy             float32 (k, Y, 384) sums per cluster and year (years YEARS[0]..YEARS[-1]; earlier
                              years go into the first bin)
  <name>_ysize.npy            int64 (k, Y)
Centroid = sum / size; its length (< 1) says how tight the cluster is (1 = every paper the same).
CPU, root env: uv run python geometry/manifold/centroids.py eom_mcs200_ms3 eom_mcs50_ms3
"""
import sys, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/geometry/medembed_small"
OUT = D / "manifold"; OUT.mkdir(exist_ok=True)
YEARS = np.arange(1975, 2027)  # bin 0 also holds everything before 1975
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

X = np.load(D / "X.npy", mmap_mode="r"); pm = np.load(D / "pmids.npy"); n = len(pm)
yf = OUT / "year.npy"
if not yf.exists():
    t = pq.read_table(REPO / "genesearch/index/abstracts.parquet", columns=["pmid", "year"])
    ap, ay = t.column("pmid").to_numpy(), t.column("year").to_numpy(zero_copy_only=False)
    o = np.argsort(ap); ap, ay = ap[o], ay[o]
    i = np.searchsorted(ap, pm).clip(0, len(ap) - 1); on = ap[i] == pm
    year = np.where(on, ay[i], 0).astype(np.int16); np.save(yf, year)
    log(f"years: {on.mean():.2%} found, range {year[year > 0].min()}-{year.max()}")
year = np.load(yf)
ybin = np.clip(year - YEARS[0], 0, len(YEARS) - 1); ybin[year == 0] = -1

C = 1_000_000
names = sys.argv[1:]
labs = {nm: np.load(D / f"clusters/{nm}_all.npy").astype(np.int64) for nm in names}
ks = {nm: int(l.max()) + 1 for nm, l in labs.items()}
acc = {nm: (np.zeros((ks[nm], 384), np.float64), np.zeros((ks[nm] * len(YEARS), 384), np.float64)) for nm in names}
total = np.zeros(384, np.float64)
for s in range(0, n, C):
    blk = np.asarray(X[s:s + C], np.float32); m = len(blk)
    total += blk.sum(0, dtype=np.float64)
    for nm in names:
        lab = labs[nm][s:s + C]
        A = sp.csr_matrix((np.ones(m, np.float32), (lab, np.arange(m))), shape=(ks[nm], m))
        acc[nm][0][:] += A @ blk
        yb = ybin[s:s + C]; ok = yb >= 0
        B = sp.csr_matrix((np.ones(ok.sum(), np.float32), (lab[ok] * len(YEARS) + yb[ok], np.flatnonzero(ok))),
                          shape=(ks[nm] * len(YEARS), m))
        acc[nm][1][:] += B @ blk
    log(f"{s + m:,}/{n:,}")
np.save(OUT / "mean.npy", (total / n).astype(np.float32))
for nm in names:
    lab, k = labs[nm], ks[nm]
    np.save(OUT / f"{nm}_sum.npy", acc[nm][0].astype(np.float32))
    np.save(OUT / f"{nm}_size.npy", np.bincount(lab, minlength=k))
    np.save(OUT / f"{nm}_ysum.npy", acc[nm][1].reshape(k, len(YEARS), 384).astype(np.float32))
    ys = np.zeros(k * len(YEARS), np.int64); ok = ybin >= 0
    np.add.at(ys, lab[ok] * len(YEARS) + ybin[ok], 1)
    np.save(OUT / f"{nm}_ysize.npy", ys.reshape(k, len(YEARS)))
    cen = acc[nm][0] / np.bincount(lab, minlength=k)[:, None]
    log(f"{nm}: k={k}, centroid length median {np.median(np.linalg.norm(cen, axis=1)):.3f}")
log(f"mean vector length {np.linalg.norm(total / n):.3f}")
