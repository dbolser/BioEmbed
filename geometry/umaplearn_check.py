"""Is the gallery's UMAP result a cuML quirk? Re-run four settings with umap-learn (the reference CPU UMAP) on
the 1% subset and score them exactly as gallery.py does.
Writes data/geometry/medembed_small/gallery/0.01_umaplearn_k<k>_md<md>.npy and prints the scores.
CPU (~7 min): uv run --with umap-learn python geometry/umaplearn_check.py
"""
import time
from pathlib import Path
import numpy as np, umap
from gallery import Score

REPO = Path(__file__).resolve().parent.parent
D = REPO / "data/geometry/medembed_small"
X = np.load(D / "X.npy", mmap_mode="r"); e = D / "eval"
field, ptr, genes = np.load(e / "field.npy"), np.load(e / "genes_ptr.npy"), np.load(e / "genes.npy")
u = np.random.default_rng(1).random(len(X)).astype(np.float32)
rows = np.flatnonzero(u < 0.01); Xs = np.asarray(X[rows], np.float32)
score = Score(Xs, rows, field, ptr, genes)
for k, md in [(5, 0.0), (15, 0.0), (15, 0.1), (50, 0.0)]:
    t = time.time()
    Y = np.asarray(umap.UMAP(n_neighbors=k, min_dist=md, metric="cosine", random_state=0).fit_transform(Xs), np.float32)
    np.save(D / f"gallery/0.01_umaplearn_k{k}_md{md:g}.npy", Y)
    print(k, md, round(time.time() - t), {a: round(b, 3) for a, b in score(Y).items()}, flush=True)
