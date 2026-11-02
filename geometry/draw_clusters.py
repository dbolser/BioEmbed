"""Describe and draw a full clustering from cluster_sweep.py --assign on the t-SNE map of tsne_full.py.

For each cluster: size, share HDBSCAN clustered itself (core), its 3 most typical genes (most papers, among the
genes at least 5x more common in the cluster than overall) and its most common journal field.
Writes clusters/<name>_summary.csv and clusters/<name>.png (each cluster a colour, the 40 biggest labelled at
their median position). CPU: uv run python geometry/draw_clusters.py eom_mcs200_ms3
"""
import sys
from pathlib import Path
import json
import numpy as np, pandas as pd, pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
D = REPO / "data/geometry/medembed_small"
E, C = D / "eval", D / "clusters"
name = sys.argv[1]
lab, core = np.load(C / f"{name}_all.npy").astype(np.int64), np.load(C / f"{name}_core.npy")
Y = np.load(D / "sweep/tsne_p30_d2.npy")
field, ptr, genes = np.load(E / "field.npy"), np.load(E / "genes_ptr.npy"), np.load(E / "genes.npy")
g = pq.read_table(REPO / "genesearch/index/genes.parquet", columns=["entrez_id", "symbol"]).to_pandas()
sym = dict(zip(g.entrez_id, g.symbol))
fname = {f["id"]: f["name"] for f in json.loads((REPO / "genesearch/index/landscape/meta.json").read_text())["fields"]}
k = lab.max() + 1

# gene x cluster counts from the CSR
row_of = np.repeat(np.arange(len(lab)), np.diff(ptr))
gu, gi = np.unique(genes, return_inverse=True)
pair = pd.DataFrame({"c": lab[row_of], "g": gi}).value_counts().rename("n").reset_index()
size = np.bincount(lab, minlength=k)
overall = np.bincount(gi, minlength=len(gu)) / len(lab)
pair["lift"] = pair.n / size[pair.c] / overall[pair.g]
top = (pair[pair.lift >= 5].sort_values(["c", "n"], ascending=[True, False]).groupby("c").head(3)
       .assign(sym=lambda d: [sym.get(int(gu[x]), str(gu[x])) for x in d.g]).groupby("c").sym.agg(" ".join))
fl = pd.DataFrame({"c": lab, "f": field})
fl = fl[fl.f > 0].value_counts().rename("n").reset_index().sort_values("n", ascending=False).drop_duplicates("c")
summ = pd.DataFrame({"cluster": np.arange(k), "size": size, "core": np.bincount(lab, core, k) / size,
                     "x": pd.Series(Y[:, 0]).groupby(lab).median().values,
                     "y": pd.Series(Y[:, 1]).groupby(lab).median().values})
summ["genes"] = summ.cluster.map(top).fillna("")
summ["field"] = summ.cluster.map(dict(zip(fl.c, fl.f.map(fname)))).fillna("")
summ.sort_values("size", ascending=False).to_csv(C / f"{name}_summary.csv", index=False)
print(summ.sort_values("size", ascending=False).head(15).to_string(index=False))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
col = plt.get_cmap("hsv")(rng.random(k)) * [0.85, 0.85, 0.85, 1]
i = rng.choice(len(lab), 1_500_000, replace=False)
fig, ax = plt.subplots(figsize=(14, 14), dpi=110)
ax.scatter(*Y[i].T, s=0.15, c=col[lab[i]], lw=0, rasterized=True)
lo, hi = np.percentile(Y[i], [0.2, 99.8], axis=0); ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
for _, r in summ.sort_values("size", ascending=False).head(40).iterrows():
    ax.text(r.x, r.y, r.genes or r.field, fontsize=7, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
ax.set_axis_off()
ax.set_title(f"{k:,} clusters ({name}, 12-D UMAP) on the t-SNE map of 7.76M human-gene papers; "
             "the 40 biggest labelled by their typical genes", fontsize=10)
fig.savefig(C / f"{name}.png", bbox_inches="tight")
