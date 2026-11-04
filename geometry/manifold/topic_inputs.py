"""Evidence for naming each topic (cluster): MeSH headings most over-represented in it, its genes and field,
its median year, and the titles of the 8 papers nearest its centroid.
Writes data/geometry/medembed_small/manifold/<name>_topic_inputs.jsonl (local only: holds paper titles) and
<name>_topic_mesh.csv (cluster, top MeSH with lift; no third-party text, safe to share).
CPU, root env: uv run python geometry/manifold/topic_inputs.py eom_mcs200_ms3
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/geometry/medembed_small"
M = D / "manifold"
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
name = sys.argv[1]

X = np.load(D / "X.npy", mmap_mode="r"); pm = np.load(D / "pmids.npy"); n = len(pm)
lab = np.load(D / f"clusters/{name}_all.npy").astype(np.int64); k = lab.max() + 1
size = np.bincount(lab, minlength=k)
cen = np.load(M / f"{name}_sum.npy") / size[:, None]
year = np.load(M / "year.npy")
sim = np.empty(n, np.float32)
for s in range(0, n, 1_000_000):
    sim[s:s + 1_000_000] = np.einsum("ij,ij->i", np.asarray(X[s:s + 1_000_000], np.float32), cen[lab[s:s + 1_000_000]])
df = pd.DataFrame({"row": np.arange(n), "c": lab, "s": sim})
top = df.sort_values("s", ascending=False).groupby("c").head(8)
want = np.sort(pm[top.row.values])
t = pq.read_table(REPO / "genesearch/index/abstracts.parquet", columns=["pmid", "title"],
                  filters=[("pmid", "in", want.tolist())]).to_pandas().drop_duplicates("pmid").set_index("pmid").title
top["title"] = [t.get(int(p), "") for p in pm[top.row.values]]
log("central titles")

mp = pq.read_table(M / "mesh_papers.parquet", columns=["row", "mesh"])
rows = mp.column("row").to_numpy(); mesh = mp.column("mesh").combine_chunks()
mo, mv = mesh.offsets.to_numpy(), mesh.values.to_numpy().astype(np.int64)
F = sp.csr_matrix((np.ones(len(mv), np.float32), (np.repeat(rows, np.diff(mo)), mv)), shape=(n, mv.max() + 1))
F.sum_duplicates(); F.data[:] = 1
L = sp.csr_matrix((np.ones(n, np.float32), (lab, np.arange(n))), shape=(k, n))
CM = (L @ F).toarray()  # (k, descriptors)
has = np.bincount(lab[rows[np.diff(mo) > 0]], minlength=k)  # papers with any MeSH, per cluster
p_all = np.asarray(F.sum(0)).ravel() / len(rows)
desc = pq.read_table(M / "mesh_desc.parquet", columns=["ui", "name"]).to_pandas().set_index("ui").name
summ = pd.read_csv(D / f"clusters/{name}_summary.csv").set_index("cluster")
out, mrows = [], []
for c in range(k):
    frac = CM[c] / max(has[c], 1); lift = frac / np.maximum(p_all, 1e-9)
    ok = (CM[c] >= max(10, 0.03 * has[c]))
    idx = np.flatnonzero(ok); idx = idx[np.argsort(-(lift[idx] * np.sqrt(frac[idx])))][:12]
    ms = [(desc.get(int(u), str(u)), round(float(frac[u]), 3), round(float(lift[u]), 1)) for u in idx]
    yc = year[lab == c]; yc = yc[yc > 0]
    rec = dict(cluster=c, size=int(size[c]), genes=str(summ.genes.get(c, "")), field=str(summ.field.get(c, "")),
               median_year=int(np.median(yc)) if len(yc) else None, mesh=ms,
               titles=top[top.c == c].title.tolist())
    out.append(rec); mrows.append(dict(cluster=c, size=int(size[c]), mesh="; ".join(f"{a} ({l}x)" for a, _, l in ms[:8])))
with open(M / f"{name}_topic_inputs.jsonl", "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
pd.DataFrame(mrows).to_csv(M / f"{name}_topic_mesh.csv", index=False)
log(f"wrote {k} topics")
