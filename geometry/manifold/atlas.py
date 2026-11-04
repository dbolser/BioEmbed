"""Bundle what the Manifold explorer (explorer.py) needs into data/geometry/medembed_small/manifold/atlas/:
  papers.npz     a random 1.5M-paper sample: rows, pmid, year, topic, t-SNE x/y, vectors (fp16, raw unit vectors)
  titles.parquet their titles (local only: third-party text, never committed)
  map.npz        200k of them for the background picture: x, y, topic
  topics.parquet topic id, label, blurb, kind, size, median year, t-SNE median x/y; topics.npy centred centroids
  concepts.parquet / concepts.npy   MeSH (>= 50 papers) and GO concepts, "name: definition" vectors
  axes.json / axes.npy   every named axis: id, kind, unit (one step), names when axis naming has run
CPU, root env: uv run python geometry/manifold/atlas.py eom_mcs200_ms3
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/geometry/medembed_small"
M = D / "manifold"
OUT = M / "atlas"; OUT.mkdir(exist_ok=True)
name = sys.argv[1]
rng = np.random.default_rng(0)

X = np.load(D / "X.npy", mmap_mode="r"); pm = np.load(D / "pmids.npy"); n = len(pm)
lab = np.load(D / f"clusters/{name}_all.npy").astype(np.int32)
year = np.load(M / "year.npy"); Y = np.load(D / "sweep/tsne_p30_d2.npy")
mu = np.load(M / "mean.npy")
rows = np.sort(rng.choice(n, 1_500_000, replace=False))
np.savez(OUT / "papers.npz", rows=rows, pmid=pm[rows], year=year[rows], topic=lab[rows],
         x=Y[rows, 0].astype(np.float32), y=Y[rows, 1].astype(np.float32), vec=np.asarray(X[rows], np.float16), mean=mu)
t = pq.read_table(REPO / "genesearch/index/abstracts.parquet", columns=["pmid", "title"],
                  filters=[("pmid", "in", pm[rows].tolist())]).to_pandas().drop_duplicates("pmid").set_index("pmid").title
pd.DataFrame({"pmid": pm[rows], "title": [t.get(int(p), "") for p in pm[rows]]}).to_parquet(OUT / "titles.parquet")
mp = np.sort(rng.choice(len(rows), 200_000, replace=False))
np.savez(OUT / "map.npz", x=Y[rows[mp], 0].astype(np.float32), y=Y[rows[mp], 1].astype(np.float32),
         topic=lab[rows[mp]].astype(np.int16), idx=mp)

size = np.load(M / f"{name}_size.npy"); S = np.load(M / f"{name}_sum.npy")
cen = S / size[:, None] - mu
lb = pd.read_csv(M / f"{name}_labels.csv")
k = len(size)
tops = pd.DataFrame({"topic": np.arange(k), "size": size,
                     "x": pd.Series(Y[:, 0]).groupby(lab).median().reindex(range(k)).values,
                     "y": pd.Series(Y[:, 1]).groupby(lab).median().reindex(range(k)).values,
                     "median_year": pd.Series(year).where(year > 0).groupby(lab).median().reindex(range(k)).values})
tops = tops.merge(lb.rename(columns={"cluster": "topic"}), on="topic", how="left")
tops.to_parquet(OUT / "topics.parquet"); np.save(OUT / "topics.npy", cen.astype(np.float32))

cdf = pd.read_parquet(M / "concepts.parquet"); C = np.load(M / "concept_def.npy")
use = (((cdf.source == "mesh") & (cdf.npapers >= 50)) | (cdf.source == "go")).values
cdf[use].reset_index(drop=True)[["source", "id", "name", "npapers"]].to_parquet(OUT / "concepts.parquet")
np.save(OUT / "concepts.npy", C[use].astype(np.float32))

# axes: unit vectors in the centred 384-D space, with a step size
z = np.load(M / f"{name}_axes.npz")
Xs = np.asarray(X[rows[:200_000]], np.float32) - mu
axes, meta = [], []
def add(aid, kind, v, unit):
    v = v / np.linalg.norm(v); axes.append(v.astype(np.float32)); meta.append(dict(id=aid, kind=kind, unit=float(unit)))
for i in range(12): add(f"W{i}", "within", z["W_vecs"][:, i], (Xs @ z["W_vecs"][:, i]).std())
for i in range(12): add(f"B{i}", "between", z["B_vecs"][:, i], (Xs @ z["B_vecs"][:, i]).std())
tu = z["tvec"] / np.maximum(np.linalg.norm(z["tvec"], axis=1, keepdims=True), 1e-12)
g = tu[z["tn"] > 200].mean(0); g /= np.linalg.norm(g)
add("time", "time", g, (Xs @ g).std())
an = np.load(M / f"{name}_analogies.npz"); th = json.load(open(M / f"{name}_analogies.json"))["themes"]
E, comp = an["edges"], an["comp"]
for i, v in enumerate(an["axes"]):
    m = np.flatnonzero(comp == i)  # comp ids follow the order themes were found, before sorting by size
    vv = cen[E[m, 1]] - cen[E[m, 0]]
    add(f"A{i}", "analogy", v, np.median(np.abs(vv @ (v / np.linalg.norm(v)))))
# the 20 largest analogy themes among the 3,349 finer topics (named in <fine>_themes_named.json)
FINE = "eom_mcs50_ms3"
if (M / f"{FINE}_themes_named.json").exists():
    fz = np.load(M / f"{FINE}_analogies.npz"); fth = json.load(open(M / f"{FINE}_analogies.json"))["themes"]
    fS, fn = np.load(M / f"{FINE}_sum.npy"), np.load(M / f"{FINE}_size.npy"); fcen = fS / fn[:, None] - mu
    fnamed = {t["theme"]: t for t in json.load(open(M / f"{FINE}_themes_named.json"))["themes"]}
    for i, v in enumerate(fz["axes"][:20]):
        m = np.flatnonzero(fz["comp"] == i); vv = fcen[fz["edges"][m, 1]] - fcen[fz["edges"][m, 0]]
        add(f"F{i}", "fine_analogy", v, np.median(np.abs(vv @ (v / np.linalg.norm(v)))))
        meta[-1] |= dict(pairs=fth[i]["pairs"][:8], edges=fth[i]["edges"], fine=fnamed.get(f"F{i}", {}))
np.save(OUT / "axes.npy", np.array(axes))
named = M / f"{name}_axes_named.json"
names = {a["axis"]: a for a in json.load(open(named))["axes"]} if named.exists() else {}
for mta in meta:
    if mta["kind"] == "fine_analogy":
        f = mta.pop("fine"); x, _, y = f.get("name", "").partition("->")
        mta |= dict(name=f.get("name", ""), minus_pole=x.strip(), plus_pole=y.strip(), description=f.get("description", ""),
                    confidence=f.get("confidence", ""))
    else:
        mta |= {k_: names.get(mta["id"], {}).get(k_, "") for k_ in ("name", "minus_pole", "plus_pole", "description", "confidence")}
for i, mta in enumerate(meta):
    if mta["kind"] == "analogy": mta["edges"], mta["pairs"] = th[int(mta["id"][1:])]["edges"], th[int(mta["id"][1:])]["pairs"][:8]
# the learned dictionary (sae.py), for "ideas in this point"
if (M / "sae/sae.pt").exists():
    import shutil
    shutil.copy(M / "sae/sae.pt", OUT / "sae.pt"); shutil.copy(M / "sae/features.csv", OUT / "sae_features.csv")
(OUT / "axes.json").write_text(json.dumps(meta, indent=1))
print("atlas written:", len(rows), "papers,", k, "topics,", int(use.sum()), "concepts,", len(axes), "axes")
