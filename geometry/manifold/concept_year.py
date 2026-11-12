"""Does a concept's own text say WHEN it was studied? MeSH concepts (>= 50 papers) placed by "name: definition"
(concept_axes.py coordinates, sd among concepts) against:
  - the median publication year of the papers indexed with them (mesh_papers.parquet)
  - the year NLM introduced the heading (desc2026 DateIntroduced; parse_mesh.py)
Reports Spearman for the time axis and for every other named axis (W, B, A), and a ridge regression on the full
384-D centred text vectors (5-fold cross-validated, alpha chosen inside each fold) as the ceiling for a linear read-out.
CPU, root env: uv run python geometry/manifold/concept_year.py eom_mcs200_ms3
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict

M = Path(__file__).resolve().parents[2] / "data/geometry/medembed_small/manifold"
name = sys.argv[1]
year = np.load(M / "year.npy")
t = pq.read_table(M / "mesh_papers.parquet", columns=["row", "mesh"])
rows = t.column("row").to_numpy(); mesh = t.column("mesh").combine_chunks()
df = pd.DataFrame({"ui": mesh.values.to_numpy(), "y": np.repeat(year[rows], np.diff(mesh.offsets.to_numpy()))})
med = df[df.y > 0].groupby("ui").y.median()
md = pd.read_parquet(M / "mesh_desc.parquet").set_index("ui")
c = pd.read_parquet(M / f"{name}_concept_coords.parquet")
m = c[(c.source == "mesh") & (c.npapers >= 50)].copy()
m["ui"] = m.id.str[1:].astype(int)  # integers: NLM ids are D + 6 digits (older) or D + 9 digits (newer)
m["medyear"] = m.ui.map(med); m["introduced"] = m.ui.map(md.introduced)
axes = [k for k in c.columns if k == "time" or k[:1] in "WBA" and k[1:].isdigit()]
out = {"concepts": int(len(m))}
for target in ("medyear", "introduced"):
    s = m[m[target] > 0]
    r = {k: float(spearmanr(s[k], s[target]).statistic) for k in axes}
    best = sorted(((abs(v), k) for k, v in r.items() if k != "time"), reverse=True)[:3]
    C = np.load(M / "concept_def.npy").astype(np.float64); C = C - C.mean(0)
    pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), C[s.index.values], s[target].values, cv=5)
    ridge = float(spearmanr(pred, s[target]).statistic)
    out[target] = dict(n=int(len(s)), time=r["time"], next_best=[(k, round(r[k], 3)) for _, k in best], ridge_cv=ridge)
    print(f"{target}: n {len(s):,}; time axis Spearman {r['time']:+.3f}; next best single axes "
          f"{[(k, round(r[k], 3)) for _, k in best]}; ridge on all 384-D (5-fold CV) {ridge:+.3f}")
s = m[m.introduced > 0].assign(dec=lambda d: (d.introduced // 10) * 10)
dec = s.groupby("dec").time.agg(["count", "mean"]).round(2)
out["introduced_by_decade"] = {int(k): dict(n=int(v["count"]), mean_time=float(v["mean"])) for k, v in dec.iterrows()}
print(dec.to_string())
(M / f"{name}_concept_year.json").write_text(json.dumps(out, indent=1))
