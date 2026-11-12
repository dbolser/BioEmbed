"""Texts for the model check: 100 random papers per topic (seed 0) as "title\\n\\nabstract" (as MedEmbed saw them),
plus every concept's "name: definition", split into N shards for N GPU VMs.
Writes data/geometry/manifold_check/{papers_<i>.parquet, concepts_<i>.parquet, subset_rows.npy}
(local only: abstracts are third-party text, never committed).
CPU, root env: uv run python geometry/manifold/subset_texts.py eom_mcs200_ms3 [--per 100] [--shards 2]
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/geometry/medembed_small"
OUT = REPO / "data/geometry/manifold_check"; OUT.mkdir(exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--per", type=int, default=100)
ap.add_argument("--shards", type=int, default=2); a = ap.parse_args()
lab = np.load(D / f"clusters/{a.name}_all.npy"); pm = np.load(D / "pmids.npy")
rng = np.random.default_rng(0)
rows = np.sort(np.concatenate([rng.choice(np.flatnonzero(lab == g), min(a.per, (lab == g).sum()), replace=False)
                               for g in range(lab.max() + 1)]))
np.save(OUT / "subset_rows.npy", rows)
t = pq.read_table(REPO / "genesearch/index/abstracts.parquet", columns=["pmid", "title", "abstract"],
                  filters=[("pmid", "in", pm[rows].tolist())]).to_pandas().drop_duplicates("pmid").set_index("pmid")
text = [f"{t.title.get(p, '') or ''}\n\n{t.abstract.get(p, '') or ''}" for p in pm[rows]]
df = pd.DataFrame({"row": rows, "pmid": pm[rows], "text": text})
c = pd.read_parquet(D / "manifold/concepts.parquet")
cdf = pd.DataFrame({"idx": np.arange(len(c)), "text": [f"{n}: {d}" if d else n for n, d in zip(c.name, c.definition)]})
for i in range(a.shards):
    df.iloc[i::a.shards].to_parquet(OUT / f"papers_{i}.parquet"); cdf.iloc[i::a.shards].to_parquet(OUT / f"concepts_{i}.parquet")
print(f"{len(df):,} papers ({np.mean([len(x) for x in text]):.0f} chars mean), {len(cdf):,} concepts, {a.shards} shards")
