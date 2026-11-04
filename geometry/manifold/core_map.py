"""A map of core biological ideas: the 72 GO slim (generic) biological processes placed by their own text on two
named axes, B0 (whole-body physiology <-> cancer and cell biology) and the shared time vector (older <-> newer
style of research). Coordinates are sd against all concepts (concept_axes.py). Also writes a table of every GO slim
term's coordinates on the main axes.
CPU, root env: uv run --with adjustText python geometry/manifold/core_map.py eom_mcs200_ms3
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text

REPO = Path(__file__).resolve().parents[2]
M = REPO / "data/geometry/medembed_small/manifold"
name = sys.argv[1]
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
c = pd.read_parquet(M / f"{name}_concept_coords.parquet")
cdf = pd.read_parquet(M / "concepts.parquet")
c["slim"], c["ns"] = cdf.slim.values, cdf.tag.values
core = c[c.slim].copy()
named = {a["axis"]: a for a in json.load(open(M / f"{name}_axes_named.json"))["axes"]}
cols = ["time", "W0", "W1", "W2", "W3", "B0", "B1", "B2"]
core[["name", "ns"] + cols].round(2).sort_values("time").to_csv(M / f"{name}_core_concepts.csv", index=False)
bp = core[core.ns == "biological_process"]
fig, ax = plt.subplots(figsize=(12, 10), facecolor=SURF); ax.set_facecolor(SURF)
ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
ax.scatter(bp.B0, bp.time, s=28, color="#2a78d6", edgecolor=SURF, linewidth=1.5, zorder=3)
texts = [ax.text(x, y, n, fontsize=8.5, color=INK2) for x, y, n in zip(bp.B0, bp.time, bp.name)]
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
b0, t = named["B0"], named["time"]
ax.set_xlabel(f"B0:  − {b0['minus_pole']}   ↔   + {b0['plus_pole']}   (sd among concepts)", color=INK2)
ax.set_ylabel(f"time:  − {t['minus_pole']}   ↔   + {t['plus_pole']}   (sd among concepts)", color=INK2)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.tick_params(colors=MUTED)
ax.set_title("Core biological ideas (GO slim processes) placed by their own text in the paper space",
             color=INK, fontsize=13, loc="left")
fig.tight_layout(); fig.savefig(REPO / "geometry/figures/manifold_core_ideas.jpg", dpi=110, facecolor=SURF)
r = bp[["time", "B0"]]
print(bp.sort_values("time")[["name", "time", "B0"]].head(6).to_string(index=False))
print(bp.sort_values("time")[["name", "time", "B0"]].tail(6).to_string(index=False))
