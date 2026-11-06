"""Figures for the README.
(1) manifold_analogies.jpg: the top four analogy themes, each in its own plane: x = position along the theme
    vector, y = the main remaining direction among topics. Grey dots: all 780 topics; arrows: the theme's pairs.
    Directions are not kept by 2-D maps (t-SNE keeps neighbourhoods), so the themes are drawn in 384-D projections.
(2) manifold_time.jpg: papers' within-topic coordinate on the shared time vector, by publication year (median and
    middle 50% of the 1.5M-paper sample).
Colours: dataviz reference palette (categorical slots 1-4 in fixed order; one ink colour per mark otherwise).
CPU, root env: uv run python geometry/manifold/figures.py eom_mcs200_ms3
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/geometry/medembed_small"
M = D / "manifold"; A = M / "atlas"
FIG = REPO / "geometry/figures"
name = sys.argv[1]
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED})
tops = pd.read_parquet(A / "topics.parquet"); cen = np.load(A / "topics.npy").astype(np.float64)
P = cen - cen.mean(0)
th = json.load(open(M / f"{name}_analogies.json"))["themes"]
an = np.load(M / f"{name}_analogies.npz"); E, comp = an["edges"], an["comp"]
named = {a["axis"]: a for a in json.load(open(M / f"{name}_axes_named.json"))["axes"]}

fig, axs = plt.subplots(2, 2, figsize=(13, 11), facecolor=SURF)
for i, ax in enumerate(axs.ravel()):
    m = np.flatnonzero(comp == i); a = an["axes"][i] / np.linalg.norm(an["axes"][i])
    v = P[E[m, 1]] - P[E[m, 0]]; s = np.sign(v @ a); fr = np.where(s > 0, E[m, 0], E[m, 1]); to = np.where(s > 0, E[m, 1], E[m, 0])
    R = P - np.outer(P @ a, a); b = np.linalg.svd(R - R.mean(0), full_matrices=False)[2][0]
    x, y = P @ a, P @ b
    ax.set_facecolor(SURF); ax.scatter(x, y, s=6, c="#c3c2b7", alpha=0.6, linewidths=0)
    for f, t in zip(fr, to):
        ax.annotate("", xy=(x[t], y[t]), xytext=(x[f], y[f]),
                    arrowprops=dict(arrowstyle="-|>", color=CAT[i], lw=2, mutation_scale=11, shrinkA=2, shrinkB=2))
    # name three well-separated pairs
    for j in np.argsort(-(y[fr]))[:: max(1, len(fr) // 3)][:3]:
        ax.annotate(tops.label[to[j]], (x[to[j]], y[to[j]]), xytext=(4, 3), textcoords="offset points", fontsize=8, color=INK2)
        ax.annotate(tops.label[fr[j]], (x[fr[j]], y[fr[j]]), xytext=(-4, -9), textcoords="offset points", fontsize=8,
                    color=MUTED, ha="right")
    nm = named[f"A{i}"]
    ax.set_title(f"A{i}: {nm['name']}  ({len(m)} pairs, {th[i]['regions']} regions)", color=INK, fontsize=11, loc="left")
    ax.set_xlabel(f"along the theme vector  (− {nm['minus_pole']}  →  + {nm['plus_pole']})", fontsize=9)
    ax.set_ylabel("main other direction among topics", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.suptitle("Analogy themes: the same move between topics, starting from different places (grey: all 780 topics)",
             color=INK, fontsize=13, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(FIG / "manifold_analogies.jpg", dpi=110, facecolor=SURF); plt.close(fig)

Pp = np.load(A / "papers.npz"); AV = np.load(A / "axes.npy"); AX = json.load(open(A / "axes.json"))
i = [a["id"] for a in AX].index("time")
vec = Pp["vec"].astype(np.float32) - Pp["mean"]
z = ((vec - np.load(A / "topics.npy")[Pp["topic"]]) @ AV[i]); z = (z - z.mean()) / z.std()
df = pd.DataFrame({"year": Pp["year"], "z": z}); df = df[(df.year >= 1975) & (df.year <= 2026)]
g = df.groupby("year").z.quantile([0.25, 0.5, 0.75]).unstack()
r = np.corrcoef(df.year, df.z)[0, 1]
fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURF); ax.set_facecolor(SURF)
ax.fill_between(g.index, g[0.25], g[0.75], color=CAT[0], alpha=0.18, linewidth=0, label="middle 50% of papers")
ax.plot(g.index, g[0.5], color=CAT[0], lw=2, label="median")
ax.axhline(0, color=GRID, lw=1); ax.grid(axis="y", color=GRID, lw=0.6)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.set_xlabel("publication year"); ax.set_ylabel("position on the time vector, within topic (sd)")
ax.set_title(f"Inside topics, papers move along one shared time vector year by year (r = {r:.2f}, 1.5M papers)",
             color=INK, fontsize=12, loc="left")
ax.text(1976, g[0.75].max() * 0.95, "+ evidence synthesis, ML, outcomes", color=INK2, fontsize=9, va="top")
ax.text(1976, g[0.25].min() * 0.95, "− serum immunochemistry, chromatography", color=INK2, fontsize=9, va="bottom")
ax.legend(frameon=False, loc="lower right", labelcolor=INK2)
fig.tight_layout(); fig.savefig(FIG / "manifold_time.jpg", dpi=110, facecolor=SURF); plt.close(fig)
print("figures written")
