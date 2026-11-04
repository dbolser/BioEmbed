"""Concept coordinates: where MeSH and GO concepts sit on the named axes, and which concepts name each axis.

Concept vectors: "name: definition" embedded with MedEmbed-small (concepts.py), centred by the concept mean
(concepts.py: own-centroid top-1 60% / top-10 88% among 22,142 MeSH paper centroids). Axes: the common
within-topic axes W0.., the between-topic axes B0.. (axes.py), the shared time vector, and the analogy themes A0..
(analogies.py). Each concept's coordinate on an axis = its centred vector . axis, in standard deviations over
concepts. Writes data/geometry/medembed_small/manifold/<name>_concept_coords.parquet (one column per axis) and
<name>_axis_concepts.json (the 12 concepts at each end of every axis), and prints them.
CPU, root env: uv run python geometry/manifold/concept_axes.py eom_mcs200_ms3
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/geometry/medembed_small"
M = D / "manifold"


def axes_of(name, nW=12, nB=12):
    z = np.load(M / f"{name}_axes.npz")
    ax = {f"W{i}": z["W_vecs"][:, i] for i in range(nW)} | {f"B{i}": z["B_vecs"][:, i] for i in range(nB)}
    tu = z["tvec"] / np.maximum(np.linalg.norm(z["tvec"], axis=1, keepdims=True), 1e-12)
    g = tu[z["tn"] > 200].mean(0); ax["time"] = g / np.linalg.norm(g)
    a = np.load(M / f"{name}_analogies.npz")["axes"]
    for i, v in enumerate(a[:22]): ax[f"A{i}"] = v / np.linalg.norm(v)
    # orient the W and B axes so that + is the end the MeSH labels call "+" (label_axes.py used the same vectors)
    return ax


def main(name):
    cdf = pd.read_parquet(M / "concepts.parquet")
    C = np.load(M / "concept_def.npy").astype(np.float64); C -= C.mean(0)
    ax = axes_of(name)
    coords = pd.DataFrame({"source": cdf.source, "id": cdf.id, "name": cdf.name, "npapers": cdf.npapers})
    # keep concepts that are about something in this corpus: MeSH with >= 50 papers, or any GO term
    use = ((cdf.source == "mesh") & (cdf.npapers >= 50)) | (cdf.source == "go")
    out = {}
    for k, v in ax.items():
        s = C @ v; s = (s - s[use].mean()) / s[use].std(); coords[k] = s.astype(np.float32)
        o = np.argsort(s); o = o[use.values[o]]
        out[k] = dict(plus=[f"{cdf.name[i]} ({cdf.source[i]})" for i in o[::-1][:12]],
                      minus=[f"{cdf.name[i]} ({cdf.source[i]})" for i in o[:12]])
        print(f"\n{k}\n  + " + "; ".join(out[k]["plus"][:8]) + "\n  - " + "; ".join(out[k]["minus"][:8]))
    coords.to_parquet(M / f"{name}_concept_coords.parquet")
    (M / f"{name}_axis_concepts.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
