"""Automatic topic names from MeSH alone (no LLM): each topic's two headings with the highest lift x sqrt(share)
(topic_inputs.py's ranking), e.g. "Celiac Disease / Gliadin". Used for the 3,349 fine topics.
Writes data/geometry/medembed_small/manifold/<name>_labels.csv (cluster, label, blurb, kind, mixed).
"""
import json, sys
from pathlib import Path
import pandas as pd

M = Path(__file__).resolve().parents[2] / "data/geometry/medembed_small/manifold"
name = sys.argv[1]
rows = []
for line in open(M / f"{name}_topic_inputs.jsonl"):
    r = json.loads(line); ms = [m[0] for m in r["mesh"]]
    rows.append(dict(cluster=r["cluster"], label=" / ".join(ms[:2]) or r["genes"], blurb="; ".join(ms[:6]),
                     kind="", mixed=False))
pd.DataFrame(rows).to_csv(M / f"{name}_labels.csv", index=False)
print(len(rows), "labels")
