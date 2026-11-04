"""Collect the evidence about every named axis into one JSON file for naming (no paper text, only MeSH/GO names
and our topic labels): MeSH headings and publication types at each end (label_axes.py), concepts at each end
(concept_axes.py), correlation with year, and for analogy themes their topic pairs and the MeSH branches that
move when the theme vector is applied to other topics (analogy_check.py).
Writes data/geometry/medembed_small/manifold/<name>_axis_evidence.json
"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
M = REPO / "data/geometry/medembed_small/manifold"
name = sys.argv[1]
lab = json.load(open(M / f"{name}_axis_labels.json"))
con = json.load(open(M / f"{name}_axis_concepts.json"))
ana = json.load(open(M / f"{name}_analogies.json"))["themes"]
chk = {c["theme"]: c for c in json.load(open(M / f"{name}_analogy_check.json"))}
out = {}
for k in con:
    e = {"axis": k, "kind": {"W": "within-topic (papers vary along it inside most topics)",
                             "B": "between-topic (where topics sit relative to each other)",
                             "t": "the shared time vector (older -> newer, within topics)",
                             "A": "analogy theme (a between-topic vector that recurs across the map)"}[k[0]],
         "concepts_plus": con[k]["plus"][:10], "concepts_minus": con[k]["minus"][:10]}
    if k in lab:
        e["mesh_plus"] = [f"{a} ({b:+.2f} sd)" for a, b, _ in lab[k]["plus"][:12]]
        e["mesh_minus"] = [f"{a} ({b:+.2f} sd)" for a, b, _ in lab[k]["minus"][:12]]
        e["year_corr"] = lab[k]["year_corr"]
    if k[0] == "A":
        i = int(k[1:]); t = ana[i]
        e |= {"edges": t["edges"], "regions": t["regions"], "pairs_minus_to_plus": t["pairs"][:12]}
        if i in chk:
            e["held_out_forward_up"] = chk[i]["forward"]["up"][:5]; e["held_out_backward_up"] = chk[i]["backward"]["up"][:5]
        e["note"] = ("pairs go from the - end topic to the + end topic. NB the concept poles were computed with the "
                     "axis sign as stored; if they contradict the pairs, trust the pairs and held-out shifts.")
    out[k] = e
(M / f"{name}_axis_evidence.json").write_text(json.dumps(out, indent=1))
print(len(out), "axes")
