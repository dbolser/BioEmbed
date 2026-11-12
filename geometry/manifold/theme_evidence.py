"""Evidence for naming the analogy themes of the finer topics (analogies.py on eom_mcs50_ms3).

Per theme F<i>: edges, regions, coherence; its topic pairs from - to + (each topic named by its two top MeSH
headings, with its six top headings in brackets, from topic_inputs.py); and the held-out shifts from
analogy_check.py (MeSH branches that rise when the theme vector is applied forward / backward to topics that did not
define it: [branch, change in share, z against real between-topic steps, z against random directions]).
MeSH only: no titles or abstracts.
CPU, root env: uv run python geometry/manifold/theme_evidence.py eom_mcs50_ms3   (after analogy_check.py --themes N)
Writes data/geometry/medembed_small/manifold/<name>_theme_evidence.json
"""
import json, sys
from pathlib import Path

M = Path(__file__).resolve().parents[2] / "data/geometry/medembed_small/manifold"
name = sys.argv[1]
th = json.load(open(M / f"{name}_analogies.json"))["themes"]
ck = {c["theme"]: c for c in json.load(open(M / f"{name}_analogy_check.json"))}
blurb = {}
for line in open(M / f"{name}_topic_inputs.jsonl"):
    r = json.loads(line); blurb[" / ".join(m[0] for m in r["mesh"][:2]) or r["genes"]] = "; ".join(m[0] for m in r["mesh"][:6])
out = {}
for i, t in enumerate(th):
    if i not in ck: continue
    out[f"F{i}"] = dict(edges=t["edges"], regions=t["regions"], coherence=t["coherence"],
                        pairs_minus_to_plus=[[f"{a} [{blurb.get(a, '')}]", f"{b} [{blurb.get(b, '')}]"] for a, b in t["pairs"][:14]],
                        held_out_forward_up=ck[i]["forward"]["up"][:5], held_out_backward_up=ck[i]["backward"]["up"][:5])
json.dump(out, open(M / f"{name}_theme_evidence.json", "w"), indent=1)
print(f"{len(out)} themes")
