"""Per-category and per-task comparison: best embedder vs best LLM.

Restricted to models with FULL suite coverage plus (footnoted) qwen3.6-35b
which lacks R2MED for context-length reasons.
"""

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
df = pd.read_csv(REPO / "results" / "bio_scores.csv")

pivot = df.pivot_table(index=["model", "model_type"], columns="task", values="score")
full = pivot.dropna().reset_index()
models = set(full["model"]) | {"qwen3.6-35b-a3b"}
sub = df[df["model"].isin(models)]

print("=== Category means (full-coverage models; qwen35b lacks R2MED) ===")
cat = sub.pivot_table(index=["model", "model_type"], columns="category",
                      values="score").round(3)
cat["Overall"] = cat.mean(axis=1).round(3)
print(cat.sort_values("Overall", ascending=False).to_string())

print("\n=== Best per task ===")
for task, grp in sub.groupby("task"):
    rows = []
    for mt in ["embedding", "llm"]:
        g = grp[grp.model_type == mt]
        if len(g):
            b = g.loc[g.score.idxmax()]
            rows.append(f"{mt}: {b.model.split('__')[-1]} {b.score:.3f}")
    print(f"{task:38s} {' | '.join(rows)}")
