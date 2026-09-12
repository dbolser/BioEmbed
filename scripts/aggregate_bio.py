"""Aggregate BioMTEB(LLM) scores from all sources into results/bio_scores.csv.

Sources:
  1. vendor/embedders-dilemma/data/scores.csv — paper's 36 models on the
     Tier A tasks (lifted).
  2. results/embedding/**/<task>.json — our embedding runs (Tier A for extra
     models; Tier B/C for everyone, when present).
  3. results/llm/**/<task>.json — our LLM runs (Tier B/C, when present).

Output: one row per (model, model_type, task, category, score, source).
Where a model appears in both the paper's data and ours (validation overlap),
our run wins only for tasks the paper lacks; overlap rows keep source=paper.
"""

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
VENDOR = REPO / "vendor" / "embedders-dilemma"

# task -> category for the bio suite (Tier A lifted names + our new tasks)
SUITE = {
    "LLMBIOSSES": "STS",
    "LLMSTSBenchmark": "STS",
    "LLMBiorxivClusteringP2PV2": "Clustering",
    "LLMMedrxivClusteringP2PV2": "Clustering",
    "LLMMedrxivClusteringS2SV2": "Clustering",
    "LLMToxicConversationsClassification": "Classification",
    "LLMPublicHealthQA": "Retrieval",
    # Tier B/C (result files appear as runs complete)
    "BioMeSHClassification": "Classification",
    "PubChemSynonymPC500": "PairClassification",
    "R2MEDBiologyPooled": "Retrieval",
    "GOPubMedRetrieval": "Retrieval",
    "GOProteinPairCls": "PairClassification",
}


def paper_rows() -> pd.DataFrame:
    df = pd.read_csv(VENDOR / "data" / "scores.csv")
    df = df[df["task"].isin(SUITE)].copy()
    df["category"] = df["task"].map(SUITE)
    df["source"] = "paper"
    return df[["model", "model_type", "task", "category", "score", "source"]]


def our_rows(kind: str) -> pd.DataFrame:
    rows = []
    root = REPO / "results" / kind
    if not root.exists():
        return pd.DataFrame(columns=["model", "model_type", "task", "category", "score", "source"])
    for f in root.rglob("*.json"):
        if f.name in ("model_meta.json",):
            continue
        task = f.stem
        if task not in SUITE:
            continue
        d = json.loads(f.read_text())
        try:
            s = d["scores"]["test"][0]
            # Retrieval: recall@1 on both paradigms (the paper's common
            # metric — LLM results' main_score is already recall@1; the
            # embedding-side mteb main_score is nDCG@10 and must not be mixed)
            if SUITE[task] == "Retrieval":
                score = s.get("recall_at_1", s.get("recall@1"))
            else:
                score = s["main_score"]
            if score is None:
                continue
        except (KeyError, IndexError):
            continue
        model = f.relative_to(root).parts[0]  # mteb layout: <model>/<rev>/<task>.json
        rows.append({
            "model": model,
            "model_type": "embedding" if kind == "embedding" else "llm",
            "task": task,
            "category": SUITE[task],
            "score": score,
            "source": "ours",
        })
    return pd.DataFrame(rows)


def main() -> None:
    paper = paper_rows()
    ours = pd.concat([our_rows("embedding"), our_rows("llm")], ignore_index=True)

    # paper rows win on overlap (they are the published reference)
    key = ["model", "task"]
    ours = ours[~ours.set_index(key).index.isin(paper.set_index(key).index)]
    df = pd.concat([paper, ours], ignore_index=True).sort_values(["model_type", "model", "task"])

    out = REPO / "results" / "bio_scores.csv"
    df.to_csv(out, index=False)

    # summary: BioScore = macro mean over available suite tasks
    pivot = df.pivot_table(index=["model", "model_type"], columns="task", values="score")
    n_tasks = pivot.notna().sum(axis=1)
    summary = pd.DataFrame({
        "bio_score": pivot.mean(axis=1),
        "n_tasks": n_tasks,
    }).reset_index().sort_values("bio_score", ascending=False)
    summary.to_csv(REPO / "results" / "bio_summary.csv", index=False)

    full = summary[summary["n_tasks"] == len(SUITE)]
    print(f"{len(df)} rows, {df['model'].nunique()} models; "
          f"{len(full)} models cover the full {len(SUITE)}-task suite")
    print(summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
