#!/usr/bin/env python
"""Score GeneDiseaseHaystack runs and the PubTator lookup baselines.

Ranked runs (<haystack>/runs/*.parquet: qid, pmid, rank) get recall@k and
nDCG@10 over the curated gold, raw and 'condensed' (silver-not-gold PMIDs,
i.e. Europe PMC text-mined links absent from the gold, are dropped from the
run before scoring so incomplete gold does not penalise them).

Lookup baselines are unranked sets: scored by recall of the whole set and
its median size per query.

Run:  uv run python scripts/score_haystack.py
Output: results/haystack/scores.csv
"""

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HAYSTACK = ROOT / "data" / "haystack" / "GeneDiseaseHaystack"
OUT = ROOT / "results" / "haystack" / "scores.csv"
KS = [10, 100, 1000]
TIERS = ["exact", "descendants", "parents", "ancestors"]


def ranked_metrics(run: pd.DataFrame, gold: dict[str, set[str]]) -> dict[str, float]:
    ranked = {q: g.sort_values("rank").pmid.tolist() for q, g in run.groupby("qid")}
    out = {}
    for k in KS:
        out[f"recall@{k}"] = sum(
            len(set(ranked.get(q, [])[:k]) & g) / len(g) for q, g in gold.items()
        ) / len(gold)
    ndcg = 0.0
    for q, g in gold.items():
        dcg = sum(1 / math.log2(i + 2) for i, p in enumerate(ranked.get(q, [])[:10]) if p in g)
        ndcg += dcg / sum(1 / math.log2(i + 2) for i in range(min(10, len(g))))
    out["ndcg@10"] = ndcg / len(gold)
    return out


def main() -> None:
    qrels = pd.read_parquet(HAYSTACK / "qrels.parquet")
    silver = pd.read_parquet(HAYSTACK / "silver.parquet")
    gold = qrels.groupby("qid").pmid.agg(set).to_dict()
    unlabelled = silver.merge(qrels, how="left", indicator=True).query("_merge == 'left_only'")
    drop = set(zip(unlabelled.qid, unlabelled.pmid))

    rows = []
    for f in sorted((HAYSTACK / "runs").glob("*.parquet")):
        model, track = f.stem.rsplit("__", 1)
        run = pd.read_parquet(f)
        keep = [(q, p) not in drop for q, p in zip(run.qid, run.pmid)]
        condensed = run[keep].copy()
        condensed["rank"] = condensed.groupby("qid")["rank"].rank(method="first")
        rows.append({"system": model, "track": track, "variant": "raw", **ranked_metrics(run, gold)})
        rows.append({"system": model, "track": track, "variant": "condensed",
                     **ranked_metrics(condensed, gold)})

    lookup = pd.read_parquet(HAYSTACK / "lookup.parquet")
    gene = pd.read_parquet(HAYSTACK / "gene_pmids.parquet")
    sets = {"pubtator_gene": gene}
    for i, t in enumerate(TIERS):
        sets[f"pubtator_gene+disease<={t}"] = lookup[lookup.tier.isin(TIERS[: i + 1])]
    for name, s in sets.items():
        found = s.groupby("qid").pmid.agg(set).to_dict()
        rows.append({
            "system": name, "track": "lookup", "variant": "set",
            "recall@set": sum(len(found.get(q, set()) & g) / len(g) for q, g in gold.items()) / len(gold),
            "median_set": s.groupby("qid").size().reindex(list(gold), fill_value=0).median(),
        })

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(OUT, index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
