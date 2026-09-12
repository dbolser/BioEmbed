# BioEmbed: The Embedder's Dilemma, for Biology

Recreating the cost-aware LLM-vs-embedding-model comparison of
[*The Embedder's Dilemma: LLMs Are Better, but at What Cost?*](https://arxiv.org/abs/2608.12875)
(El Assadi, Muennighoff, Lee — COLM 2026) on **biological text**.

Question: what does the quality-vs-USD Pareto frontier look like for
biology tasks — open embedders (22M–8B), bio-specialised embedders,
and LLMs prompted directly?

## The BioMTEB(LLM) suite (12 tasks, 5 categories)

| Tier | Tasks | LLM results |
|---|---|---|
| A — lifted from the paper | BIOSSES, STSBenchmark, Biorxiv/Medrxiv clustering ×3, ToxicConversations, PublicHealthQA | published (free) |
| B — new, from public sources | BioMeSHClassification (30-class), PubChemSynonymPC500, R2MEDBiologyPooled | to run |
| C — new, human-curated GAF gold | GOPubMedRetrieval, GOProteinPairCls | to run |

Both paradigms see identical seed-42 data. Tier A uses the paper's public
`mteb/llm-eval-*` subsets; Tier B/C datasets are in `data/tasks/` (committed,
built by `scripts/build_tier_*.py`, adversarially verified).

## Layout

- `eval/` — embedding-side runs, pinned `mteb==2.11.6` (the version behind
  the paper's embedding results; harness validated by reproducing their
  multilingual-e5-small numbers exactly).
- `llm/` — LLM-side runs via the paper's `llm_judge` framework (OpenRouter,
  needs `llm/.env` with a key).
- `scripts/` — dataset builders, own-tokenizer token counting
  (`count_tokens_bio.py`), aggregation (`aggregate_bio.py`), Pareto plot
  (`pareto_bio.py`).
- `vendor/embedders-dilemma` — the paper's public repo (clone; gitignored).
- `results/` — our result JSONs, merged CSVs, figures.

## Reproduce

```bash
uv sync                                   # analysis env
(cd eval && uv sync && uv run python run_embedders.py --suite all)
uv run python scripts/count_tokens_bio.py
uv run python scripts/aggregate_bio.py
uv run python scripts/pareto_bio.py       # results/figures/bio_pareto.png
```

Cost accounting follows the paper: LLMs at API token prices (cached input
at 10%, generated incl. thinking at output rate); embedders at
own-tokenizer token counts × H100-rental $/MTok (paper's measured
throughput; params-nearest proxy, flagged, for models it didn't measure).

Status: embedding sweep in progress; LLM side wired, awaiting API key.
