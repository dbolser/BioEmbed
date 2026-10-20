# Cost vs quality: how the numbers were made

Files: `results/writeup/cost_quality.csv` (one row per system) and
`docs/writeup_frontier.png`. Both are built by `scripts/cost_quality.py`.

**Exploratory.** None of these cost numbers was pre-registered or tested.
The confirmatory results are only the claims in `results/claims.md`.

## Quality columns

| column | what it is |
|---|---|
| `graded_ndcg10` | Graded nDCG@10: a 0-1 score for how good the top 10 papers are. Uses the 0/1/2 grades of two LLM judges (Gemini, DeepSeek), merged. From `results/haystack/graded/stats.csv` (variant `pooled`). 493 queries: those with at least one relevant paper. |
| `graded_ci_low`, `graded_ci_high` | 95% range from resampling queries (bootstrap). Blank for 3-seed means. |
| `seed_min`, `seed_max` | Lowest and highest of the 3 seeds, for our 3-seed fine-tunes. |
| `curator_ndcg10` | nDCG@10 counting only papers Open Targets curators linked (variant `raw`). |
| `graded_ndcg10_other_judge` | LLM rerankers only. The reranking LLM is also one of the two judges, so we re-score with only the other judge's grades (`stats_haystack.py --judgments judgments_merged_no-<judge>.parquet`). |

Almost every system scores lower with no-Gemini grades (all but the two GO
fine-tunes) and every one scores higher with no-DeepSeek grades. So compare an "other judge" score only with other systems under the
same grades:

| system | both judges | no-Gemini grades | no-DeepSeek grades |
|---|---|---|---|
| BM25, tuned | 0.759 | 0.737 | 0.788 |
| First pass (BM25 + bge-base) | 0.750 | 0.731 | 0.775 |
| First pass + fine-tuned MedCPT cross-encoder (ours) | 0.861 | 0.837 | 0.876 |
| First pass + Qwen3-Reranker-4B | 0.827 | 0.812 | 0.839 |
| First pass + Gemini rerank | 0.853 | **0.834** | 0.858 |
| First pass + DeepSeek rerank | 0.841 | 0.826 | **0.843** |
| BM25 + Gemini rerank | 0.844 | **0.825** | 0.852 |
| BM25 + DeepSeek rerank | 0.832 | 0.816 | **0.840** |

Bold = the fair score for that LLM (its own grades left out). The figure uses
"both judges" for every point.

## Cost columns

`run_cost_usd` = `first_pass_cost_usd` + `step_cost_usd`: the cost to run
all 500 test queries once.

- **One GPU price for everything: $0.94 per L4-hour.** That is a
  g2-standard-8 VM with one L4, on-demand, in europe-west1, where the
  embedders ran. The fine-tune and rerank VMs ran in europe-west3 ($1.006,
  7% more). One price means the same work costs the same.
- **GPU time = whole job time** in `data/gcp_logs/<vm>/runner.log`, from the
  start of the last try to its exit. This covers loading the model,
  embedding the 1,699,080 candidate abstracts once, embedding the 500
  queries, and ranking. Not counted: VM start-up, setup, syncing and failed
  tries (a few minutes per VM).
- **10 test runs have no log** (made 2026-10-15 on Spot VMs). We estimate
  them:
  - bge-base, MedEmbed-base, bge-base-GO: the same model shape as our
    bge-base fine-tunes, so we use those logged test jobs (mean 1.77 h).
  - bge-small, MedEmbed-small, bge-small-GO: likewise from our bge-small
    fine-tunes (0.96 h).
  - MiniLM, e5-small, PubMedBERT, Qwen3-Embedding-0.6B: the model's
    practice-set embedding time x 5.38. 5.38 is the median of (test job
    time / practice embedding time) over the 6 models that have both
    (range 5.12-5.43). Check: this rule gives bge-base 1.81 h against 1.77 h
    logged for the same shape (2% apart).
- **BM25:** runs on a CPU; not priced, so $0. The figure draws it at the left
  edge of the log axis (log scales cannot show $0).
- **First pass** (BM25 + bge-base, fused): the bge-base cost, $1.665. BM25
  and the fusion run on a CPU, $0.
- **Cross-encoder rerankers:** first pass + the logged job that reranks the
  top 100 per query (`data/gcp_logs/hs-rr`, test jobs 10-16).
- **LLM rerankers:** first pass + the API charges OpenRouter reported,
  summed over the 500 queries (`cost` column of
  `results/llm_v2/haystack_rerank/GeneDiseaseHaystack/*.parquet`): Gemini
  $0.877, DeepSeek $0.074. DeepSeek's BM25 run was charged $0.126 for
  almost the same tokens (same provider; its price changed between runs).
- **Our 3-seed fine-tunes:** mean of the 3 logged test jobs (they differ by
  under 1%).

## Training cost (`train_cost_usd`)

Our gene-disease fine-tunes only: one model's training job in
`data/gcp_logs/hs-ft/runner.log` x $0.94 per hour. It is a one-off, so it is
not on the figure's x axis.

| model | L4 hours | USD |
|---|---|---|
| bge-base, fine-tuned (mean of 3 seeds) | 0.33 | 0.31 |
| bge-small, fine-tuned (mean of 3 seeds) | 0.15 | 0.14 |
| MedCPT cross-encoder, fine-tuned | 0.43 | 0.40 |
| bge-reranker-base, fine-tuned | 0.48 | 0.46 |

- bge-small seed 42 is the recipe-grid run that won (`data/finetune_gd/runs/chosen.json`).
- Not included: the recipe search and practice checks. All hs-ft jobs except
  the test runs took 3.77 L4-hours ($3.54).
- The GO fine-tunes were trained on 2026-09-25, outside the logged job
  queues. Their training cost is unknown, so the cell is blank.

## Not counted anywhere

The LLM judging that made the grades (~$8.60: test $5.83, practice $2.77),
building the data sets, the PubTator gene lookup that picks each query's
candidates, and storage.

## More queries change the picture

Embedding the 1.7M abstracts is done once; the vectors are kept. Another 500
queries cost almost nothing for an embedder. A reranker pays again every
time: $0.04-$1.52 per 500 queries for a cross-encoder, $0.07-$0.88 for an
LLM. So the more queries, the more the reranker's cost matters.

## Which rows

- All rows asked for, plus **2 extra rows** (`requested` = False): the same
  Gemini and DeepSeek reranks applied to BM25's top 20, with no GPU at all.
  They are on the frontier, so leaving them out would change it.
- Left out: untuned BM25 (same cost as tuned, lower score), MedCPT scored by
  cosine (same vectors as MedCPT), and the other BM25 + embedder fusions.
- Not run: cross-encoders on BM25's top 100 alone. A fine-tuned MedCPT
  cross-encoder there would cost about $0.04 (its rerank time), but its
  quality is unknown.

## The frontier line

A point is on the frontier if no other point is both at least as cheap and
better. It uses graded nDCG@10 (both judges) and run cost (training not
counted): BM25 -> BM25 + DeepSeek rerank -> BM25 + Gemini rerank -> first pass
+ fine-tuned MedCPT cross-encoder. Without the 2 extra rows it would be BM25
-> fine-tuned bge-base -> first pass + fine-tuned MedCPT cross-encoder. Many
neighbouring points differ by less than their 95% ranges (about +/-0.02), and
none of these gaps was tested. The line's slope from BM25 depends on where
$0 is drawn; it is only a guide for the eye.
