# Results v1 — The Embedder's Dilemma, for Biology

*Updated 2026-09-25. 21 embedding models + 4 LLMs (3 run by us on OpenRouter, the
rest lifted from the paper's published results) on the 12-task BioMTEB(LLM)
suite. LLM spend: $8.8 OpenRouter + ~$26 Vertex (Gemini Pro); GPU: $0.82 AWS + ~$1.35 GCP.*

## Headline

The paper's division of labour **reproduces on biological text**, with one
domain twist. On the full 12-task bio suite:

| model | type | params | BioScore | $/pass (measured L4) |
|---|---|---|---|---|
| **Gemini 3.1 Pro** (Vertex) | LLM | — | **0.689** | $39 |
| Qwen3-Embedding-8B | embedding | 7.6B | 0.662 | $0.28 |
| F2LLM-v2-1.7B | embedding | 1.7B | 0.656 | $0.08 |
| Qwen3-Embedding-4B | embedding | 4B | 0.645 | $0.18 |
| F2LLM-v2-0.6B | embedding | 596M | 0.631 | $0.045 |
| Qwen3-Embedding-0.6B | embedding | 596M | 0.616 | $0.044 |
| **bge-base-gaf (ours: bge-base + 172k GAF pairs)** | embedding | 109M | 0.604 | $0.006 |
| bge-small-gaf (ours) | embedding | 33M | 0.598 | $0.002 |
| MedEmbed-base | embedding | 109M | 0.592 | $0.006 |
| DeepSeek-V4-Flash | LLM | — | 0.566 | $0.94 |
| Gemini 3.1 Flash-Lite | LLM | — | 0.538 | $2.34 |

(Full table: `results/bio_summary.csv`. Gemini 3.1 Pro now covers all 12
tasks (~$26 on Vertex): it leads by 2.7 points at ~140× the cost of
Qwen3-E-8B — the paper's dilemma, exactly. Qwen3.6-35B-A3B scores 0.683 over 11 tasks but cannot run R2MED-pooled — its 271k-token corpus-in-context
prompt exceeds the 262k context window; cost $4.85.)

The best affordable LLM is **43× the cost of the best embedder and 1,000×
the cost of an embedder that outscores it**. On the paper's lifted subset,
Gemini 3.1 Pro (0.758) still edges the best embedding model (0.745) — at
~100× the cost — so the frontier shape (all embedders + one frontier LLM)
carries over to biology.

## By category (best model per side)

- **Retrieval — LLMs win at the top rank only.** On GO-evidence retrieval
  the LLM's first pick is gold 85–93% of the time (best embedder: 84%,
  MiniLM 71%), which wins recall@1 (DeepSeek 0.331 vs bge-large 0.292);
  but LLMs return 1–3 documents, so embedders dominate every deeper cut
  (recall@5: 0.64–0.77 vs 0.41–0.60). "LLMs lead retrieval" is true under
  the paper's recall@1 convention and reverses under recall@5/nDCG.
- **Classification — split.** On 30-class MeSH, LLMs win big (Qwen35B
  0.818 vs best embedder MedCPT 0.689; MTEB's 8-samples-per-label logreg
  protocol is hard on 30 classes, and the Qwen3-E family curiously scores
  lowest of all embedders here, ~0.51). On binary ToxicConversations,
  Qwen3-E-4B (0.923) beats the best LLM (0.910) — the paper's
  big-embedders-win-classification result returns with embedder scale.
- **Clustering — embedders win** (0.51 vs 0.50/0.32/0.20). The non-reasoning
  Flash-Lite collapses to 0.196 — the paper's exact finding.
- **Pair classification — embedders win, but by less than AP suggests.**
  Qwen3-E-4B tops chemical synonymy (0.707 AP) ahead of BioLORD (0.657);
  LLMs sit at ~0.46–0.48 AP — but LLMs emit binary 0/1 scores, and AP
  punishes that granularity: at F1 the real gap is 0.66–0.67 (LLM) vs 0.71
  (embedder oracle threshold). Part of the residual LLM "error" is PubChem
  depositor noise the models reject with chemically correct reasoning, and
  LLMs alone decode exotic notation (WLN strings). Caveat: LLM
  max_precision/max_recall values in these result files are threshold-sweep
  tie artifacts; trust AP and F1 only.
- **STS — a tie** (Flash-Lite 0.900 vs bge-large 0.877 on suite means).

## Retrieval pipelines (nDCG@10, both paradigms scored identically)

| protocol | GO-evidence | R2MED-bio | LLM cost/task |
|---|---|---|---|
| best embedder alone (Qwen3-E-8B) | 0.841 | 0.786 | — |
| LLM reads whole corpus, ranked list (DeepSeek) | 0.835 | 0.840 | $0.15–0.45 |
| Qwen3-E-0.6B top-20 → Flash-Lite rerank | **0.871** | **0.870** | $0.13–0.31 |
| MiniLM top-20 → Flash-Lite rerank | 0.828 | 0.851 | same |

Retrieve-then-rerank with a $0.04 embedder and a non-reasoning reranker
beats both the 8B embedder and corpus-in-context, and it scales to real
corpora. DeepSeek reranking spent 200–370k thinking tokens per task for
the same nDCG Flash-Lite reached with ~4k output tokens and no thinking.
(`results/llm_ranked`, `results/llm_rerank`.)

## Thinking tax (`results/thinking_tax.csv`)

Disabling reasoning on DeepSeek-V4-Flash cut generated tokens ~66% at
−0.01 average quality (MeSH *improved* +0.006). Qwen3.6-35B paid more
(−0.068 on MeSH) for a ~75% token cut. Matches the paper: lower reasoning
budgets often preserve quality, but it is model-dependent.

## The domain twist

Full-suite Pareto frontier (measured L4 costs): MiniLM → **bge-small-gaf** →
**bge-base-gaf** → Qwen3-E-0.6B → F2LLM-0.6B → F2LLM-1.7B → Qwen3-E-8B → Gemini 3.1 Pro.
As in the paper: all embedders plus one frontier LLM at the top. Our own fine-tunes — bge-small/base trained one epoch on
172k human-curated (GO term ↔ evidence abstract / protein function) pairs
from the GAF, strict test exclusion, ~1 GPU-hour total — sit on the frontier,
beating MedEmbed and their parents (+0.026 small, +0.015 base). Gains
concentrate on the GO tasks that share the training pair type (+0.07–0.11
on GO–protein pairs, +0.03–0.07 on GO-evidence retrieval), with ±0.01 noise
elsewhere: a targeted fine-tune buys exactly the task family you train.
MedEmbed's ~$20-of-compute synthetic-triplet fine-tunes beat their bge
parents at every size (small +0.016, base +0.004, large +0.000 BioScore) —
domain fine-tuning of small embedders is real but its margin shrinks as
the base model grows, and the newest general instruction-tuned embedder
(Qwen3-E-0.6B) beats all of them. MedCPT (PubMed search specialist) wins
nothing outside PubMed-style retrieval.

## Biology-specific findings (v1.1 analysis)

- **Biology punishes LLMs ~4.6× harder than embedders.** Moving from the
  30 general tasks to the 5 strictly-bio lifted tasks, embedders drop
  −0.014 on average, LLMs −0.066; the biggest rank-losers are all LLMs
  (Gemini 3 Flash falls 18 places).
- **On strictly-bio tasks, model rankings genuinely change** (Spearman ρ
  vs general ranking drops to 0.59): the F2LLM-v2 embedder family sweeps
  the strict-bio top 5 — F2LLM-1.7B (0.729, $0.037/pass) beats every 8B
  embedder and every LLM there, while ranking ~21st on general tasks.
- **Why embedders lose the bio top-1**: curated biology links texts with
  zero lexical overlap. MiniLM's recall on GO–protein positives whose
  function text shares no word with the GO term: 0.67, vs 0.95 with
  overlap (e.g. "zinc ion binding" ↔ zinc-finger TFs whose text never
  says "zinc"; "estrogen catabolic process" ↔ sulfotransferase papers).
  LLMs bridge those with background knowledge — resolving unnamed
  proteins (CD79A from a CD79B description) — yet stay ultra-conservative
  (≤2% false-positive rate on hierarchy-aware negatives).
- **Medical fine-tuning does not transfer to molecular biology.**
  MedEmbed's gains are Tier A literature/medical only (+0.03 small);
  on the GO tasks its deltas are noise (±0.009). Specialists are niche
  weapons: MedCPT is the worst model overall (catastrophic forgetting,
  −0.29 on STSB) yet the best embedder on MeSH-30 and GO–protein pairs.
- **Chemistry inverts the specialist story**: PubMed-*pretrained* models
  (BioLORD +0.065, PubMedBERT +0.045 vs bge-base) win chemical synonymy
  that MedEmbed's clinical *fine-tune* loses (−0.01..−0.04) — pretraining
  coverage of nomenclature beats task fine-tuning.
- **Clustering is the LLMs' worst bio failure** (−0.24 vs embedders on
  BiorxivP2P; even Gemini 3.1 Pro at $13/pass loses to a $0.04 embedder).
- **The cost knee**: jina-v5-nano ($0.012) captures ~90% of the lifted
  frontier's range; the last +0.013 (Gemini Pro) costs 100×. On the full
  bio suite the frontier ends at Qwen3-E-4B with no LLM on it.

## Method notes & caveats

- Both paradigms score identical seed-42 data; retrieval scored recall@1
  on both sides (the paper's convention). Embedding results validated by
  reproducing the paper's multilingual-e5-small numbers exactly.
- LLM costs: measured OpenRouter usage at current prices (lifted-task share
  recomputed from the paper's raw per-task usage at the same prices).
  Embedding costs: own-tokenizer tokens × $/MTok from throughput measured
  on one GCP Spot L4 ($0.48/hr, seq 512, largest batch; results/
  embedding_throughput_gpu.csv). L4 rates run 2–5× the paper's H100
  proxies with the same ordering; the LLM/embedder cost ratios above are
  therefore conservative.
- R2MED-pooled is far easier than full-corpus R2MED (363-doc pool, no
  out-of-pool distractors) — scores are not comparable to the R2MED paper.
- GOPubMedRetrieval gold is GO-Consortium experimental-evidence GAF links
  (human-curated); GOProteinPairCls negatives are GO-hierarchy-aware.
- Not yet run: gated embeddinggemma on the new tasks; Gemini 3 Flash anywhere new. Single seed.

### Uncertainty (`scripts/bootstrap_bio.py`)

Task-level paired bootstrap, the paper's method (10k resamples of the task
set, seed 42; `results/bootstrap_scores.csv`, `results/bootstrap_pairs.csv`).
Item-level CIs are not possible: the result JSONs hold only aggregate
metrics, and the paper's lifted scores are one number per task.

With 12 (full) or 7 (lifted) tasks, each model's BioScore has a 95% CI
about ±0.12 wide (median width 0.23 on both suites) — wider than most gaps
between models. Paired differences are tighter, since the same tasks hit
both models. Of the headline comparisons only one survives at 95%:

- **Qwen3-E-4B > Qwen3-E-0.6B**: +0.030 [+0.007, +0.054], p=0.008 — significant.
- Qwen3-E-0.6B > MedEmbed-base: +0.023 [−0.040, +0.073], p=0.42 — not.
- MedEmbed-base > DeepSeek-V4-Flash: +0.027 [−0.038, +0.096], p=0.43 — not.
  Even Qwen3-E-4B > DeepSeek (+0.080 [−0.016, +0.171], p=0.11) is not.
- MedEmbed vs bge parents: small +0.015 [−0.001, +0.033], p=0.06 (borderline);
  base +0.004, p=0.71; large −0.000, p=0.94 — none significant.
- Gemini 3.1 Pro > Qwen3-E-8B (lifted): +0.013 [−0.024, +0.057], p=0.61 — not.

Read the rankings as suggestive orderings, not settled ones. The
embedder-vs-LLM ordering on the full suite rests on where each side wins
(category-level), not on a significant BioScore gap; and the paper's own
Pro-vs-best-embedder gap is a tie here too.
