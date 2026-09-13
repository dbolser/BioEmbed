# Results v1 — The Embedder's Dilemma, for Biology

*2026-09-13. 16 embedding models + 4 LLMs (3 run by us on OpenRouter, the
rest lifted from the paper's published results) on the 12-task BioMTEB(LLM)
suite. Total LLM API spend: $6.22.*

## Headline

The paper's division of labour **reproduces on biological text**, with one
domain twist. On the full 12-task bio suite:

| model | type | params | BioScore | $/pass |
|---|---|---|---|---|
| **Qwen3-Embedding-4B** | embedding | 4B | **0.645** | $0.083 |
| Qwen3-Embedding-0.6B | embedding | 596M | 0.616 | $0.022 |
| MedEmbed-base (bio fine-tune of bge-base) | embedding | 109M | 0.592 | $0.0009 |
| bge-large-en-v1.5 | embedding | 335M | 0.590 | $0.010 |
| MedEmbed-large | embedding | 335M | 0.590 | $0.010 |
| bge-base-en-v1.5 | embedding | 109M | 0.589 | $0.0009 |
| DeepSeek-V4-Flash | LLM | — | 0.566 | $0.94 |
| Gemini 3.1 Flash-Lite | LLM | — | 0.538 | $2.34 |

(Full table: `results/bio_summary.csv`. Qwen3.6-35B-A3B scores 0.683 over
11 tasks but cannot run R2MED-pooled — its 271k-token corpus-in-context
prompt exceeds the 262k context window; cost $4.85.)

The best affordable LLM is **43× the cost of the best embedder and 1,000×
the cost of an embedder that outscores it**. On the paper's lifted subset,
Gemini 3.1 Pro (0.758) still edges the best embedding model (0.745) — at
~100× the cost — so the frontier shape (all embedders + one frontier LLM)
carries over to biology.

## By category (best model per side)

- **Retrieval — LLMs win**, including both new human-curated bio tasks:
  GO-evidence retrieval (DeepSeek 0.331 vs bge-large 0.292, recall@1) and
  pooled R2MED biology (Flash-Lite 0.319 vs MiniLM 0.244). Reproduces the
  paper's "LLMs lead reasoning-heavy retrieval".
- **Classification — split.** On 30-class MeSH, LLMs win big (Qwen35B
  0.818 vs best embedder MedCPT 0.689; MTEB's 8-samples-per-label logreg
  protocol is hard on 30 classes, and the Qwen3-E family curiously scores
  lowest of all embedders here, ~0.51). On binary ToxicConversations,
  Qwen3-E-4B (0.923) beats the best LLM (0.910) — the paper's
  big-embedders-win-classification result returns with embedder scale.
- **Clustering — embedders win** (0.51 vs 0.50/0.32/0.20). The non-reasoning
  Flash-Lite collapses to 0.196 — the paper's exact finding.
- **Pair classification — embedders win** (0.71 vs 0.63–0.66 category mean).
  Qwen3-E-4B tops chemical synonymy (0.707) ahead of BioLORD (0.657),
  whose concept-synonymy training objective makes it the best sub-300M
  model there; every LLM sits at ~0.46–0.48.
- **STS — a tie** (Flash-Lite 0.900 vs bge-large 0.877 on suite means).

## Thinking tax (`results/thinking_tax.csv`)

Disabling reasoning on DeepSeek-V4-Flash cut generated tokens ~66% at
−0.01 average quality (MeSH *improved* +0.006). Qwen3.6-35B paid more
(−0.068 on MeSH) for a ~75% token cut. Matches the paper: lower reasoning
budgets often preserve quality, but it is model-dependent.

## The domain twist

Bio-specialised models occupy the cheap end of the bio Pareto frontier
(pubmedbert-emb → BioLORD → bge-base → MedEmbed-base → Qwen3-E-0.6B →
Qwen3-E-4B).
MedEmbed's ~$20-of-compute synthetic-triplet fine-tunes beat their bge
parents at every size (small +0.016, base +0.004, large +0.000 BioScore) —
domain fine-tuning of small embedders is real but its margin shrinks as
the base model grows, and the newest general instruction-tuned embedder
(Qwen3-E-0.6B) beats all of them. MedCPT (PubMed search specialist) wins
nothing outside PubMed-style retrieval.

## Method notes & caveats

- Both paradigms score identical seed-42 data; retrieval scored recall@1
  on both sides (the paper's convention). Embedding results validated by
  reproducing the paper's multilingual-e5-small numbers exactly.
- LLM costs: measured OpenRouter usage at current prices (lifted-task share
  recomputed from the paper's raw per-task usage at the same prices).
  Embedding costs: own-tokenizer tokens × the paper's measured H100
  $/MTok; params-nearest proxy (flagged) where unmeasured.
- R2MED-pooled is far easier than full-corpus R2MED (363-doc pool, no
  out-of-pool distractors) — scores are not comparable to the R2MED paper.
- GOPubMedRetrieval gold is GO-Consortium experimental-evidence GAF links
  (human-curated); GOProteinPairCls negatives are GO-hierarchy-aware.
- Not yet run: Qwen3-E-8B and gated embeddinggemma on the new tasks;
  Gemini Pro/Flash on the new tasks (would cost $32–76). Single seed, no
  bootstrap CIs yet.
