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
  Embedding costs: own-tokenizer tokens × the paper's measured H100
  $/MTok; params-nearest proxy (flagged) where unmeasured.
- R2MED-pooled is far easier than full-corpus R2MED (363-doc pool, no
  out-of-pool distractors) — scores are not comparable to the R2MED paper.
- GOPubMedRetrieval gold is GO-Consortium experimental-evidence GAF links
  (human-curated); GOProteinPairCls negatives are GO-hierarchy-aware.
- Not yet run: Qwen3-E-8B and gated embeddinggemma on the new tasks;
  Gemini Pro/Flash on the new tasks (would cost $32–76). Single seed, no
  bootstrap CIs yet.
