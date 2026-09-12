# BioEmbed experimental design

Recreation of *The Embedder's Dilemma* (COLM 2026, arXiv:2608.12875) on
biological text. Their code is vendored read-only at `vendor/embedders-dilemma`
(MIT); we reuse their protocol, cost model, and — where tasks overlap — their
published results.

## The question

Cost-aware Pareto frontier of quality vs USD for biology text tasks:
open embedding models (22M–8B) vs bio-specialised embedders vs LLMs prompted
directly. Second question (novel): does a cheap domain fine-tune of a small
embedder dominate a frozen 8B model per dollar?

## Protocol (theirs, unchanged)

Both paradigms see identical seed-42 data subsets.

| Category | Embedding pipeline | LLM |
|---|---|---|
| Retrieval | encode corpus+queries, cosine top-k, nDCG@10 | whole corpus in context, pick relevant doc IDs (LOFT-style) |
| Classification | kNN over embedded train split | zero-shot labeling, labels in prompt |
| Clustering | k-means with true k | all docs in one prompt, assign cluster IDs |
| STS | cosine similarity | rate similarity as float on native scale |
| PairClassification | cosine + threshold sweep | binary judgment per pair |

LLM outputs are JSON `{reasoning, output}` with schema validation, 2 retries.
Cost: API tokens at provider rates (cached input at 10%, all generated tokens
incl. thinking at output rate); embeddings at own-tokenizer token counts ×
GPU-rental $/token derived from measured throughput (H100 @ $2.49/hr).

## Task suite: BioMTEB(LLM)

**Tier A — already in the paper, biomedical; results lifted from their repo
(zero API cost, 36 models):**
- LLMBIOSSES (STS), LLMBiorxivClusteringP2PV2, LLMMedrxivClusteringP2PV2,
  LLMMedrxivClusteringS2SV2 (Clustering)

**Tier B — public bio MTEB tasks, sampled seed-42 with their recipe.**
Retrieval corpora additionally subsampled (all gold docs for the ≤100 sampled
queries + random distractors) to ≈500 docs so corpus-in-context fits ~128k:
- NFCorpus, SciFact (standard bio retrieval)
- R2MEDBiologyRetrieval, R2MEDBioinformaticsRetrieval (reasoning-heavy —
  the category where LLMs must win for the paper's story to transfer)
- SciRepEvalMeSHDescriptorsClassification (fine-grained many-class),
  SciRepEvalDRSMClassification (Classification)

**Tier C — novel molecular-biology tasks from human-curated cross-references
(IUK-IPF data at `/outsee5/home/danbolser/Work/IUK-IPF`, read-only):**
- GOUniProtRetrieval: query = GO term name+definition; corpus = UniProt
  function texts; gold = GO-Consortium GAF links (1.28M human-curated triples)
- BioKarlQARetrieval: query = grounded bio question; gold = gold_record_ids
  over the curated 90k-record corpus (silver labels)
- GOProteinPairCls: (GO term text, protein function text) linked-or-not,
  hierarchy-aware negatives (PairClassification)

Aggregate: macro mean over tasks (BioScore) + per-category means.

## Models

Embedding ladder (`src/bioembed/models.py`): MiniLM-22M → bge ladder →
EmbeddingGemma-300M → arctic-l-v2 → Qwen3-Embedding-0.6B/4B/8B; bio
specialists: PubMedBERT-emb, MedCPT, MedEmbed-S/B/L, BioLORD-2023,
BMRetriever-410M/1B. Phase 2: our own fine-tune of bge-small/base on GAF
synthetic pairs (Gecko-style: LLM-generated queries from bio passages).

LLMs (phase 3, OpenRouter, ~3–5): frontier reasoning model, cheap reasoning
model, same-family non-reasoning baseline (isolates the thinking tax), one
open MoE. Tier A LLM results are lifted, so LLM spend is Tier B+C only
(~small corpora — single-digit dollars per model).

## Cost accounting for our hardware

- In-paper models: reuse their measured H100 throughput → $/MTok.
- New models (bio specialists): measure tokens/s on our CPU for all models,
  fit CPU→H100 ratio on the overlapping models, extrapolate; optionally
  validate with one $2 GPU spot hour. Report both.

## Phases

1. **Embedding sweep** (no keys, CPU): all models × Tier A/B/C tasks.
2. **Fine-tune** (no keys / local): synthetic-pair fine-tune of small bge;
   add to sweep.
3. **LLM sweep** (OpenRouter key): Tier B+C via their `llm_judge`, vendored
   env pointed at OpenRouter.
4. **Analysis**: bio Pareto frontier, category radar, thinking-tax ablation
   (reasoning budget off/low on retrieval), paired bootstrap significance.
