# Hugging Face models worth knowing about for BioEmbed

Survey date: 2026-11-04. Scope: open-weight models on the Hugging Face Hub that could matter to any BioEmbed
thread (A: Paper 1, the Embedder's Dilemma for biology; B: genesearch; C/D/E: geometry, gene-tag errors, cross-model
maps). API-only models (Gemini, Voyage, Cohere, OpenAI embeddings) are out of scope.

**Nothing here was run on our tasks.** Every number comes from a model card, a paper or the MTEB results repo
(github.com/embeddings-benchmark/results). Our own gold may disagree. Costs are estimates scaled from
`results/embedding_throughput_gpu.csv`, not measurements.

How it was made: eight search agents, one per model category, looked up every candidate live on the Hub (metadata,
card, paper, leaderboard). A second agent per category then tried to refute each entry: it re-ran the lookups,
re-checked the headline number at its source and dropped duplicates and off-target repos. A last agent looked for
gaps. The first gene/protein agent failed, so that category was run again. 142 distinct repos survived (137 models, 5
datasets), plus one late-interaction model I added after a verifier flagged it; I re-checked that every one exists. Full records, with sources for every number:
[`docs/hf-model-survey.json`](hf-model-survey.json). "Downloads" means the last 30 days.

## TLDR: what we should do

No new model clearly changes the paper's story. The main lesson is that BM25 still wins because it matches exact
tokens, and late-interaction models now do that better. Two of our planned checks may also be biased.

Short term (this week, almost no cost):

1. Protect what we already have. Check which suite models trained on BRIGHT data, since R2MED Biology is the same
   set (finding 1). Find out which GNorm2 model PubTator3 runs before we calibrate on NLM-Gene (finding 5).
2. Run the cheap checks that answer reviewer questions: BioHiCL-Base as the same-backbone control for our fine-tune
   (CPU minutes), and bge-reranker-v2-gemma and gpt-oss-120b as rerank arms (under $1 each). gpt-oss-120b also fixes
   the problem that our two judges currently grade their own reranks. Cite the Dictyostelium paper.
3. Compare our literature gene centroids with the 68 precomputed gene embeddings in the BoevaLab dataset. No GPU,
   and it gives thread D a new way to find mis-tags.

Medium term (next few weeks, tens of GPU hours):

1. Add sparse and MaxSim scoring to `eval/haystack.py`, then run SPARSEUP, LateOn and DenseOn. They share a backbone
   and training data, so they show which architecture closes the gap to BM25. Then try topk-embed-v1-small as a
   top-100 reranker. This is the most likely new section for Paper 1.
2. Run Nemotron-3-Embed-1B and 8B on the main task and suite. They are the new best open health embedders.
3. Repeat our fine-tune on UltraBERT-Embed-150M to see whether the gain stacks on a better base. Use HunFlair2 as a
   second-opinion tagger on the flagged bare symbols (SE, AST, TNM, CAR, ACE).

Long term (only if the evidence supports it):

1. Swap genesearch off MedEmbed-small only if harrier-oss-v1-270m (GPU) or bekko (CPU) beats it on our main task.
   The 35M re-embed is the cost.
2. Build the cross-model maps with Nemotron-3-Embed-1B as the large modern model, a static PubMed model as the
   floor, and SemCSE or sciembed as a different training signal.
3. Add protein axes (ESMC-6B, ProTrek) to the gene maps only once the free dataset comparison shows something.
   Re-run this survey in about six months; nobody has yet released a successor to MedCPT or MedEmbed.

## Five findings that matter more than any one model

1. **R2MED Biology is BRIGHT Biology.** Same 57,359 documents, same 103 queries (checked on the Hub's dataset
   server). Several new models trained on data derived from BRIGHT: Nemotron-3-Embed (BRIGHT is a seed set for its
   synthetic queries), llama-nv-embed-reasoning-3b (queries synthesised from the BRIGHT corpus) and Diver-GroupRank-7B
   (lists BRIGHT as training data). Their scores on the suite's R2MEDBiologyPooled task may be inflated. Worth
   checking the training data of the embedders already in the suite too.
2. **Related work to cite:** *Benchmarking Literature Retrieval for a Model Organism: A Dictyostelium Case Study*
   (arXiv 2610.03130, Oct 2026). Gene-literature retrieval for curation. BM25 is strong there too, gene-synonym query
   expansion helps, and the best reranker is BAAI/bge-reranker-v2-gemma (MRR@10 0.707 against zero-shot MedCPT-CE
   0.650 and BM25 0.593).
3. **BM25 still beats almost every single-vector embedder on health retrieval.** On RTEB's EnglishHealthcare1, BM25
   scores 68.75. Of the models in the MTEB results repo, only one single-vector model beats it (Nemotron-3-Embed-8B, 71.30). Two late-interaction models are
   far ahead: topk-embed-v1-small 83.28 and -xsmall 81.29 (I checked these in the results repo). This echoes our
   haystack result and points at token-level matching as the missing piece.
4. **An open LLM rerank can avoid the self-grading problem.** Today both LLM rerank arms (Gemini Flash-Lite,
   DeepSeek-V4-Flash) are also the two judges. openai/gpt-oss-120b (Apache-2.0) is served by about ten Hub inference
   providers from about $0.04 per million input tokens, so both judges could grade it.
5. **NLM-Gene may flatter PubTator3.** GNorm2 (PubTator3's gene normaliser) ships two model sets: one "trained by the
   training set only" and one "trained by all datasets" (its README). If PubTator3 runs the all-data model, scoring it
   on the NLM-Gene test split (the geometry TODO's calibration plan) overstates its accuracy. HunFlair2 also trained on
   NLM-Gene. Find out which model PubTator3 uses before trusting that calibration.

## What to try first

| # | Model | Question it answers | Thread | Rough cost |
|---|---|---|---|---|
| 1 | [nvidia/Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) | A cheaper frontier point near Qwen3-Embedding-8B? The "large modern model" for the cross-model maps? | A, E | 7.76M papers ≈ 63 L4-h (~$30 Spot) |
| 2 | [nvidia/Nemotron-3-Embed-8B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16) | Does the new best open health embedder beat Qwen3-Embedding-8B on our tasks? | A | main task + suite ≈ 10-30 L4-h |
| 3 | [BAAI/bge-reranker-v2-gemma](https://huggingface.co/BAAI/bge-reranker-v2-gemma) | Open reranker vs the Gemini top-20 rerank and our fine-tuned MedCPT-CE; then a LoRA fine-tune | A | under $1 zero-shot |
| 4 | [Linkup SPARSEUP](https://huggingface.co/Linkup-Platform/linkup-sparseup-embed-v1) + [LightOn LateOn](https://huggingface.co/lightonai/LateOn) + [DenseOn](https://huggingface.co/lightonai/DenseOn) | Why does BM25 come so close? Same backbone and training data, three architectures (sparse, late interaction, dense) | A | ~2 L4-h each on the 1.7M haystack; needs sparse and MaxSim scoring paths |
| 5 | [topk-io/topk-embed-v1-small](https://huggingface.co/topk-io/topk-embed-v1-small) | Does the health-retrieval leader (late interaction) transfer to gene-disease papers? Use it as a reranker over the top 100 | A | GPU; token vectors too big for the full haystack |
| 6 | [dmis-lab/UltraBERT-Embed-150M](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M) | Our one-epoch gene-disease fine-tune on a better 150M base: does the +0.09 gain stack? | A | ~1 L4-h |
| 7 | [cross-encoder/ettin-reranker-150m-v1](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1) | Is a modern general cross-encoder a better fine-tune base than MedCPT-CE at the same size? Also a CPU rerank stage for genesearch | A, B | under $1 |
| 8 | [LunaLan07/BioHiCL-Base](https://huggingface.co/LunaLan07/BioHiCL-Base) | bge-base plus generic MeSH supervision vs bge-base plus our task supervision: the clean control a reviewer will ask for (BiCA-base is built on gte-base, so it is not that control) | A | CPU minutes |
| 9 | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | A rerank both judges can grade (finding 4) | A | well under $1 via providers |
| 10 | [microsoft/harrier-oss-v1-270m](https://huggingface.co/microsoft/harrier-oss-v1-270m) | Should genesearch replace MedEmbed-small? Test on the main task first | B | 35M re-embed ≈ 25-75 L4-h |
| 11 | [hunflair/hunflair2-ner](https://huggingface.co/hunflair/hunflair2-ner) | A second-opinion gene tagger for the bare-symbol audit (SE, AST, TNM, CAR, ACE) | D | CPU, on the flagged subsets |
| 12 | [hanhainebula/reason-embed-qwen3-8b-0928](https://huggingface.co/hanhainebula/reason-embed-qwen3-8b-0928) | The R2MED leader (avg 43.18, Biology 54.01). Mind finding 1 | A | ~1 L4-h for the suite task |

## By category

Priority is the verifiers' judgement for this project: high = answers an open question cheaply; medium = a good
option, not urgent. Low-priority models are listed after each table.

### General-purpose dense embedders

| Model | Size | Released | Licence | Why | Priority |
|---|---|---|---|---|---|
| [nvidia/Nemotron-3-Embed-8B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16) | 8.0B | 2026-07 | OpenMDW-1.1 | Top open single-vector model on RTEB(Health): 75.54 vs Qwen3-Embedding-8B 71.31. Successor to the covered llama-embed-nemotron-8b | high |
| [nvidia/Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) | 1.1B | 2026-07 | OpenMDW-1.1 | Distilled from the 8B. Beats Qwen3-8B on EnglishHealthcare1 (67.19 vs 63.54) but not on RTEB(Health) overall (67.87 vs 71.31). 2048-d, sliceable to 512 | high |
| [microsoft/harrier-oss-v1-270m](https://huggingface.co/microsoft/harrier-oss-v1-270m) | 268M | 2026-03 | MIT | Best bio evidence of any small model: CUREv1 45.0 vs MedEmbed-small 26.0, TRECCOVID 82.1 vs 77.7. 640-d, no Matryoshka | high |
| [dmis-lab/UltraBERT-Embed-150M](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M) | 151M | 2026-10 | Apache-2.0 | MTEB(eng,v2) 67.24 vs bge-base 65.14 at similar size; 32K context. Needs transformers ≥ 5.20 or trust_remote_code | high |
| [microsoft/harrier-oss-v1-0.6b](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) | 596M | 2026-03 | MIT | Drop-in rival to Qwen3-Embedding-0.6B (MTEB eng v2 70.64 vs 70.47); mixed on bio (SciFact +6.5, TRECCOVID −3.6) | medium |
| [perplexity-ai/pplx-embed-v1-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b) | 596M | 2026-01 | MIT | Takes no instruction prompt, so it removes a prompt-choice confound from cross-model maps. Native int8: 7.76M × 1024 ≈ 8 GB. No bio numbers. trust_remote_code | medium |
| [MongoDB/voyage-4-nano](https://huggingface.co/MongoDB/voyage-4-nano) | 346M | 2026-01 | Apache-2.0 | Shares an embedding space with the Voyage-4 API models: embed documents locally, queries with voyage-4-large. A new kind of cost point. No bio numbers for nano | medium |
| [hotchpotch/bekko-embedding-v1-a25m](https://huggingface.co/hotchpotch/bekko-embedding-v1-a25m) | 123M (25M active) | 2026-07 | MIT | 384-d like MedEmbed-small and built for CPU (ONNX, OpenVINO). CUREv1 42.1 vs 26.0, but TRECCOVID 56.8 vs 77.7. The only full 35M re-embed that is realistic on our CPUs (~11 h, estimate) | medium |

Low: [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (multimodal successor, no better on
text), [zeroentropy/zembed-1-embedding](https://huggingface.co/zeroentropy/zembed-1-embedding) (4B, distilled from a
reranker), [microsoft/bitnet-embedding-270m](https://huggingface.co/microsoft/bitnet-embedding-270m) (ternary weights,
CPU), [lightonai/DenseOn](https://huggingface.co/lightonai/DenseOn) (needed for the trio above),
[microsoft/harrier-oss-v1-27b](https://huggingface.co/microsoft/harrier-oss-v1-27b) (tops MTEB Multilingual v2, 74.32;
a >8B ceiling probe), [perplexity-ai/pplx-embed-v1-4b](https://huggingface.co/perplexity-ai/pplx-embed-v1-4b),
[Octen/Octen-Embedding-4B](https://huggingface.co/Octen/Octen-Embedding-4B),
[KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5](https://huggingface.co/KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5),
[infgrad/Jasper-Token-Compression-600M](https://huggingface.co/infgrad/Jasper-Token-Compression-600M),
[Kingsoft-LLM/QZhou-Embedding](https://huggingface.co/Kingsoft-LLM/QZhou-Embedding) and
[codefuse-ai/F2LLM-v2-160M](https://huggingface.co/codefuse-ai/F2LLM-v2-160M) (both high MTEB averages, weak on
TRECCOVID and SciFact: useful negative controls).

### Biomedical, scientific and reasoning embedders

| Model | Size | Released | Licence | Why | Priority |
|---|---|---|---|---|---|
| [LunaLan07/BioHiCL-Base](https://huggingface.co/LunaLan07/BioHiCL-Base) | 109M | 2026-04 | none stated | bge-base + MeSH hierarchical contrastive training (ACL 2026). The same-backbone control for our fine-tune. BEIR-bio IR avg 0.543 vs bge-base 0.529. Use mean pooling, as its card does | high |
| [hanhainebula/reason-embed-qwen3-8b-0928](https://huggingface.co/hanhainebula/reason-embed-qwen3-8b-0928) | 7.6B | 2025-10 | Apache-2.0 | The checkpoint behind the R2MED leader (avg 43.18, Biology 54.01). Use it, not the older BAAI 0923 checkpoint. 4B sibling exists | medium |
| [nvidia/llama-nv-embed-reasoning-3b](https://huggingface.co/nvidia/llama-nv-embed-reasoning-3b) | 3.2B | 2026-02 | cc-by-4.0 (check card) | BRIGHT Biology 63.4, the highest seen, but trained on queries synthesised from BRIGHT (finding 1) | medium |
| [AQ-MedAI/Diver-Retriever-4B-1020](https://huggingface.co/AQ-MedAI/Diver-Retriever-4B-1020) | 4.0B | 2025-10 | Apache-2.0 | Qwen3-Embedding-4B tuned for reasoning retrieval: a paired test against our best embedder. BRIGHT avg 31.9 | medium |
| [sentence-transformers/embeddinggemma-300m-medical](https://huggingface.co/sentence-transformers/embeddinggemma-300m-medical) | 303M | 2025-09 | Apache-2.0 | One-variable test against the covered embeddinggemma-300m (MIRIAD nDCG@10 0.886 vs 0.834). Does a medical-QA fine-tune close its haystack gap to BM25? | medium |
| [xdawang/edel_po](https://huggingface.co/xdawang/edel_po) | 108M | 2026-05 | none stated | Retrieves abstracts that support curated knowledge-base records (CIViC, OncoKB): the closest published task to ours. The reusable part is the recipe: graded margins, and hard negatives that share some entities (same gene, other disease) | medium |
| [NeuML/biomedbert-small-embeddings](https://huggingface.co/NeuML/biomedbert-small-embeddings) | 23M | 2026-04 | Apache-2.0 | 384-d, half MedEmbed-small's layers, PubMed-distilled. Only in-domain similarity scores so far | medium |
| [CLAUSE-Bielefeld/SemCSE_cosine](https://huggingface.co/CLAUSE-Bielefeld/SemCSE_cosine) | 184M | 2025-07 | MIT | Scientific abstracts embedded by content, not citations: a different training signal for the cross-model maps. 7.76M ≈ 4-6 L4-h | medium |
| [J0nasW/sciembed-full](https://huggingface.co/J0nasW/sciembed-full) | 149M | 2026-08 | MIT | Citation-context supervision on ModernBERT; another signal for the maps (SciRepEval 66.85) | medium |

Low: [NeuML/biomedbert-hash-nano-embeddings](https://huggingface.co/NeuML/biomedbert-hash-nano-embeddings) (under 1M
parameters), [NeuML/biomedbert-base-embeddings](https://huggingface.co/NeuML/biomedbert-base-embeddings) (also a
fine-tune base, below), [minetta/nemotron-3-embed-8b-medical](https://huggingface.co/minetta/nemotron-3-embed-8b-medical)
(same retrieval weights as its base), [ikim-uk-essen/stm_phi](https://huggingface.co/ikim-uk-essen/stm_phi) and
[yerim0210/REPAIR_Scientific_Retrievers_7B](https://huggingface.co/yerim0210/REPAIR_Scientific_Retrievers_7B)
(BMRetriever-style LLM retrievers, gains of 0.3-3 points),
[NeuML/bioclinical-modernbert-base-embeddings](https://huggingface.co/NeuML/bioclinical-modernbert-base-embeddings)
(full-text only), [allenai/specter2_base](https://huggingface.co/allenai/specter2_base) (citation-trained, for the
maps), [BAAI/bge-reasoner-embed-qwen3-8b-0923](https://huggingface.co/BAAI/bge-reasoner-embed-qwen3-8b-0923)
(superseded by 0928), [BASF-AI/ChEmbed-prog](https://huggingface.co/BASF-AI/ChEmbed-prog) (chemistry, for the PubChem
synonym task only; check overlap; CC-BY-NC),
[MohammadKhodadad/MedTE-cl15-step-8000](https://huggingface.co/MohammadKhodadad/MedTE-cl15-step-8000),
[uiyunkim-hub/qwen3-bio-embedding-0.6b](https://huggingface.co/uiyunkim-hub/qwen3-bio-embedding-0.6b) (trained on
claim → supporting papers, close to our task format; no licence, no evaluation).

### Rerankers

| Model | Size | Released | Licence | Why | Priority |
|---|---|---|---|---|---|
| [BAAI/bge-reranker-v2-gemma](https://huggingface.co/BAAI/bge-reranker-v2-gemma) | 2.5B | 2024-03 | Apache-2.0 | The only open reranker with gene-literature evidence (finding 2). Note: that benchmark used claim-sentence queries and zero-shot MedCPT-CE; ours is fine-tuned | high |
| [cross-encoder/ettin-reranker-150m-v1](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1) | 149M (17M to 1B siblings) | 2026-05 | Apache-2.0 | Small cross-encoders distilled from mxbai-rerank-large-v2. MTEB retrieval rerank 0.599 vs bge-reranker-v2-m3 0.553. CPU: 14 pairs/s (150m), 31 (68m) | high |
| [zeroentropy/zerank-2-reranker](https://huggingface.co/zeroentropy/zerank-2-reranker) | 4.0B | 2025-11 | Apache-2.0 | Vendor says biomedical 0.72 vs Gemini 2.5 Flash 0.54; an independent MTEB run puts it below bge-reranker-v2-m3. Only our gold can settle it | medium |
| [jinaai/jina-reranker-v3.5](https://huggingface.co/jinaai/jina-reranker-v3.5) | 597M | 2026-07 | CC-BY-NC-4.0 | Listwise: all 20 abstracts in one pass, like the Gemini step, at 0.6B. BEIR 63.20 | medium |
| [BAAI/bge-reranker-v2.5-gemma2-lightweight](https://huggingface.co/BAAI/bge-reranker-v2.5-gemma2-lightweight) | 9.2B | 2024-07 | Gemma | BEIR 63.67 vs v2-gemma 60.71; layer cut-off and compression knobs give several cost points from one checkpoint. Tight on an L4 | medium |
| [AQ-MedAI/Diver-GroupRank-7B](https://huggingface.co/AQ-MedAI/Diver-GroupRank-7B) | 7.6B | 2025-11 | Apache-2.0 | Reasoning reranker that roughly ties Gemini 2.5 Flash listwise on BRIGHT Biology (56.85 vs 56.6), but trained on BRIGHT | medium |
| [liuwenhan/reasonrank-7B](https://huggingface.co/liuwenhan/reasonrank-7B) | 7.6B | 2025-08 | MIT | Listwise reasoning reranker with R2MED numbers on top of E5-mistral, which we already ran | medium |
| [Alibaba-NLP/E2Rank-0.6B](https://huggingface.co/Alibaba-NLP/E2Rank-0.6B) | 596M | 2025-10 | Apache-2.0 | One model for both first pass and listwise rerank; BEIR rerank 52.09 | medium |

Low: [mixedbread-ai/mxbai-rerank-large-v2](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2),
[Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base),
[nvidia/Nemotron-3.5-Rerank-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16) (Oct 2026; visual
documents only), [nvidia/llama-nemotron-rerank-1b-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2),
[KaLM-Embedding/KaLM-Reranker-V1-Small-R2](https://huggingface.co/KaLM-Embedding/KaLM-Reranker-V1-Small-R2),
[Qwen/Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) (no English gain over the 4B),
[ContextualAI/ctxl-rerank-v2-instruct-multilingual-2b](https://huggingface.co/ContextualAI/ctxl-rerank-v2-instruct-multilingual-2b)
(takes a free-text instruction; non-commercial),
[lightonai/LightOn-rerank-PW-2B](https://huggingface.co/lightonai/LightOn-rerank-PW-2B),
[castorini/gaggle-reranker-20261005](https://huggingface.co/castorini/gaggle-reranker-20261005) (fully open recipe),
[NeuML/biomedbert-base-reranker](https://huggingface.co/NeuML/biomedbert-base-reranker) (the only new biomedical
cross-encoder since MedCPT-CE; no ranking benchmark),
[OpenSciLM/OpenScholar_Reranker](https://huggingface.co/OpenSciLM/OpenScholar_Reranker),
[naver/xprovence-reranker-bgem3-v2](https://huggingface.co/naver/xprovence-reranker-bgem3-v2) (keeps only the
relevant sentences; useful for error analysis).

### Sparse, late-interaction and static models

| Model | Size | Released | Licence | Why | Priority |
|---|---|---|---|---|---|
| [topk-io/topk-embed-v1-small](https://huggingface.co/topk-io/topk-embed-v1-small) | 2.2B (xsmall 0.85B) | 2026-09 | Apache-2.0 | Late interaction on Qwen3.5-2B. EnglishHealthcare1 83.28 vs best single-vector 71.30 and BM25 68.75. Multimodal, trust_remote_code, CUDA bf16. Rerank top-100 rather than index 1.7M | high |
| [Linkup-Platform/linkup-sparseup-embed-v1](https://huggingface.co/Linkup-Platform/linkup-sparseup-embed-v1) | 188M | 2026-09 | Apache-2.0 | Learned sparse, same backbone and data as LateOn and DenseOn. BEIR-13 56.4 vs SPLADE-v3 51.7. Shows which expansion terms fire per query | high |
| [lightonai/LateOn](https://huggingface.co/lightonai/LateOn) | 149M | 2026-03 | Apache-2.0 | The late-interaction leg of that trio (BEIR-15 57.22). MaxSim shows whether the gene symbol or the disease term drives a match | high |
| [opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte) | 137M | 2025-06 | Apache-2.0 | Inference-free: the query costs what BM25 costs. Does learned document expansion alone close the gap? | medium |
| [NeuML/pubmedbert-base-splade](https://huggingface.co/NeuML/pubmedbert-base-splade) | 110M | 2025-07 | Apache-2.0 | The only PubMed-trained learned-sparse model. It lowercases, so CAR = car | medium |
| [naver/splade-v3](https://huggingface.co/naver/splade-v3) | ~110M | 2024-03 | CC-BY-NC-SA-4.0 | The standard sparse baseline reviewers will expect | medium |
| [DataScience-UIBK/uibk-embedding-v1](https://huggingface.co/DataScience-UIBK/uibk-embedding-v1) | 179M | 2026-10 | Apache-2.0 | Dense and ColBERT in one index; BEIR-15 59.79 (self-reported) | medium |
| [NeuML/pubmedbert-base-embeddings-8M](https://huggingface.co/NeuML/pubmedbert-base-embeddings-8M) | 7.8M | 2025-01 | Apache-2.0 | Static PubMed embedder: all 7.76M papers on CPU in about an hour. A "no context" floor for the cross-model maps | medium |
| [minishlab/potion-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M) | 32M | 2025-01 | MIT | The general static anchor at the bottom of the cost frontier; minutes on CPU | medium |

Our `eval/haystack.py` is dense-only, so every sparse or late-interaction run needs a new scoring path first.

Low: [NeuML/biomedbert-base-colbert](https://huggingface.co/NeuML/biomedbert-base-colbert),
[ibm-granite/granite-embedding-30m-sparse](https://huggingface.co/ibm-granite/granite-embedding-30m-sparse),
[utahnlp/laconic-1b](https://huggingface.co/utahnlp/laconic-1b) (LLM-scale sparse; no licence stated),
[mixedbread-ai/mxbai-edge-colbert-v0-17m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-17m) (CPU-sized
late interaction; could highlight matched tokens in genesearch),
[sentence-transformers/static-retrieval-mrl-en-v1](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1),
[lightonai/Reason-ModernColBERT](https://huggingface.co/lightonai/Reason-ModernColBERT),
[Qdrant/minicoil-v1](https://huggingface.co/Qdrant/minicoil-v1),
[Alibaba-NLP/UEmbed-4B](https://huggingface.co/Alibaba-NLP/UEmbed-4B) (dense and sparse from one pass),
[DatologyAI/luxical-one](https://huggingface.co/DatologyAI/luxical-one) (very fast lexical-dense CPU embedder),
[doc2query/msmarco-t5-base-v1](https://huggingface.co/doc2query/msmarco-t5-base-v1) (document expansion for BM25;
the gap agent notes LLM query rewriting lifts BM25 far more on R2MED Biology: 59.65 vs 19.19 with o3-mini).

### Base encoders for fine-tuning

The gene-disease fine-tune is about 15.7k rows for one epoch: too little to turn a raw language model into a good
embedder. So the fair "which base gains more" sweep is among models that are already embedders (UltraBERT-Embed,
gte-modernbert, NeuML biomedbert-base). Raw encoders fit better as cross-encoder bases.

| Model | Size | Released | Licence | Why | Priority |
|---|---|---|---|---|---|
| [dmis-lab/UltraBERT-Embed-150M](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M) | 151M | 2026-10 | Apache-2.0 | See above: the best embedder start point at this size | high |
| [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | 149M | 2025-01 | Apache-2.0 | Second general arm for the sweep (MTEB eng v2 66.26) | medium |
| [NeuML/biomedbert-base-embeddings](https://huggingface.co/NeuML/biomedbert-base-embeddings) | 109M | 2026-04 | Apache-2.0 | Domain arm: a further-tuned pubmedbert-base-embeddings (already in the suite) | medium |
| [dmis-lab/UltraBERT-150M](https://huggingface.co/dmis-lab/UltraBERT-150M) | 190M | 2026-10 | Apache-2.0 | Raw encoder; best BEIR after fine-tuning at ≤ 200M (45.2 vs ModernBERT 42.6). A cross-encoder base | medium |
| [michiyasunaga/BioLinkBERT-large](https://huggingface.co/michiyasunaga/BioLinkBERT-large) | ~340M | 2022-03 | Apache-2.0 | Still first on BLURB (84.30). Citation-linked PubMed pretraining; a cross-encoder base to set against MedCPT-CE | medium |

Low: [almanach/ModernBERT-bio-base](https://huggingface.co/almanach/ModernBERT-bio-base),
[thomas-sounack/BioClinical-ModernBERT-base](https://huggingface.co/thomas-sounack/BioClinical-ModernBERT-base) (PubMedBERT
still wins short-abstract tasks), [ibm-granite/granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2),
[ibm-granite/granite-embedding-english-r2](https://huggingface.co/ibm-granite/granite-embedding-english-r2),
[dmis-lab/UltraBERT-380M](https://huggingface.co/dmis-lab/UltraBERT-380M),
[microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract),
[LiquidAI/LFM2.5-Encoder-350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M),
[jhu-clsp/ettin-encoder-150m](https://huggingface.co/jhu-clsp/ettin-encoder-150m),
[BSC-LT/MrBERT-biomed](https://huggingface.co/BSC-LT/MrBERT-biomed),
[Simonlee711/Clinical_ModernBERT](https://huggingface.co/Simonlee711/Clinical_ModernBERT).

### Open-weight LLMs (rerank, judge, topic names)

Hub inference providers now sell sparse mixture-of-experts models at $0.03 to $0.15 per million input tokens, against
Gemini 3.1 Flash-Lite at $0.25 ($0.125 batch). So "cheaper than Flash-Lite" is easy; quality and independence from
the judges are the open questions.

| Model | Size | Released | Licence | Why | Priority |
|---|---|---|---|---|---|
| [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | 117B (5.1B active) | 2025-08 | Apache-2.0 | A rerank arm neither judge produced (finding 4). Reasoning cannot be switched off (lowest: "low") | high |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) + [FreedomIntelligence/HuatuoGPT-3-9B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-3-9B) | 9.7B / 9.4B | 2026-02 / 2026-09 | Apache-2.0 | A clean pair: does medical RL post-training help gene-disease relevance judging? HuatuoGPT-3 lifts HealthBench Total from 43.6 to 67.2 but MMLU-Pro (Med) only 81.0 to 85.3 | medium |
| [google/medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it) | 27B | 2025-05 | HAI-DEF (gated) | The standard medical LLM; overlaps with the HuatuoGPT pair, so run one of the two. No MedGemma 2 exists yet | medium |
| [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | 26B (3.8B active) | 2026-03 | Apache-2.0 | Could name genesearch topics offline (4-bit GGUF 14.4 GB). A Google model, so not independent of the Gemini judge | medium |
| [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | 31B | 2026-03 | Apache-2.0 | A strong dense judge that self-hosts on one L4 at 4-bit. Same caveat | medium |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 28B | 2026-08 | Apache-2.0 | Successor to the Qwen3.6-27B we used. Up to 1M context on some providers: one gene's ~1,100 abstracts in one call | medium |
| [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | 21B (3.6B active) | 2025-08 | Apache-2.0 | The cheapest open rerank point; a size ablation beside the 120b | medium |

Low: [lion-ai/MedDecider-4B](https://huggingface.co/lion-ai/MedDecider-4B),
[FreedomIntelligence/HuatuoGPT-3-Grader-8B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-3-Grader-8B) (rubric
grader), [FreedomIntelligence/HuatuoGPT-3-27B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-3-27B),
[Intelligent-Internet/II-Medical-8B-1706](https://huggingface.co/Intelligent-Internet/II-Medical-8B-1706),
[meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) (Meta's 2026 open line; an
independent-family judge), [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (cheap 1M-context
corpus-in-context), [baichuan-inc/Baichuan-M3-235B](https://huggingface.co/baichuan-inc/Baichuan-M3-235B),
[google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it),
[google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) (small enough for CPU topic naming),
[autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B) (one-pass true/false judge).

### Gene NER, entity linking and relation extraction (thread D)

No model on the Hub answers the exact question "is this bare short symbol a gene here?". The useful tools are a
context-aware second tagger, a zero-shot tagger given competing labels, and a gene-vs-family tagger.

| Model | Size | Released | Licence | Why | Priority |
|---|---|---|---|---|---|
| [hunflair/hunflair2-ner](https://huggingface.co/hunflair/hunflair2-ner) | ~110M | 2024-02 | not stated | Context-aware second opinion: where PubTator3 tagged SE, AST, TNM, CAR or ACE as a gene, does HunFlair2 tag a gene? Shares training corpora with PubTator3 (NLM-Gene, GNormPlus), so not fully independent. PubTator still wins gene F1 (86.02 vs 76.75 on tmVar v3) | high |
| [Ihor/gliner-biomed-large-v1.0](https://huggingface.co/Ihor/gliner-biomed-large-v1.0) | ~450M | 2025-02 | Apache-2.0 | Zero-shot: offer "gene", "lab test", "staging system", "chemical element", "cell therapy" and see which label wins. Cased, so CAR ≠ car | medium |
| [Glasgow-AI4BioMed/bioner_gnormplus](https://huggingface.co/Glasgow-AI4BioMed/bioner_gnormplus) | 109M | 2025-07 | MIT | Tags gene families apart from genes (the "cathepsins" problem). Family precision is about 0.50, so a plural rule may do as well | medium |
| [ktiwari-el0303/re-gene-disease](https://huggingface.co/ktiwari-el0303/re-gene-disease) | 333M | 2026-09 | MIT | Gene-disease relation extraction (18 predicates) on sentences: a feature or filter for the main task. Trained on LLM silver labels, one author, custom loader | medium |

Low: [hunflair/biosyn-sapbert-bc2gn](https://huggingface.co/hunflair/biosyn-sapbert-bc2gn),
[Dash00/BeLink-reranker-nlm-gene-8B](https://huggingface.co/Dash00/BeLink-reranker-nlm-gene-8B) (trained on NLM-Gene),
[cambridgeltl/SapBERT-from-PubMedBERT-fulltext](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext)
and [RenaissanceComputing/SapBERT-PubMedBERT-BabelSynonyms-v2025-01-23](https://huggingface.co/RenaissanceComputing/SapBERT-PubMedBERT-BabelSynonyms-v2025-01-23)
(ignore context, so cannot settle ACE vs ace),
[lingbionlp/AIONER-0415](https://huggingface.co/lingbionlp/AIONER-0415) and
[steventango/GNorm2-docker](https://huggingface.co/steventango/GNorm2-docker) (PubTator3's own NER and normaliser:
would show which stage makes a bad tag),
[OpenMed/OpenMed-NER-GenomeDetect-SuperClinical-434M](https://huggingface.co/OpenMed/OpenMed-NER-GenomeDetect-SuperClinical-434M),
[OpenMed/OpenMed-ZeroShot-NER-Protein-Large-459M](https://huggingface.co/OpenMed/OpenMed-ZeroShot-NER-Protein-Large-459M),
[microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL](https://huggingface.co/microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL)
(links a mention in context: AST to a lab test, TNM to the staging system),
[Glasgow-AI4BioMed/bioner_medmentions_st21pv](https://huggingface.co/Glasgow-AI4BioMed/bioner_medmentions_st21pv),
[thealper2/flan-t5-base-medal-abbreviation-expansion](https://huggingface.co/thealper2/flan-t5-base-medal-abbreviation-expansion),
[anthonyyazdaniml/umls-agent-qwen3.5-4b-thinking](https://huggingface.co/anthonyyazdaniml/umls-agent-qwen3.5-4b-thinking),
[pritamdeka/S1-PubMedBERT](https://huggingface.co/pritamdeka/S1-PubMedBERT).

### Gene- and protein-level embeddings (a stretch for this project)

**Start with data, not models.** [BoevaLab/Gene-Embedding-Hub-dataset](https://huggingface.co/datasets/BoevaLab/Gene-Embedding-Hub-dataset)
(2026-07, no licence at repo level) holds 68 precomputed gene-embedding matrices aligned to 20,097 protein-coding
genes: protein models (ESM2, ProtT5), single-cell models (Geneformer, scGPT, UCE, scFoundation, scPRINT), text
(GenePT, BioConceptVec), networks (STRING), GO, DepMap, GTEx and DNA models. So "does the literature map agree with
sequence, expression and curated knowledge?" costs no GPU: average each gene's MedEmbed paper vectors into a
literature centroid, then compare neighbourhoods (kNN overlap, CKA). For thread D, a gene whose literature neighbours
disagree with all its biology neighbours is a mis-tag candidate. Prior work to cite: PRISME (arXiv 2507.07367) found
that omics, literature and knowledge-graph gene embeddings carry largely non-overlapping signal.

Models matter only for axes the hub lacks:

| Model | Size | Released | Licence | Why | Priority |
|---|---|---|---|---|---|
| [westlake-repl/ProTrek_650M](https://huggingface.co/westlake-repl/ProTrek_650M) | 650M | 2024-05 | MIT | Protein and text in one space, so each (paper, tagged gene) pair gets a score: cos(abstract, the gene's protein). Bare-symbol mis-tags should score low. Compare first with the free baseline: the abstract's distance to the gene's own literature centroid | medium |
| [biohub/ESMC-6B](https://huggingface.co/biohub/ESMC-6B) | 6.4B (300M, 600M siblings) | 2026-05 | MIT + notices | The newest protein model; the hub only has ESM2-650M and older. One vector per gene by mean-pooling the canonical protein. ~1-2 L4-h for 20k proteins (estimate). No benchmark numbers published yet | medium |
| [mims-harvard/PROTON](https://huggingface.co/mims-harvard/PROTON) | 578M (card) | 2025-11 | MIT | Precomputed 512-d knowledge-graph vectors for 147k nodes, genes included (link prediction AUROC 0.91). Part of its graph comes from literature, so it is not fully independent | medium |
| [Bitbol-Lab/ProteomeLM-L](https://huggingface.co/Bitbol-Lab/ProteomeLM-L) | 328M | 2025-08 | none on HF | Protein vectors in the context of the whole proteome: interaction context from sequence alone, a cleaner contrast with literature than STRING (which includes text-mining) | medium |
| [biolm/human-proteome-esmc-embeddings](https://huggingface.co/datasets/biolm/human-proteome-esmc-embeddings) (dataset) | n/a | 2025-10 | CC-BY-4.0 | Precomputed 2024 ESM C vectors for every human protein isoform: an ESM C axis with no GPU | medium |

Low: [ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer),
[facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D),
[Rostlab/prot_t5_xl_half_uniref50-enc](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc),
[westlake-repl/SaProt_650M_AF2](https://huggingface.co/westlake-repl/SaProt_650M_AF2),
[genbio-ai/GB.Cell-100M](https://huggingface.co/genbio-ai/GB.Cell-100M), [tdc/scGPT](https://huggingface.co/tdc/scGPT),
[tahoebio/Tahoe-x1](https://huggingface.co/tahoebio/Tahoe-x1) (mostly already in the hub as vectors); datasets
[jghedley/gene-embeddings-mlgenx2026-v1](https://huggingface.co/datasets/jghedley/gene-embeddings-mlgenx2026-v1)
(gene summaries embedded with Qwen3-Embedding-8B, KaLM-Gemma3-12B and llama-embed-nemotron-8b, all models in our
suite), [genbio-ai/foundation-models-perturbation](https://huggingface.co/datasets/genbio-ai/foundation-models-perturbation)
(non-commercial) and [anushaggs/Mahi_gene_embeddings](https://huggingface.co/datasets/anushaggs/Mahi_gene_embeddings).
Dropped: arcinstitute/SE-600M (its gene table is ESM2), a scELMo pickle (GenePT, already in the hub), C2S-Scale (no
gene vector) and Evo 2 (no realistic gene-level output). Not on the Hub: TranscriptFormer, TxGNN, an open ESM3-large.

## Looked for, not found

- No successor to MedCPT, BMRetriever, BiCA, MedEmbed or BioLORD; no SPECTER3.
- No Qwen3.5 or 3.6 text embedder (Qwen's 2026 embedder is the multimodal Qwen3-VL-Embedding, weaker on text); no
  jina-embeddings-v5 base or large; no new text-only BGE, GTE, nomic, mxbai or Snowflake embedder.
- No MedGemma 2 and no Gemma-4-based MedGemma.
- No open model trained for gene-disease retrieval or literature-based discovery.
- No open reranker trained on BioASQ or TREC biomedical data.

Besides the four gene/protein drops above, five entries were dropped by the verifiers: bioelx-nlp/bioelx_retriever (cross-lingual, off-target),
AleksanderObuchowski/medembed-small-onnx (a format copy of MedEmbed-small whose card documents the wrong pooling),
endless-frontier/Fx-Bio, dmis-lab/bern2-ner (stale since 2021) and OpenMed/GLiNER2.5-BioNER-Disease-Base-v1
(off-target).

## Licences to watch

Non-commercial (fine for the paper, not for a product): jina-reranker-v3.5, naver/splade-v3,
Reason-ModernColBERT, ContextualAI rerankers, ChEmbed, xprovence (no derivatives), S1-PubMedBERT. NVIDIA's OpenMDW-1.1
allows commercial use. MedGemma needs the HAI-DEF terms accepted. Several small repos state no licence at all
(BioHiCL-Base, edel_po, HunFlair2, laconic).
