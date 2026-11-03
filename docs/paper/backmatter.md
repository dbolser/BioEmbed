# Back matter

*Draft, 2026-11-03. Same conventions as `methods.md`: each number is
followed by an HTML comment naming its source; `TODO(check)` marks a fact or
decision not recorded in the repo; `TODO(ref)` marks a work with no
`refs.bib` entry.*

## Data and code availability

**TODO(check): Dan to decide.** Everything in this section is a proposal,
not an agreed release plan. Paths are where each item lives in the repo
today; "tracked" means it is in git, "local" means it is on disk but
git-ignored.

**Proposed text.** Code, the answer keys, the judges' grades and the
fine-tuned models are released; PubMed abstract texts are not (NLM and
publisher terms). Each released file lists PMIDs, so the texts can be
fetched from PubMed. Code is cited at commit
`67e649c6b8379613ec2dfa2681c7d4a29cf546ec` <!-- git rev-parse HEAD on main, 2026-11-03 -->
of <https://github.com/dbolser/BioEmbed> (public) <!-- gh repo view: visibility PUBLIC -->.
When this was written the working tree held uncommitted edits, so the commit
to cite is the one that holds the final paper: TODO(check). An archived copy
with a DOI (e.g. Zenodo): TODO(check).

| Item | Proposal | Where it is now |
|---|---|---|
| Scripts (builders, runners, statistics, figures) | release | `scripts/`, `eval/`, `llm/`, `labeller/` (tracked) |
| Vendored evaluator of *The Embedder's Dilemma* | cite upstream, do not re-host | `vendor/embedders-dilemma/` (local), upstream `github.com/embeddings-benchmark/embedders-dilemma` at commit 1a94652 <!-- git -C vendor/embedders-dilemma remote -v; log -1 --> |
| Queries and PMIDs: test and practice pairs, gene candidate sets, PubTator lookup sets, judging pool | release | `data/haystack/GeneDiseaseHaystack/` (test) and `data/haystack/GeneDiseaseHaystackDev/` (practice): `queries.parquet`, `gene_pmids.parquet`, `lookup.parquet`, `pool.parquet`, `strata.parquet` (test only); all local |
| Curator answer key (qrels) and silver key | release, with the source licences of the four curated Open Targets sets (`methods.md`, Data sources) | `qrels.parquet`, `qrels_clean.parquet` (test only), `silver.parquet` in the same folders (local); the full curated table `data/gold/ot_curated_gene_disease_pmid.parquet` (local) |
| Judge grades, per judge and merged | release | `judgments.parquet` (per judge, with each judge's reason), `judgments_merged.parquet`, `judgments_merged_no-gemini.parquet`, `judgments_merged_no-deepseek.parquet`, `judgments_conflicts.parquet` in the same folders (local). About 1% of the judges' reasons quote 8 or more words verbatim from the abstract: 1.2% of a random 3,000 test judgments (DeepSeek 0.6%, Gemini 1.8%; median reason 93 characters) <!-- ad hoc check 2026-11-03, not in a script: data/haystack/GeneDiseaseHaystack/judgments.parquet sampled with pandas random_state=0; lowercased \w+ tokens; a hit is any shared 8-token run with the PMID's text in docs_gene.parquet or pool_extra_docs.parquet -->, so reasons need screening or dropping before release. Put this check in a script: TODO(check) |
| Practice/test split | release | haystack: the two folders above; small task: batch 0 practice, batches 1–4 test, `data/tasks/GeneDiseaseRetrieval_v2_{0..4}/` (tracked: `queries`, `qrels`, `doc_roles.parquet`, `manifest.json`) |
| Ranked runs | release | `data/haystack/*/runs/*.parquet` (local; 46 test and 58 practice files) <!-- ls data/haystack/*/runs/*.parquet -->; scores and statistics in `results/` (tracked) |
| Fine-tuning data | release PMIDs, anchors and negative PMIDs only (the rows also hold abstract text) | `data/finetune_gd/train.parquet`, `train_silver.parquet`, `valid.parquet`, `no_test_disease/`, `manifest.json` (local) |
| Fine-tuned weights | release on Hugging Face under each base model's licence (Table S1); the MedCPT cross-encoder's base is marked public domain on its card but ships an empty LICENSE file (Table S1), so its licence, and ours, need checking: TODO(check) | `models/` (local): the 10 test models (`bioembed-bge-{base,small}-gd-s{42,43,44}`, `bioembed-bge-{base,small}-gaf`, `bioembed-medcpt-ce-gd-s42`, `bioembed-bge-reranker-base-gd-s42`). Whether to also release the 4 practice-selection models and the `*-ckpt` checkpoints: TODO(check) |
| Raw LLM responses (reranks, judges, small task, memory probe, judge wording check) | release responses with request hashes and the PMIDs shown; strip prompts | `llm/llm_results/` (local, except 12 tracked suite sample files) <!-- git ls-files llm/llm_results -->. Each cached file stores the full request, whose prompt holds abstract text, so prompts must be removed or replaced by PMIDs before release. Parsed rankings are already tracked in `results/llm_v2/` |
| Human grades (Dan and three colleagues) | release only with each grader's consent, graders pseudonymised, after calibration ends: TODO(check) | `labeller/labels_calibration.sqlite` and its backup `labeller/labels_calibration.backup-20261019-1558.sqlite` (local; `labels.sqlite` is empty) <!-- sqlite3 row counts, 2026-11-03: labels_calibration.sqlite labels 350 rows, 4 labellers; labels.sqlite labels and log 0 rows -->. Rows carry the grader's name (first name, or full name for one grader) and a free-text note |
| Abstract texts and document vectors | not released | texts: built from a local PubMed records table outside the repo (`methods.md`, Data sources); per-haystack `docs_gene.parquet` and `pool_extra_docs.parquet` (local); vectors: `data/haystack/*/vectors/` (local) |

**Conflict to resolve before release: TODO(check), Dan to decide.** The
public repo already tracks abstract texts. The corpus files under
`data/tasks/` (GeneDiseaseRetrieval, its five v2 batches and
GOPubMedRetrieval) have `title` and `text` columns. BioMeSHClassification
has title and abstract in one `text` column (2,000 train and 500 test rows)
<!-- data/tasks/BioMeSHClassification/manifest.json: sizes, protocol "text = title + '\n\n' + abstract" -->.
Three tracked suite sample files
(`llm/llm_results/*/…/BioMeSHClassification_default_samples.json`) also
quote abstracts, in `input_text` <!-- git ls-files llm/llm_results -->.
`labeller/items_calibration.parquet` and `labeller/items_demo.parquet` have
`title` and `abstract` columns <!-- git ls-files; column names read 2026-11-03 -->.
R2MEDBiologyPooled holds non-PubMed passages, not abstracts <!-- data/tasks/R2MEDBiologyPooled/corpus: _id values are file paths, titles empty; manifest.json: from R2MED/Biology -->.
`items_calibration.parquet` also holds each item's stratum (curator, silver
or unlabelled) and repeat flag. It has been in the public repo since commit
157de0f (2026-10-17) <!-- git log --diff-filter=A -- labeller/items_calibration.parquet; git branch -r --contains 157de0f: origin/main; gh repo view: PUBLIC -->,
so the calibration design is already public. Removing these files from the
current tree does not remove them from git history.

## Funding

TODO(check): funder, grant number, or "no specific funding". Who paid for
Google Cloud (GPU VMs, Vertex AI) and OpenRouter use is not recorded in the
repo: TODO(check).

## Competing interests

TODO(check): Dan to state employment, affiliation and any interests. Not
recorded in the repo.

## Acknowledgements

TODO(check): three colleagues graded subsets of the calibration sample
<!-- docs/paper/results.md: Judges checked against human graders -->. The
labelling data records two colleagues by first name and one by full name
<!-- labeller/labels_calibration.sqlite: labels.labeller -->. Dan to supply full names
and confirm each one consents to being named.

TODO(check): compute and credits to acknowledge (Google Cloud, OpenRouter).

TODO(check): AI-assistance statement. 112 of the 120 commits up to
`67e649c` carry a `Co-Authored-By: Claude` trailer <!-- git log --format=%B | grep -c "^Co-Authored-By: Claude"; git rev-list --count HEAD -->;
code and drafts were written with Claude Code (Anthropic). Wording, and
whether the journal or bioRxiv asks for it in a set place: TODO(check).

## Author contributions

TODO(check): author list and affiliations. `methods.md` names one author,
D.B. <!-- docs/paper/methods.md: Human calibration -->. Proposed CRediT roles
for D.B. (conceptualisation, methodology, software, investigation, data
curation, formal analysis, writing): TODO(check). Whether the three graders
are authors or acknowledged: TODO(check).

## Supplementary Table S1: models

Every model we ran, with its Hugging Face id or API id, parameter count and
licence. Licences were read from each Hugging Face model card's metadata and
text (Hugging Face API, fetched 2026-11-03), or from the provider's model
listing for API models. Parameter counts come from
`results/embedding_throughput_gpu.csv` where the model was measured there;
otherwise from the Hugging Face API (`safetensors.total`, fetched
2026-11-03), from the model card, or by counting tensors in our local
weights. Models scored in the 12-task suite only from the published table of
*The Embedder's Dilemma* [@elassadi2026embedders] are not listed.

Revisions are given only for embeddinggemma and MedCPT. Our own loading
code passes a revision for embeddinggemma and BMRetriever (mteb's registry
revision); for MedCPT it records one but does not pass it
<!-- eval/run_embedders.py: GEMMA_MIRROR; eval/bmretriever_model.py: from_pretrained(revision=meta.revision); eval/medcpt_model.py: META.revision, _encoder() -->.
Whether mteb's own loaders pin the other models: TODO(check). Pooling,
similarity, prompts, max tokens and precision are not in this table;
`methods.md` (Systems) gives them where they differ from the model card.

*Used in*: H, haystack test run; R, haystack reranker (top 100 for
cross-encoders, top 20 for LLMs); S, small task (v2); B, 12-task suite, our
own run; P, closed-book memory probe; J, judge; F, base of one of our
fine-tunes. A claim number in brackets (C1–C9) marks an arm of that Table 1
claim <!-- results/claims.csv arm_a, arm_b -->; BM25 is also an arm (C2) but
is not a model.

**Dense embedders (published)**

| Model | Used in | Hugging Face id | Parameters | Licence | Reference |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | H S B | `sentence-transformers/all-MiniLM-L6-v2` | 22.7M | Apache-2.0 <!-- params: results/embedding_throughput_gpu.csv; licence: HF card; runs: cost_quality.csv, v2_scores.csv, bio_scores.csv source=ours --> | [@wang2020minilm; @sbert2021allminilm] |
| multilingual-e5-small | H S B | `intfloat/multilingual-e5-small` | 117.7M | MIT <!-- params: results/embedding_throughput_gpu.csv; licence: HF card --> | [@wang2024multilinguale5] |
| bge-small-en-v1.5 | H S B F | `BAAI/bge-small-en-v1.5` | 33.4M | MIT <!-- params: results/embedding_throughput_gpu.csv; licence: HF card --> | [@xiao2024cpack] |
| bge-base-en-v1.5 | H S B F (C5, C7) | `BAAI/bge-base-en-v1.5` | 109.5M | MIT <!-- params: results/embedding_throughput_gpu.csv; licence: HF card; also the dense half of the first pass --> | [@xiao2024cpack] |
| bge-large-en-v1.5 | S B | `BAAI/bge-large-en-v1.5` | 335.1M | MIT <!-- params: results/embedding_throughput_gpu.csv; licence: HF card --> | [@xiao2024cpack] |
| embeddinggemma-300m | H S (C1, C2, C3a) | `google/embeddinggemma-300m` (weights from the byte-identical mirror `unsloth/embeddinggemma-300m`, revision bfa3c84) | 302.9M | Gemma Terms of Use (gated on Hugging Face) <!-- params: HF API safetensors.total; licence: HF card license:gemma, mirror the same; mirror: eval/run_embedders.py GEMMA_MIRROR --> | [@vera2025embeddinggemma] |
| snowflake-arctic-embed-l-v2.0 | S B | `Snowflake/snowflake-arctic-embed-l-v2.0` | 567.8M | Apache-2.0 <!-- params: results/embedding_throughput_gpu.csv; licence: HF card --> | [@yu2024arcticembed2] |
| F2LLM-v2-0.6B | S B | `codefuse-ai/F2LLM-v2-0.6B` | 596.0M | Apache-2.0 <!-- params: results/embedding_throughput_gpu.csv; licence: HF card --> | [@zhang2026f2llmv2] |
| F2LLM-v2-1.7B | S B | `codefuse-ai/F2LLM-v2-1.7B` | 1.72B | Apache-2.0 <!-- params: results/embedding_throughput_gpu.csv; licence: HF card --> | [@zhang2026f2llmv2] |
| Qwen3-Embedding-0.6B | H S B | `Qwen/Qwen3-Embedding-0.6B` | 595.8M | Apache-2.0 <!-- params: results/embedding_throughput_gpu.csv; licence: HF card --> | [@zhang2025qwen3embedding] |
| Qwen3-Embedding-4B | S B | `Qwen/Qwen3-Embedding-4B` | 4.02B | Apache-2.0 <!-- params: results/embedding_throughput_gpu.csv; licence: HF card --> | [@zhang2025qwen3embedding] |
| Qwen3-Embedding-8B | S B (C9) | `Qwen/Qwen3-Embedding-8B` | 7.57B | Apache-2.0 <!-- params: results/embedding_throughput_gpu.csv; licence: HF card; C9: results/claims.csv arm_b --> | [@zhang2025qwen3embedding] |
| MedCPT (query and article encoders; dot product, and cosine as a variant) | H S B | `ncbi/MedCPT-Query-Encoder`, `ncbi/MedCPT-Article-Encoder` (query encoder revision d83a36c is recorded in our code but not passed when the weights load; article encoder revision not recorded: TODO(check)) <!-- eval/medcpt_model.py: META.revision d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc; _encoder() calls Transformer(name) with no revision --> | 109.5M each | Public domain (NCBI notice in each repo's LICENSE) <!-- params: results/embedding_throughput_gpu.csv (ncbi/MedCPT); HF API safetensors.total for both encoders; licence: HF card license_name public-domain and LICENSE text; ids: eval/medcpt_model.py --> | [@jin2023medcpt] |
| BiCA-small | H S | `bisectgroup/BiCA-small` | 33.4M | none stated: the card's licence line is commented out as "Unknown" and the repo has no LICENSE file; base is `thenlper/gte-small`. TODO(check) <!-- params: HF API safetensors.total; licence: HF card and file list, fetched 2026-11-03 --> | [@sinha2026bica] |
| BiCA-base | H S (C6, C7) | `bisectgroup/BiCA-base` | 109.5M | MIT <!-- params: HF API safetensors.total; licence: HF card --> | [@sinha2026bica] |
| BMRetriever-410M | H S (C3b) | `BMRetriever/BMRetriever-410M` | 353.8M | MIT <!-- params: HF API safetensors.total; licence: HF card --> | [@xu2024bmretriever] |
| BMRetriever-1B | S | `BMRetriever/BMRetriever-1B` | 908.8M | MIT <!-- params: HF API safetensors.total; licence: HF card --> | [@xu2024bmretriever] |
| BMRetriever-2B | S | `BMRetriever/BMRetriever-2B` | 2.51B | MIT <!-- params: HF API safetensors.total; licence: HF card --> | [@xu2024bmretriever] |
| BMRetriever-7B | S | `BMRetriever/BMRetriever-7B` | 7.11B | MIT <!-- params: HF API safetensors.total; licence: HF card --> | [@xu2024bmretriever] |
| MedEmbed-small-v0.1 | H S B | `abhinand/MedEmbed-small-v0.1` | 33.4M | Apache-2.0 <!-- params: results/embedding_throughput_gpu.csv; licence: HF card --> | [@balachandran2024medembed] |
| MedEmbed-base-v0.1 | H S B | `abhinand/MedEmbed-base-v0.1` | 109.5M | Apache-2.0 <!-- params: results/embedding_throughput_gpu.csv; licence: HF card --> | [@balachandran2024medembed] |
| MedEmbed-large-v0.1 | S B | `abhinand/MedEmbed-large-v0.1` | 335.1M | Apache-2.0 <!-- params: results/embedding_throughput_gpu.csv; licence: HF card --> | [@balachandran2024medembed] |
| PubMedBERT embeddings | H S B | `NeuML/pubmedbert-base-embeddings` | 109.5M | Apache-2.0 <!-- params: results/embedding_throughput_gpu.csv; licence: HF card --> | [@gu2021pubmedbert] |
| BioLORD-2023 | H S B | `FremyCompany/BioLORD-2023` | 109.5M | Other: the author's work is MIT, but the card says users need UMLS and SNOMED CT licences, as the training data come from them <!-- params: results/embedding_throughput_gpu.csv; licence: HF card license_name ihtsdo-and-nlm-licences and License section --> | [@remy2024biolord] |

**Cross-encoder rerankers (published)**

| Model | Used in | Hugging Face id | Parameters | Licence | Reference |
|---|---|---|---|---|---|
| MedCPT Cross-Encoder | R F | `ncbi/MedCPT-Cross-Encoder` | 109.5M | Card metadata says public domain, but the repo's LICENSE file is empty. TODO(check) <!-- params: tensor count of the local safetensors snapshot 75e855e; licence: HF card license_name public-domain; LICENSE blob is empty --> | [@jin2023medcpt] |
| bge-reranker-v2-m3 | R | `BAAI/bge-reranker-v2-m3` | 567.8M | Apache-2.0 <!-- params: HF API safetensors.total; licence: HF card --> | [@chen2024m3; @baai2024rerankerv2m3] |
| bge-reranker-base | R F | `BAAI/bge-reranker-base` | 278.0M | MIT <!-- params: HF API safetensors.total; licence: HF card --> | [@baai2023rerankerbase] |
| Qwen3-Reranker-0.6B | R | `Qwen/Qwen3-Reranker-0.6B` | 595.8M | Apache-2.0 <!-- params: HF API safetensors.total; licence: HF card --> | [@zhang2025qwen3embedding] |
| Qwen3-Reranker-4B | R | `Qwen/Qwen3-Reranker-4B` | 4.02B | Apache-2.0 <!-- params: HF API safetensors.total; licence: HF card --> | [@zhang2025qwen3embedding] |

**LLMs (via API)**

| Model | Used in | API id (route) | Parameters | Licence or terms | Reference |
|---|---|---|---|---|---|
| DeepSeek-V4-Flash | R S P J B | `deepseek/deepseek-v4-flash` (OpenRouter; pinned to provider Baidu, no fallback, for R, S, P and J; the suite run B was not pinned) <!-- pin: llm/judge_pool.py, rerank_haystack_llm.py, run_gdv2_llm.py, closed_book_probe.py; none in llm/run_bio_llm.py, llm_common.py, tasks_bio.py; pin first added in 2d0d111 (2026-10-15), suite results committed 2026-09-12 (4ad513c, d2cb569) -->; weights `deepseek-ai/DeepSeek-V4-Flash` | 284B total, 13B active (MoE) | Weights MIT; API use under OpenRouter's terms: TODO(check) <!-- ids: llm/judge_pool.py JUDGES, llm/rerank_haystack_llm.py MODELS; OpenRouter /api/v1/models hugging_face_id; params and licence: HF card --> | TODO(ref) |
| Gemini 3.1 Flash-Lite | R S P J B (C1, C3a, C3b, C8) | `google/gemini-3.1-flash-lite` (OpenRouter) | not disclosed | Proprietary, no released weights; API terms: TODO(check) <!-- ids: llm/judge_pool.py JUDGES; OpenRouter /api/v1/models: no hugging_face_id --> | TODO(ref) |
| Gemini 3.1 Pro (preview) | B (C9) | `google/gemini-3.1-pro-preview` (Vertex AI, OpenAI-compatible endpoint) | not disclosed | Proprietary, no released weights; Google Cloud terms: TODO(check) <!-- id: llm/.env.vertex model; C9: results/claims.csv arm_a; OpenRouter /api/v1/models: no hugging_face_id --> | TODO(ref) |
| Qwen3.6-35B-A3B | B (our run of 4 of its 11 tasks) | `qwen/qwen3.6-35b-a3b`; weights `Qwen/Qwen3.6-35B-A3B` | 35B total, 3B active (MoE) | Apache-2.0 <!-- runs: results/bio_scores.csv source=ours (4 rows); params and licence: HF card; route via OpenRouter inferred from commit d2cb569 message, not recorded: TODO(check) --> | TODO(ref) |

Qwen3.6-35B-A3B appears in the suite figure but not in `methods.md`. Keep
it in the paper or drop it: TODO(check).

**Our fine-tunes** (all `bioembed/` names are local; none is on Hugging Face
yet)

| Model | Used in | Local id (base) | Parameters | Proposed licence | Reference |
|---|---|---|---|---|---|
| bge-base-gd, seeds 42, 43, 44 | H (C5, C6) | `bioembed/bge-base-gd-s{42,43,44}` (`BAAI/bge-base-en-v1.5`) | 109.5M | MIT, as base: TODO(check) <!-- params: tensor count of models/bioembed-bge-base-gd-s42; base: models/*/README.md base_model --> | this work |
| bge-small-gd, seeds 42, 43, 44 | H | `bioembed/bge-small-gd-s{42,43,44}` (`BAAI/bge-small-en-v1.5`) | 33.4M | MIT, as base: TODO(check) <!-- params: tensor count of models/bioembed-bge-small-gd-s42; base: README.md base_model --> | this work |
| bge-base-gaf (GO fine-tune) | H S B | `bioembed/bge-base-gaf` (`BAAI/bge-base-en-v1.5`) | 109.5M | MIT, as base: TODO(check) <!-- params: tensor count of models/bioembed-bge-base-gaf; base: README.md base_model --> | this work |
| bge-small-gaf (GO fine-tune) | H S B | `bioembed/bge-small-gaf` (`BAAI/bge-small-en-v1.5`) | 33.4M | MIT, as base: TODO(check) <!-- params: tensor count of models/bioembed-bge-small-gaf; base: README.md base_model --> | this work |
| MedCPT cross-encoder, fine-tuned, seed 42 | R (C8) | `bioembed/medcpt-ce-gd-s42` (`ncbi/MedCPT-Cross-Encoder`) | 109.5M | base: card says public domain, LICENSE empty; our licence: TODO(check) <!-- params: tensor count of models/bioembed-medcpt-ce-gd-s42; base: eval/rerank_haystack.py FINETUNED --> | this work |
| bge-reranker-base, fine-tuned, seed 42 | R | `bioembed/bge-reranker-base-gd-s42` (`BAAI/bge-reranker-base`) | 278.0M | MIT, as base: TODO(check) <!-- params: tensor count of models/bioembed-bge-reranker-base-gd-s42; base: eval/rerank_haystack.py FINETUNED --> | this work |
| Recipe-selection models (curated or curated + silver, 0 or 3 negatives), seed 42 | practice only | `bioembed/bge-small-gd-{train,silver}-n{0,3}-s42` (`BAAI/bge-small-en-v1.5`) | 33.4M | release or not: TODO(check) <!-- models/ directory listing; data/finetune_gd/runs/chosen.json --> | this work |

Not models, so not in the table: BM25 (bm25s 0.3.13) and reciprocal-rank
fusion, both run on CPU (`methods.md`, Systems).
