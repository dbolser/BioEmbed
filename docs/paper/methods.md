# Methods

*Draft, 2026-10-16. Citations are Pandoc keys into `refs.bib`. Each number is
followed by an HTML comment naming its source file. `TODO(result)` marks work
not yet run; `TODO(ref)` marks a work with no `refs.bib` entry;
`TODO(prereg)` marks a rule not yet registered in `docs/PREREG.md`;
`TODO(check)` marks a fact not yet recorded in the repo.*

## Data sources

Inputs are local, read-only snapshots, recorded in each set's manifest.

- **PubMed** 2026 baseline only (pubmed26n0001–1334, no update files) <!-- /outsee5/home/danbolser/Work/IUK-IPF/data/sources/pubmed/manifest.json: archives_seen, results -->: 28,332,361 abstracts with distinct PMIDs <!-- data/haystack/GeneDiseaseHaystack/manifest.json: stats.local_abstracts -->. Texts come from a local records table that a separate pipeline builds from the XML <!-- manifests: provenance.corpus, /outsee5/home/danbolser/Work/IUK-IPF/data/embeddings-catalogue/records/*.parquet -->. The abstract is the record text after its "Abstract:" marker; a PMID with several records keeps the one with the lowest record hash <!-- scripts/build_tier_c_haystack.py: docs_gene query -->. Which XML fields are used and how structured-abstract sections are joined: TODO(check). Licence: NLM terms; abstracts may carry publisher copyright. <!-- licence not recorded in repo: verify -->
- **Open Targets Platform 26.06** [@buniello2025opentargets]: target and disease indexes and nine literature evidence sets (four curated, five used only as guards; below). Only these were downloaded <!-- /outsee5/home/danbolser/Work/IUK-IPF/data/sources/opentargets/26.06/; scripts/build_tier_c_gene_disease_v2.py: CURATED_SOURCES, LINK_SOURCES -->. Licence: CC0 1.0. <!-- licence not recorded in repo: verify --> The four curated sources keep their own terms: Orphanet and UniProt CC BY 4.0 <!-- licence not recorded in repo: verify -->; PanelApp and Gene2Phenotype: TODO(check).
- **PubTator3** [@wei2024pubtator3], release 2026-10-06: gene and disease tags per PMID (per its README, abstract mentions plus curated resources such as gene2pubmed). Licence: public domain. <!-- /data/public/PubTator3/2026-10-06/README.txt -->
- **HGNC** complete set, 2026-10-14: Ensembl to NCBI Gene, alias and previous symbols. Licence: free use, no restrictions. <!-- licence not recorded in repo: verify -->
- **Mondo** [@vasilevsky2025mondo] SSSOM mappings, 2026-10-14. Licence: CC BY 4.0. <!-- licence not recorded in repo: verify -->
- **ClinGen** gene–disease validity [@strande2017clingen], file of 2026-10-06, for strata only. Licence: CC0 1.0. <!-- date: file header of /outsee5/home/danbolser/Work/IUK-IPF/data/sources/clingen/gene-validity-2026-10-06.csv; licence not recorded: verify -->

Model licences: see the per-model table, TODO(check). **Availability**
(what is released, e.g. PMIDs, qrels and build scripts but no abstract
texts; fine-tuned weights under their base's licence; raw LLM responses; the
git commit): TODO(check).

A document is `{title}\n\n{abstract}`. A query is `{SYMBOL} ({approved gene
name}) in {disease name}` with Open Targets names, e.g. "HFE (homeostatic iron
regulator) in hereditary hemochromatosis". A *pair* is a gene–disease query;
an *item* is a (pair, PMID).

## Gold standard

**Curator citations.** A gold link is a (gene, disease, PMID) triple from one
of four curated literature sources in Open Targets
[@ochoa2021opentargets; @buniello2025opentargets]: Orphanet
[@rath2012orphanet], Gene2Phenotype [@thormann2019g2p], Genomics England
PanelApp [@martin2019panelapp] and UniProt [@uniprot2025]
(`scripts/build_tier_c_gene_disease.py`, which writes
`data/gold/ot_curated_gene_disease_pmid.parquet`). Every confidence tier is
kept, and only numeric PMIDs. Evidence rows with citations: Gene2Phenotype
definitive 2,833, strong 1,065, moderate 338, limited 674; Genomics England
green 29,309, amber 4,278; UniProt high 6,196, medium 496; Orphanet assessed
6,889, not yet assessed 5 <!-- /outsee5/home/danbolser/Work/IUK-IPF/data/sources/opentargets/26.06/evidence_*: confidence, rows with literature -->.
They cite 38,392 PMIDs for 17,723 pairs and 5,402 genes; 17,352 pairs and
34,839 PMIDs have a local abstract <!-- data/finetune_gd/manifest.json: funnel_curated.curated, .local_abstract -->.
No text mining, LLM output or synthetic text enters the gold.

**Silver guard.** An abstract that Europe PMC text mining links to a pair,
but curators do not cite, may still be relevant. On the haystack the
*condensed* curator-only score removes such PMIDs from a run and closes up
the ranks before scoring [@sakai2007bpref]; the *raw* score counts them as
non-relevant, and raw is the score that picked BM25 and the fine-tune recipe
<!-- scripts/score_haystack.py; scripts/ft_select.py: METRIC -->. In the test
haystack 18,669 of 19,137 silver links are not gold <!-- data/haystack/GeneDiseaseHaystack/silver.parquet, qrels.parquet -->.
The small-task and fine-tuning guards never use them as negatives, and are
wider: curated links plus those from Europe PMC, EVA, Cancer Gene Census,
gene burden and GWAS credible sets <!-- scripts/build_tier_c_gene_disease_v2.py: LINK_SOURCES -->,
to the query disease or any related disease (defined below).

**Clean gold** (a check): gene-tagged, cited for fewer than 10 genes, and
rated green if Genomics England cites it. 1,916 of the 2,431 test golds are clean <!-- data/haystack/GeneDiseaseHaystack/qrels_clean.parquet -->.
Only Genomics England's lower tier is dropped; Gene2Phenotype "limited" and
UniProt "medium" stay.

**Exclusion registry** (`scripts/build_exclusion_registry.py`): every
identifier in any evaluation set, namely 1,109 genes, 1,148 pairs, 856
diseases, 29,074 task, gold and silver PMIDs, and 1,836,144 haystack candidate
PMIDs, plus GO and UniProt ids for the GO tasks <!-- data/exclusion/manifest.json: distinct_ids_by_kind -->.
The v2 builder uses it to keep genes disjoint. The practice haystack predates
it and drops the test genes directly <!-- scripts/build_tier_c_haystack.py: --exclude-genes-from; mtimes GeneDiseaseHaystackDev/manifest.json 2026-10-15 18:25 vs data/exclusion/registry.parquet 19:47 -->.
The fine-tuning builder drops every gene, pair, PMID and haystack document in
it, but not its diseases (below).

## Test sets

*Practice* is the gene-disjoint development split: the practice haystack,
v2 batch 0 and practice calibration items. It is never trained on and is used
only for the choices named below. Folders: test haystack
`data/haystack/GeneDiseaseHaystack`, practice haystack
`data/haystack/GeneDiseaseHaystackDev`, small task
`data/tasks/GeneDiseaseRetrieval_v2_{0..4}` (0 is practice). Genes are
disjoint across all of them; diseases and documents are not (below).

**Haystack (test).** Built on 2026-10-14 and frozen <!-- docs/PREREG.md: Rules -->
(`scripts/build_tier_c_haystack.py`, seed 42): 500 pairs sampled from the
17,242 curated pairs with 1–50 local golds and a gene mapped to NCBI Gene
<!-- data/haystack/GeneDiseaseHaystack/manifest.json: protocol.selection -->.
They cover 473 genes and 420 diseases (334 MONDO, 44 Orphanet, 22 HP, 20 EFO;
by pair 386, 47, 44, 23) <!-- data/haystack/GeneDiseaseHaystack/queries.parquet -->,
with 2,431 gold links to 2,222 PMIDs, median 3 per pair, range 1–41 <!-- data/haystack/GeneDiseaseHaystack/qrels.parquet -->.
A pair's candidates are the local abstracts PubTator tags with its gene's
human NCBI Gene ID (Open Targets to HGNC to NCBI Gene): median 1,087.5 per
pair, 1,699,080 distinct <!-- data/haystack/GeneDiseaseHaystack/manifest.json: provenance.gene_map, stats.gene_pmids -->.
This filter keeps 87.7% of gold per pair on average <!-- same manifest: stats.gene_pmids.recall -->,
a ceiling shared by every ranker. Of the 372 golds it loses, 76 have no gene
tag and 296 are tagged only with other genes; 128 of those carry a non-human
gene tag, often an ortholog in a model-organism paper, which the rubric would
grade 2 <!-- data/haystack/GeneDiseaseHaystack/qrels_clean.parquet: gene_tag; non-human = gene2pubtator3 id not an HGNC entrez_id, counted 2026-10-16 -->.
Each system returns its top 1,000 candidates per pair.

Strata are labels only (`scripts/haystack_strata.py`): disease ontology; main
curation source (Genomics England 280, Orphanet 82, Gene2Phenotype 77,
UniProt 61); gold count (1: 151; 2–3: 156; 4–9: 130; 10+: 63); gene-set size
quartile; ClinGen class, matched for 57 pairs; latest gold year (2025 or
later: 11 pairs) <!-- data/haystack/GeneDiseaseHaystack/strata.parquet -->.

**Practice haystack.** The same build with seed 43: 200 pairs from 14,066,
after removing 633 genes <!-- data/haystack/GeneDiseaseHaystackDev/manifest.json: protocol -->:
all test genes, and genes PubTator tags in over 20,000 local abstracts (6% of
test pairs exceed that <!-- docs/PREREG.md: change log 2026-10-15 -->). It has
197 genes, 187 diseases, 1,009 gold links <!-- data/haystack/GeneDiseaseHaystackDev/queries.parquet, qrels.parquet -->,
a median of 816 candidates per pair (326,337 distinct) and a filter recall of
88.0% <!-- data/haystack/GeneDiseaseHaystackDev/manifest.json: stats -->.
It shares 32 diseases with the test haystack, and 241 of its gold PMIDs are
test candidates <!-- queries.parquet, qrels.parquet, gene_pmids.parquet of both haystacks -->.
Chosen on it: the BM25 settings, the fine-tune recipe and every haystack
"best" system. Fixed in advance, not tuned: rerank depths, model instructions
(mteb's default) and the fine-tune checkpoint (end of epoch). Each choice is
then run once on the test set, but the ten dense test runs predate any
practice dense run (see Disclosures).

**Small task (v2).** Five batches of 75 pairs, each with its own 300-document
corpus, 375 pairs in all (`scripts/build_tier_c_gene_disease_v2.py`, seeds
42–46) <!-- data/tasks/GeneDiseaseRetrieval_v2_{0..4}/manifest.json: sizes -->.
A candidate pair has exactly one curated PMID with a local abstract (5,338 of
17,352 pairs), tagged with the gene and cited for fewer than 10 genes; a
MONDO, Orphanet or EFO disease (no HP phenotypes); a gene used by neither
haystack nor v1 (734 genes excluded); enough same-gene negatives (938 pairs
left); and at least one same-disease negative. 640 pairs qualify <!-- data/tasks/GeneDiseaseRetrieval_v2_0/manifest.json: exclusion_counts.candidate_funnel -->.
So the small task covers only single-gold pairs whose gene is curated for
other, unrelated diseases; C3 holds for that population only.
Each query brings four documents:

- *gold*: the pair's one curated PMID;
- *same gene* (×2): a PMID curated for the gene with a disease unrelated to
  the query disease *d*, tagged with the gene, whose text names neither *d*,
  an exact synonym of *d* (≥ 5 characters) nor a twin of *d*;
- *same disease* (×1): a PMID curated for *d* with another gene, not tagged
  with the query gene and not linked to it by any of the nine evidence sets.

Two Open Targets diseases are *twins* if they share a concept
cross-reference (one of 21 prefixes, e.g. OMIM, Orphanet, UMLS, MeSH; not
ICD) or one lists the other's id among its cross-references. Disease *o* is
*related* to *d* if (ontology) *o* is reached from *d* by twin links plus
only ancestor steps, or plus only descendant steps; (category) *o* is in the
family (ancestors, descendants), or a twin of it, of a disease that shares an
ICD-9, ICD-10 or ICD-O code with *d* or a twin; (names) every stemmed word of
one name, minus modifiers (type, autosomal, congenital…) and numbers, lies
inside a word of the other; or *o* is missing from the Open Targets disease
index <!-- data/tasks/GeneDiseaseRetrieval_v2_0/manifest.json: protocol.related; scripts/build_tier_c_gene_disease_v2.py: CONCEPT_XREFS, CATEGORY_XREFS, NAME_MODIFIERS -->.
Prefix and modifier lists: supplementary table, TODO(check). No corpus
document may be linked, by any of the nine sets, to a query's gene with a
disease related to that query's, unless it is that query's gold; this is
asserted for every case. Pairs are accepted greedily in seeded order: one per
gene overall, one per disease within a batch. Batch 0 is for
practice; batches 1–4 (300 pairs) are the test set <!-- docs/PREREG.md: change log 2026-10-15 22:45 -->.
Diseases and documents can recur across batches: the 375 pairs use 286
diseases, 22 batch-0 diseases recur in test batches, 47 PMIDs appear in more
than one batch (15 shared by batch 0 and test), and in 17 cases one batch's
gold is another's negative <!-- data/tasks/GeneDiseaseRetrieval_v2_{0..4}/doc_roles.parquet -->. The score is hit@1
over all 300 documents; chance is 1/300, or 1/4 for a method that finds the
four documents sharing the gene or disease.

**Small task v1 (appendix only).** 75 pairs, 300 documents, 78 gold links;
three negatives per query, linked to its gene with a disease other than the
query disease, its ancestors or descendants <!-- data/tasks/GeneDiseaseRetrieval/manifest.json -->.
Its guard covered only the exact pair, so a negative could be linked to the
gene with a related disease; v2 fixes this. v1 shares 7 genes with the test
haystack and 4 with the practice haystack <!-- data/exclusion/registry.parquet: gene ids by source -->.

## Graded answer key

**Pool** (`llm/judge_pool.py`), per test pair: the top 10 of every run (bar
the MedCPT run made before its pooling fix), all curator-cited papers, and up
to 5 PubTator lookup hits per disease tier, a seeded hash sample of the tier's
non-gold hits that does not move when runs are added, each weighted by the
inverse of its chance of being drawn. The test pool came from the 24 runs
that existed on 2026-10-16: BM25 v1.3 (full query, k1 = 1.5, b = 0.75), tuned
BM25, ten dense runs (the eight below plus bge-small-gaf and bge-base-gaf),
ten fusions of tuned BM25 with each dense run, and the two LLM reranks of
tuned BM25 <!-- data/haystack/GeneDiseaseHaystack/runs/ -->. It held 26,614
items <!-- data/haystack/GeneDiseaseHaystack/pool.parquet: rows -->. Each
run's share of the pool: supplementary table, TODO(check). Runs added later
(MedCPT, BiCA, embeddinggemma, BMRetriever, BioLORD, the fine-tunes, the
cross-encoders and the C1 reranks) are graded under the add-later rule
below: TODO(result).

**Judges.** DeepSeek-V4-Flash (via Baidu only, no provider fallback) and
Gemini 3.1 Flash-Lite, two model families via OpenRouter, grade the pooled
items at temperature 0 with reasoning off, at most 2,048 output tokens; a
response with no usable grade is retried once <!-- llm/judge_pool.py: JUDGES, MAX_TOKENS, send_all -->.
A request holds one query and up to 10 abstracts, labelled A, B, C… in an
order shuffled per judge and query, with the gene symbol, name and HGNC
aliases (minus those of 2 characters or fewer, any approved or shared symbol,
and any disease name), the disease name and exact synonyms, and the
pre-registered rubric verbatim (the script stops if it changes). Practice
requests made on 2026-10-15 (607 items) used one order and grouping per query
for both judges <!-- llm/judge_pool.py: docstring; llm/llm_results/judge/GeneDiseaseHaystackDev/*/: seed 42 -->:

- **2**: the abstract reports evidence that this gene is involved in this
  disease (or a synonym, subtype, or the same phenotype in a model organism
  of the gene);
- **1**: the abstract discusses both, but the link is not its evidence (a
  review mention, a candidate in a screen with no result, or a related
  disease);
- **0**: otherwise.

PREREG still lists this rubric, and the relevance threshold, as awaiting
sign-off <!-- docs/PREREG.md: Waiting on Dan -->. Judges never see rank,
system, or curator or text-mining status, and give a reason of at most 25
words before each grade. Curator-cited papers are graded like any other.

**Merge rule.** One grade per item: the mean of the two grades, rounded down
(1 and 2 give 1; 0 and 2 give 1; 0 and 1 give 0). Items without both grades
are left out of the merged key and count as unjudged.

**No self-grading.** A run reranked by an LLM is never graded by a judge of
its family: a comparison involving it uses only the other judge's grades for
both sides, or curator-only scores if no judge is left. Runs added later are
graded in requests filled up to 10 with already-graded items, never on their
own; an item keeps its first grade.

**Grading done.** Test: both judges graded the pool on 2026-10-16,
20:58–21:14 BST <!-- llm/llm_results/judge/GeneDiseaseHaystack/*/*.json: created -->:
DeepSeek 26,614 items, Gemini 26,610, so 26,610 merged (grade 0: 16,148;
1: 4,453; 2: 6,009) <!-- data/haystack/GeneDiseaseHaystack/judgments.parquet, judgments_merged.parquet -->.
Practice: 7,190 items over all 200 pairs, 7,189 merged <!-- data/haystack/GeneDiseaseHaystackDev/judgments.parquet, judgments_merged.parquet -->.
The 4 test items and 1 practice item without a Gemini grade are all non-gold.

**Human calibration.** One author (D.B.), who also wrote the rubric, grades a
blind sample in a local labelling app (`labeller/`) with the same rubric: 20
test pairs with 10 abstracts each, 200 items, 30 of them shown twice to
measure self-consistency <!-- labeller/items_calibration.parquet -->. There is
one human grader. Items are spread over curation label (gold, silver,
unlabelled) by best pooled rank (1–3, 4–10, 11 or unranked) as evenly as each
pair's pool allows (7–30 per stratum) <!-- labeller/items_calibration.parquet: stratum -->;
the stratum is hidden. For reweighting, each item counts its stratum's pool
size over its sample size; the pool counts are printed by
`labeller/make_calibration_items.py`, not saved: TODO(check). Choices of
rubric, prompt or judges may be tuned on practice items only; the test items
were built after the protocol was committed
<!-- mtime of labeller/items_calibration.parquet 2026-10-16 20:59 vs commit d11cc65 20:58 -->.
Agreement with the judges, by the statistics used between judges (exact
agreement, linear and quadratic weighted kappa, kappa at grade ≥ 1 and at
grade 2, 95% intervals from resampling pairs) <!-- llm/judge_pool.py: agreement -->:
TODO(result).

## Systems

**PubTator lookup ($0).** Unranked sets: candidates also tagged with the
disease (Open Targets MeSH/OMIM cross-references) at four tiers: exact, plus
descendants, plus parents, plus ancestors. Scored by recall and set size.

**BM25** [@robertson2009bm25] in bm25s 0.3.13 [@lu2024bm25s] <!-- uv.lock -->,
English stopwords, Snowball stemming, one index over the gene-tagged
abstracts. Three query formats (the full query; `{SYMBOL} {disease}`; symbol,
HGNC aliases, disease name and synonyms) × k1 ∈ {0.9, 1.2, 1.5} × b ∈ {0.4,
0.75} gave 18 practice runs; the best by curator-only nDCG@10 (raw) was
`{SYMBOL} {disease}`, k1 = 0.9, b = 0.75 (0.3689, against 0.3685 for
k1 = 1.2); on condensed nDCG@10, k1 = 1.5, b = 0.75 would have won (0.4210) <!-- results/haystack/GeneDiseaseHaystackDev/scores.csv -->.
It was run once on the test set.

**Dense embedders.** General: all-MiniLM-L6-v2, multilingual-e5-small,
bge-small/base/large-en-v1.5, embeddinggemma-300m,
snowflake-arctic-embed-l-v2.0, F2LLM-v2 (0.6B, 1.7B) and Qwen3-Embedding
(0.6B, 4B, 8B) [@zhang2025qwen3embedding] <!-- TODO(ref): MiniLM, e5, bge, embeddinggemma, snowflake-arctic, F2LLM -->.
Biomedical or PubMed-trained: MedCPT [@jin2023medcpt], BiCA-small/base
<!-- TODO(ref): BiCA, arXiv 2511.08029 -->, BMRetriever (410M–7B)
[@xu2024bmretriever], MedEmbed-small/base/large [@balachandran2024medembed],
PubMedBERT embeddings [@gu2021pubmedbert] and BioLORD-2023
[@remy2024biolord]. Ours, as observations: the GO fine-tunes bge-small-gaf
and bge-base-gaf. All 27 configurations (24 published models, our 2 GO
fine-tunes, plus MedCPT by cosine) ran on the small task <!-- results/v2_scores.csv -->.
The haystack runs a chosen subset, not every model up to 0.6B parameters.
MiniLM, e5-small, bge-small/base, MedEmbed-small/base, PubMedBERT and
Qwen3-Embedding-0.6B have test runs (also bge-small-gaf and bge-base-gaf),
picked when those runs were made on 2026-10-15 <!-- data/haystack/GeneDiseaseHaystack/runs/ -->.
MedCPT, MedCPT by cosine, BiCA-small/base, embeddinggemma-300m,
BMRetriever-410M and BioLORD-2023 are queued, practice first, added on
2026-10-16 <!-- scripts/gcp_jobs/hs-q1.txt, hs-q2.txt, hs-q3.txt; commit cd52d2c -->: TODO(result).
bge-large, MedEmbed-large, snowflake-arctic-embed-l-v2.0 and F2LLM-v2-0.6B
(all ≤ 0.6B) <!-- results/embedding_throughput_gpu.csv: params --> have no
haystack run, and no reason for leaving them out is recorded: TODO(check).
Bigger models are left to reranking (Qwen3-Embedding-8B would need about 900
L4-hours on the haystack) <!-- RESULTS.md: haystack notes -->.

Each model is loaded as its card specifies, except as noted: mteb's loader
and prompts for registered models; otherwise the model's own
sentence-transformers configuration, and no silent mean pooling. MedCPT uses
its query encoder (64 tokens) and article encoder (512), [CLS] pooling and
dot product, with an article encoded as one `title\n\nabstract` segment, not
the card's [title, abstract] pair <!-- eval/medcpt_model.py -->. BMRetriever
uses the authors' templates, last-token pooling and dot product, but mteb's
task instruction, `title\n\nabstract` passages, and bfloat16 rather than the
authors' fp16 autocast <!-- eval/bmretriever_model.py -->. BiCA uses mean
pooling and cosine; embeddinggemma the card prompts. Instruction models
(Qwen3-Embedding, F2LLM, BMRetriever) get mteb's default instruction,
"Retrieve text based on user query.", as our tasks set none; it was not
tuned <!-- tests/test_pooling.py: BMR_INSTRUCTION; eval/bmretriever_model.py -->.
Unit tests check pooling for bge-small, MedEmbed-small, e5-small, MiniLM,
Qwen3-Embedding-0.6B, F2LLM-v2-0.6B, BiCA and embeddinggemma, and vectors
against the card's code for BiCA, embeddinggemma, all four BMRetriever sizes
and MedCPT's query encoder (its article encoder: pooling and length only).
The other models are untested <!-- tests/test_pooling.py -->. An earlier
mean-pooled MedCPT run is kept out of pools and fusions. Only MedCPT's and
embeddinggemma's weights are pinned to a revision in our code <!-- eval/medcpt_model.py: META.revision; eval/run_embedders.py: GEMMA_MIRROR -->;
a per-model table of revision, licence, prompts, max tokens, pooling,
similarity and precision: TODO(check). Haystack encoding runs in bfloat16 on
NVIDIA L4 GPUs; document vectors are cached in float16 <!-- eval/haystack.py -->.

**Software**, per uv environment. Builders, BM25 and statistics: mteb
2.20.11, bm25s 0.3.13, PyStemmer 3.1.0, duckdb 1.5.5 <!-- uv.lock -->.
Embedding, reranking and fine-tuning: mteb 2.11.6, sentence-transformers
6.0.1, transformers 5.17.0 <!-- eval/uv.lock -->. LLM scripts: mteb 2.6.5,
openai 3.13.0 <!-- llm/uv.lock -->. The eval lockfile pins CPU PyTorch 2.14.0;
GPU VMs drop that pin and install PyPI's CUDA build, whose version is not
recorded <!-- eval/uv.lock: torch 2.14.0+cpu; scripts/gcp_runner.sh, scripts/gcp_haystack.sh -->: TODO(check).
Packages added with `uv run --with` are unpinned (duckdb and rapidfuzz for
the LLM scripts, accelerate for fine-tuning): TODO(check). The vendored
evaluator of *The Embedder's Dilemma* is at commit 1a94652 <!-- vendor/embedders-dilemma: git log -->.

**Fusion.** Reciprocal-rank fusion <!-- TODO(ref): Cormack et al. 2009 --> of
tuned BM25 with each dense run: score = Σ 1/(60 + rank), 0 for a document
outside a run's top 1,000, ties broken by PMID <!-- scripts/rrf_haystack.py -->.

**First pass for C1.** Of tuned BM25, each dense run and each fusion, the one
with the best practice recall@100 (curator-only) <!-- docs/PREREG.md: change log 2026-10-15 22:45 -->;
raw or condensed: TODO(prereg). Result: TODO(result).

**Cross-encoders** [@nogueira2019monobert] re-sort the first pass's top 100
(fixed, not tuned); lower ranks keep their order. MedCPT-Cross-Encoder
[@jin2023medcpt] and bge-reranker-v2-m3 <!-- TODO(ref): bge-reranker-v2-m3 -->
score (query, abstract) pairs at 512 tokens by raw logit; Qwen3-Reranker-0.6B
and 4B [@zhang2025qwen3embedding] use the card's yes/no prompt and default
instruction, up to 8,192 tokens, scored as logit(yes) − logit(no), which
orders as the card's P(yes) without saturating. Weights run in bfloat16, the
score head in fp32 <!-- eval/rerank_haystack.py -->. Which reranker C1 uses,
and how it is chosen: TODO(prereg). Results: TODO(result).

**LLM listwise rerankers** [@sun2023rankgpt] re-sort the first pass's top 20
(fixed). DeepSeek-V4-Flash or Gemini 3.1 Flash-Lite (temperature 0, reasoning
off) sees gene and disease names and synonyms and the 20 abstracts (300 words
each, query-seeded shuffle), and returns a JSON ranking. Ids it leaves out
follow in first-pass order <!-- llm/rerank_haystack_llm.py -->. Reranks of tuned BM25
exist for both haystacks, as observations; reranks of the C1 first pass:
TODO(result).

**Corpus-in-context LLMs (small task).** With the unchanged evaluator of
*The Embedder's Dilemma* [@elassadi2026embedders] and LOFT's format
[@lee2024loft], each query is sent with all 300 documents of its batch, in
sorted-PMID order (`llm/run_gdv2_llm.py`). The prompt asks for every document
that helps, not a ranking; hit@1 is whether the first id the model lists is
the gold. Both LLMs run with reasoning off at the provider's default
temperature (1.0 on OpenRouter), one sample per query. Answers often list
several ids, mostly in prompt order: on the test batches Gemini gave 98
multi-id answers out of 300 (96 ascending), DeepSeek 89 (63 ascending) <!-- results/llm_v2/summary.csv: n_multi, n_multi_ascending -->.
So for these queries the registered hit@1 partly reflects document order, not
the model's judgment; two order-free scores are kept as observations: gold
listed at all, and hit@1 with listed ids tie-broken at random. On batch 0 the
best LLM was Gemini 3.1 Flash-Lite (59 of 75, 57 for DeepSeek), and
embeddinggemma-300m and BMRetriever-410M tied as best embedder (63 of 75), so
C3 is tested against both, as C3a and C3b <!-- docs/PREREG.md: change log 2026-10-16 -->.

**Closed-book memory probe.** With no documents, each LLM names up to 3
papers that are the key evidence for the pair, as JSON (title, first author,
year, PMID), at temperature 0 (DeepSeek with reasoning off, Gemini at its
default reasoning, not set in the request) <!-- llm/closed_book_probe.py -->. A gold paper is recalled if its PMID
is listed or a listed title matches with rapidfuzz token_sort_ratio ≥ 90; a
pair is *memorised* if any gold paper is recalled. On template titles that
know nothing of the literature ("{SYMBOL} mutations in {disease}"…), a looser
rule (token_set_ratio ≥ 90) gave false hits on 23% of haystack pairs (output
not saved) <!-- llm/closed_book_probe.py: docstring; docs/PREREG.md: change log 2026-10-15 -->,
this one on 1% <!-- results/memorisation_probe.parquet: template_hit, GeneDiseaseHaystack -->.
The probe ran on the test haystack, v1 and all v2 batches. C4 needed at least 15 memorised test pairs;
Gemini had 9, so C4 is dropped <!-- docs/PREREG.md: change log 2026-10-16; results/memorisation_probe.parquet -->.
The 13 v2 probe replies that did not stop (12 Gemini, 1 DeepSeek) <!-- llm/probe_loop_recheck.py; results/llm_v2/summary.csv: n_probe_cut -->
will be re-sent at temperature 1 as a sensitivity check: TODO(result). The
registered label stands.

## Fine-tuning

**Data** (`scripts/build_finetune_gene_disease.py`; no LLM calls, no synthetic
text). Rows are (anchor, positive abstract, up to 3 hard negatives). A
positive is a curated PMID for the pair with a local abstract, tagged with the
gene and cited for fewer than 10 genes; at most 10 per pair.

**Exclusions**, for positives and negatives alike: every gene of the
haystacks, v1 and the v2 batches; every registry pair and PMID; every haystack
candidate; every PMID tagged with, or curated for, an excluded gene; every
PMID any Open Targets literature evidence links to a registry pair. In all
2,914,504 PMIDs are banned <!-- data/finetune_gd/manifest.json: exclusion.pmids_banned -->.
Of the 17,352 curated pairs with a local abstract, 7,990 survive, with 3,207
genes and 14,630 PMIDs <!-- data/finetune_gd/manifest.json: funnel_curated -->.
Test diseases with another gene are allowed: 2,425 training pairs use 543
distinct diseases that are registry query diseases or their twins (856
registry diseases, 1,209 with twins) <!-- data/finetune_gd/manifest.json: sizes.train, exclusion.registry_diseases_with_twins -->.
A variant without them was built (`data/finetune_gd/no_test_disease/`, 17,677
curated rows) <!-- data/finetune_gd/no_test_disease/manifest.json: sizes.train -->
but is not used; whether to run it as a sensitivity check: TODO(check). Half
the anchors use the test query template (below). Published PubMed-trained
models may have seen test PMIDs, or the citations behind the gold, in
pretraining; we do not control this.

**Anchors**, one per pair: 50% the test template, 25% `{SYMBOL} {disease}`,
25% the template with an HGNC alias (never another gene's symbol) or an exact
disease synonym.

**Hard negatives**: two same-gene slots and one same-disease slot, topped up
from the other kinds. A same-gene negative is a curated PMID for the gene with
an unrelated disease, or, at most once per row, a BM25 hit (`{SYMBOL}
{disease}`, k1 = 0.9, b = 0.75) in the top 10 over a seeded sample of 500 of
the gene's tagged abstracts. Both kinds obey rules stricter than v2's: outside
the gene's guard; not naming the disease or a twin, as a phrase or by all
stemmed core words; no PubTator tag of the disease, a twin, a descendant or a
narrow ancestor (≤ 100 descendants), nor a tag mention matching their names.
HP anchors get no same-gene negatives.
Same-disease negatives follow v2. An audit used the practice grades available
when the data was built (2026-10-16 17:54) <!-- mtime of data/finetune_gd/manifest.json -->:
of 360 non-gold practice abstracts graded by both judges, 111 got a 2 from at
least one. The v2 rule (guard and name match only) would keep 184 of the 360
as negatives (11 graded 2 by either judge, 5 by both); this rule keeps 91 (1,
by both) <!-- data/finetune_gd/manifest.json: filter_audit_practice -->. Rule
choices were checked against this audit (a co-curation rule was dropped as it
caught no extra judged positive) <!-- scripts/build_finetune_gene_disease.py: docstring -->,
so the negatives carry the judges' view of relevance on practice data.

**Split and size.** 5% of genes (160) form a validation set; training never
uses a validation positive. The training file has 23,237 rows, 7,574 pairs,
3,044 genes and 12,050 PMIDs; validation 1,366 rows and 402 pairs <!-- data/finetune_gd/manifest.json: sizes -->.
The *silver* variant adds, for training pairs only, up to 5 Europe PMC PMIDs
per pair (title or abstract co-mention, same rules, highest resourceScore):
34,788 rows, 11,551 from Europe PMC <!-- data/finetune_gd/manifest.json: sizes.train_silver -->.
A recipe with 3 negatives drops rows with fewer, so the chosen recipe (below)
trains on 15,699 silver rows (45%; 5,579 from Europe PMC), 3,789 pairs, 1,884
genes and 10,497 PMIDs <!-- data/finetune_gd/train_silver.parquet where negative_3 is not null; data/finetune_gd/runs/logs/bge-small-gd-silver-n3-s42.log -->.

**Training** (`eval/finetune.py`): CachedMultipleNegativesRankingLoss in
sentence-transformers <!-- TODO(ref): MNRL / GradCache --> (mini-batch 32,
default scale 20), one epoch, learning rate 2e-5, 10% warm-up then linear
decay, AdamW (trainer defaults), batch 128, bfloat16, the base's max length
(512 for bge) <!-- scripts/gcp_jobs/hs-ft.txt; eval/finetune.py -->. Length, pooling and
similarity are checked against the base (`scripts/ft_select.py`); prompts
are copied from the base's mteb prompts (bge: its query instruction), not
checked <!-- eval/finetune.py: mteb_prompts -->. So a fine-tune encodes as its
base. Since an in-batch negative can be another pair's known positive, a
guarded sampler never batches two rows if one's documents are linked by Open
Targets evidence to the other's pair; rows that fit no batch sit out the
epoch (385 of 15,699, 2.5%, for bge-small seed 42) <!-- data/finetune_gd/runs/logs/bge-small-gd-silver-n3-s42.log: GuardedBatchSampler -->.

**Selection on practice.** On bge-small-en-v1.5, seed 42, {curated, curated +
silver} × {0, 3} hard negatives; each model re-ranked the practice BM25 top
1,000 (test runs score all gene candidates). Curated + silver with 3
negatives won on raw curator-only nDCG@10, 0.426 against 0.424 for curated
with 3; it also wins on condensed, 0.468 against 0.463 <!-- data/finetune_gd/runs/chosen.json -->.
This recipe is trained on bge-base-en-v1.5 (the C5 base) and bge-small, seeds
42, 43 and 44. Checkpoints are saved every 50 steps but none is selected: the
end-of-epoch model is run on all gene candidates <!-- scripts/gcp_jobs/hs-ft.txt -->.
For C5 and C6 a fine-tune's per-pair score is the mean over its seeds; seed
spread is reported. PREREG names C8 too, but the cross-encoder fine-tunes
have one seed: TODO(prereg). bge-base fine-tunes and test runs: TODO(result).

**Cross-encoder fine-tunes** (`eval/finetune_ce.py`, C8): MedCPT-Cross-Encoder
and bge-reranker-base <!-- TODO(ref): bge-reranker-base --> on the selected
data, binary cross-entropy with the positive weighted by negatives per
positive, 3 hard negatives, batch 32, learning rate 2e-5, one epoch, seed 42
<!-- eval/finetune_ce.py; scripts/gcp_jobs/hs-ft.txt -->, compared with their
bases on the practice BM25 top 100. TODO(result).

## Cost model

**Embedders.** Throughput was measured on 2026-09-25 <!-- git log of results/embedding_throughput_gpu.csv -->
on one NVIDIA L4 at a Google Cloud Spot rate of $0.48 per hour (zone not
recorded), the rate the cost model uses <!-- results/embedding_throughput_gpu.csv; RESULTS.md: method notes -->:
float16, inputs padded or cut to 512 tokens, the largest batch that fits,
median tokens per second over at least 8 timed batches (`eval/throughput.py`).
Dollars per million tokens are the hourly rate over millions of tokens per
hour: bge-small 326,407 tokens/s, $0.000408; MedCPT 139,126 tokens/s,
$0.000958; Qwen3-Embedding-8B 2,609 tokens/s, $0.0511 <!-- results/embedding_throughput_gpu.csv -->.
A run costs its own-tokenizer token count times its rate. Actual runs differ:
they use bfloat16 and each model's own max length (Qwen3-Reranker up to 8,192
tokens), and the phase-2 queues run on on-demand L4s, which cost more than
the Spot rate <!-- scripts/gcp_queue.sh: --provisioning-model STANDARD; scripts/gcp_haystack.sh (phase 1): Spot -->.
Haystack token counts, rates for models with no L4 measurement (BiCA,
BMRetriever, embeddinggemma-300m, the fine-tunes <!-- results/embedding_throughput_gpu.csv: absent or status=error -->)
and cross-encoder costs: TODO(result). BM25 and lookups run on CPU; CPU time
is not priced ($0).

**LLMs.** Small task, reranks, probe and judges: the charges OpenRouter
reports per request, summed. For the judges failed attempts count too, and a
request that raised an error is counted at a list-price ceiling <!-- llm/judge_pool.py: send_all; llm/rerank_haystack_llm.py; llm/run_gdv2_llm.py; llm/closed_book_probe.py -->.
C9's 12-task suite: list price × tokens <!-- scripts/pareto_bio.py -->, with
Gemini 3.1 Pro run on Vertex AI (about $26 billed) <!-- RESULTS.md: header -->.
List prices per million input, cached-input and output tokens, with the date
recorded (cached-input rates from `scripts/pareto_bio.py` only): Gemini 3.1
Flash-Lite $0.25, $0.025, $1.50 (2026-09-12, unchanged
2026-10-16) <!-- scripts/pareto_bio.py: OR_PRICES; llm/judge_pool.py: PRICES; git log -S -->;
Gemini 3.1 Pro $2.00, $0.20, $12.00 (2026-09-25) <!-- scripts/pareto_bio.py: OR_PRICES; git log -S -->;
DeepSeek-V4-Flash $0.066, $0.013, $0.131 (2026-09-12) <!-- scripts/pareto_bio.py, llm/estimate_cost.py; git log -S -->,
later $0.091 input and $0.182 output (2026-10-16) <!-- llm/judge_pool.py: PRICES; git log -S -->.
Gemini's charges match its list price. DeepSeek, on its pinned provider, was
charged $0.1265 for 3,232,606 input and 32,000 output tokens on the test
rerank, below either price <!-- results/llm_v2/haystack_rerank/GeneDiseaseHaystack/llm-deepseek-v4-flash-top20-of-bm25-symbol_disease-k0.9-b0.75.parquet: sums -->.
TODO(check): one dated price table for the paper.

## Metrics and statistics

**Metrics.** Small task: hit@1. Haystack main score: nDCG@10 on the graded
key, gain = merged grade (a 2 counts twice a 1), unjudged items 0 (including
those one judge left ungraded), ideal ranking from the pair's judged grades,
pairs with no positive grade dropped; the share of the top 10 that is judged
is shown alongside (`scripts/stats_haystack.py`). Also planned: graded
recall@100, precision@10 and bpref [@buckley2004bpref], with relevant meaning
merged grade ≥ 1 by default (pending sign-off <!-- docs/PREREG.md: Waiting on Dan -->).
Ranks 11–100 are mostly unjudged, so graded recall@100's denominator and its
handling of unjudged items: TODO(prereg). Graded recall@100, precision@10 and
bpref are not yet implemented: TODO(result). Curator-only recall@10,
recall@100 and nDCG@10, raw and condensed, are shown too. How much curators
miss will be estimated from the lookup sample, each hit weighted by its
inverse chance of being drawn <!-- llm/judge_pool.py: lookup_weight -->;
the estimator: TODO(prereg).

**Selection metric.** Until graded practice labels existed, settings were
selected on curator-only nDCG@10 <!-- docs/PREREG.md: change log 2026-10-15 22:45 -->,
in its raw variant: BM25 and the fine-tune recipe were picked on raw. Graded
practice labels now exist (7,189 merged grades over all 200 practice pairs)
<!-- data/haystack/GeneDiseaseHaystackDev/judgments_merged.parquet -->, so by
that rule the remaining picks (best embedder for C1, C2 and C7; the C6
comparator; the C1 reranker; the C8 off-the-shelf reranker) would use graded
practice nDCG@10. This reading is not yet registered: TODO(prereg).

**Claims.** "Best" means best on the practice set, chosen before the test
run; "similar size" means under 150M parameters, counted per encoder, as the
C6 list implies (MedCPT has two 109M encoders, 219M in total) <!-- eval/medcpt_model.py: META.n_parameters -->.

| claim | comparison | test set | test |
|---|---|---|---|
| C1 | first pass + reranker vs best single embedder | haystack | sign-flip |
| C2 | best embedder vs practice-tuned BM25 | haystack | sign-flip |
| C3a | best LLM, corpus in context, vs embeddinggemma-300m | small task, batches 1–4 | McNemar |
| C3b | best LLM, corpus in context, vs BMRetriever-410M | small task, batches 1–4 | McNemar |
| C4 | LLM edge larger on memorised pairs (dropped) | small task, batches 1–4 | label permutation |
| C5 | our fine-tune vs bge-base-en-v1.5 | haystack | sign-flip |
| C6 | our fine-tune vs best of MedCPT, BiCA-base, MedEmbed-base, PubMedBERT | haystack | sign-flip |
| C7 | best PubMed-trained vs best general model, both < 150M | haystack | sign-flip |
| C8 | our fine-tuned reranker vs best off-the-shelf reranker | haystack | sign-flip |
| C9 | Gemini 3.1 Pro vs Qwen3-Embedding-8B | 12-task suite | exact sign-flip |

<!-- docs/PREREG.md: claims table and change log -->

**Candidate sets**, not yet registered: TODO(prereg). As the text stands,
the best embedder for C1 and C2 is drawn from the dense models with practice
haystack runs, GO fine-tunes excluded. C7's general models under 150M are
MiniLM, e5-small, bge-small and bge-base; its PubMed-trained ones are MedCPT,
BiCA-small/base, MedEmbed-small/base and PubMedBERT, plus BioLORD-2023 if it
counts (it is trained on ontology definitions, not PubMed). For C8: the
first pass, which fine-tuned reranker (MedCPT-Cross-Encoder or
bge-reranker-base), and which off-the-shelf rerankers compete.

**Tests.** Haystack claims: two-sided paired sign-flip permutation test on
per-pair nDCG@10, 10,000 random sign vectors (seed 0) shared by all
comparisons, p = (1 + #{|permuted| ≥ |observed|}) / (1 + 10,000), with its
Monte Carlo error <!-- scripts/stats_haystack.py -->. Small-task claims: exact
two-sided McNemar test on per-pair hit@1. C9: exact sign-flip over the 12
tasks: the seven lifted from *The Embedder's Dilemma*
[@elassadi2026embedders] (LLMBIOSSES, LLMSTSBenchmark,
LLMBiorxivClusteringP2PV2, LLMMedrxivClusteringP2PV2,
LLMMedrxivClusteringS2SV2, LLMToxicConversationsClassification,
LLMPublicHealthQA) and five we added (BioMeSHClassification,
GOProteinPairCls, GOPubMedRetrieval, PubChemSynonymPC500,
R2MEDBiologyPooled) <!-- results/bio_scores.csv: task -->. C9 tests
superiority only: PREREG expects no difference, but with no margin
registered a non-significant result would not show the two are equivalent.
Holm–Bonferroni correction holds the family-wise error at 5%. C3a/C3b made the
family 10; a claim not run is dropped and the correction recomputed, so with
C4 dropped the family is nine tests: C1, C2, C3a, C3b, C5–C9 <!-- docs/PREREG.md: claims; change log 2026-10-16 -->.
Comparisons with LLM-reranked runs use the no-self-grading grades. The script
for the confirmatory family: TODO(result).

**Uncertainty.** 95% percentile bootstrap intervals for each system and
difference, from 10,000 resamples of pairs and from 10,000 resamples of genes
that keep a gene's pairs together (500 test pairs share 473 genes) <!-- scripts/stats_haystack.py; data/haystack/GeneDiseaseHaystack/queries.parquet -->.
Exploratory tables add all-pairs Holm tests and a compact letter display
<!-- TODO(ref): Piepho 2004 -->. Strata, curator-only versus graded
rankings, per-source gold quality, the GO fine-tunes and unfiltered BM25 are
observations, not claims.

**Disclosures.** The change log records results seen before a rule was
fixed; commit times are the record. Draft 2 (2026-10-15 20:37, commit
bbb9a7b) has no recorded sign-off. Before it, the bug-fix reruns, the BM25
selection with one BM25 test run, and the memory probe had run (touching C2
and C4). Test-haystack results were also computed and looked at before the
rules that use them:

- Curator-only (condensed) test recall@10, recall@100 and nDCG@10 for ten
  dense runs (the eight above plus bge-small-gaf and bge-base-gaf), the
  mean-pooled MedCPT run and BM25 v1.3 were in RESULTS.md by 2026-10-15
  16:59 (commits 9631d7f, 40e8b48), before draft 1 (18:18).
- Test bootstrap intervals and Holm-vs-best p-values for those runs and
  tuned BM25 were committed at 22:37 (2d0d111) <!-- results/haystack/stats.csv, stats_pairs.csv; computed 2026-10-15 19:19 -->,
  before the "similar size" rule (cdf1ff1, 22:38) and the C6 list (cd52d2c,
  2026-10-16 20:47). The size rule excludes Qwen3-Embedding-0.6B, the best
  of these on test.
- Ten fusion runs (made 2026-10-15 22:54) and the two LLM reranks of tuned
  BM25 were scored curator-only on test by 2026-10-16 12:41 <!-- results/haystack/scores.csv -->.

These touch C1 (first-pass candidates and best single embedder), C2, C5
(bge-base), C6 (MedEmbed-base, PubMedBERT) and C7. No practice dense or
fusion run exists yet; the eight models have practice runs queued
<!-- scripts/gcp_jobs/hs-q1.txt -->, and "best" will be picked on those. On
the small task, the test batches had been scored for all embedders, and both
LLMs' test answers collected with their hit@1 printed (2026-10-15
22:45–22:53; DeepSeek 0.817, Gemini 0.740) <!-- llm/llm_results/llm_v2/*/; llm/llm_results/logs/gdv2_*.log -->,
before the practice tie was resolved (1540a28, 2026-10-16 16:29). So no
tie-break could be blind; hence C3a and C3b. The LLM pick (Gemini, best on
practice) stands although DeepSeek scored higher on test. The test pool was
graded on 2026-10-16 20:58–21:14, after the protocol commit (d11cc65,
20:58:27) <!-- PREREG change-log stamp for this entry says 21:30, later than both the commit and the grading; fix the stamp -->,
while PREREG still lists the rubric and the relevance threshold as awaiting
sign-off. The threshold must be fixed before any graded test recall,
precision or bpref is computed. Departures from the PREREG rule that every
setting is chosen on practice: model instructions are mteb's default and the
fine-tune checkpoint is the end-of-epoch model, neither tuned; these are not
yet in the change log: TODO(prereg). C9's inputs were seen before
registration (Gemini 3.1 Pro − Qwen3-Embedding-8B +0.027, p = 0.35 <!-- docs/PREREG.md: Starting numbers -->).
