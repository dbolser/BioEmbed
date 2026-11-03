# Introduction

*Draft, 2026-11-03. Same conventions as `methods.md`: each number is followed
by an HTML comment naming its source; results numbers are taken from
`results.md`. Only the Table 1 claims are confirmatory; everything else is
exploratory.*

A curator who adds a gene–disease link to a database needs the papers that
show it, and a researcher who meets the link needs the same papers to judge
it. Finding them means searching PubMed. Our local copy holds 28,332,361
abstracts <!-- data/haystack/GeneDiseaseHaystack/manifest.json: stats.local_abstracts -->.
Even after keeping only abstracts that PubTator3 [@wei2024pubtator3] tags
with the gene, a test pair has a median of 1,088 candidates
<!-- data/haystack/GeneDiseaseHaystack/manifest.json: stats.gene_pmids, median 1,087.5 -->.
Curators cite a median of 3 papers per pair, and the gene filter keeps 88% of
them (mean over pairs) <!-- data/haystack/GeneDiseaseHaystack/qrels.parquet: papers per qid, median 3; manifest.json: stats.gene_pmids.recall 0.8771 -->.
Curation is moving towards AI-assisted triage, in which models rank and
extract and people decide [@wood2026empowering]. PubMind uses LLMs to extract
variant–disease evidence from all of PubMed [@wang2026pubmind]. dgiLIT, which
curates drug–gene interactions, filters PubMed by PubTator3 gene tags, ranks
by keyword counts and passes the top 100 abstracts to an LLM
[@cannon2026dgilit]. Such pipelines rest on a ranking step that is rarely
measured against curated evidence.

*The Embedder's Dilemma* [@elassadi2026embedders] asked when an LLM is worth
its cost over a text embedder. Over 37 tasks the best LLM and the best
embedder were effectively tied (77.6 against 77.2). LLMs led only on
reasoning-heavy retrieval. The best LLM, Gemini 3.1 Pro, cost 1,431 times as
much as an embedder of similar quality ($154 against $0.11 per benchmark
pass) <!-- docs/The-Embedders-Dilemma-2026/sections/abstract.tex; sections/results.tex: Pareto frontier -->.
Its LLMs retrieve by reading the whole corpus in context [@lee2024loft],
which works for corpora of 82–415 documents but cannot reach 28M abstracts
<!-- docs/The-Embedders-Dilemma-2026/sections/appendix.tex: (iii) Small-corpus retrieval -->.
For a full-corpus task the real options are cascades: a cheap first pass by
keyword search [@robertson2009bm25] or a dense embedder, then a
cross-encoder [@nogueira2019monobert] or an LLM [@sun2023rankgpt] reranks a
short list. The study also tested such cascades on general benchmarks: an LLM
reranker helped on reasoning-heavy queries, while on semantic ones the
strongest embedder alone scored highest
<!-- docs/The-Embedders-Dilemma-2026/sections/results.tex: sec:rerank; tables/reranker_matrix.tex -->.
It did not test rerankers fine-tuned for a task, or a domain task like ours.

We built a benchmark from the papers curators cite. Gold links come from four
curated sources aggregated by Open Targets [@buniello2025opentargets]:
Orphanet [@rath2012orphanet], Gene2Phenotype [@thormann2019g2p], Genomics
England PanelApp [@martin2019panelapp] and UniProt [@uniprot2025]. The main
test is a haystack of 500 gene–disease pairs searched over PubMed; a small
task asks systems, among them an LLM reading the whole corpus, to find one
paper among 300. Curators cite a sample of the good papers, not all of them,
so two LLM judges from different model families graded the pooled top
results, and we checked the judges against human graders. Nine claims were
pre-registered. One was split in two and one was dropped, so nine tests were
Holm-corrected together <!-- docs/PREREG.md: change log 2026-10-16; results/claims.csv -->.
For three of them (C3a, C3b, C9) the claim's own test numbers were seen
before its rule was fixed. Curator-only test scores that bear on five more
(C1, C2, C5–C7) were seen before the rules that use them were fixed
(Disclosures).

Our main result is that fine-tuning small models on curated gene–disease
pairs gave the best system. A MedCPT cross-encoder [@jin2023medcpt],
fine-tuned by us and reranking the top 100 of a cheap first pass (BM25 fused
with bge-base), scored graded nDCG@10 0.861, up from 0.750 off the shelf.
That was the highest of 46 test runs, but not separable from the Gemini 3.1
Flash-Lite rerank of the same first pass (0.853 on both judges' grades)
<!-- results/writeup/cost_quality.csv: graded_ndcg10; results/haystack/graded/stats.csv: graded_ndcg@10, pooled, 46 systems; Gemini rerank diff_vs_best −0.0078, p_holm_vs_best 0.19, tied_with_best -->.
On the non-Gemini judge's grades, which C8 uses, it scored 0.837 against
Gemini's 0.834 (difference +0.003, 95% CI −0.008 to +0.014)
<!-- results/claims.csv: C8 -->. C8 tested superiority, so no difference was
shown; that is not evidence the two are equal. Its rerank step (top 100)
cost $0.04 per 500 queries; the Gemini rerank of the same first pass
(top 20) cost $0.88. Training cost $0.40
<!-- results/writeup/cost_quality.csv: step_cost_usd 0.043 vs 0.877, train_cost_usd 0.403 -->.
Fine-tuning helped embedders too: bge-base [@xiao2024cpack] gained 0.090
(C5) and beat BiCA-base, a PubMed-trained embedder of similar size
[@sinha2026bica], by 0.065 (C6) <!-- results/claims.csv: C5, C6 -->.
Fine-tunes on Gene Ontology annotations scored about 0.12 below their
off-the-shelf bases <!-- results/writeup/cost_quality.csv: GO rows, bge-base-gaf 0.554 vs 0.675, bge-small-gaf 0.540 vs 0.658 -->.
Three caveats apply. Our training negatives were checked against the judges'
practice grades, which may favour our models on the graded key. On the
curator-only key the embedder gain shrinks to 0.02, while the reranker gains
hold (MedCPT 0.393 to 0.477); there, though, the fine-tuned MedCPT ranks
fourth, behind three off-the-shelf rerankers (top: the Gemini rerank, 0.497).
Each reranker was trained with one seed
<!-- results/writeup/cost_quality.csv: curator_ndcg10, bge-base 0.381 to 0.400; results/haystack/graded/stats.csv: ndcg@10, raw -->.
This fits a study of medical procedure search, where a fine-tuned
109M-parameter cross-encoder beat a 4B-parameter LLM reranker
[@fainzilber2026listwise].

Reranking the first pass with Gemini beat the
best off-the-shelf embedder, embeddinggemma-300m [@vera2025embeddinggemma],
by 0.133 (C1). Tuned BM25 beat that embedder by 0.042 (C2) and scored
above all 14 off-the-shelf embedders on the graded key. On the curator-only
key it scored below seven of them by point estimate (embeddinggemma-300m
0.430 against BM25's 0.372; not tested), so the answer key may decide this
comparison
<!-- results/claims.csv: C1, C2; results/writeup/cost_quality.csv: curator_ndcg10, group single -->.
On the small task, Gemini 3.1 Flash-Lite reading all 300 abstracts put the
right paper first for 74.0% of 300 test queries, against 79.3% and 71.7% for
the two embedders that tied best on the practice batch; neither difference
was shown (C3a, C3b)
<!-- results/claims.csv: C3a, C3b; practice tie: docs/PREREG.md change log 2026-10-16, results/v2_scores.csv batch 0 -->.
Over all 194 gradeable labels from one author (D.B.), the judges matched
70%. Three colleagues matched him on 75–79%, but on different, smaller sets
(20–47 labels), so these are not like-for-like
<!-- results/figures/judge_agreement.csv: dan vs judges n 194 exact 0.701; human vs human dan–colleague n 20/42/47, exact 0.75/0.786/0.787 -->.
On the 91 abstracts a colleague also graded, the judges matched the author
on 80% and the colleague did so on 78% (difference +2 points, 95% CI −8 to
+12) <!-- results/figures/judge_agreement_overlap.csv: analysis overlap, 109 abstract–colleague pairs -->.
Where the judges differed from the author they mostly graded lower (44
lower, 14 higher, of 194), largely because of DeepSeek, so graded scores
are, if anything, strict <!-- results/figures/judge_agreement_overlap.csv: analysis direction; DeepSeek: results/judge_wording_check.csv, arm same -->.

Our contributions are:

1. **A curated-citation benchmark for gene–disease evidence.** Gold from
   38,392 PMIDs that four curated sources cite for 17,723 pairs
   <!-- data/finetune_gd/manifest.json: funnel_curated.curated -->; a frozen
   500-pair test haystack over PubMed with a gene-disjoint practice haystack;
   and a small corpus-in-context task with 300 test queries.
2. **A graded answer key checked against people.** Two judges from different
   model families graded all 33,531 pooled test items
   <!-- data/haystack/GeneDiseaseHaystack/judgments_merged.parquet: rows; judgments.parquet: 33,531 per judge -->.
   Any claim on an LLM-reranked run (C1, C8) used only the other judge's
   grades. We compared the judges with four human graders.
3. **Quality for 46 test runs, and cost for 35 of them,** on one dollar
   axis: tuned BM25, a BM25–embedder fusion, 14 off-the-shelf embedders,
   five cross-encoders, two LLM rerankers and our fine-tunes
   <!-- results/haystack/graded/stats.csv: graded_ndcg@10, pooled, 46 systems; results/writeup/cost_quality.csv: run_ids, 35 runs -->.
4. **A small fine-tuned reranker not shown different from an LLM reranker**
   (C8: +0.003, 95% CI −0.008 to +0.014; superiority test, one seed)
   <!-- results/claims.csv: C8 --> at about a twentieth of its rerank-step
   cost <!-- results/writeup/cost_quality.csv: step_cost_usd 0.043 vs 0.877 -->,
   and fine-tuned embedders that beat a PubMed-trained one (C5, C6).
5. **Nine Holm-corrected tests of pre-registered claims**, and exploratory
   signs that the answer key (graded or curator-only) may decide whether
   keyword search beats embedders.

<!-- What is released (PMIDs, qrels, build scripts, fine-tuned weights, raw LLM responses) is not recorded in the repo: TODO(check), see methods.md "Availability". -->
