# Discussion

*Draft, 2026-11-03. Same conventions as `methods.md`: each number is followed
by an HTML comment naming its source. Numbers come from `results.md` and
`docs/WRITEUP.md`. Only the named claims (C1, C2, C3a, C3b, C5–C9) are
confirmatory; every other statement here is exploratory.*

**Fine-tuned small rerankers are the practical choice.** The best system was
built by fine-tuning small models on curated gene–disease pairs. Our
fine-tuned MedCPT cross-encoder, reranking the cheap first pass, scored
graded nDCG@10 0.861, top of 46 test runs, up from 0.750 off the shelf
<!-- results/haystack/graded/stats.csv: graded_ndcg@10, pooled; results/writeup/cost_quality.csv -->.
On DeepSeek's grades alone, so that Gemini did not grade its own output, the
fine-tuned MedCPT pipeline was not shown different from the Gemini 3.1
Flash-Lite rerank (C8: 0.837 against 0.834, +0.003 [−0.008, +0.014])
<!-- results/claims.csv: C8 -->. The interval excludes a gap larger than
0.008 in Gemini's favour or 0.014 in ours.
Its rerank step cost about $0.04 per 500 queries against $0.88 for the
Gemini rerank of the same first pass, and training cost $0.40 <!-- results/writeup/cost_quality.csv: step_cost_usd 0.043, 0.877; train_cost_usd 0.403 -->.
Its total of $1.71 is nearly all the first pass's embedding of 1.7M
abstracts ($1.67), which is paid once; more queries add only the rerank step
<!-- results/writeup/cost_quality.csv: run_cost_usd 1.709, first_pass_cost_usd 1.665; results/writeup/cost_notes.md -->.
Along the cost frontier, BM25 ($0, 0.759) is followed by BM25 reranked by
DeepSeek-V4-Flash (~$0.13, 0.832), which needs no GPU, then BM25 reranked by
Gemini ($0.88 total run cost, no GPU; 0.844), then the fine-tuned MedCPT
pipeline ($1.71, 0.861)
<!-- results/writeup/cost_notes.md: The frontier line; results/writeup/cost_quality.csv -->.
These scores use both judges' grades, which may flatter the LLM reranks (see
Limitations), and none of these gaps was tested. Fine-tuning helped embedders
too: bge-base gained 0.090 (C5) and beat BiCA-base, a PubMed-trained model of
similar size (C6, +0.065), while BiCA-base, built on a different base model
(gte-base), beat bge-base by 0.024 (C7), so C7 is not a clean PubMed-training
effect <!-- results/claims.csv: C5-C7; Hugging Face model card bisectgroup/BiCA-base: base_model thenlper/gte-base -->. The training task matters: fine-tunes on
Gene Ontology annotation pairs lost about 0.12 against their bases
<!-- results/writeup/cost_quality.csv: GO rows 0.554, 0.540 vs 0.675, 0.658 -->.
A listwise-tuned 109M cross-encoder beating a 4B reranker on medical
procedure search [@fainzilber2026listwise] points the same way. Three caveats
apply; see Limitations.

**Good rerankers win, but most off-the-shelf cross-encoders did not.** The
Gemini rerank of the first pass beat the best off-the-shelf embedder by 0.133
(C1), and the top seven of 46 test runs are reranked
<!-- results/claims.csv: C1; results/haystack/graded/stats.csv -->. Yet of
five off-the-shelf cross-encoders only Qwen3-Reranker-4B scored above the
first pass (0.827 against 0.750); MedCPT (0.750) and Qwen3-Reranker-0.6B
(0.749) were level with it and bge-reranker-v2-m3 (0.703) and
bge-reranker-base (0.608) scored lower, none of it tested
(Results, "Good rerankers win") <!-- results/writeup/cost_quality.csv: group rerank -->.
A general-purpose reranker can undo a good first pass. Fine-tuning fixed this
for both models we trained; bge-reranker-base went from 0.608 to 0.838 <!-- results/writeup/cost_quality.csv -->. We did not test why the
off-the-shelf models fail.

**The answer key decides BM25 against embedders.** Tuned BM25 beat the best
off-the-shelf embedder by 0.042 on the graded key (C2)
<!-- results/claims.csv: C2 -->, and scores above all 14 such embedders there; on
the curator-only key it scores below seven of them (embeddinggemma-300m 0.430
against 0.372), untested <!-- results/writeup/cost_quality.csv: curator_ndcg10, group single -->.
Two readings fit. Curators cite a median of 3 papers per pair
<!-- data/haystack/GeneDiseaseHaystack/qrels.parquet, via methods.md -->
and may miss papers that keyword search finds; or the judges may favour
abstracts that name both the gene and the disease, which is what BM25
rewards. We did not test between them; human grades on the papers where the
keys disagree would. C2 is a result about the graded key only.

**Reading the whole corpus.** With all 300 small-task abstracts in its
context, Gemini 3.1 Flash-Lite put the right paper first for 74.0% of test
queries, against 79.3% for embeddinggemma-300m and 71.7% for
BMRetriever-410M; neither difference was shown (C3a, C3b)
<!-- results/claims.csv: C3a, C3b -->. The answer format weakens this test.
About a third of LLM answers listed several papers, often in the order given,
so hit@1 partly measures input order (Results, "The small task")
<!-- results/llm_v2/summary.csv: n_multi, batches 1-4 -->.
The small task did not separate LLMs from embedders. We did not run
corpus-in-context on the haystack, where a pair has a median of 1,088
candidates <!-- data/haystack/GeneDiseaseHaystack/manifest.json: stats.gene_pmids.median_set 1087.5 -->.

**Relation to *The Embedder's Dilemma*.** That study found the best LLM and
the best embedder tied overall (Gemini 3.1 Pro 77.6, Octen-8B 77.2), with
Gemini 3.1 Pro ahead on retrieval (64.5 against 56.0 for the best embedder),
most of all on reasoning-heavy retrieval [@elassadi2026embedders]. LLM
listwise reranking lifted a strong embedder on BRIGHT from 22.3 to 35.1
nDCG@10, but on BEIR the embedder alone
(63.1) beat every reranked set-up (60.3)
<!-- docs/The-Embedders-Dilemma-2026/sections/results.tex: Overall Performance; Task-Category Analysis; Retrieve-then-Rerank -->.
On our 12-task suite, which reuses seven of its tasks, Gemini 3.1 Pro was not
shown better than Qwen3-Embedding-8B (C9, +0.027), in line with the overall
tie; those numbers were seen before registration <!-- results/claims.csv: C9; docs/paper/methods.md: Disclosures -->.
On the haystack, LLM reranking behaved as on BRIGHT, not BEIR: the Gemini
rerank beat the best embedder by 0.133 (C1) and scored about 0.10 above its
own first pass (0.834 against 0.731 on non-Gemini grades; exploratory)
<!-- results/claims.csv: C1; results/writeup/cost_notes.md: other-judge table -->;
most off-the-shelf cross-encoders did not beat the first pass. Judging
whether an abstract is evidence for a link, not just about the gene, may be
the kind of reasoning that rerankers supply; we did not test this. Our
corpus-in-context LLM was Flash-Lite, which that study scored well below Pro
(64.5 overall against 77.6)
<!-- docs/The-Embedders-Dilemma-2026/sections/results.tex: Overall Performance -->,
so C3 is not a test of its retrieval finding. Our addition concerns the
second stage of the hybrid design it recommends: where curated pairs exist,
a small fine-tuned cross-encoder was not shown different from an LLM rerank
at a twentieth of its rerank-step cost (C8, above) <!-- results/claims.csv: C8 -->.

**What to use when** (exploratory). With no GPU, or for a few hundred
queries, BM25 with a DeepSeek-V4-Flash rerank of the top 20 scores 0.832 for
about $0.13 per 500 queries (a later DeepSeek rerank of almost the same
tokens was charged $0.07)
<!-- results/writeup/cost_quality.csv; results/writeup/cost_notes.md -->. That
0.832 includes DeepSeek's grades of its own output; on Gemini's grades alone
it scores 0.840, against 0.876 for the fine-tuned MedCPT pipeline
<!-- results/writeup/cost_notes.md: other-judge table -->. For
many queries with a GPU, embed the candidates once and rerank the first pass
with a fine-tuned cross-encoder (0.861; $0.04 per 500 queries after a one-off
$1.67 and $0.40 of training). To reproduce curators' choices, the Gemini
rerank led the curator-only key (0.497), untested. Off-the-shelf
cross-encoders and embedders alone should be checked on practice data first;
most did not beat BM25 or the first pass. A fine-tuned cross-encoder over
BM25's top 100 alone would cost about $0.04 per 500 queries with no
embedding step, but we did not run it <!-- results/writeup/cost_notes.md: Which rows -->.
Curation pipelines that pass abstracts to an LLM for extraction
[@cannon2026dgilit; @wang2026pubmind] could use such a reranker to choose
which abstracts it reads, with a human checking the output
[@wood2026empowering]; we did not test this.

## Limitations

- **LLM judges are also rerankers.** Gemini 3.1 Flash-Lite and
  DeepSeek-V4-Flash made the graded key and are also reranked systems, so
  their graded scores may be flattered. C1 and C8 use only the other judge's
  grades, as registered. The cost frontier and the run ranking use both
  judges' grades, so the LLM reranks there may be flattered (on Gemini's
  grades alone BM25 + DeepSeek scores 0.840, against 0.876 for the fine-tuned
  MedCPT pipeline). The grade sets shift every system (no-Gemini grades
  lower, no-DeepSeek higher), so scores compare only under the same grades
  <!-- results/writeup/cost_notes.md: other-judge table -->.
- **Judges grade lower than the expert.** They match an author (D.B.) on 70%
  of his 194 labels, against 75–79% for each colleague on smaller, different subsets
  (20–47 labels), so these are not like-for-like <!-- results/figures/judge_agreement.csv -->. On the 91
  abstracts he and a colleague both graded, the judges matched him on 80% and
  the colleague on 78% (difference +2 points, 95% CI −8 to +12)
  <!-- results/figures/judge_agreement_overlap.csv: analysis overlap -->.
  Where they differ on his 194 labels, they are lower 44 times and higher 14
  <!-- results/figures/judge_agreement_overlap.csv: analysis direction -->.
  Resending the human guide's wording barely changed this (70% to 71% on the
  same 184 abstracts); the gap is mostly DeepSeek's (mean grade 0.95 against
  his 1.16 and Gemini's 1.15), carried by the round-down merge
  <!-- results/judge_wording_check.csv -->. Graded scores are, if anything,
  strict. The calibration set is small (200 abstracts, 20 pairs)
  <!-- labeller/items_calibration.parquet: 200 rows, 20 gene–disease pairs -->.
- **The graded key covers pooled papers only.** It grades the top 10 of the
  pooled runs, every curator-cited paper and up to 5 sampled PubTator lookup
  hits per disease tier; an ungraded paper counts as 0 (`methods.md`, Graded
  answer key).
- **One first pass for the cross-encoders.** All cross-encoders, and the LLM
  reranks in C1 and C8, reorder the same BM25 + bge-base fusion; the LLMs
  also reranked tuned BM25's top 20 (exploratory). The fusion's top 100
  (cross-encoders) holds 76% of curator-cited papers per pair on average, its
  top 20 (LLM rerankers) 58%
  <!-- results/haystack/graded/stats.csv: raw recall@100 0.7604; recall@20 0.579: mean per pair of qrels.parquet PMIDs at rank ≤ 20 in runs/rrf-bm25-symbol_disease-k0.9-b0.75+BAAI__bge-base-en-v1.5__gene.parquet (recomputed 2026-11-03; not yet in a script output) -->,
  so the LLMs reranked a shorter list. No cross-encoder ran on BM25 alone.
- **The gene-tag filter sets a ceiling.** Candidates are abstracts PubTator
  tags with the gene, which keep 87.7% of curator-cited papers per pair on
  average <!-- data/haystack/GeneDiseaseHaystack/manifest.json: stats.gene_pmids.recall 0.8771 -->.
  Of the 372 lost, 128 carry only a non-human gene tag, often a
  model-organism ortholog <!-- docs/paper/methods.md: Haystack (test) -->.
- **Some test scores were seen before rules were fixed.** Curator-only test
  scores for ten embedders, a mean-pooled MedCPT run and an older BM25 were
  seen before the first draft. Test intervals for those runs and tuned BM25
  were seen before the C6 size rule. Curator-only test scores of fusions and
  BM25 + LLM reranks were seen before the candidate sets were registered.
  Small-task hit@1 was seen before the practice tie was settled (hence C3a
  and C3b). C9's inputs were seen before registration. The C1, C2, C6, C7
  and C8 candidate sets were registered on 2026-10-17, after many test runs
  had been made (`methods.md`, Disclosures; `docs/PREREG.md`, change log).
  The rubric and threshold were signed off on 2026-11-03, after the test pool
  was graded (2026-10-16), unchanged <!-- docs/PREREG.md: sign-off, change log 2026-11-03; docs/paper/methods.md: Grading done -->.
  Every "best" was picked on practice scores, though not always blind to test
  scores.
- **Training negatives saw the judges' view.** The fine-tuning negatives were
  filtered by rules checked against the judges' practice grades (gene-disjoint
  from test), which may favour our models on the graded key. On the
  curator-only key the embedder gain shrinks to +0.02 (bge-base 0.381 to
  0.400) while the reranker gain holds (MedCPT 0.393 to 0.477), though there
  the Gemini rerank (0.497) and Qwen3-Reranker-4B (0.489) score higher,
  untested <!-- docs/paper/methods.md: Fine-tuning; results/writeup/cost_quality.csv: curator_ndcg10 -->.
  Test diseases, with other genes, were allowed in training. Retrained
  without them (on a third less data), the embedder lost 0.015 on graded
  nDCG@10 and nothing on curator-only, and the cross-encoder gained 0.009:
  both within the 0.02 rule, registered after the test results were seen
  (Results, sensitivity check) <!-- results/ntd_readout.csv -->.
- **One seed for rerankers.** The embedder fine-tunes have three seeds
  (0.761–0.767); each cross-encoder fine-tune has one, so C8's interval omits
  seed variation <!-- results/writeup/cost_quality.csv: seed_min, seed_max -->.
- **Costs are rough.** One GPU price ($0.94 per L4-hour) for all runs; 10
  runs without logs are estimated; the rerank and fine-tune VMs ran about 7%
  dearer. Not counted: grading (~$8.60), data building, storage and the
  recipe search ($3.54) <!-- results/writeup/cost_notes.md -->. API charges
  vary: two DeepSeek reranks of almost the same tokens were charged $0.126
  and, later, $0.074 <!-- results/writeup/cost_notes.md: LLM rerankers; docs/paper/methods.md: LLMs -->.
- **Abstracts only.** Documents are title and abstract. Evidence that is only
  in the full text is invisible to every system and judge; labellers named
  abstract-only grading as a grey area <!-- docs/PREREG.md: change log 2026-11-03 -->.
- **English only.** Queries, rubric and judge prompts are in English; we did
  not test other languages.
- **One source mix.** The gold comes from four Open Targets curated sources
  built for rare, mostly Mendelian disease (by main source, Genomics England
  280 of 500 test pairs, Orphanet 82, Gene2Phenotype 77, UniProt 61)
  <!-- data/haystack/GeneDiseaseHaystack/strata.parquet, via methods.md -->.
  Results may differ for common-disease or GWAS-derived links, and for
  PubMed snapshots other than the 2026 baseline.

# Conclusion

For finding the papers behind a gene–disease link, a cheap first pass
reranked by a small cross-encoder fine-tuned on curated pairs scored highest
of 46 test runs (graded nDCG@10 0.861) and was not shown different from an
LLM reranker (C8), at about $0.04 against $0.88 (Gemini rerank of the same
first pass) per 500 queries for the rerank step and $0.40 of training <!-- results/claims.csv: C8; results/writeup/cost_quality.csv -->.
Without a GPU, BM25 with a cheap LLM rerank came close (0.832, a score that
includes DeepSeek's grades of its own output) for about $0.13. Most
off-the-shelf cross-encoders did not help. The best off-the-shelf embedder
trailed tuned BM25 on the graded key (C2), but on the curator-only key seven
of the 14 scored above BM25 (untested). These conclusions rest on LLM judges
that grade more strictly than an expert, on one first pass for the
cross-encoders and on abstracts only.
