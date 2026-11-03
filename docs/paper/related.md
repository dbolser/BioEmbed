# Related work

*Draft, 2026-11-03, revised after the results (first draft 2026-10-15).
Citations are Pandoc keys into `refs.bib`. Every entry was checked against
Crossref, PubMed, arXiv, medRxiv, bioRxiv, the NIST TREC browser or the
publisher. Our numbers follow `methods.md` (source in an HTML comment) and
match `results.md`. Only the Table 1 claims (C1, C2, C3a, C3b, C5–C9) are
confirmatory; every other result named here is exploratory.*

**Gene–disease literature retrieval.** The TREC Genomics track is the
closest precedent. Its first year used NCBI GeneRIF citations as
pseudo-relevance judgements [@hersh2003trec]. The 2004 and 2005 ad hoc
tasks switched to pooled expert judgements over a 4.59M-record MEDLINE
subset [@hersh2004trec; @hersh2005trec]. One 2005 template, "find articles
describing the role of a gene involved in a given disease", is exactly our
information need, but it had ten topics. TREC Precision Medicine
(2017–2020) used synthetic oncology topics: patient cases in 2017–19 (ten
2019 topics were non-cancer genetic conditions) and disease–gene–treatment
triples in 2020. It searched full MEDLINE snapshots of 26.8–29.1M abstracts
[@roberts2017trecpm; @roberts2018trecpm; @roberts2019trecpm;
@roberts2020trecpm]. Relevance turned on treatment, prevention or
prognosis; only its graded disease field (exact, more general, more
specific) mirrors our rubric.
BioASQ pairs expert questions with gold PubMed documents
[@tsatsaronis2015bioasq], and RELISH holds author judgements of
article-to-article similarity for over 180k articles [@brown2019relish].
Most embedders report biomedical results on BEIR's TREC-COVID, NFCorpus
and SciFact [@thakur2021beir; @voorhees2020treccovid; @boteva2016nfcorpus;
@wadden2020scifact] and on MTEB's medical tasks[^mteb-medical]
[@muennighoff2023mteb; @enevoldsen2025mmteb]. MedTEB is a separate,
MTEB-style medical benchmark [@khodadad2025medteb].
BRIGHT [@su2025bright] and R2MED [@zhang2025r2med] add reasoning-heavy
queries, biology among them. None of these asks, at PubMed scale and with
hundreds of topics, for the evidence behind one curated gene–disease link
while scoring against distractors that share the gene. The closest
design we know of is concurrent: a retrieval benchmark built from dictyBase
curator queries linked to PubMed, for one model organism [@wang2026dicty].

**Curated citations as gold.** Using the papers a database cites as
relevance labels has a long history: GeneRIFs above, NFCorpus's links from
NutritionFacts pages to the papers they cite [@boteva2016nfcorpus], and
SciFact's claims written from citing sentences [@wadden2020scifact]. Gene
Ontology annotations carry the same kind of evidence. The 2004 Genomics
track asked systems to triage articles with experimental evidence for GO
terms [@hersh2004trec], and BioCreative asked for the passages behind GO
annotations [@blaschke2005biocreative; @mao2014biocreative]. We use GO
experimental-evidence links [@go2023] for one retrieval task and as
fine-tuning data. Our gene–disease gold comes from curated sources that
Open Targets aggregates [@ochoa2021opentargets; @buniello2025opentargets]:
Genomics England PanelApp [@martin2019panelapp], UniProt [@uniprot2025],
Gene2Phenotype [@thormann2019g2p] and Orphanet [@rath2012orphanet].
ClinGen's validity classes [@strande2017clingen] give one of our strata.
Curator lists are a sample, not a census. In the first Genomics track,
relevant articles without a GeneRIF outnumbered those with one about four
to one in the top 20 of the organisers' best training run (42.5% against
10.5%) [@hersh2003trec]. LitDD machine-mapped about 69,000 PubMed case reports to
Gene2Phenotype's developmental-disorder entries [@yates2025litdd], while
Gene2Phenotype's evidence in Open Targets 26.06 cites 12,024 PMIDs across
all its panels <!-- /outsee5/home/danbolser/Work/IUK-IPF/data/sources/opentargets/26.06/evidence_gene2phenotype: distinct numeric PMIDs in literature, counted 2026-11-03 -->.

**Mining gene–disease associations.** A separate line extracts
associations from text: BeFree and DisGeNET
[@bravo2015befree; @pinero2017disgenet; @pinero2020disgenet], DISEASES
[@pletscher2015diseases], the Europe PMC component of Open Targets
[@kafkas2017literature], and relation corpora such as BioRED
[@luo2022biored]. LLMs now do such extraction at full scale: PubMind
extracts variant–function–disease associations, with supporting evidence,
from over 41M PubMed abstracts and 5M full texts [@wang2026pubmind]. These
systems decide whether a link exists. We take the link as given and rank
the evidence for it. We use text-mined links in two ways. As a guard: an
abstract that Europe PMC links to the query but curators do not cite is
never a negative in training or in the small task, and the condensed
curator-only score removes it (the raw score counts it as wrong). And as
extra positives: the chosen fine-tuning recipe adds up to five Europe PMC
abstracts to each training pair
<!-- methods.md: Silver guard; data/finetune_gd/manifest.json: sizes.train_silver -->.

**Entity-aware search at PubMed scale.** PubTator3 tags genes, diseases and
relations in about 36M abstracts, and reports that its entity-pair search
beats PubMed and Google Scholar on top-20 precision [@wei2024pubtator3].
LitSense 2.0 adds dense sentence and paragraph search over 38M abstracts
[@yeganova2025litsense2], and MedCPT trains a retriever and reranker on
255M PubMed click logs [@jin2023medcpt]. Several systems already search all
of PubMed for genetic evidence. AMELIE parses all 29M abstracts, plus full
texts, for evidence that a variant causes a Mendelian disease, and ranks a
patient's candidate genes with it [@birgmeier2020amelie]. LitVar 2.0
normalises variant mentions across PubMed, PMC full text and supplementary
files [@allot2023litvar2]. LitDD, described above, retrieves about 70% of
the papers in independent curated sets [@yates2025litdd]. dgiLIT, for
drug–gene interactions, has the same shape as our cascades: PubTator3 gene
tags cut PubMed down to abstracts that mention the genes, interaction-word
counts rank them, and an LLM curates the top 100; one expert judged 134 of
its 137 extracted interactions valid [@cannon2026dgilit]. None of these is
a benchmark on which other rankers can be scored. We use PubTator3's gene
tags as a first-stage filter and compare rankers inside it. Its disease
tags make a weak filter: 148 of the 500 test query diseases, mostly Mondo
and HPO terms, have no MeSH or OMIM identifier in Open Targets
<!-- RESULTS.md: haystack notes, "148/500"; recounted 2026-11-03 from data/haystack/GeneDiseaseHaystack/queries.parquet diseaseId against Open Targets 26.06 disease dbXRefs (MESH:, OMIM:): 91 MONDO, 44 HP, 12 EFO, 1 Orphanet -->,
and Mondo's cross-references, built to bridge these
vocabularies [@vasilevsky2025mondo], recover only one more query disease
<!-- RESULTS.md: haystack notes, "MONDO's own mappings add one disease" -->.

**Cost and quality: embedders, rerankers, LLMs.** *The Embedder's Dilemma*
compared 10 LLMs with 26 embedders on 37 tasks. It found the two tied
overall. The one LLM on its cost–quality frontier, Gemini 3.1 Pro, cost
1,431× as much as an embedder of similar quality, and LLMs led only on
reasoning-heavy retrieval [@elassadi2026embedders]. Its main LLM retrieval
test puts the whole corpus (82–415 documents) in context, following LOFT
[@lee2024loft]; a side test found that an LLM reranker helped on
reasoning-heavy BRIGHT but not on BEIR. Reading the whole corpus cannot
reach 28M abstracts. At that scale the real options are lexical search
[@robertson2009bm25; @lu2024bm25s], a dense embedder
[@zhang2025qwen3embedding], cross-encoder reranking
[@nogueira2019monobert; @zhang2025qwen3embedding] or listwise LLM
reranking [@sun2023rankgpt] of a short list, or a cascade that spends
large-model compute only where it pays [@chen2023frugalgpt]. Biomedical
embedders sit on the same axis: MedCPT, BioLORD-2023 [@remy2024biolord],
MedEmbed [@balachandran2024medembed], PubMedBERT-based encoders
[@gu2021pubmedbert], BMRetriever [@xu2024bmretriever] and BiCA
[@sinha2026bica]. We moved the Dilemma's comparison from sampled corpora
to a full-corpus domain task. Where corpus-in-context reading was possible
we did not see an LLM win: on a small task (300 test pairs, each batch a
300-abstract corpus) Gemini 3.1 Flash-Lite was not shown different from the
two embedders tied best on practice (hit@1 0.740 against 0.793 and 0.717;
C3a, C3b), and on a 12-task suite Gemini 3.1 Pro was not shown better than
Qwen3-Embedding-8B (C9). At PubMed scale the cascade mattered more than the
single model: an LLM rerank of a cheap first pass beat the best
off-the-shelf embedder (picked on practice) by 0.133 graded nDCG@10, scored
without Gemini's grades (C1)
<!-- results/claims.csv: C1, C3a, C3b, C9 -->. A large LLM is not the only
good reranker. On medical procedure search, a 109M cross-encoder
fine-tuned with a listwise loss beat a 4B LLM reranker tuned by prompt
optimisation [@fainzilber2026listwise]. Our task points the same way
(pointwise loss): our fine-tuned MedCPT cross-encoder scored highest of 46
test runs (exploratory)
<!-- results/haystack/graded/stats.csv: graded_ndcg@10, pooled; results/writeup/cost_quality.csv: graded_ndcg10 -->
and was not shown different from the Gemini 3.1 Flash-Lite rerank (C8), at
a small fraction of its rerank-step cost
<!-- results/claims.csv: C8; results/writeup/cost_quality.csv: step_cost_usd 0.043 vs 0.877, train_cost_usd 0.403 -->.
Off the shelf, only one of five cross-encoders scored above the first pass;
two were level with it (0.750, 0.749) and two scored lower (0.703, 0.608)
<!-- results/writeup/cost_quality.csv: group rerank, graded_ndcg10 0.827, 0.750, 0.749, 0.703, 0.608 vs first pass 0.750 -->.

**Domain fine-tuning transfer.** In-domain pretraining helps biomedical NLP
[@gu2021pubmedbert], and pseudo-labelled adaptation helps dense retrieval
on BEIR sets, the biomedical ones included [@wang2022gpl]. BEIR also showed
that models trained on one retrieval dataset often transfer poorly to others
[@thakur2021beir]. BiCA fine-tunes small general embedders with hard
negatives mined from PubMed citation links [@sinha2026bica]. We compared
three kinds of model of one size (under 150M parameters) on gene–disease
search: a general embedder, bge-base [@xiao2024cpack]; the best
PubMed-trained one, BiCA-base; and bge-base fine-tuned by us on curated
gene–disease pairs (curator-cited plus text-mined abstracts), with every
test gene, pair and abstract removed from training (test diseases were
allowed with other genes). BiCA-base, a PubMed-trained model built on a
different base (GTE-base, TODO(ref)), scored 0.024 above bge-base (C7). Our
fine-tune, built on bge-base itself, scored 0.090 above it (C5) and 0.065
above BiCA-base (C6)
<!-- results/claims.csv: C5-C7; BiCA-base model card (Hugging Face bisectgroup/BiCA-base): base_model thenlper/gte-base -->.
Both gains carry caveats (Results; Discussion, Limitations).
A neighbouring biological task did not transfer. Our earlier fine-tunes on
Gene Ontology evidence (a GO term paired with an abstract or protein text
that supports it) scored about 0.12 below their bases on gene–disease
search (bge-base 0.554 against 0.675; bge-small 0.540 against 0.658;
exploratory) <!-- results/writeup/cost_quality.csv: GO rows -->, although
they scored above their bases on our 12-task suite, which includes two GO
tasks (mean over 12 tasks: bge-base +0.015, bge-small +0.026) <!-- docs/PREREG.md: Starting numbers; results/bio_scores.csv -->.
They were built for the GO tasks, with those tasks' test items removed but
not the gene–disease test set's <!-- data/finetune/manifest.json: exclusion -->,
so any leak would favour them.

**Incomplete judgements and LLM judges.** Pooling misses relevant
documents and overstates recall [@zobel1998reliable]. When pools are
shallow for the size of the collection, the judged set skews toward
documents that contain the topic's title words [@buckley2007bias]. Bpref
was designed for incomplete judgements [@buckley2004bpref]; Sakai then
showed that standard metrics on condensed lists, with unjudged documents
removed, are more robust than bpref [@sakai2007bpref]. Our condensed
curator-only score applies that idea to the text-mined abstracts above;
the curator-only scores in Results are raw. The bias is concrete for dense
models in biomedicine: on TREC-COVID, dense retrievers left 14–32% of their top-10
results unjudged, against 6.4% for BM25 [@thakur2021beir]. LLM judges now
fill such gaps. Tuned against real searcher feedback, LLM labels match
human labellers [@thomas2024llm]; UMBRELA reproduces that method openly
[@upadhyay2024umbrela], and at TREC 2024 RAG its labels ranked 77 runs
much as NIST's assessors did [@upadhyay2025largescale]. The risks are
documented too. Using an LLM only for the holes moves new runs up or down,
depending on the LLM, so the whole pool should be judged
[@abbasiantaeb2024holes]. LLM judges can be fooled by query words planted
in a passage [@alaofi2024fooled], which matters here because many of our
negatives name both the gene and the disease. They favour LLM-based
rankers [@balog2025rankers; @frobe2025agree] and their own outputs
[@panickssery2024self], and they agree with one another more than with
human assessors [@frobe2025agree], so agreement between judges does not
show they are right. The evaluation also becomes circular when the systems
under test use LLMs [@faggioli2023perspectives; @clarke2025llm]. We
therefore judged the whole pool (the top 10 of every run and every
curator-cited abstract) with two judges from different model families,
DeepSeek-V4-Flash and Gemini 3.1 Flash-Lite. In the claims (C1, C8) a run
reranked by one of them is scored on the other judge's grades alone;
elsewhere such runs are scored on both judges' grades, which may flatter
them. We checked the judges (exploratory) against blind labels from four
human graders rather than against each other: an author graded 200
calibration abstracts and three colleagues graded subsets
<!-- docs/PREREG.md: change log 2026-11-03 -->. On the 91 abstracts that
the author and a colleague both graded, the judges matched the author about
as often as the colleague did (80% against 78%; difference +2 points, 95% CI
−8 to +12). The author matched himself on 78% of 27 usable repeats (of 30)
<!-- results/figures/judge_agreement_overlap.csv: overlap; results/figures/judge_agreement.csv: dan vs dan, repeat; methods.md: Human calibration -->.
The two judges agreed with each other on 80% of 199 abstracts; the author
and each colleague agreed on 75–79%, on smaller, different sets (20–47
labels)
<!-- results/figures/judge_agreement.csv: exact; judge vs judge n 199; human vs human, author vs each colleague, n 20, 42, 47 -->.
Across all 194 of the author's gradeable labels, the judges graded lower
than him 44 times and higher 14 times. An exploratory check traced this
mostly to DeepSeek, not to the rubric's wording (Results, "Judges checked
against human graders")
<!-- results/figures/judge_agreement_overlap.csv: direction; results/judge_wording_check.csv: arm same vs human, mean_grade -->.
We report curator-only scores beside the graded ones, and the key
mattered: tuned BM25 beat all 14 off-the-shelf embedders on the graded
key but trailed seven of them on the curator-only key (exploratory)
<!-- results/writeup/cost_quality.csv: curator_ndcg10, group single -->.
This fits either curators missing papers that keyword search finds or the
judges favouring abstracts that name both the gene and the disease
[@alaofi2024fooled]; we did not test between them.

**Memorisation.** The gold PMIDs for curated gene–disease links sit on
public database pages, so an LLM that ranks well may be remembering, not
reading. Memorisation grows with model size and duplication
[@carlini2023quantifying]. It can be detected from token probabilities
[@shi2024detecting] or by prompting [@golchin2024time], and contamination
must be measured benchmark by benchmark [@sainz2023contamination].
Memorisation of medical adaptation data, from continued pretraining or
fine-tuning, is common and higher than in the general domain
[@li2026memorization]. Free recall of citations is unreliable: 55% of
GPT-3.5's citations and 18% of GPT-4's were fabricated
[@walters2023fabrication]. LLM gene knowledge is uneven: in gene
prioritisation, highly cited genes such as BRCA1 and TP53 are more likely
to be predicted [@kim2024llmgene], and on GeneTuring the best results come
from an LLM connected to NCBI tools rather than from memory alone
[@shang2025geneturing]. We ran a closed-book probe that gave each LLM a
gene–disease pair and no abstracts, and asked for its key papers. Gemini
3.1 Flash-Lite, the LLM tested in C3, named a correct PMID or title for 9
of the 300 small-task test pairs <!-- results/memorisation_probe.parquet: GeneDiseaseRetrieval_v2_1..4, gemini-3.1-flash-lite, memorised -->.
That is below the registered minimum of 15, so the planned test of whether
an LLM's edge is larger on pairs it can recall (C4) was dropped
<!-- docs/PREREG.md: change log 2026-10-15, 2026-10-16; results/claims.csv: C4 -->.

**What is new here.** AMELIE, LitVar 2.0 and LitDD search all of PubMed,
but each serves one use (Mendelian diagnosis, variant lookup,
developmental-disorder curation) and scores its own pipeline; TREC 2005
asked our question with ten topics. A recent review of AI-assisted
biocuration names scarce training data among the main problems and calls
for shared benchmarks and harmonised evaluation [@wood2026empowering]. To
our knowledge, ours is the first model-agnostic benchmark of curated
gene–disease evidence retrieval at PubMed scale, with hundreds of
pair-level topics, same-gene distractors and a cost axis
<!-- "open" left out until the release is agreed: TODO(check), backmatter.md: Data and code availability -->.
It is built from the 38,392 PMIDs that four curation sources cite for
17,723 gene–disease pairs
<!-- data/finetune_gd/manifest.json: funnel_curated.curated -->. The main
test has 500 pairs. For each, systems rank the abstracts that PubTator3
tags with the gene, out of 28,332,361 PubMed abstracts
<!-- data/haystack/GeneDiseaseHaystack/manifest.json: stats.local_abstracts; docs/PREREG.md: test sets -->.

[^mteb-medical]: The `mteb` package groups them as the `MTEB(Medical, v1)`
    benchmark (12 retrieval, reranking and clustering tasks), which has no
    paper of its own.
