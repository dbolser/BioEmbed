# Related work

*Draft, 2026-10-15. Citations are Pandoc keys into `refs.bib`. Every entry
was checked against Crossref, PubMed, arXiv, medRxiv, the NIST TREC browser
or the publisher.*

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
to one in the best training run's top 20 (42.5% against 10.5%)
[@hersh2003trec]. LitDD machine-mapped about 69,000 PubMed case reports to
Gene2Phenotype's developmental-disorder entries [@yates2025litdd], while
Gene2Phenotype's evidence in Open Targets 26.06 cites 12,024 PMIDs across
all its panels.

**Mining gene–disease associations.** A separate line extracts
associations from text: BeFree and DisGeNET
[@bravo2015befree; @pinero2017disgenet; @pinero2020disgenet], DISEASES
[@pletscher2015diseases], the Europe PMC component of Open Targets
[@kafkas2017literature], and relation corpora such as BioRED
[@luo2022biored]. These systems decide whether a link exists. We take the
link as given and rank the evidence for it. We use text-mined links only as
a guard: an abstract that Europe PMC links to the query but curators do not
cite is never a negative, and is dropped before scoring.

**Entity-aware search at PubMed scale.** PubTator3 tags genes, diseases and
relations in about 36M abstracts, and reports that its entity-pair search
beats PubMed and Google Scholar on top-20 precision [@wei2024pubtator3].
LitSense 2.0 adds dense sentence and paragraph search over 38M abstracts
[@yeganova2025litsense2], and MedCPT trains a retriever and reranker on
255M PubMed click logs [@jin2023medcpt]. Three systems already search all
of PubMed for genetic evidence. AMELIE parses all 29M abstracts, plus full
texts, for evidence that a variant causes a Mendelian disease, and ranks a
patient's candidate genes with it [@birgmeier2020amelie]. LitVar 2.0
normalises variant mentions across PubMed, PMC full text and supplementary
files [@allot2023litvar2]. LitDD, described above, retrieves about 70% of
the papers in independent curated sets [@yates2025litdd]. None of the
three is a benchmark on which other rankers can be scored. We use
PubTator3's gene tags as a first-stage filter and compare rankers inside
it. Its disease tags make a weak filter: many Orphanet, HPO and EFO
diseases have no MeSH or OMIM identifier, and Mondo's cross-references,
built to bridge these vocabularies [@vasilevsky2025mondo], recover only one
more query disease.

**Cost and quality: embedders, rerankers, LLMs.** *The Embedder's Dilemma*
compared 10 LLMs with 26 embedders on 37 tasks. It found the two tied
overall, with LLMs costing up to 1,431× more and leading only on
reasoning-heavy retrieval [@elassadi2026embedders]. Its LLM retrieval puts
the whole corpus in context, following LOFT [@lee2024loft]. That cannot
reach 28M abstracts. At that scale the real options are lexical search
[@robertson2009bm25; @lu2024bm25s], a dense embedder
[@zhang2025qwen3embedding], cross-encoder reranking
[@nogueira2019monobert; @zhang2025qwen3embedding] or listwise LLM
reranking [@sun2023rankgpt] of a short list, or a cascade that spends
large-model compute only where it pays [@chen2023frugalgpt]. Biomedical
embedders sit on the same axis: MedCPT, BioLORD-2023 [@remy2024biolord],
MedEmbed [@balachandran2024medembed], PubMedBERT-based encoders
[@gu2021pubmedbert] and BMRetriever [@xu2024bmretriever]. We move the
Dilemma's frontier from sampled corpora to a full-corpus domain task, where
the choice is between cascades rather than single models.
<!-- TODO: confirm when run (reranker arms, C1/C7) -->

**Domain fine-tuning transfer.** In-domain pretraining helps biomedical NLP
[@gu2021pubmedbert], and pseudo-labelled adaptation helps dense retrieval
on BEIR sets, the biomedical ones included [@wang2022gpl]. BEIR also showed
that models trained on one retrieval dataset often transfer poorly to others
[@thakur2021beir]. We test a narrower case: does fine-tuning on one kind of
curated evidence (GO term to abstract) transfer to another (gene–disease
link to abstract), with every test gene, gene–disease pair and PMID
excluded from training? <!-- TODO: confirm when run (registry-clean
fine-tunes, C5a/C5b) -->

**Incomplete judgements and LLM judges.** Pooling misses relevant
documents and overstates recall [@zobel1998reliable]. When pools are
shallow for the size of the collection, the judged set skews toward
documents that contain the topic's title words [@buckley2007bias]. Bpref
was designed for incomplete judgements [@buckley2004bpref]; Sakai then
showed that standard metrics on condensed lists, with unjudged documents
removed, are more robust than bpref [@sakai2007bpref]. Our condensed
scoring applies that idea to abstracts that text mining links to the query
but curators do not cite. The bias is concrete for dense models in
biomedicine: on TREC-COVID, dense retrievers left 14–32% of their top-10
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
therefore judge the whole pool, use two judges from different model
families plus a third family on LLM-reranked documents, calibrate the
judges against blind expert labels rather than against each other, and
report gold-only recall alongside pooled nDCG. <!-- TODO: confirm when run
(judges, rubric and labelling time pending Dan's sign-off in PREREG) -->

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
[@shang2025geneturing]. We add a closed-book probe that asks each LLM for
the key PMIDs per query, and test whether the LLM–embedder gap differs on
queries the LLM can recall. <!-- TODO: confirm when run (probe, C4) -->

**What is new here.** AMELIE, LitVar 2.0 and LitDD search all of PubMed,
but each serves one use (Mendelian diagnosis, variant lookup,
developmental-disorder curation) and scores its own pipeline; TREC 2005
asked our question with ten topics. To our knowledge, this is the first
open, model-agnostic benchmark of curated gene–disease evidence retrieval
at PubMed scale, with hundreds of pair-level topics, same-gene distractors
and a cost axis. Specifically:

1. **A curated-citation gold standard.** 17.7k gene–disease pairs citing
   38.4k PubMed articles, from four curation sources via Open Targets, with
   text-mined links used as a guard. It yields two tasks: a 300-document
   task whose distractors share the gene or the disease, and a 28.3M-abstract
   PubMed haystack with 500 frozen test queries, a gene-disjoint dev set and
   a PubTator3 gene-filtered track.
2. **A cost–quality frontier at full scale.** BM25, general and biomedical
   embedders, cross-encoder and LLM rerankers, and corpus-in-context LLMs on
   one dollar axis. This extends *The Embedder's Dilemma* from sampled
   corpora to cascades over all of PubMed. <!-- TODO: confirm when run
   (rerankers C1/C7, corpus-in-context LLMs C3) -->
3. **Evaluation that assumes the gold is incomplete.** Condensed scoring,
   LLM-judged pooling with cross-family judges calibrated against expert
   labels, and pre-registered claims with Holm correction. <!-- TODO:
   confirm when run (pooled judging) -->
4. **Two controls earlier biomedical retrieval benchmarks lack.** A
   closed-book memorisation probe for the LLMs, and a test of whether
   GO-evidence fine-tuning transfers to gene–disease retrieval under a
   strict exclusion registry. <!-- TODO: confirm when run (C4, C5a/C5b) -->

[^mteb-medical]: The `mteb` package groups them as the `MTEB(Medical, v1)`
    benchmark (12 retrieval, reranking and clustering tasks), which has no
    paper of its own.
