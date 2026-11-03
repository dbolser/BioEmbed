# Title options

*Draft, 2026-11-03. Same conventions as `methods.md`: each number is followed
by an HTML comment naming its source. Results numbers are taken from
`results.md` or the result files it cites; data facts from `methods.md`.
Only the Table 1 claims are confirmatory.*

1. **A small fine-tuned reranker rivals an LLM at finding gene–disease
   evidence in PubMed** (13 words). *Recommended*: it leads with the main
   result. Note that C8 tested superiority only (no equivalence margin) and
   its fine-tuned arm has one seed; "rivals" rests on the interval, −0.008 to
   +0.014, which excludes a gap larger than 0.008 in the LLM's favour or
   0.014 in ours.
2. **BioEmbed: finding the papers behind a gene–disease link** (8 words). The
   working title; plain, but does not say what we found.
3. **Embedders, rerankers and LLMs for gene–disease evidence retrieval: cost
   and quality** (11 words). Neutral; closest to *The Embedder's Dilemma*.

# Abstract

We know of no open benchmark for finding the papers behind curated
gene–disease links at PubMed scale<!-- related.md: Gene–disease literature retrieval; What is new here -->.
We built one from papers four curated sources cite<!-- intro.md: Open Targets sources; methods.md -->: 500 test gene–disease
pairs, each searched over PubMed abstracts tagged with its gene<!-- methods.md: Haystack (test); data/haystack/GeneDiseaseHaystack/manifest.json -->,
graded by two LLM judges checked against human graders. We tested nine
pre-registered claims. For three, their own test numbers were seen before the
rule was fixed; for others, related ones were<!-- results.md: Confirmatory claims; Table 1 ‡ and §; methods.md: Disclosures -->.
Our fine-tuned MedCPT cross-encoder, reranking a cheap first pass, scored
graded nDCG@10 0.861, top of 46 test runs (0.750 off the shelf)<!-- results.md: Table 2; "Good rerankers win"; results/haystack/graded/stats.csv: graded_ndcg@10, pooled, 46 systems -->.
On the non-Gemini judge's grades it was not shown different from a Gemini 3.1
Flash-Lite rerank of the same first pass (+0.003, 95% CI −0.008 to +0.014;
C8, superiority test only)<!-- results.md: Table 1 C8; results/claims.csv: C8 -->.
Its rerank step cost $0.04 per 500 queries against $0.88 for that Gemini
step, plus $0.40 training<!-- results.md: "The cheapest strong system needs no GPU"; results/writeup/cost_quality.csv: step_cost_usd 0.043 (MedCPT fine-tuned) vs 0.877 (Gemini rerank of first pass), train_cost_usd 0.403 -->.
Fine-tuned rerankers had one seed, and training negatives checked against the
judges' practice grades may favour our models; on curator citations alone the
reranker gain holds (0.393 to 0.477)<!-- results.md: Table 2 curator-only; "Fine-tuning helps rerankers most"; methods.md: Fine-tuning -->.
Tuned BM25 beat the best off-the-shelf embedder by 0.042 on the graded key
(C2) but trailed seven of 14 on curator citations<!-- results.md: Table 1 C2; The answer key section -->. On 91 abstracts graded
by one author and a colleague, the judges matched the author's grade about
as often as the colleague (80% vs 78%; difference 95% CI −8 to +12)<!-- results.md: Table 3; overlap paragraph; results/figures/judge_agreement_overlap.csv: analysis overlap -->,
but graded lower overall<!-- results.md: Judges checked against human graders; results/figures/judge_agreement_overlap.csv: analysis direction, 44 lower, 14 higher -->.
Whether small fine-tuned rerankers can replace LLM rerankers, at a twentieth
of the rerank step's cost, needs an equivalence test with more seeds<!-- results.md: C8 paragraph; results/writeup/cost_quality.csv: step_cost_usd 0.043 vs 0.877 -->.
