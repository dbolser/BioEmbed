# Pre-registration: claims, metrics and tests (gene–disease paper)

Written 2026-10-15, before any of the runs it governs (v2 tasks, pooled
judging, rerankers, new fine-tunes). Committed to git so the timestamp is
the record. Any later change is an **amendment**: add it at the bottom with
its date and reason; never edit the text above it.

Defaults below were set by Claude and are pending Dan's sign-off. The calls
that are Dan's are marked **[Dan]**. If he changes one before the result it
governs exists, that is an amendment, not a deviation.

## Test sets (frozen)

- **Haystack test**: the 500 queries in `data/haystack/GeneDiseaseHaystack`
  as built on 2026-10-14 (seed 42). Never resampled. Strata are added as
  labels only.
- **Haystack dev**: a separate 200-query, gene-disjoint set. All selection
  happens here and only here: BM25 query format and k1/b, dense
  instructions, fine-tune checkpoints, reranker depth.
- **GeneDiseaseRetrieval v2**: 5 gene-disjoint 300-doc instances (seeds
  42–46), disjoint from the test and dev haystack genes. Pooled over
  instances for testing.
- **v1 (75 queries)** is superseded by v2. It is reported as an appendix
  with its known builder issue.

## Relevance labels

Rubric (judges and Dan use the same text) **[Dan: rubric wording]**:

- **2**: the abstract reports evidence that this gene is involved in this
  disease (or a synonym, subtype, or the same phenotype in a model organism
  of the gene).
- **1**: the abstract discusses both, but the link is not its evidence (a
  review mention, a candidate in a screen with no result, or a related
  disease).
- **0**: otherwise.

Judges: DeepSeek-V4-Flash (thinking off) and Gemini 3.1 Flash-Lite. A
third-family judge (Qwen3.6-35B) is used on LLM-reranker top-10 docs to
control self-preference. Judge labels are calibrated against Dan's blind
labels on 150–200 stratified pairs **[Dan: labelling time]**.

## Metrics

- **Haystack primary**: pooled graded nDCG@10 (gains 0/1/2), mean over the
  500 test queries. Graded gains mean no relevance threshold is needed for
  the primary metric.
- **Haystack secondary**: pooled P@10 and recall@100 at threshold ≥1, and
  again at =2 **[Dan: which threshold is primary for binary metrics; default
  ≥1]**; gold-only raw recall@10/100 (comparable to v1.3); bpref.
- **v2 primary**: hit@1 (one gold per query, 4 candidates per query share
  the gene or disease), pooled over 5 instances. Secondary: MRR.
- **Similarity**: each model's documented similarity function. MedCPT uses
  CLS pooling with dot product, as on its model card; cosine is also
  reported.

## Claim family (Holm-corrected, α = 0.05)

All tests are paired over queries (two-sided). Haystack: sign-flip
permutation test on per-query nDCG@10 (10,000 permutations). v2: exact
McNemar on per-query hit@1, with the instance-clustered bootstrap reported
alongside.

| id | claim (H1) | data | test |
|---|---|---|---|
| C1 | best first stage + best reranker > best single embedder | haystack test | sign-flip |
| C2 | best single embedder > dev-tuned BM25 | haystack test | sign-flip |
| C3 | best corpus-in-context LLM ≠ best embedder | v2 | McNemar |
| C4 | LLM−embedder gap differs between probe-memorised and other queries | v2 | permutation of the interaction |
| C5a | registry-clean bge-small-gaf (3-seed mean) ≠ bge-small | haystack test | sign-flip |
| C5b | registry-clean bge-base-gaf (3-seed mean) ≠ bge-base | haystack test | sign-flip |
| C6 | gene-anchor GAF control > GAF (the drop is a query-format effect) | haystack test | sign-flip |
| C7 | gene–disease fine-tuned reranker > best off-the-shelf reranker | haystack test | sign-flip |
| C8 | gene–disease fine-tuned bge-base > bge-base | haystack test | sign-flip |
| C9 | Gemini 3.1 Pro ≠ Qwen3-E-8B on the 12-task suite | suite | exact sign-flip over tasks |

"Best" in C1–C3, C7 means best **on dev**, fixed before the test run. If a
claim's arm is not run (e.g. no Gemini Pro budget), the claim is dropped
from the family and the drop is recorded here; Holm is applied to the rest.

## Exploratory (never sold as confirmatory)

Strata (ontology, curated source, gold-count band, gene-set size, ClinGen
class, gold year), the Recent slice, TREC coverage, gold-only vs pooled
Kendall τ, gold precision by source, the unfiltered BM25 track, API
embedders, and any analysis not in the table above.

## Fixed before any new run

- MedCPT: CLS pooling, dot product, query max length 64, article 512.
- Doc text: `title\n\nabstract`, title not prepended again by the loader.
- Exclusion registry: every test/dev PMID, gene–disease pair and gene is
  excluded from all fine-tune data; overlap counts are logged.
- Starting numbers (exact sign-flip over 12 tasks, `results/bio_scores.csv`):
  Pro − Qwen3-E-8B +0.027 (p = 0.35, 6/12 wins); bge-small-gaf − bge-small
  +0.026 (p = 0.008, 10/12); bge-base-gaf − bge-base +0.015 (p = 0.11).

## Amendments

(none yet)
