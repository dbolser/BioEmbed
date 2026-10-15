# Pre-registration (draft 2, plain language)

**What this is.** We write down the questions, the test sets, the scores and
the yes/no tests *before* running the experiments, and commit it to git so
the date is on record. That stops us (or a reviewer suspecting us) from
picking the flattering analysis after seeing the results. Changes are
allowed, but each one goes in the change log at the bottom with its date.

**Waiting on Dan:** (1) is the relevance rubric right? (2) for the yes/no
scores, does "relevant" mean grade 1 or 2 (default), or only 2?

## The three questions

1. **Finding the papers.** For a gene–disease pair, how well can each
   method find the papers that support the link in PubMed, and what does it
   cost? Methods: keyword search (BM25), embedding models, rerankers, LLMs.
2. **What fine-tuning buys.** How much better is a model trained on
   biomedical text than the same kind of model off the shelf? We compare:
   general models (e.g. bge), published PubMed-trained models (MedCPT,
   BiCA, BMRetriever, MedEmbed), and our own model fine-tuned on curated
   gene–disease evidence.
3. **The answer key.** Curators' citations are a free answer key, but they
   miss papers. How much do they miss, and does it change which method
   looks best?

## The test sets

| name | what it is | size | used for |
|---|---|---|---|
| **Haystack** (test) | For each gene–disease pair, every PubMed abstract that PubTator tags with the gene (median ~1,100 per pair). The method must rank the supporting papers to the top. | 500 pairs, 1.7M abstracts | the main results |
| **Practice haystack** | Same kind, but 200 *other* pairs with *other* genes. | 200 pairs | tuning settings only (like a practice exam) |
| **Small task** (v2) | Each pair comes with 4 papers: 1 right answer, 2 about the same gene but an unrelated disease, 1 about the same disease but another gene. Pick the right one. Small enough for an LLM to read everything. | 375 pairs in 5 batches of 75 | LLM vs embedder |
| Old small task (v1) | Earlier version with a bug in how wrong answers were picked. | 75 pairs | appendix only |

Rules:

- The test haystack was built on 2026-10-14 and is **never changed**.
- Every setting (keyword query style, model instructions, which fine-tuned
  checkpoint, how many results to rerank) is chosen on the **practice**
  set, then run once on the test set.
- No gene appears in more than one of the haystack, the practice haystack
  and the small-task batches. Nothing in any test set is used to fine-tune.

## The answer key (labels)

- **Curators' citations** (Open Targets: Orphanet, Gene2Phenotype, Genomics
  England, UniProt) are the starting answer key.
- They miss papers, so we also **grade** the papers methods actually
  return: the top 10 from every method, pooled. Two LLMs grade each one 0,
  1 or 2. Dan grades a blind sample of ~200 in the labelling app, and we
  report how well the LLM grades agree with his.

Rubric (the same text for LLMs and Dan):

- **2**: the abstract reports evidence that this gene is involved in this
  disease (or a synonym, subtype, or the same phenotype in a model organism
  of the gene).
- **1**: the abstract discusses both, but the link is not its evidence (a
  review mention, a candidate in a screen with no result, or a related
  disease).
- **0**: otherwise.

## How we score

- **Small task: hit@1.** How often the method puts the right paper first.
  Guessing scores 0.25.
- **Haystack, main score: nDCG@10.** Looks at the top 10 results. A paper
  graded 2 counts twice as much as a 1, and higher positions count more.
  1.0 is perfect.
- **Haystack, also shown:**
  - *recall@100*: the share of known good papers that are in the top 100.
    This matters when a reranker re-sorts the top 100.
  - *precision@10*: the share of the top 10 that are relevant.
  - *bpref*: a standard score built for answer keys with gaps.
  - *curator-only scores*: the same scores counting only curator-cited
    papers as right (the way RESULTS.md v1.3 scored). Shown so you can see
    how much the graded answer key changes the picture.
- Each model is used the way its authors intend (e.g. MedCPT: first-token
  vector, dot product).

## The claims we test

Each claim is a yes/no question. We compare two methods pair by pair
(paired test, 10,000 random shuffles). Because we test 9 claims, the bar is
raised so a lucky result can't pass (Holm correction, overall 5%).
"Best" always means best **on the practice set**, chosen before the test run.

| # | question | test set |
|---|---|---|
| C1 | Does a cheap first pass + a reranker beat the best single embedder? | haystack |
| C2 | Does the best embedder beat keyword search (BM25, tuned on practice)? | haystack |
| C3 | Does the best LLM, reading all 4 papers, beat the best embedder? | small task |
| C4 | Is an LLM's edge bigger on pairs where it can name the paper from memory? | small task |
| C5 | Does our gene–disease fine-tune beat the same model off the shelf? | haystack |
| C6 | Does our fine-tune beat the best published PubMed-trained model of similar size? | haystack |
| C7 | Does the best published PubMed-trained model beat the best general model of similar size? | haystack |
| C8 | Does our fine-tuned reranker beat the best off-the-shelf reranker? | haystack |
| C9 | Is Gemini 3.1 Pro better than Qwen3-Embedding-8B on the 12-task suite? (expect: no) | suite |

If a claim's experiment is never run, it is dropped here and the bar is
recomputed for the rest.

Precise version, for reviewers: two-sided paired sign-flip permutation
tests on per-query nDCG@10 (haystack) or exact McNemar on per-query hit@1
(small task), 10,000 permutations, Holm–Bonferroni at α = 0.05 across
C1–C9; C4 tests the interaction by permuting the memorised label; C9 is an
exact sign-flip over the 12 tasks.

## Looked at, but not claims

These are reported as observations, never as proven:

- Breakdowns by disease type, curation source, number of known papers,
  gene popularity, and recent papers only (2025 or later).
- Whether the method ranking changes between the curator-only and the
  graded answer key.
- How much each curation source's citations hold up when graded.
- Our earlier GO fine-tunes (bge-gaf) on gene–disease search.
- Keyword search over all 28M abstracts with no gene filter.
- Paid embedding APIs, if run.

## Starting numbers (for the record)

12-task suite, exact sign-flip over tasks, from `results/bio_scores.csv`:
Gemini 3.1 Pro − Qwen3-E-8B +0.027 (p = 0.35, wins 6/12);
bge-small-gaf − bge-small +0.026 (p = 0.008, 10/12);
bge-base-gaf − bge-base +0.015 (p = 0.11, 7/12).

## Change log

- 2026-10-15 18:18: draft 1.
- 2026-10-15: draft 2, before sign-off. Rewritten in plain words at Dan's
  request. Fine-tuning claims refocused on what fine-tuning buys (Dan's
  priority), so the GO-transfer claims moved to "not claims", and published
  PubMed-trained models (BiCA, BMRetriever) added. Bug-fix items removed
  (they are fixes, not choices). At this point the only new results were
  the bug-fix reruns, BM25 setting selection on the practice set with one
  selected BM25 test run, and the memory probe. The claims those touch (C2,
  C4) did not change.
- 2026-10-15 (after phase 1 builds, before any claim's test run except the
  BM25 one above):
  - **C4 memory label.** A model "remembers" a pair if, given no papers, it
    names a right PMID or a right title (fuzzy match, rapidfuzz
    token_sort_ratio ≥ 90; a looser rule gave 23% false hits on made-up
    titles, this one 1%). C4 uses each LLM's own label. If fewer than 15
    small-task pairs are remembered by the LLM tested, C4 is dropped.
  - **Small task (v2), as built.** Phenotype-only (HP) diseases are left
    out. A wrong answer on the same gene may not name the query disease, and
    may not be linked by any Open Targets source to it or a related disease.
    Chance is 1 in 300 for a method that ranks everything, 1 in 4 for one
    that already knows the gene and disease.
  - **Practice haystack, as built.** 200 pairs, no shared genes with the
    test haystack, genes with over 20,000 tagged abstracts left out (6% of
    test pairs are bigger than that).
  - **Clean answer key** (a check, not a claim): curator papers that PubTator
    tags with the gene, cited for fewer than 10 genes, and from Genomics
    England only if rated green.
- 2026-10-15 22:45 (before phase 2 runs):
  - **Small task practice batch.** Batch 0 (75 pairs) is the practice set
    for C3/C4: the best embedder and the best LLM are picked on it. C3 and
    C4 are tested on batches 1–4 (300 pairs).
  - **Rerank depths, fixed now.** Cross-encoders and fine-tuned rerankers
    re-sort the top 100 of the first pass; LLM rerankers re-sort the top 20.
    Not tuned.
  - **First-pass candidates for C1:** tuned BM25, each dense embedder, and
    BM25 + dense fused by reciprocal rank (k = 60). The best on the practice
    haystack by recall@100 (curator-only) is the first pass for C1.
  - **"Similar size" for C6/C7:** under 150M parameters (bge-base class).
  - **Haystack selection metric** until graded practice labels exist:
    curator-only nDCG@10 on the practice haystack.
