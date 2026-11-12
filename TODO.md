# BioEmbed: roadmap across the threads

Written 2026-11-02. One list, so the threads stay countable. Details live with each thread.

## The threads, and what each becomes

| Thread | Where | State | Becomes |
|---|---|---|---|
| A. The Embedder's Dilemma, for biology | `README.md` Findings, `docs/WRITEUP.md`, `docs/PREREG.md` | 9 claims run and Holm-corrected; write-up drafted; judge agreement with Dan's grades in progress (`labeller/`) | **Paper 1**, the closest to done |
| B. genesearch | `genesearch/` | live on outsee4:8766; list, maps, two whole-literature maps | the tool behind A and C; not a paper of its own (an application note at most, later) |
| C. Geometry of the human-gene literature | `geometry/README.md`, `geometry/TODO.md` | one model done: density, UMAP vs t-SNE, 780 clusters | **Paper 2**, with D as its lead result |
| D. Embedding maps expose gene-tagging errors | `geometry/TODO.md` item 2 | found: SQLE via "SE", TNM, AST, Fe, CAR...; not yet measured over all genes | the most new and useful part of Paper 2 |
| E. Across embedding models | `geometry/TODO.md` item 1 | not started; PubMedBERT vectors on disk | Paper 2's second half, or a later paper; not needed for D |

So: two papers, one tool. New ideas go under a paper or wait in "Later".

## Next, in order

1. **Paper 1 out.** Finish the judge-agreement check with Dan's grades, settle the write-up, bioRxiv.
2. **D measured** (`geometry/TODO.md` item 2): over every gene, the share of its papers tagged only by a bare short
   symbol or a family plural; the worst offenders; how many of the 7.76M "human-gene papers" mention no gene. A few
   days, CPU only.
3. **D checked**: hand-check a sample of flagged and unflagged clusters (does off-topic mean mis-tagged?), and score
   PubTator3 on the NLM-Gene test set (100 hand-annotated abstracts) as a calibration of its gene tags.
4. **Paper 2 drafted** around D, with C's maps and clusters as the method. E only if it is cheap by then.

## Smaller items, captured

- [ ] Merge PR #8, then restart :8766 so the MedEmbed map tab goes live (`landscape.py medembed` already built).
- [ ] NLM-Gene as a BioEmbed test, beside the search tasks: which abstracts are about a gene, hand-labelled
      (test split only: PubTator's tagger likely trained on the rest).
- [ ] Better cluster names: litmap's LLM naming (cached) for the 780 clusters, instead of their typical genes.
- [ ] A static version of the MedEmbed map on GitHub Pages: background image, cluster names, and per-gene dots from
      precomputed files (the full search needs the server).
- [ ] Clean the gene tags (from step 2), then rebuild the corpus, maps and scores.
- [ ] genesearch, disease specificity: rank down papers tagged with many diseases, e.g. an information-content
      weight (fewer diseases tagged = more specific), next to the yes/no "named in the title or abstract" filter.
- [ ] genesearch, review filter: show only (or hide) review papers. Europe PMC gives a publication type
      (`pubTypeList`, e.g. "review") and PubMed has the "Review" publication type.
- [ ] genesearch, hybrid ranking: show the semantic similarity (and BM25 score) as well as the two ranks. Today
      semantic mode shows "semantic match 0.79" but hybrid mode only "semantic rank 5", so the strength is hidden.
- [ ] genesearch, Genes and Diseases tabs: merge the diseases from PubTator3 and Europe PMC into one list, so the same
      disease from both sources is one row (raised 2026-11-12 on the dev copy, branch t3/explore-disease-gene-facets).

## Later (not now)

- genesearch, design question: can a reader tell "a tenuous link supported by a couple of papers" from "core
  disease biology"? A ranked list always has a top paper, whatever the query. Options: how many of the disease's
  papers match well (above a similarity threshold), compared with the same query's rate over all papers (an
  enrichment, like the specificity weight above); or a clear "few matches" message. Raised while linking the
  IUK-IPF term labeller to genesearch with a term's name and definition as the query.
- Cross-model geometry (E) beyond PubMedBERT; Leiden and 384-D cosine HDBSCAN baselines; seeds and error bars
  (all in `geometry/TODO.md`).
