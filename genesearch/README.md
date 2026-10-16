# genesearch

Type a gene, see its literature: PubMed abstracts tagged by PubTator3, Open Targets curated diseases, and an
"and..." box that ranks the gene's papers by keywords, by meaning (AI embeddings), or both. A Map tab groups
the current list into topics.

1. Build the index once (~10-15 min, 61 GB in `index/`): `uv run python build_index.py`
2. Run: `cd genesearch && uv run python app.py --port 8766 --host 0.0.0.0`, then open http://localhost:8766
3. Type a symbol, alias, old symbol or name (`/` jumps to the box); pick from the list if several genes match.
4. Optional "and..." words rank the gene's papers (e.g. ZCCHC8 + pulmonary fibrosis); empty = newest first.
   **Rank by**: Keywords (BM25, only papers with the words), Meaning (every paper, closest first), or Both (hybrid).
5. Click a curated or text-mined disease to filter (this clears the "and..." words). Click the green **Curated**
   badge to show only Open Targets curated papers. Active filters are listed above the results, each with a ✕.
6. **Map** tab: the current list as a topic map. "Download CSV" saves the whole list. URLs are shareable
   (`?gene=CFTR&q=sweat+chloride&rank=hybrid`, `&view=map`).

How it works

- Papers: PubTator3 gene tags (NCBI text mining, precomputed) plus Open Targets curated links.
- Keywords: BM25 computed live over the gene's papers (`app.py`). With several words, the count is papers
  containing *all* of them (a link shows any-word matches). No AI.
- Meaning: `abhinand/MedEmbed-small-v0.1` (loaded as `eval/run_embedders.py` does: CLS pooling, normalised,
  no query prompt). Query = the "and..." words; papers = `title\n\nabstract`; cosine similarity (`semantic.py`).
  Hybrid = reciprocal-rank fusion, 1/(60 + keyword rank) + 1/(60 + meaning rank).
- Paper vectors: from `index/emb_medembed_small/` (whole PubMed: float16 `.npy` shards + `pmids.npy`,
  memory-mapped) once that exists and is complete. Until then they are computed on CPU for one gene's papers
  when first needed (~50-130 papers/s, shown as a progress bar), capped at the 30,000 best keyword matches,
  and kept in `index/cache/emb/`. Worker count: `GENESEARCH_EMBED_WORKERS` (12) x `GENESEARCH_EMBED_THREADS` (4).
- Map (`litmap.py`): up to 20,000 papers (a fixed random sample beyond that), UMAP (cosine) to 12-D, HDBSCAN
  topics (at most 8, one colour each; papers HDBSCAN leaves out join the nearest topic unless they are among
  the 10% it rates least typical: those stay grey), UMAP to 2-D for drawing; topics named by c-TF-IDF top terms, or by Gemini 3.1
  Flash-Lite via OpenRouter (key in `llm/.env.or-google_gemini-3.1-flash-lite`). Maps and names are cached in
  `index/cache/`.

Papers with no abstract in the local PubMed copy are listed last with a small "no abstract" tag.
