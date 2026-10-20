# genesearch

Type a gene (or click "Gene ⇄" for a disease), see its literature: PubMed abstracts tagged by PubTator3, Open
Targets curated diseases (genes, for a disease), and an "Optionally rank by" box that ranks the papers by keywords,
by meaning (AI embeddings), or both. A Map tab groups the current list into topics; Map Explorer puts the papers on
axes you choose. `/about` lists the data and how the map is made.

1. Build the index once (~10-15 min, 61 GB in `index/`; needs `data/gold/ot_curated_gene_disease_pmid.parquet`
   from the gene-disease work): `cd genesearch && uv run python build_index.py`
2. Run: `cd genesearch && uv run python app.py --port 8766 --host 0.0.0.0`, then open http://localhost:8766
3. Type a symbol, alias, old symbol or name (`/` jumps to the box); pick from the list if several genes match.
   Disease search (click the box's label): Open Targets diseases by name, synonym or PubTator3 wording
   (`diseases.parquet`, see `build_index.py diseases`), plus PubTator3 ids no Open Targets disease covers.
4. Optional words rank the papers (e.g. ZCCHC8 + pulmonary fibrosis); empty = newest first. **Rank method**:
   Semantic (the default; every paper, closest in meaning first), Keyword (BM25, only papers with the words), or
   Hybrid (reciprocal rank fusion).
5. Facet boxes on the left: **Named in the abstract** (only papers whose title or abstract names the gene or disease),
   **OT** (Open Targets curated) and **ePMC** (Europe PMC text-mined). Click a disease (a gene, in disease search) to
   filter (with keyword ranking this clears the words), or "Any" for all of that box's papers; one in each box = papers
   in both, and each box's counts then follow the other box's pick. Papers carry OT and ePMC chips that do the same.
   Active filters are listed above the results, each with a ✕. "How this search works" is under the (i).
6. **Map** tab: the current list as a topic map (without the facet boxes: picking or hovering over one greys out
   the papers it leaves out; `/api/map_facets`). Play years steps through time; Shift over a dot picks out its
   topic, Esc clears; "Drill in" re-maps one topic's papers. **Map Explorer**: each axis is closeness in meaning to
   your words or to a topic's centre (cosine of the MedEmbed vectors). "Download CSV" saves the whole list. URLs are
   shareable (`?gene=CFTR&q=sweat+chloride&rank=hybrid`, `?disease=cystic+fibrosis`, `&ot=any`, `&ep=<id>`, `&named=1`, `&view=map&topic=3`,
   `&view=explore&ax=cancer&ay=topic:2`).

How it works

- Papers: PubTator3 gene tags (NCBI text mining, precomputed) plus Open Targets curated links; the ePMC box
  uses Open Targets' Europe PMC evidence, which can add papers PubTator did not tag.
- Keywords: BM25 computed live over the gene's papers (`app.py`). With several words, the count is papers
  containing *all* of them (a link shows any-word matches). No AI.
- Semantic: `abhinand/MedEmbed-small-v0.1` (loaded as `eval/run_embedders.py` does: CLS pooling, normalised,
  no query prompt). Query = the ranking words; papers = `title\n\nabstract`; cosine similarity (`semantic.py`).
  Hybrid = reciprocal-rank fusion, 1/(60 + keyword rank) + 1/(60 + semantic rank).
- Named in the abstract: the paper's PubTator3 mentions of the gene (or disease) plus its symbol (name), found
  whole in the title or abstract, as the page highlights them (`named_pmids` in `app.py`).
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
