# genesearch

Type a gene (or click "Gene ⇄" for a disease), see its literature: PubMed abstracts tagged by PubTator3, Open
Targets curated diseases (genes, for a disease), and an "Optionally rank by" box that ranks the papers by keywords,
by meaning (AI embeddings), or both. A Map tab groups the current list into topics; Map Explorer puts the papers on
axes you choose. `/about` lists the data and how the map is made.

1. Build the index once (~10-15 min, 61 GB in `index/`; needs `data/gold/ot_curated_gene_disease_pmid.parquet`
   from the gene-disease work): `cd genesearch && uv run python build_index.py`.
   Then `uv run python build_named.py` (~1 h on the 754 GB box, 369 MB; ~0.9 GB in the app's memory): 'Named in
   the abstract' for every gene and disease, so a search reads no abstracts for it (on a cloud disk the live count
   took up to 2 min for a gene and held up the next search; breast cancer took 18 s of CPU). `--genes-only`: the
   genes alone in ~4 min (54 MB), diseases then counted live. Build it on a big machine and copy it to a server
   (with any index file that changed, keeping file times: `rsync -a`), with the server's Python version: its 16
   workers take up to ~5 GB each for a gene, more for the biggest diseases (neoplasm: 4.4M papers). Rerun it after
   changing the naming code or rewriting an index file it reads (a new size or time): until then the app counts
   live, and its startup line and `/api/about` say why (`named_stamp` in `app.py`).
   Then, optionally, `uv run python build_maps.py` (a few hours with 14 workers on the 112-core box, ~16 GB, ~$12 of
   LLM topic names): every gene's and disease's map, so the Map tab opens at once. Copy `index/cache/maps` (and
   `maps_manifest.jsonl`) to a server with the index. Rerun it after an index rebuild (a map is remade only if its
   list of papers changed: if only vectors, titles or years did, bump `MAP_VERSION` in `app.py` first).
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
   `&view=explore&ax=cancer&ay=topic:2`, `&view=map&pmid=27083884`: the paper clicked on a map, opened under it).
7. **PubMedBERT map** tab (`&view=landscape`): the current list (facets applied) on the published map of PubMed by
   González-Márquez et al. (Patterns 2024): PubMedBERT embeddings of 23.4M abstracts, t-SNE to 2-D, fields from the
   journal. Behind the dots: its 6.26M human-gene papers (PubTator3 tags of HGNC genes) in their field colours. The map
   ends in early 2024; newer papers are counted, not drawn. At most 50,000 dots (a fixed sample beyond that). Build
   once (~2 min) with `uv run python landscape.py`, from `/data/public/PubMedLandscape/2024_v2/` (see its README).
8. **MedEmbed map** tab (`&view=medmap`): the same, on our own map (`../geometry/README.md`): a t-SNE of the search's
   MedEmbed-small embeddings of the 7.76M human-gene papers, coloured by 780 clusters found in a 12-D UMAP of the same
   vectors and named by their most typical genes (some names are PubTator tagging errors, e.g. "TNM", "AST"). The 25
   clusters holding most of the list's papers are labelled. Build (~10 s, after the geometry runs) with
   `uv run python landscape.py medembed`; from `data/geometry/medembed_small/`.

How it works

- Papers: PubTator3 gene tags (NCBI text mining, precomputed), Open Targets curated links, and Open Targets'
  Europe PMC text-mined links (the ePMC box), which add papers PubTator did not tag. The list and the map hold
  all three.
- Keywords: BM25 computed live over the gene's papers (`app.py`). With several words, the count is papers
  containing *all* of them (a link shows any-word matches). No AI.
- Semantic: `abhinand/MedEmbed-small-v0.1` (loaded as `eval/run_embedders.py` does: CLS pooling, normalised,
  no query prompt). Query = the ranking words; papers = `title\n\nabstract`; cosine similarity (`semantic.py`).
  Hybrid = reciprocal-rank fusion, 1/(60 + keyword rank) + 1/(60 + semantic rank).
- Named in the abstract: the paper's PubTator3 mentions of the gene (or disease) plus its symbol, aliases,
  previous symbols and HGNC names (a disease: name and synonyms), found whole in the title or abstract, as the page
  highlights them (`named_pmids` in `app.py`). HGNC names: the approved name ('aquaporin 4') and the alias and
  previous names that aren't disease names (`gene_names`: TP53's alias name is 'Li-Fraumeni syndrome'). PubTator3
  writes Greek letters as names and Unicode dashes and spaces as plain ones ('TGF-beta1' for TGF-β1), HGNC writes
  'aquaporin 4' where papers write 'aquaporin-4', and a few names keep the letter ('β-actin'): a Greek letter and its
  name match each other, and any run of dashes and spaces matches any other (`loose` in `app.py` and in
  `index.html`). Each name's pattern goes to RE2 for the common ones and to Python's `re` for the long tail of rare
  spellings (`word_re`), so both find the same papers (but for a few with the Turkish ı or İ, which Python's `re`
  takes for i). Before the Greek fix IFNG, TGFB1 and IL1B had about half their naming papers missed; before the HGNC
  names, papers linked only by MeSH or curation that wrote 'aquaporin-4' counted as not naming AQP4.
- A paper that does not name it says why it is listed, one sentence per source: PubTator3 tagged it in the full
  text (its words not being in the title or abstract), a curated source PubTator3 merges in (MeSH indexing,
  gene2pubmed, GeneRIF, CTD, BioGRID...: the `resource` column of `gene_pmids`/`disease_pmids`, `tag_sources` in
  `app.py`), Open Targets curation, or Europe PMC text mining. With an index built before that column, papers don't
  name the curated source (the app says so at startup): rebuild the two files with
  `uv run python build_index.py gene_pmids disease_pmids`.
- Paper vectors: from `index/emb_medembed_small/` (whole PubMed: float16 `.npy` shards + `pmids.npy`,
  memory-mapped) once that exists and is complete. Until then they are computed on CPU for one gene's papers
  when first needed (~50-130 papers/s, shown as a progress bar), capped at the 30,000 best keyword matches,
  and kept in `index/cache/emb/`. Worker count: `GENESEARCH_EMBED_WORKERS` (12) x `GENESEARCH_EMBED_THREADS` (4).
- Map (`litmap.py`): up to 20,000 papers (a fixed random sample beyond that), UMAP (cosine) to 12-D, HDBSCAN
  topics (at most 8, one colour each; papers HDBSCAN leaves out join the nearest topic unless they are among
  the 10% it rates least typical: those stay grey), UMAP to 2-D for drawing; topics named by c-TF-IDF top terms, or by Gemini 3.1
  Flash-Lite via OpenRouter (key in `llm/.env.or-google_gemini-3.1-flash-lite`). Maps and names are cached in
  `index/cache/`.

Papers with no abstract carry a small "no abstract" tag; those not in the local PubMed copy at all are listed last.
The copy ends at a date and holds only papers with an abstract: `backfill_abstracts.py` fetches the rest (title,
abstract, year, journal) from PubMed, merges them in and embeds them on CPU as one more shard.
