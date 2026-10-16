# genesearch

Type a gene, see its literature: PubMed abstracts tagged by PubTator3, Open Targets curated diseases, BM25 "and..." search.

1. Build the index once (~10-15 min, 61 GB in `index/`): `uv run python build_index.py`
2. Run: `cd genesearch && uv run python app.py --port 8766 --host 0.0.0.0`, then open http://localhost:8766
3. Type a symbol, alias, old symbol or name (`/` jumps to the box); pick from the list if several genes match.
4. Optional "and..." box ranks the gene's papers by BM25 (e.g. ZCCHC8 + pulmonary fibrosis); empty = newest first.
5. Click a curated disease to filter; "Download CSV" saves the whole current list. URLs are shareable (`?gene=CFTR&q=sweat`).

Notes: with several words, the count is papers containing *all* of them (a link shows any-word matches).
Papers listed as "no abstract in the local PubMed copy" are mostly older items with no abstract (letters,
editorials), plus some newer than the copy; they have no title or year here and are listed last.
