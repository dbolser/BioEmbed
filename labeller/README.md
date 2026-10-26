# labeller

Blind 0/1/2 relevance labelling of PubMed abstracts for gene–disease pairs (gold labels for judge calibration).

```bash
cd labeller && uv run python make_demo_items.py                       # -> items_demo.parquet (3 rounds x 10)
uv run python app.py --items items_demo.parquet --port 8765 --host 0.0.0.0   # open http://localhost:8765
# keys: 2/1/0 rate, U unsure, F flag, N note, Backspace/Left back, Right forward, C continue, R guide (narrow windows)
uv run python app.py --export                                          # -> export/labels.csv + labels.parquet
# repeats: a round with is_repeat items is split into 2 equal parts; part 2 mixes the repeats with fresh
# abstracts and opens >= --repeat-gap rounds and --repeat-hours (8) after part 1. Test now: --repeat-hours 0
```

Labels live in `labels.sqlite` (`labels` = latest answer per labeller/item/pass, `log` = every submission).
The browser only ever sees opaque handles: no item_id, stratum or repeat flag, and an abstract's PMID
(for the PubMed link) only once it is rated.
Item builders should clean `gene_aliases` as `make_demo_items.py` does (drop ≤2 chars, approved or
shared symbols, disease names), since aliases drive the gene highlighting.

Several labellers: `--full dan` (default) get every round with repeats; anyone else gets a block of
`--block 5` rounds, whole, no repeats, overlapping their neighbours' blocks by a round, and can ask
for another block when done (`blocks` table). Every abstract shows its PMID, a PubMed link and its publication year (`--records`), with
a reminder to grade from the abstract alone.
