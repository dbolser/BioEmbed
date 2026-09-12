# llm/ — LLM-side evaluation of the BioEmbed tasks

Runs the five locally built bio tasks (`data/tasks/`) through the paper's
LLM-judge framework (`vendor/embedders-dilemma/llm_judge`, pinned mteb==2.6.5).
Nothing under `vendor/` is modified; `run_bio_llm.py` puts the vendor repo on
`sys.path` and reuses its evaluators, client, and `_DummyEncoder` pattern.

## Setup

```bash
cd llm
uv sync
cp .env.example .env   # then fill in token= and model=
```

`.env` fields map to `llm_judge.settings.Settings` (see `.env.example` for the
OpenRouter template and the planned model set). The checked-out `.env` ships
with dummy values so offline imports/verification work; `run_bio_llm.py`
refuses to run with them.

## Run (one model per pass)

```bash
uv run --env-file .env python run_bio_llm.py
```

- Scores land in `results/llm/<model-slug>/` (slug = `settings.model` with `/`
  replaced by `__`), in the vendor's mteb result-cache layout.
- The vendor evaluators also drop a few debug-sample JSONs under
  `./llm_results/` relative to the CWD.
- Reruns are cached per task by the mteb result cache; switch `model=` in
  `.env` to start a fresh pass in a new folder.

## Offline verification (no API calls)

```bash
uv run python verify_bio_tasks.py
```

## Per-pass token/cost expectation

Estimated with the actual prompt templates and a len(text)/4 token proxy:

| Task                  | Items          | Input tokens/pass | Per-prompt size |
|-----------------------|----------------|------------------:|-----------------|
| GOPubMedRetrieval     | 75 q × 300 docs  |       8,157,612 | ~109k (corpus in context) |
| R2MEDBiologyPooled    | 100 q × 363 docs |      27,470,766 | ~275k (corpus in context) |
| BioMeSHClassification | 500 test docs    |         268,900 | ~540 |
| PubChemSynonymPC500   | 500 pairs        |          37,609 | ~75 |
| GOProteinPairCls      | 500 pairs        |         202,322 | ~400 |
| **Total**             |                  |  **36,137,209** | |

- The two corpus-in-context retrieval tasks dominate: ~98.6% of all input
  tokens; R2MEDBiologyPooled alone is ~76%.
- Cost per model pass ≈ 36M × (input $/M token). E.g. at $0.25/M that is ~$9,
  at $1.25/M ~$45 — plus output/thinking tokens (small by comparison unless
  reasoning effort is high). Providers with implicit prompt caching charge much
  of the repeated corpus prefix at the cached rate, so real cost is often
  well below the naive estimate.
- **Context limits**: each R2MEDBiologyPooled query is a ~275k-token prompt —
  it does not fit 128k/200k-context models (same situation as the vendor's
  BuiltBench note). Use a long-context model (e.g. Gemini 2.5, gpt-4.1
  family). GOPubMedRetrieval (~109k) just fits 128k models.
