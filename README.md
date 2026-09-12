# BioEmbed: The Embedder's Dilemma, for Biology

Recreating the cost-aware LLM-vs-embedding-model comparison of
[*The Embedder's Dilemma: LLMs Are Better, but at What Cost?*](https://arxiv.org/abs/2608.12875)
(El Assadi, Muennighoff, Lee) on **biological text**: PubMed, bioRxiv,
UniProt, and ontology (OBO) corpora.

Goal: map the Pareto frontier of retrieval/embedding quality vs cost,
specifically for biology, across:

- open embedding models from ~30M to multi-B parameters,
- biomedical-specialised embedders,
- LLMs used directly for the same tasks (via OpenRouter),

on biomedical MTEB tasks (retrieval, reasoning-retrieval, STS, clustering,
classification, pair classification) plus a custom UniProt/GO/PubMed
cross-reference retrieval benchmark.

## Setup

```bash
uv sync
```

Status: early scaffold — design in progress.
