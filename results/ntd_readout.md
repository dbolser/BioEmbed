| role | a | b | score | n | mean a | mean b | a - b [95% CI] | rule |
|---|---|---|---|---|---|---|---|---|
| registered | bioembed__bge-base-gd-ntd-s42__gene | bioembed__bge-base-gd-s42__gene | judgments_merged.parquet | 493 | 0.750 | 0.765 | -0.015 [-0.021, -0.008] | within 0.02 |
| registered | bioembed__bge-base-gd-ntd-s42__gene | bioembed__bge-base-gd-s42__gene | curator-only | 500 | 0.398 | 0.399 | -0.001 [-0.009, +0.007] |  |
| registered | ce-bioembed__medcpt-ce-gd-ntd-s42-top100-of-rrf-bm25-symbol_disease-k0.9-b0.75+BAAI__bge-base-en-v1.5__gene | ce-bioembed__medcpt-ce-gd-s42-top100-of-rrf-bm25-symbol_disease-k0.9-b0.75+BAAI__bge-base-en-v1.5__gene | judgments_merged.parquet | 493 | 0.869 | 0.861 | +0.009 [+0.004, +0.014] | within 0.02 |
| registered | ce-bioembed__medcpt-ce-gd-ntd-s42-top100-of-rrf-bm25-symbol_disease-k0.9-b0.75+BAAI__bge-base-en-v1.5__gene | ce-bioembed__medcpt-ce-gd-s42-top100-of-rrf-bm25-symbol_disease-k0.9-b0.75+BAAI__bge-base-en-v1.5__gene | judgments_merged_no-gemini.parquet | 494 | 0.847 | 0.836 | +0.011 [+0.006, +0.016] | within 0.02 |
| registered | ce-bioembed__medcpt-ce-gd-ntd-s42-top100-of-rrf-bm25-symbol_disease-k0.9-b0.75+BAAI__bge-base-en-v1.5__gene | ce-bioembed__medcpt-ce-gd-s42-top100-of-rrf-bm25-symbol_disease-k0.9-b0.75+BAAI__bge-base-en-v1.5__gene | curator-only | 500 | 0.489 | 0.477 | +0.012 [+0.002, +0.023] |  |
| context (C5 base) | bioembed__bge-base-gd-ntd-s42__gene | BAAI__bge-base-en-v1.5__gene | judgments_merged.parquet | 493 | 0.750 | 0.674 | +0.076 [+0.062, +0.089] |  |
| context (C6 BiCA-base) | bioembed__bge-base-gd-ntd-s42__gene | bisectgroup__BiCA-base__gene | judgments_merged.parquet | 493 | 0.750 | 0.699 | +0.051 [+0.039, +0.063] |  |
| context (C8 Gemini rerank) | ce-bioembed__medcpt-ce-gd-ntd-s42-top100-of-rrf-bm25-symbol_disease-k0.9-b0.75+BAAI__bge-base-en-v1.5__gene | llm-gemini-3.1-flash-lite-top20-of-rrf-bm25-symbol_disease-k0.9-b0.75+BAAI__bge-base-en-v1.5__gene | judgments_merged_no-gemini.parquet | 494 | 0.847 | 0.834 | +0.014 [+0.003, +0.025] |  |

Paired bootstrap over queries, 10,000 resamples, seed 0; no p-values (PREREG 2026-11-04). Rule: a drop of 0.02 or less on graded nDCG@10 (each set of grades) reads 'does not depend on test diseases'.
