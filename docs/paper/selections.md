# Practice-set picks (pre-registered rules, docs/PREREG.md)

Made 2026-10-17 from `results/haystack/GeneDiseaseHaystackDev/stats.csv`
(raw tables: `results/haystack/GeneDiseaseHaystackDev/selections_raw.txt`).
No test-set claim result had been computed when these were recorded.

| pick | rule | chosen | score (practice) | runner-up |
|---|---|---|---|---|
| Best embedder (C1, C2) | graded nDCG@10, 14 dense runs | google/embeddinggemma-300m | 0.774 | Qwen3-Embedding-0.6B 0.753 |
| C6 comparator | graded nDCG@10 among MedCPT, BiCA-base, MedEmbed-base, PubMedBERT | bisectgroup/BiCA-base | 0.752 | MedCPT 0.744 |
| C7 PubMed-trained arm | graded nDCG@10, PubMed-trained < 150M | bisectgroup/BiCA-base | 0.752 | MedCPT 0.744 |
| C7 general arm | graded nDCG@10, general < 150M | BAAI/bge-base-en-v1.5 | 0.725 | bge-small 0.711 |
| C1 first pass | curator-only raw recall@100 (BM25, 14 dense, 14 BM25+dense RRF) | RRF(BM25 symbol_disease k0.9 b0.75 + bge-base) | 0.778 | RRF(BM25 + Qwen3-E-0.6B) 0.777; BM25 alone 0.750 |

## Reranker picks (2026-10-17, after grading practice reranker runs)

Graded practice nDCG@10 on the C1 first pass, under each grade set
(`results/haystack/GeneDiseaseHaystackDev/selections_rerankers_raw.txt`).
An LLM reranker is only compared using grades from the other judge.

| system | both judges | no DeepSeek | no Gemini |
|---|---|---|---|
| fine-tuned MedCPT-CE (ours) | 0.876 | 0.888 | 0.861 |
| Gemini 3.1 Flash-Lite LLM rerank, top 20 | 0.875 | 0.881 | 0.863 |
| Qwen3-Reranker-4B | 0.862 | 0.869 | 0.848 |
| DeepSeek-V4-Flash LLM rerank, top 20 | 0.860 | 0.864 | 0.851 |
| fine-tuned bge-reranker-base (ours) | 0.854 | 0.865 | 0.841 |
| Qwen3-Reranker-0.6B | 0.795 | 0.814 | 0.775 |
| MedCPT-CE | 0.780 | 0.823 | 0.770 |
| first pass alone (BM25 + bge-base RRF) | 0.780 | 0.801 | 0.765 |
| bge-reranker-v2-m3 | 0.747 | 0.778 | 0.729 |
| bge-reranker-base | 0.653 | 0.688 | 0.644 |

| pick | chosen | why |
|---|---|---|
| C1 reranker | Gemini 3.1 Flash-Lite, LLM rerank of top 20 | beats Qwen3-Reranker-4B under no-Gemini grades (0.863 vs 0.848); Qwen3-Reranker-4B beats DeepSeek under no-DeepSeek grades (0.869 vs 0.864) |
| C8 fine-tuned arm | fine-tuned MedCPT-CE (bioembed/medcpt-ce-gd-s42) | 0.876 vs 0.854 for fine-tuned bge-reranker-base |
| C8 comparator | Gemini 3.1 Flash-Lite LLM rerank (the best off-the-shelf reranker) | as C1 |

Grades for the claims: C1 (Gemini rerank vs embeddinggemma) and C8
(fine-tuned MedCPT-CE vs Gemini rerank) both use no-Gemini grades.
