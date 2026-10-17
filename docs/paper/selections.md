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

Still to pick on practice (needs reranker runs): the C1 reranker, and for
C8 the best off-the-shelf reranker and the better of the two fine-tuned
cross-encoders.
