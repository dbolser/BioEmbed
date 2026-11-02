# geometry: TODO (path to publication)

Written 2026-11-02, after the clustering (README "Clustering"). In order of importance.

## What is already new

- A biology-grounded way to score maps and clusterings: do neighbours (or cluster mates) share a human gene
  (PubTator3) or a journal field? Not just kNN recall.
- Two density findings: intrinsic dimension *falls* as papers are added (19.4 at 78k, 14.0 at 7.76M), and 2-D
  UMAP gets worse with density while t-SNE does not (about half against about 80% of the gene signal).
- A clustering recipe that works at 7.8M papers: 12-D UMAP, HDBSCAN on a 1M sample, every paper voted in;
  drawn on a separate t-SNE map, where the clusters land as coherent patches.

## 1. More embedding models (the "across embeddings" claim)

Without this it is a methods note.
- [ ] PubMedBERT: vectors already in `/data/public/PubMedLandscape/2024_v2/` for 6.26M of our papers. Run the
      whole pipeline (truth, density, UMAP 12-D, t-SNE, clustering) on the shared 6.26M papers.
- [ ] One large modern embedding model (pick on the BioEmbed tier results); needs GPU embedding of 7.76M
      abstracts.
- [ ] Compare the shapes: intrinsic dimension and its density curve, gene against field signal, how much of
      each survives the squeeze, how well the clusterings agree (ARI / NMI between models).

## 2. Clean the gene labels

PubTator3 tags bare abbreviations as genes. `gene2pubtator3` mention text, first ~2,000 mentions each:
PC (pyruvate carboxylase, NCBI 5091) is "PC" 1,709 times and "pyruvate carboxylase" 180; CS (citrate synthase)
"CS" 2,101 against 612; CP (ceruloplasmin) "CP" 1,390 against 1,104. The "AP2B1 OXT PC" cluster (33k papers)
is psychology: depression trials, primary care ("PC-PTSD"). GPT and TF are mostly real (alanine
aminotransferase, transferrin: clinical blood tests).
- [ ] Measure it: share of the 7.76M papers whose *only* gene tags are bare short symbols (2-3 letters, mention
      text = symbol).
- [ ] Filter: drop gene tags whose only mentions are bare ambiguous symbols; rebuild the corpus and the scores.
      Note: this also changes the corpus itself (some "human-gene papers" mention no gene).
- [ ] Re-score gene lift and gene sharing on the clean labels.

## 3. Robustness

- [ ] Seeds: repeat UMAP, t-SNE and HDBSCAN with 3 to 5 seeds; error bars on every headline number.
- [ ] t-SNE longer: its fit was still improving at 750 iterations.
- [ ] A third, independent label: MeSH terms (PubMed) as "same topic", alongside genes and fields.

## 4. Baselines a reviewer will ask for

- [ ] Leiden clustering on the exact kNN graph in 384-D (no UMAP at all).
- [ ] HDBSCAN straight on the 384-D vectors with cosine (Dan's cuML PR); compare with 12-D UMAP.
- [ ] k-means at matching cluster counts.

## 5. Release

- [ ] Cluster labels and names (`clusters/<name>_all.npy`, `_summary.csv`) as a table of PMID -> cluster.
- [ ] genesearch tab: our t-SNE map with the clusters (like the PubMedBERT map tab).
- [ ] Zenodo deposit with a DOI (maps, clusters, scores); tagged code release.

## 6. Write and submit

- [ ] bioRxiv preprint.
- [ ] Journal: Patterns (where González-Márquez et al. published) or NAR Genomics and Bioinformatics.

Item 1 is the real work, about a day of GPU and CPU time per model; the rest is days, not weeks.
