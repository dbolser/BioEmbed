# axeslab: automatic Map Explorer axes

The experiment behind genesearch's suggested axes (`genesearch/axes.py`, PR #20). It tried 18 ways to find two
interesting directions in a map's MedEmbed vectors, on 5 maps: ZCCHC8, SFTPC, CFTR and AQP4 (genes) and
EFO_0000768 (idiopathic pulmonary fibrosis).

- `prep.py` (genesearch env): saves each subject's map and vectors, plus 50,000 random human-gene papers as the
  background, into `data/geometry/axeslab/` (gitignored, like all results).
- `axislab.py`: the shared harness. A method only proposes directions. `record()` then names their ends (Gemini
  Flash-Lite on Vertex AI, cached), measures them, plots them, and writes `results/<method>/<subject>.{json,png}`.
- `m_<family>.py`: one script per family of methods.
  - `local`: PCA, varimax and ICA of the map's own papers.
  - `named`: the manifold work's 67 named axes.
  - `concepts`: MeSH/GO concept vectors.
  - `sae`: a 4,096-feature sparse autoencoder.
  - `supervised`: time, curated papers and LDA.
  - `contrastive`: the map against the background (contrastive PCA, generalised eigenvectors).

Three blind LLM judges scored each method's plots (a biologist, an explorer UX designer and a skeptic). Their mean
scores out of 10:

| Method | Score |
|---|---|
| contrastive_gevcore | 6.8 |
| local_pca | 6.2 |
| contrastive_cpca | 6.0 |
| contrastive_gev | 5.7 |
| sae_cpca | 5.5 |
| ... | ... |
| named_spread | 3.5 |

The app's "Discover" is gevcore, "Main spread" is local PCA, and "About it ↔ in passing" is gevcore's removed
direction (subject centroid minus the background mean).

Findings:
- ICA finds hidden sub-literatures that the map leaves grey. For CFTR these are pancreatitis and male infertility
  (CBAVD).
- The IPF and SFTPC paper lists include off-topic papers: behavioural trials and breast pathology in IPF, and
  pneumococcal "PspC" alias papers in SFTPC.
