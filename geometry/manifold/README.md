# The space of biological ideas: themes, concepts, time and a map to walk it

Started 2026-11-04 (branch `geometry-manifold`). Data: the 7.76M human-gene papers and their MedEmbed-small
vectors from `geometry/` (384 numbers per paper, unit length), and the 780 topics (clusters) of
`cluster_sweep.py --assign eom_mcs200_ms3`.

The plan was: topics → centroids → vectors between centroids ("between-topic vectors") → cluster those vectors to
find the themes behind the topics → put core concepts next to the papers as coordinates → use time vectors to
predict where the literature goes (inspired by ForeSCITE, below) → navigate.

**Explorer:** <http://outsee4.lan:8768/> (`explorer.py`). Type an idea, see where it lands, walk it along any axis,
complete an analogy, or route between topics.

## At a glance

| # | Question | Answer | Against |
|---|---|---|---|
| 1 | Is the space centred? | No: the mean paper vector has length 0.79. Every paper shares one big direction; everything below works on vectors minus that mean | 0 for a centred space |
| 2 | Do between-topic vectors, clustered directly, give themes? | Not on their own: one blob of 3,107 of 5,355 edges, which is just the first principal axis (clinical ↔ molecular) | A Gaussian null gives the same picture |
| 3 | Do between-topic vectors repeat across the map? | Yes: 10.9% of edges have a partner elsewhere on the map with \|cos\| > 0.6 | null: 0.0% |
| 4 | Analogy themes, calibrated so the null gives none | 22 themes; the top three: **"X in cancer → X"** (39 edges, 16 regions), **"X in the clinic → X genetics"** (26, 12), **"X in the body → X in the brain"** (21, 10) | Largest \|cos\| in 5 null draws: 0.593 |
| 5 | Do the analogy vectors work on topics that did not define them? | Yes: "X → X in cancer" applied to the 740 other topics lands on topics with +50 points more Neoplasms headings (z = 128 against random directions) | Random directions of the same length |
| 6 | Themes behind the clusters: directions papers vary along *inside* topics | 6 axes are among the top-20 local directions of most topics (93%, 83%, 82%, 74%, 65%, 52%): gene mapping ↔ drug treatment; populations ↔ mechanism; bench biochemistry ↔ patient outcomes; immune cells ↔ drug design; … | Chance: 20/384 = 5% |
| 7 | Can concepts be placed by their text alone? | Yes: a MeSH concept's "name: definition" vector finds its own papers' centroid first among 22,142 in 60% of cases, top-10 in 88% | Chance: 1 in 22,142 |
| 8 | Is there a time vector? | Yes, one shared by topics: each topic's own drift (slope of its papers on year) agrees with the others' (mean pairwise cos 0.32); a paper's position on it correlates with its year at r = 0.49 (within topic) | 0 in 384-D |
| 9 | Does it predict where topics move? | Yes: chosen on 1999-2008, then fixed, it beats "nothing moves" for 2021-23 in 90% of 749 topics (error 0.86x); 68% (0.97x) when the target is rebuilt without the clustering | Persistence |
| 10 | Does it predict which topics grow? | No: position on the time vector adds R² 0.006 to past growth | |
| 11 | Does geometry predict new gene-disease links? | Yes: among pairs never co-mentioned up to 2015, closeness adds to popularity and common neighbours (AUC 0.934 → 0.951, average precision 0.152 → 0.171); moving genes and diseases along their own time vectors helps on top | The two graph baselines |
| 12 | ForeSCITE's move: shift a topic's papers along its time vector | Right direction, tiny effect: a small shift brings 60% of topics closer to the real 2019-23 papers (another topic's vector: 30%, backwards: 14%, random: 2%), but the gain is +0.0003 cosine | Placebo directions |
| 13 | Does a topic's time vector say which MeSH headings rise in it? | No (median Spearman 0.005; simple trend: 0.025) | |
| 14 | Is it MedEmbed only? | Running: Qwen3-Embedding-8B (best embedder on BioScore) on 100 papers per topic | |

## 1. The papers sit in a narrow cone

The mean of all 7.76M paper vectors has length 0.79 (1 would mean every paper identical); a topic's centroid
has length 0.86 (median). So most of every vector is the shared "this is a biomedical abstract" direction, and
cosine similarity between any two papers is high. Differences between centroids cancel it; kNN graphs and
concept placement do not, so every step below subtracts the mean first (`centroids.py`).

Centred, the 780 topic centroids spread in many directions: the first principal axis holds 10.6% of their
variance, 45 axes hold 80%, participation ratio 27.

## 2. Topics with names

`topic_inputs.py` collects per topic the MeSH headings most over-represented in it (from the PubMed XML,
`parse_mesh.py`: 7.55M of the papers found, 81% with MeSH, median 11 headings), its genes, journal field, median
year and the titles of its 8 most central papers. 20 agents named the 780 topics from that (a name, a line, a
kind, and a flag if it lumps unrelated things): `<name>_labels.csv`. 12 are flagged mixed, e.g. "Journal
corrigenda", "Letters, replies and journal history", and **topic 125, retracted lncRNA/miRNA cancer papers
(likely paper-mill output)**: the map puts paper mills in a region of their own.

## 3. Between-topic vectors and analogy themes

**Clustering them directly does not work.** `directions.py` links each topic to its 10 nearest, takes the unit
difference of the two centroids per edge, and clusters the 5,355 edges with HDBSCAN on 1 − |cos| (an edge has no
direction, so v and −v are one axis). In 384-D one cluster takes 3,107 edges, and it is the first principal axis
(surgery, cardiology, nursing ↔ nuclear transport, splicing, cell cycle). Whitened spaces give noise. Centroids
drawn from a Gaussian with the same covariance give the same picture, so this finds nothing beyond the spread.

**But the vectors do repeat.** Allow two edges to vouch for each other only if they share no topic and no
endpoint region (20 k-means regions of the map), and use mutual neighbours only (no hub topics). Then 10.9% of
the 2,445 edges have a partner elsewhere with |cos| > 0.6; in the null, none do (`relations.py`).

**Analogy themes** (`analogies.py`). Threshold = the largest |cos| between any allowed pair of edges in 5 null
draws (0.593), so in the null not one pair passes. Themes = groups of edges linked above it (loose groups are
split by raising the bar inside them). 22 themes with 3 or more edges. Named by agents from both lines of evidence
below, with a skeptic checking each name (`<name>_axes_named.json`):

| Theme | Edges / regions | The relation | Examples (− topic → + topic) |
|---|---|---|---|
| A0 | 39 / 16 | X in cancer → X outside cancer | Metformin as anticancer agent → Metformin mechanisms; Thyroid cancer → Thyroid hormones and Graves; IGF system in cancer → IGF binding proteins; COX-2 in cancer → Leukotrienes |
| A1 | 26 / 12 | X in the clinic → X genetics | IBD biologic therapy → IBD immunology and genetics; TB diagnosis → TB immune response; Psoriasis biologics → Psoriasis immunopathology; Severe asthma management → Asthma and allergy genetics |
| A2 | 21 / 10 | X in the body → X in the brain | Alcoholic liver disease → Alcohol effects on brain; Exercise and rehab in cancer → Exercise, cognition and BDNF; Menopausal hormone therapy → Brain estrogen |
| A3 | 6 / 4 | X → microRNAs in X | Leukaemia fusion genes → MicroRNAs in leukaemia; Glioblastoma signatures → MicroRNAs in glioma |
| A7 | 5 / 4 | X in heart and vessels → X in kidney | Creatine kinase and rhabdomyolysis → NGAL acute kidney injury; Blood pressure → Chronic kidney disease |
| A14 | 3 / 5 | X under injury → X regeneration | Cardiac reperfusion injury → Cardiac regeneration; Cerebral ischaemia → Nerve regeneration |
| A20 | 3 / 4 | Drug treatment of X → immune biology of X | Tacrolimus dosing → Transplant rejection; Antiretroviral regimens → HIV pathogenesis |

**Do they work?** Two checks.
- *Completion* (leave one edge out): start at the edge's first topic, add the theme vector built from its other
  edges, and the true partner is the nearest topic 76% of the time (top-5: 97.5%), against 17% (63%) for the start
  topic's own nearest neighbour. This is partly by construction (edges were picked for being parallel).
- *Held out* (`analogy_check.py`), the real test: apply the vector to every topic that did not define it, land on the
  nearest topic, and compare its MeSH branches with the start's; z against 200 random directions of the same length.

| Theme | Step toward + : landing topic has more | Step toward − : more |
|---|---|---|
| A0 X in cancer → X | (weaker: sugars, lipids, diabetes, z 9-11) | **Neoplasms +50 points (z 128)** |
| A1 clinic → genetics | **Genetic phenomena +22 points (z 30)** | Therapeutics +12 points (z 38) |
| A2 body → brain | **Central nervous system +28 points (z 204)** | |
| A3 → microRNAs | **Antisense elements / nucleic acids +51 points (z 354)** | |
| A7 heart → kidney | Urologic diseases (z 67) | Heart diseases (z 17) |

## 4. The themes behind the clusters: axes inside every topic

`axes.py` takes, for each topic, its papers' scatter around the centroid, scales each topic to equal weight, and
averages: W. W's top directions are the ways papers differ *inside* topics, shared across topics. For most topics
they are among that topic's own top-20 directions: W0 93% of topics, W1 83%, W2 82%, W3 74%, W4 65%, W5 52%
(chance 5%). `label_axes.py` reads them off the papers' MeSH headings and publication types (effect = how far
toward + the papers with a heading sit, in sd), and `concept_axes.py` independently from where MeSH and GO concept
*texts* fall on them. The two agree:

| Axis | − end | + end | MeSH at the ends (papers) | Concepts at the ends (text) |
|---|---|---|---|---|
| W0 | Drug dosing and administration | Gene mapping and heredity | Ranibizumab, phase I trials ↔ LOD score, linkage, QTL, twin study | IV anaesthetics, analgesia ↔ genetic loci, linkage, X-linked genes |
| W1 | Population and clinical description | Molecular mechanism | Seroepidemiology, outbreaks, prevalence ↔ viral polymerase, allostery, MD simulation | Symptoms, medical records ↔ protein binding, mTORC2, ubiquitin ligase binding |
| W2 | Bench biochemistry | Patient outcome statistics | Chromatography, ultracentrifugation ↔ Mendelian randomisation, mortality, propensity scores | Electron and cryo-EM microscopy ↔ cancer survivors, risk adjustment, propensity score |
| W3 | Immune (myeloid) cells | Computational drug chemistry | Macrophages, neutrophil infiltration ↔ QSAR, docking, pharmacogenetics | Histiocytes, dendritic cells ↔ pharmacogenomic testing, drug repositioning, docking |
| time | Serum immunochemistry | Evidence synthesis and data | Immunoelectrophoresis, immunodiffusion ↔ network meta-analysis, machine learning, deep learning | Serologic tests, serum globulins ↔ precision medicine, implementation science |

W and the between-topic axes B share a lot (top-20 subspace overlap 0.72, chance 0.05): the directions that
separate topics also run through them. The B axes are the map's regions (B0 autonomic physiology ↔ cancer biology,
B1 hospital care ↔ basic science in model organisms, …): all 47 axes with names, poles and confidence are in
`eom_mcs200_ms3_axes_named.json`; the skeptic marked W4 and W7 low confidence.

## 5. Concepts as coordinates

`concepts.py` embeds every MeSH descriptor (31,110) and GO term (40,214) with MedEmbed-small, by name and as
"name: definition". MeSH concepts also get a ground-truth position: the centroid of the papers indexed with them
(22,142 with 50+ papers). Does the text land on its own papers?

| Concept text | Own centroid first (of 22,142) | Top 10 | Median rank |
|---|---|---|---|
| name | 57% | 83% | 0 |
| name, centred | 58% | 84% | 0 |
| name: definition | 56% | 87% | 0 |
| **name: definition, centred** | **60%** | **88%** | 0 |
| learned linear map, text → centroid | 41% | 77% | 1 |

So a concept can be placed among the papers from its text alone, centred by the concept mean; that is how the
explorer places typed text. On the time axis, a concept's text coordinate predicts the median year of its
papers (Spearman 0.32, 22,142 concepts; a full 384-D regression reaches 0.38; no other single axis beats 0.26).
Against the year MeSH introduced the heading it is weak (0.10; 2010s headings sit 0.54 sd toward "new").

## 6. Time vectors

The inspiration Dan remembered is most likely **ForeSCITE** (Chang, Ondov, …, Hua Xu, "Learning Scientific
Trajectories to Predict Future Biomedical Discoveries", arXiv 2610.38719, 28 Oct 2026): it clusters ~3M biomedical
conclusions into 407 topics, finds in each a direction from earlier to later findings, shifts embeddings along it
and decodes them into hypotheses. Two other candidates: Nylund, Gururangan & Smith 2024 (the "time vectors" name,
in model weights, https://arxiv.org/abs/2312.13401) and Tshitoyan et al. 2019 (embeddings of older abstracts flag
later discoveries, https://www.nature.com/articles/s41586-019-1335-8).

**Topics share one time direction.** Each topic's time vector (least-squares slope of its papers on year) points
the same way as the others' (mean pairwise cos 0.32 over 780 topics; the mean of the unit vectors has length 0.57).
44% of it lies on within-topic axis W2 (bench ↔ outcomes).

**Forecasting topic positions** (`timevec.py drift`). Train 2009-2018, predict the 2021-23 centroid, error relative
to "the topic stays where it was in 2016-18":

| Prediction | Error vs persistence (median) | Topics improved |
|---|---|---|
| own slope | 1.19 | 33% |
| shared time vector | 0.91 | 87% |
| own and shared, half each | 0.87 | 69% |
| **chosen on 1999-2008 → 2011-13, then fixed: 0.2 own + 0.8 shared, 0.75 of the step** | **0.86** | **90%** |
| the same, target rebuilt without the clustering (2021-23 papers to the nearest 2016-18 centroid) | 0.97 | 68% |

A topic's own slopes are noisy; shrunk toward the shared one they help. The clustering saw all years, which could
flatter the first target; the rebuilt target is biased the other way (nearest *old* centroid), so the truth lies
between. A topic's own past direction does point where it goes next: cos with the actual move 0.36, against 0.12
for another topic's slope.

**Growth** (`timevec.py growth`): topics at the "new" end grow faster (Spearman 0.50 with 2019-23 growth), but past
growth already says that (0.80); adding position raises R² from 0.500 to 0.506. Nothing.

**Latent knowledge: new gene-disease links** (`timevec.py links`; Tshitoyan's test on paper embeddings). Pairs of
a PubTator gene (10+ papers) and a MeSH disease heading (100+ papers) never in one paper up to the cutoff; positive
if 3+ papers co-mention them in the horizon. Gene and disease positions are centroids of their papers up to the
cutoff; "time" moves each along its own slope to the middle of the horizon.

| Score | 2010 → 2011-15: AUC | within popularity deciles | 2015 → 2016-23: AUC | within deciles |
|---|---|---|---|---|
| popularity (log papers, gene + disease) | 0.925 | 0.571 | 0.920 | 0.578 |
| common neighbours | 0.925 | 0.684 | 0.896 | 0.712 |
| cosine | 0.686 | 0.815 | 0.651 | 0.789 |
| cosine after the time-vector shift | 0.730 | 0.825 | 0.690 | 0.786 |

30.5M (2010) and 46.7M (2015) unlinked pairs; 0.23% and 0.53% became linked. Fitted on 2010, tested on 2015:

| Logistic model | AUC | Average precision | Precision of the top 1,000 |
|---|---|---|---|
| popularity + common neighbours | 0.934 | 0.152 | 0.636 |
| + cosine | 0.950 | 0.163 | 0.566 |
| + cosine after the time shift | 0.951 | 0.168 | 0.633 |
| + both | **0.951** | **0.171** | **0.646** |

Caveat: MedEmbed-small was trained after 2015 and may have read some of the future.

**ForeSCITE's move, in embedding space** (`foresee.py`, nothing after 2015 used): shift each topic's 2011-15 papers
along its own time vector (refitted on papers up to 2015) and measure how close the real 2019-23 papers (assigned
by nearest pre-2016 centroid) come to the shifted cloud. A half step helps 60% of topics; another topic's vector
30%, backwards 14%, a random direction 2%. The direction is right, but the gain is tiny (+0.0003 cosine) and larger
steps overshoot. Reading the shift as "which MeSH headings will rise in this topic" fails (median Spearman 0.005;
simple trend extrapolation 0.025).

## 7. The explorer

<http://outsee4.lan:8768/>: `atlas.py` bundles a 1.5M-paper sample (vectors, t-SNE positions, titles), the 780
named topics, 62k concepts and the 47 named axes; `explorer.py` serves them (FastAPI). Everything is computed in
the centred 384-D space, never on the 2-D map; the map only shows where results are.
- **Locate**: text, a topic, a concept → nearest topics, concepts and papers, and coordinates on the main axes (in
  sd against concepts for text, against papers for topics).
- **Walk** a point along any axis; each step names the topic, concepts and papers it reaches and how close the
  nearest real paper still is (so leaving the literature shows).
- **Analogy**: a : b :: start : ?, with the start alone shown as the baseline.
- **Path**: shortest chain of neighbouring topics.
- Colour the map by any axis (blue − end, red + end).

Examples: "insulin signalling" walked toward "X in cancer" goes Insulin receptor binding → Insulin signalling and
GLUT4 → IGF system in cancer → PI3K/AKT/mTOR in cancer. Thyroid cancer : Thyroid hormones :: Prostate cancer : SHBG
and circulating androgens (the start alone gives Androgen receptor in prostate cancer).

Run: `uv run --with fastapi --with uvicorn python geometry/manifold/explorer.py --port 8768` (about 1 minute to load).

## 8. Model check

(Running: Qwen3-Embedding-8B on 100 papers per topic and all concepts, two GCP Spot L4s, `gcp_embed.sh`.)

## Running it

All CPU scripts use the repo root env (`uv run python geometry/manifold/<script>`), in this order:

| Script | Time | What |
|---|---|---|
| `centroids.py NAME...` | 30 s | year per paper, corpus mean, topic centroids overall and per year |
| `parse_mesh.py` | 8 min | MeSH descriptors; each paper's headings and publication types from the PubMed XML |
| `directions.py NAME` | s | between-topic vectors clustered directly (the failed first try) |
| `relations.py NAME [--mutual] [--null]` | 1 min | parallel partners and HDBSCAN themes, against the null |
| `axes.py NAME` | 3 min | within-topic axes W, between-topic axes B, per-topic time vectors |
| `label_axes.py NAME` | 4 min | MeSH and publication types at each end of each axis |
| `topic_inputs.py NAME` | 2 min | naming evidence per topic (MeSH lifts, central titles: local only) |
| `analogies.py NAME` | 1 min | null-calibrated analogy themes, completion test |
| `branches.py NAME`, `analogy_check.py NAME` | 1 min | MeSH branch shares per topic; held-out test of the themes |
| `concepts.py --embed --score` | 25 min (16 threads) | MeSH and GO concepts embedded; MeSH paper centroids; placement scores |
| `concept_axes.py NAME` | s | concept coordinates on every axis |
| `timevec.py drift / drift_reassigned / growth / links` | s / 1 min / s / 50 min | the time tests |
| `foresee.py` | 2 min | ForeSCITE-style shift test |
| `axis_evidence.py NAME` | s | evidence file the axis names came from |
| `atlas.py NAME`, `explorer.py` | 3 min, 1 min | the explorer |
| `subset_texts.py NAME`, `gcp_embed.sh VM SHARD`, `gpu_embed.py` | ~2 h GPU | model check |

Outputs: `data/geometry/medembed_small/manifold/` (not in git). Topic and axis names were made by agents from MeSH,
gene and title evidence; the titles and abstracts themselves stay local.
