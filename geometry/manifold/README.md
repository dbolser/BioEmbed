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
| 2 | Do between-topic vectors, clustered directly, give themes? | Only one coarse one: a loose blob of 3,107 of 5,355 edges (coherence 0.18) along hospital care ↔ basic science (mostly the second principal axis, B1), and noise | A Gaussian null gives 36 small clusters and no blob, so the blob is real; it just hides everything finer |
| 3 | Do between-topic vectors repeat across the map? | Yes: 10.9% of edges have a partner elsewhere on the map with \|cos\| > 0.6 | null: 0.0% |
| 4 | Analogy themes, calibrated so the null gives none | 22 themes; the top three: **"X in cancer → X"** (39 edges, 16 regions), **"X in the clinic → X genetics"** (26, 12), **"X in the body → X in the brain"** (21, 10) | Largest \|cos\| in 5 null draws: 0.593 |
| 5 | Do the analogy vectors work on topics that did not define them? | Yes: "X → X in cancer" applied to the 710 other topics lands on topics with +50 points more Neoplasms headings (z = 13) | Steps of the same length along real between-topic directions |
| 5b | At 3,349 finer topics | 34.6% of edges have a partner above the null's maximum; 182 themes; the largest read as a grammar of moves: change the level of explanation (clinic → genetics, disease → its microRNAs, target → drugs), the context (non-cancer → cancer, brain → body), or the organ (gut → lung, body → mouth) | Same null procedure |
| 5c | Does a dictionary learned from single papers agree? | Yes: a sparse autoencoder (4,096 features, 84% of variance) has single features within cos 0.80-0.92 of the cancer, brain and microRNA analogy vectors | |
| 6 | Themes behind the clusters: directions papers vary along *inside* topics | 6 axes have at least half their length inside the top-20 local directions of most topics (93%, 83%, 82%, 74%, 65%, 52%): gene mapping ↔ drug treatment; populations ↔ mechanism; bench biochemistry ↔ patient outcomes; immune cells ↔ drug design; … | A random direction has 5% (20/384) of its length there on average and essentially never half |
| 7 | Can concepts be placed by their text alone? | Yes: a MeSH concept's "name: definition" vector finds its own papers' centroid first among 22,142 in 60% of cases, top-10 in 88% | Chance: 1 in 22,142 |
| 8 | Is there a time vector? | Yes, one shared by topics: each topic's own drift (slope of its papers on year) agrees with the others' (mean pairwise cos 0.32); a paper's position on it correlates with its year at r = 0.49 (within topic) | 0 in 384-D |
| 9 | Does it predict where topics move? | Yes: chosen on 1999-2008, then fixed, it beats "nothing moves" for 2021-23 in 91% of 749 topics (squared error 0.86x); 68% (0.97x) when the target is rebuilt without the clustering | Persistence |
| 10 | Does it predict which topics grow? | No: position on the time vector adds R² 0.006 to past growth | |
| 11 | Does geometry predict new gene-disease links? | Yes: among pairs never co-mentioned up to 2015, closeness adds to popularity and common neighbours (AUC 0.934 → 0.953, average precision 0.152 → 0.169), though it lowers precision of the top 1,000 (0.636 → 0.578); moving genes and diseases along their own time vectors restores that and adds a little average precision (0.173) | The two graph baselines |
| 12 | ForeSCITE's move: shift a topic's papers along its time vector | Right direction, small effect: at the topic's own past rate over the real 8-year gap, the shift brings 68% of 776 topics closer to the real 2019-23 papers (another topic's vector 41%, backwards 1%, random 1%); a quarter of that step: 90% (79%, 3%, 22%). Gains are about +0.001 cosine | The same steps in placebo directions |
| 13 | Does a topic's time vector say which MeSH headings rise in it? | A little: median Spearman 0.08 with the actual change (positive in 71% of topics), 0.07 beyond what trend extrapolation predicts (trend alone: 0.13) | Trend extrapolation |
| 14 | Is it MedEmbed only? | Running: Qwen3-Embedding-8B (best embedder on BioScore) on 100 papers per topic | |

## 1. The papers sit in a narrow cone

The mean of all 7.76M paper vectors has length 0.79 (1 would mean every paper identical); a topic's centroid
has length 0.86 (median). So most of every vector is the shared "this is a biomedical abstract" direction, and
cosine similarity between any two papers is high. Differences between centroids cancel it; kNN graphs and
concept placement do not, so the geometry below subtracts the mean first (`centroids.py`; the one exception is
picking each topic's central papers for naming, `topic_inputs.py`, which ranks by raw cosine).

**Centring changes the neighbours, not just the scale** (1M papers, 5,000 queries): two random papers have raw
cosine 0.62 (sd 0.056), centred 0.00 (sd 0.10); a paper's nearest neighbour stands 4.8 random-pair sds above
random raw, 6.9 centred; and only 80% of the 15 (79% of the 100) nearest neighbours are the same under the two.
So cosine clustering of these vectors (e.g. a cosine HDBSCAN) should centre them first.

Centred, the 780 topic centroids spread in many directions: the first principal axis holds 10.6% of their
variance, 45 axes hold 80%, participation ratio 27.

## 2. Topics with names

`topic_inputs.py` collects per topic the MeSH headings most over-represented in it (from the PubMed XML,
`parse_mesh.py`: 7.55M of the papers found, 81% with MeSH, median 13 headings among those), its genes, journal field, median
year and the titles of its 8 most central papers. 20 agents named the 780 topics from that (a name, a line, a
kind, and a flag if it lumps unrelated things): `<name>_labels.csv`. 12 are flagged mixed, e.g. "Journal
corrigenda", "Letters, replies and journal history", and topic 125, named "retracted lncRNA/miRNA cancer
papers". It is the most retraction-heavy topic (13.6% of its papers are flagged Retracted Publication, 65x the
corpus rate of 0.21%), but most retracted papers are elsewhere: neighbouring ncRNA-cancer topics hold more of them
("Cancer microRNA targeting networks" 1,217, "Cancer lncRNAs and ceRNA networks" 1,008, topic 125 413 of 15,680).

## 3. Between-topic vectors and analogy themes

**Clustering them directly does not work.** `directions.py` links each topic to its 10 nearest, takes the unit
difference of the two centroids per edge, and clusters the 5,355 edges with HDBSCAN on 1 − |cos| (an edge has no
direction, so v and −v are one axis). In 384-D one loose cluster (coherence 0.18) takes 3,107 edges: mostly the
centroids' second principal axis, B1 (cos 0.73; with the first 0.55): surgery, cardiology, nursing ↔ nuclear
transport, splicing, cell cycle. Centroids drawn from a Gaussian with the same covariance give 36 small clusters
(largest 112 edges) and no blob, so this is real: neighbouring topics most often differ along hospital care ↔ basic
science. But it swallows everything finer, and in whitened spaces the clustering gives noise (as does the null).

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

![The four largest analogy themes, each drawn in its own plane](../figures/manifold_analogies.jpg)

Directions are not kept by 2-D maps (t-SNE keeps neighbourhoods), so each theme is drawn in a projection of the
384-D space: across, the theme vector; up, the main remaining direction among topics.

**Do they work?** Two checks.
- *Completion* (leave one edge out): start at the edge's first topic, add the theme vector built from its other
  edges, and the true partner is the nearest topic 76% of the time (top-5: 97.5%), against 17% (63%) for the start
  topic's own nearest neighbour and 72% top-5 for a step of the same length along another theme's vector. This is
  partly by construction (edges were picked for being parallel).
- *Held out* (`analogy_check.py`), the real test: apply the vector to every topic that did not define it, land on the
  nearest topic, and compare its MeSH branches with the start's. Placebo: 300 steps of the same length along real
  between-topic directions (other edges of the topic graph). (Random directions in 384-D are a far weaker placebo:
  most land back on the start topic's own neighbour, which inflates z about tenfold; a first version used them.)

Over the 710 topics outside A0, 738 outside A1, 748 outside A2 (change in the landing topic's share of papers under
a MeSH branch, percentage points; z against the placebo):

| Theme | Step toward + : landing topic has more | Step toward − : more |
|---|---|---|
| A0 X in cancer → X | (weak: sugars, hormones, z 2-3) | **Neoplasms +50 points (z 13)** |
| A1 clinic → genetics | **Genetic phenomena +22, genetic structures +13 (z 5)** | Therapeutics +12, patient care (z 4-5) |
| A2 body → brain | **Central nervous system +28 (z 25), neurocognitive disorders +12 (z 35)** | |
| A3 → microRNAs | **Antisense elements / nucleic acids +51 (z 19)** | |
| A7 heart → kidney | Urologic diseases +12 (z 9) | Heart diseases +4 (z 3.5) |

**Finer topics, more themes.** On the 3,349 finer topics (named from their top MeSH headings, `mesh_labels.py`),
9,539 mutual edges, null threshold 0.540: 34.6% of edges have a partner above it, 182 themes with 3+ edges.
The 20 largest, named and checked by agents from their pairs and held-out MeSH shifts
(`eom_mcs50_ms3_themes_named.json`; one, dopamine → serotonin, rejected as a local axis), read as a grammar with
three kinds of move:

| Kind of move | Themes (topic pairs; held-out z of the main MeSH shift, against real between-topic steps) |
|---|---|
| **Change the level of explanation** | clinical care → genetics (82; genetic phenomena +18 points, z 5; backwards clinical protocols, drug therapy z 7-8); disease → its microRNA studies (58; z 132); genetic association → cell or tissue mechanism (32; backwards genotype z 14); immune → intrinsic cell biology (20; backwards z 10); drug target → medicinal chemistry (19; drug development z 32) |
| **Change the context** | non-cancer → cancer (53; neoplasms +35 points, z 11); brain → whole body (49; backwards neurocognitive z 45); cardiovascular → metabolic (38; backwards heart z 12, forward nutrition only z 3); substance use → other disorders (23; backwards z 22); kidney disease → general (20; backwards urologic z 48); pregnancy and newborn → general (17; backwards embryonic structures z 40); stem cells → native tissue (14; backwards stem cells z 30); animals and ecosystems → human toxicology (14; backwards ecological z 19) |
| **Change the organ** | gut → lung (20; lung z 16, backwards GI z 10); body → mouth and jaw (18; z 97); pancreas → stomach and colon (16; backwards pancreatic z 42, forward weak); cancer → glioma (15; nervous system neoplasms z 81); prostate → other cancers (15; backwards genital z 35); GI cancer → breast cancer (12; weak, z 5-8) |

**A learned dictionary agrees** (`sae.py`). A top-k sparse autoencoder on the centred paper vectors (4,096
features, 32 active per paper, trained on 2.9M papers capped at 4,000 per topic; no topics, no MeSH) explains 84%
of held-out variance; every feature fires (median on 0.33% of papers), and 99% have a MeSH heading at 5x lift or more where they
fire, most reading as one idea (dopamine and striatum; multiple sclerosis and EAE; postmenopausal hormone therapy; topoisomerase II
poisons; BNP and heart failure; pre-eclampsia; islet transplantation; …; `sae/features.csv`). The analogy vectors,
built from pairs of topic centroids, are close to single features: "X in cancer → X" to the tumour cell-line
feature (cos −0.92), "→ brain" to the brain / neurons / Alzheimer feature (0.89), "→ microRNAs" to the microRNA
feature (0.80); "clinic → genetics" is mutation / genotype (+0.63) against treatment outcome / RCT (−0.56), the same
pair as the within-topic axis W0 (+0.62 / −0.72): one contrast, seen between topics and inside them.

## 4. The themes behind the clusters: axes inside every topic

`axes.py` takes, for each topic, its papers' scatter around the centroid, scales each topic to equal weight, and
averages: W. W's top directions are the ways papers differ *inside* topics, shared across topics. For most topics
at least half of each lies in that topic's own top-20 directions: W0 93% of topics, W1 83%, W2 82%, W3 74%,
W4 65%, W5 52% (a random direction has 5% of its length there on average and essentially never half). W is an
average over these same topics, so this could be partly circular; rebuilding W without each topic in turn gives
the same shares (92, 83, 81, 74, 64, 52%). `label_axes.py` reads them off the papers' MeSH headings and publication types (effect = how far
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
papers (Spearman 0.32, 22,142 concepts; `concept_year.py`). Other single axes come close (analogy theme A12 0.31,
W1 0.26), and a ridge regression on all 384 centred dimensions (5-fold cross-validated) reaches 0.38. Against the
year NLM introduced the heading it is weaker (0.17; A10 and W1 do better, about 0.3), but recent headings do sit
toward "new": 2010s +0.77 sd, 2020s +0.73 sd, against −0.10 to +0.20 for the 1970s-2000s.

**Coordinates of core ideas** (`core_map.py`): the 72 GO slim biological processes placed by their own text on two
named axes. Cell and cancer biology (cell cycle, DNA replication and repair, chromosome segregation, transcription)
sit right, whole-body physiology (endocrine, renal, circulatory, digestive, lipid metabolism) left. Newer-style
research sits high (ncRNA gene silencing +2.3 sd, wound healing, autophagy, programmed cell death, protein folding),
older-style low (renal physiology, cytoplasmic translation, membrane and lysosome organisation). All 141 GO slim
terms on the main axes: `<name>_core_concepts.csv`.

![The 72 GO slim processes on two named axes](../figures/manifold_core_ideas.jpg)

## 6. Time vectors

The inspiration Dan remembered is most likely **ForeSCITE** (Chang, Ondov, …, Hua Xu, "Learning Scientific
Trajectories to Predict Future Biomedical Discoveries", arXiv 2610.38719, 28 Oct 2026): it clusters ~3M biomedical
conclusions into 407 topics, finds in each a direction from earlier to later findings, shifts embeddings along it
and decodes them into hypotheses. Two other candidates: Nylund, Gururangan & Smith 2024 (the "time vectors" name,
in model weights, https://arxiv.org/abs/2312.13401) and Tshitoyan et al. 2019 (embeddings of older abstracts flag
later discoveries, https://www.nature.com/articles/s41586-019-1335-8).

**Topics share one time direction.** Each topic's time vector (least-squares slope of its papers on year) points
the same way as the others' (mean pairwise cos 0.32 over 780 topics; the mean of the unit vectors has length 0.56).
44% of it lies on within-topic axis W2 (bench ↔ outcomes).

![Papers' within-topic position on the time vector by year](../figures/manifold_time.jpg)

The median paper is nearly flat until the mid-1980s (−1.45 sd in 1975, −1.34 in 1984), then climbs steadily to
+0.75 sd (2026), steeper after 2022.

**Forecasting topic positions** (`timevec.py drift`). Train 2009-2018, predict the 2021-23 centroid, error relative
to "the topic stays where it was in 2016-18":

| Prediction | Squared error vs persistence (median) | Topics improved |
|---|---|---|
| own slope | 1.19 | 33% |
| shared time vector | 0.91 | 87% |
| own and shared, half each | 0.87 | 69% |
| **chosen on 1999-2008 → 2011-13, then fixed: 0.2 own + 0.8 shared, 0.75 of the step** | **0.86** | **91%** |
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
| cosine | 0.718 | 0.831 | 0.670 | 0.802 |
| cosine after the time-vector shift | 0.771 | 0.831 | 0.723 | 0.790 |

30.5M (2010) and 46.7M (2015) unlinked pairs; 0.23% and 0.53% became linked. Fitted on 2010, tested on 2015:

| Logistic model | AUC | Average precision | Precision of the top 1,000 |
|---|---|---|---|
| popularity + common neighbours | 0.934 | 0.152 | 0.636 |
| + cosine | 0.953 | 0.169 | 0.578 |
| + cosine after the time shift | 0.952 | 0.172 | 0.636 |
| + both | **0.953** | **0.173** | 0.636 |

Closeness is the strongest signal once popularity is held fixed, and it adds to both graph baselines. The time shift
is a modest extra: as a single score it raises overall AUC (popular genes drift toward popular diseases), but within
popularity deciles it does not help in 2015; in the combined model it mainly repairs the top of the ranking. All
positions are centred with the mean of papers up to the cutoff (an all-years mean leans along the time vector; the
first version used it and scored slightly lower). Caveat: MedEmbed-small was trained after 2015 and may have read
some of the future.

**ForeSCITE's move, in embedding space** (`foresee.py`). Built from papers up to 2015, except that the topic seeds
come from the all-years clustering: centre with the pre-2016 mean, take each topic's pre-2016 centroid, give every
paper (1975-2023) to its nearest such centroid, refit each topic's time vector on its papers up to 2015, then shift
its 2011-15 papers along it and measure how close the real 2019-23 papers come to the shifted cloud. Step 1 = the
topic's own past rate carried over the real gap (2013 → 2021). Share of 776 topics whose future papers get closer:

| Step | Own time vector | Another topic's | Backwards | Random direction |
|---|---|---|---|---|
| 0.25 | **90%** | 79% | 3% | 22% |
| 0.5 | **86%** | 69% | 2% | 7% |
| 1 | **68%** | 41% | 1% | 1% |
| 1.5 | 48% | 21% | 1% | 0% |

Another topic's vector does well because the time direction is largely shared; the topic's own does better at every
step, so it also carries topic-specific direction. The gains are small (at most +0.001 in best-match cosine), and
the full past rate already overshoots a little. **Decoding** the shift as "which MeSH headings will rise in this
topic" (cos of the time vector with each heading's concept vector, against the change from 2011-15 to 2019-23 under
the same assignment): median Spearman 0.08, positive in 71% of 742 topics; 0.07 beyond trend extrapolation (which
alone gives 0.13). A weak but real signal. (A first version mixed two assignment rules and gave ~0; a review caught
it.)

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
| `mesh_labels.py eom_mcs50_ms3` | s | MeSH-only names for the 3,349 finer topics (after `topic_inputs.py`) |
| `sae.py` | 12 min (16 threads) | the sparse autoencoder and its feature names |
| `figures.py NAME` | 1 min | the two figures |
| `atlas.py NAME`, `explorer.py` | 3 min, 1 min | the explorer |
| `subset_texts.py NAME`, `gcp_embed.sh VM SHARD`, `gpu_embed.py`, `modelcheck.py` | ~1.5 h on two Spot L4s, 3 min | model check |

Outputs: `data/geometry/medembed_small/manifold/` (not in git). Topic and axis names were made by agents from MeSH,
gene and title evidence; the titles and abstracts themselves stay local.
