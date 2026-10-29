# geometry: the shape of the human-gene literature

The long-run question: what does "the geometry of biological knowledge" look like, and how does it change
from one embedding model to another? The first steps, below, use one model (MedEmbed-small) on the papers
about human genes, and ask a narrower question first: **when we squeeze the embeddings down to a few
dimensions to cluster and draw them, how much of their structure survives?**

Started 2026-10-28. Status: dimension reduction tuned; clustering not yet tuned.

## The papers and their vectors

- **Human-gene papers**: every PubMed paper that PubTator3 (release 2026-10-06) tags with a gene that is an
  HGNC human gene (matched on NCBI Gene id): 7,756,710 papers, `data/geometry/human_gene_pmids.parquet`
  (with each paper's genes).
- **Vectors**: MedEmbed-small (`abhinand/MedEmbed-small-v0.1`, 384 numbers per paper, unit length) of
  `title\n\nabstract`, taken from genesearch's whole-PubMed index (`genesearch/index/emb_medembed_small/`):
  7,755,620 of the papers have one. `subset.py` copies them out to `data/geometry/medembed_small/X.npy`
  (float16, 6 GB) and `pmids.npy`.
- **Reference map**: "The landscape of biomedical research" (González-Márquez, Schmidt, Schmidt, Berens &
  Kobak, Patterns 2024): PubMedBERT vectors of 23.4M PubMed abstracts and a 2-D t-SNE map of them, with
  a field per paper from its journal (40 fields; most papers have none). Downloaded to
  `/data/public/PubMedLandscape/2024_v2/` (see its README). 6.26M of our human-gene papers are on it.
  genesearch's "PubMedBERT map" tab draws it.

## Two ideas used throughout

**Neighbours in 384-D.** Each paper is a point: an arrow of 384 numbers. Two papers are close when their
arrows point the same way, measured by **cosine similarity** (1 = same direction, 0 = unrelated). A paper's
neighbours are the papers with the highest cosine similarity to it. Nothing approximate here: for 20,000
randomly chosen "query" papers, `truth.py` compares each against all 7.76 million papers (a big matrix
multiply, about 3 minutes on this box's CPUs) and keeps its true 100 nearest. Those lists are the ground
truth every map is scored against.

**Intrinsic dimension.** The vectors have 384 numbers, but the papers need not use all 384 directions.
Think of a sheet of paper crumpled in a room: it sits in 3-D but is really 2-D, because standing on it you
can only go two ways. The intrinsic dimension is how many directions you can really move in while staying
among the papers. It can be read off the neighbour distances:

- In a 1-D world (a line), if you double your radius you meet twice as many neighbours; in 2-D, four times
  as many; in d dimensions, 2^d times as many.
- So how fast the neighbour count grows with distance tells you d. **TwoNN** (Facco et al. 2017) uses only
  each paper's nearest and second-nearest neighbour distances: their ratio has a known spread that depends
  only on d. **Levina-Bickel** does the same with the nearest 10, 20 or 50.
- Here: **TwoNN 14.1; Levina-Bickel 13.6 (10 neighbours), 15.4 (20), 16.9 (50).** So the papers locally
  spread in about 14 to 17 directions, not 384. It creeps up with more neighbours because further out the
  "sheet" curves into more directions. (`data/geometry/medembed_small/eval/intrinsic.json`)

This says a squeeze to somewhere around 6 to 24 dimensions can keep most of the local structure, and that
2-D cannot.

## How a squeeze is scored

For each squeezed space, the same 20,000 query papers get their 15 (or 50) nearest neighbours again, now in
the squeezed space, and:

| Score | Meaning |
|---|---|
| Kept 15 / 50 nbrs | Of a paper's true 15 (50) neighbours, the share still among its 15 (50) nearest after the squeeze. "kNN recall", as in González-Márquez et al. |
| Share a gene | Of a paper's 15 nearest, the share that are tagged with at least one of its human genes. Biology, not geometry: the original space's own value is the ceiling. |
| Same field | Of neighbour pairs where both have a journal field, the share with the same field. |
| Global shape | Rank correlation of distances, original against squeezed, over 100,000 random pairs of papers: does far stay far? |
| Graph recall | UMAP only: the share of the true neighbours present in the approximate neighbour graph UMAP builds for itself. |

PCA, a straight-line squeeze, is the baseline.

## Results: the UMAP sweep

One full UMAP of all 7.76M papers per setting (RAPIDS cuML 26.10, one Spot NVIDIA L4, min_dist 0, cosine).
`umap_sweep.py`; table `data/geometry/medembed_small/sweep/results.csv`; k = UMAP's `n_neighbors`.

| Setting | Minutes | Kept 15 nbrs | Kept 50 nbrs | Graph recall | Share a gene | Same field | Global shape |
|---|---|---|---|---|---|---|---|
| original 384-D |  | 100% | 100% |  | 61% | 53% | 1.00 |
| PCA 2-D | 1.1 | 0% | 0% |  | 6% | 15% | 0.40 |
| PCA 6-D | 1.1 | 1% | 1% |  | 14% | 32% | 0.63 |
| PCA 12-D | 1.1 | 8% | 9% |  | 28% | 44% | 0.74 |
| PCA 24-D | 1.1 | 23% | 24% |  | 42% | 50% | 0.81 |
| UMAP k=15, 2-D | 5.7 | 6% | 9% | 91% | 31% | 37% | 0.24 |
| UMAP k=15, 6-D | 5.6 | 24% | 24% | 91% | 53% | 48% | 0.36 |
| UMAP k=15, 12-D | 5.5 | 25% | 25% | 91% | 54% | 49% | 0.36 |
| UMAP k=15, 24-D | 5.8 | 25% | 26% | 91% | 54% | 49% | 0.37 |
| UMAP k=50, 2-D | 5.9 | 4% | 7% | 94% | 29% | 36% | 0.26 |
| UMAP k=50, 6-D | 6.0 | 19% | 23% | 94% | 51% | 47% | 0.38 |
| UMAP k=50, 12-D | 6.8 | 20% | 23% | 94% | 52% | 48% | 0.38 |
| UMAP k=50, 24-D | 7.5 | 20% | 24% | 94% | 52% | 48% | 0.38 |
| UMAP k=50, 12-D (8 partitions, overlap 4) | 10.4 | 20% | 23% | 95% | 52% | 48% | 0.38 |
| UMAP k=50, 12-D (nn-descent degree 128, 40 iterations) | 9.9 | 20% | 23% | 94% | 52% | 48% | 0.38 |

k=100 and k=150 ran out of GPU memory (the L4 has 23 GB) at 7.76M papers.

**What it says**

- **Best: k=15 at 6 to 12 dimensions.** More dimensions stop helping at about 6, below the intrinsic
  dimension, so 12-D is safe but not needed.
- **The biology survives the squeeze.** At 12-D, 54% of a paper's neighbours share one of its genes, against
  61% in the full 384-D space. PCA needs 24 dimensions to reach 42%.
- **Only a quarter of the exact neighbours survive**, yet the biology mostly does: the neighbours that
  replace them are mostly papers on the same genes. Exact neighbour identity in a space this crowded is
  partly noise.
- **More neighbours (k=50) hurts.** It loses local detail and barely improves global shape (0.36 to 0.38).
- **UMAP's own neighbour graph is not the problem.** It finds 91 to 95% of the true neighbours, and making
  it more thorough changes nothing.
- **2-D loses most of the structure** (6% kept, 31% share a gene). The first full map (k=15, 12-D UMAP,
  HDBSCAN with leaf selection and minimum cluster size 200, then 2-D UMAP for drawing) looked like a blob
  with 6,203 tiny clusters and 60% noise. That blob comes from the 2-D drawing and the clustering settings,
  not from the 12-D squeeze.

## Is t-SNE worth trying?

González-Márquez et al. chose t-SNE for their 2-D map because, in their tests, it kept neighbours better
than UMAP. `tsne_check.py` tests that on our papers. Pool: the 6.26M human-gene papers on their map.
Queries: the 16,094 fixed queries in the pool. Each space finds neighbours within the pool. "Kept 15 nbrs"
is against each map's own source vectors: their t-SNE against PubMedBERT, our UMAP against MedEmbed.
Table: `data/geometry/medembed_small/sweep/tsne_check.csv`.

| Space | Kept 15 nbrs of its source | Share a gene | Same field |
|---|---|---|---|
| MedEmbed 384-D (our source) | 100% | 61% | 53% |
| PubMedBERT 768-D (their source) | 100% | 49% | 58% |
| their t-SNE 2-D (from PubMedBERT) | 9% | 33% | 50% |
| our UMAP k=15 2-D (from MedEmbed) | 7% | 32% | 36% |
| our UMAP k=15 12-D (from MedEmbed) | 26% | 54% | 49% |

- **For a 2-D picture, t-SNE wins.** Their map keeps more of its source's neighbours (9% against 7%), keeps
  two thirds of its source's gene signal (33% of 49%) where our 2-D UMAP keeps half (32% of 61%), and keeps
  nearly all its source's field signal (50% of 58%) where ours keeps two thirds (36% of 53%).
- **For clustering, UMAP at 12-D still wins.** t-SNE is in practice limited to 2-D (its fast versions do
  not go much beyond 2 or 3 dimensions), and no 2-D map comes close to 12-D UMAP's 54%.
- **The two models see different things.** MedEmbed's neighbours share genes more (61% against 49%);
  PubMedBERT's share journal fields more (58% against 53%). That difference is the first data point on
  "the geometry of biological knowledge" across models.
- **Caveat.** This compares two maps made from different vectors. The fair test is t-SNE on our MedEmbed
  vectors (openTSNE on CPU, as they did, or cuML's TSNE on GPU), scored with `umap_sweep.py`'s scores.

So: cluster in 12-D UMAP, draw in 2-D t-SNE.

## Clustering: notes for the next step

- Cluster on the 12-D (or 6-D) k=15 UMAP; draw separately. Score clusters with the same gene and field
  measures (do papers in a cluster share genes more than chance?), so the cluster sweep has an objective
  too.
- HDBSCAN took 85 minutes on 7.76M papers in 12-D on an L4 (leaf selection). Try excess-of-mass selection
  and much bigger minimum cluster sizes; sweep on a 1M sample first.
- **Cosine in HDBSCAN.** cuML's HDBSCAN only takes Euclidean distance (cuML DBSCAN also takes cosine). On
  unit-length vectors this matters less than it seems: Euclidean distance squared = 2 - 2 x cosine, so the
  two put every pair of papers in the same order. HDBSCAN's tree depends only on that order, so on
  unit-length vectors Euclidean HDBSCAN builds the same tree as cosine HDBSCAN; only cluster "stability",
  which uses the distance values, can pick slightly different clusters. On UMAP output cosine is the wrong
  measure anyway: UMAP lays its points out by Euclidean distance. A cosine option matters for clustering the
  original vectors directly, which at 384-D runs into the crowding the intrinsic dimension describes.

## Running it

| Script | Where | What |
|---|---|---|
| `subset.py` | here, root env | the human-gene vectors out of the genesearch index |
| `truth.py` | here, root env (~3 min) | queries, exact neighbours, intrinsic dimension, fields and genes per paper |
| `map_gpu.py` | GPU, this dir's env | one full map: UMAP to 12-D, HDBSCAN, UMAP to 2-D |
| `umap_sweep.py` | GPU, this dir's env | the sweep above; resumes after a preemption |
| `gcp_babysit.sh VM ZONE [SEEN]` | here | restarts a preempted Spot VM, relaunches the sweep, prints each finished row |
| `tsne_check.py` | here, root env (~10 min) | the t-SNE comparison |

GPU runs used GCP Spot `g2-standard-16` VMs with one L4 (project `agentic-ai-iuk-ipf`, image family
`common-cu129-ubuntu-2204-nvidia-580`): `uv sync` in a copy of this directory installs cuML. The sweep took
about 2.5 hours including two preemptions, at Spot prices well under $5. Spot L4 capacity in Europe comes and
goes: a stopped VM sometimes cannot restart for half an hour.
