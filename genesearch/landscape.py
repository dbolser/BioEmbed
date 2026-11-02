"""The whole-literature map tabs: the list's papers placed on a fixed 2-D map of PubMed.

Two maps, built once each:
  pubmedbert  "The landscape of biomedical research" (González-Márquez, Schmidt, Schmidt, Berens & Kobak,
              Patterns 2024): a 2-D t-SNE of the PubMedBERT embeddings of 23.4M PubMed abstracts (Zenodo
              record 12794883, data v2 2024). Positions and field labels (40, from journal names; most
              papers unlabelled) are theirs. Reads LANDSCAPE (their metadata table as parquet). ~2 min.
  medembed    ours (geometry/README.md): a 2-D t-SNE (openTSNE) of the MedEmbed-small embeddings of the
              7.76M human-gene papers, coloured by 780 clusters found in a 12-D UMAP of the same vectors
              (HDBSCAN, every paper voted in) and named by their most typical genes. Reads
              data/geometry/medembed_small/ (tsne_full.py, cluster_sweep.py --assign, draw_clusters.py).

  uv run python landscape.py [pubmedbert|medembed]
  writes index/landscape/ (pubmedbert) or index/landscape_medembed/:
           pmid.npy     int64, sorted, one row per PMID on the map (duplicates: the first)
           xy.npy       float32 (n, 2), its position
           field.npy    int16, index into meta.json's fields (journal field, or cluster); 0 = none
           meta.json    fields (name, colour, centre and count over the human-gene papers), extent
           background.png  the human-gene papers drawn in their field colours (none = grey)
Human-gene papers: index/gene_pmids.parquet + index/genes.parquet (PubTator3 tags of HGNC genes).
The app loads these (load(name)) and draws the current list's papers over the background.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import numpy as np

HERE = Path(__file__).resolve().parent
MAPS = {"pubmedbert": HERE / "index" / "landscape", "medembed": HERE / "index" / "landscape_medembed"}
LANDSCAPE = "/data/public/PubMedLandscape/2024_v2/pubmed_landscape_data_2024_v2.parquet"
GEOMETRY = HERE.parent / "data" / "geometry" / "medembed_small"
CLUSTERS = "eom_mcs200_ms3"  # the clustering drawn on the medembed map (geometry/README.md)
WIDTH = 2400  # background image width in pixels (height follows the data's aspect)
UNLABELLED = "unlabeled"
SMOOTH = 3  # pixels
MAX_ALPHA = 0.75  # strongest background colour: lighter, so the list's dots stand out on top


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#").ljust(6, "0")  # (their cancer colour is '#00000')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def read_pubmedbert(con) -> tuple:
    """Their positions and journal fields; fields as (name, colour), the unlabelled first (= 0)."""
    fields = con.sql(f"""SELECT Labels, any_value(Colors) FROM '{LANDSCAPE}' GROUP BY Labels
                         ORDER BY Labels = '{UNLABELLED}' DESC, Labels""").fetchall()
    t = con.sql(f"""
        SELECT PMID AS pmid, x::FLOAT AS x, y::FLOAT AS y, Labels AS label
        FROM '{LANDSCAPE}' WHERE PMID IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY PMID ORDER BY x, y) = 1
        ORDER BY PMID""").fetchnumpy()
    lut = {n: i for i, (n, _) in enumerate(fields)}
    field = np.array([lut[v] for v in t["label"]], np.int16)
    return t["pmid"].astype(np.int64), np.stack([t["x"], t["y"]], 1).astype(np.float32), field, fields


def read_medembed(con) -> tuple:
    """Our t-SNE positions and 12-D UMAP clusters; each cluster named by its typical genes, else its field."""
    import pandas as pd
    pmid = np.load(GEOMETRY / "pmids.npy").astype(np.int64)
    xy = np.load(GEOMETRY / "sweep/tsne_p30_d2.npy").astype(np.float32)
    lab = np.load(GEOMETRY / f"clusters/{CLUSTERS}_all.npy").astype(np.int64)
    summ = pd.read_csv(GEOMETRY / f"clusters/{CLUSTERS}_summary.csv").set_index("cluster").sort_index()
    import colorsys
    rng = np.random.default_rng(0)  # (fixed, distinct-ish colours: golden-ratio hues, varied lightness)
    hue = (np.arange(len(summ)) * 0.618034) % 1
    colours = ["#%02x%02x%02x" % tuple(int(255 * c) for c in colorsys.hls_to_rgb(h, 0.42 + 0.16 * rng.random(), 0.75))
               for h in hue]
    name = lambda r: " · ".join(str(r.genes).split()) if isinstance(r.genes, str) and r.genes else f"({r.field})"
    fields = [("no cluster", "#d3d3d3")] + [(name(r), c) for r, c in zip(summ.itertuples(), colours)]
    o = np.argsort(pmid, kind="stable")
    return pmid[o], xy[o], (lab[o] + 1).astype(np.int16), fields


def build(name: str = "pubmedbert") -> None:
    from PIL import Image
    from scipy.ndimage import uniform_filter

    t0 = time.time()
    out = MAPS[name]
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(config={"threads": 32})
    pmid, xy, field, fields = (read_pubmedbert if name == "pubmedbert" else read_medembed)(con)
    human = con.sql(f"""
        SELECT DISTINCT g.pmid FROM '{HERE}/index/gene_pmids.parquet' g
        JOIN (SELECT DISTINCT entrez_id::BIGINT AS ncbi FROM '{HERE}/index/genes.parquet' WHERE entrez_id IS NOT NULL) h
        USING (ncbi)""").fetchnumpy()["pmid"].astype(np.int64)
    on = np.isin(pmid, human)
    print(f"{len(pmid):,} papers on the {name} map; {on.sum():,} of {len(human):,} human-gene papers", flush=True)

    # Extent: the whole landscape's, so every paper fits, padded 2%.
    lo, hi = xy.min(0), xy.max(0)
    pad = (hi - lo) * 0.02
    lo, hi = lo - pad, hi + pad
    height = round(WIDTH * (hi[1] - lo[1]) / (hi[0] - lo[0]))

    # Background: per pixel, the mean colour of its labelled human-gene papers over a grey for the
    # unlabelled ones; strength grows with log density. Then washed out towards white.
    hx, hf = xy[on], field[on]
    px = np.clip(((hx[:, 0] - lo[0]) / (hi[0] - lo[0]) * WIDTH).astype(int), 0, WIDTH - 1)
    py = np.clip(((hi[1] - hx[:, 1]) / (hi[1] - lo[1]) * height).astype(int), 0, height - 1)  # (image rows top-down)
    cell = py * WIDTH + px
    size = WIDTH * height
    rgb = np.array([hex_rgb(c) for _, c in fields], np.float64)
    lab = hf > 0
    # Counts smoothed over a SMOOTH-pixel square: at ~1 paper per pixel the raw counts speckle.
    blur = lambda w: uniform_filter(np.bincount(cell, weights=w, minlength=size).reshape(height, WIDTH), SMOOTH).ravel()
    n_lab, n_unl = blur(lab.astype(float)), blur((~lab).astype(float))
    mean = np.stack([blur(np.where(lab, rgb[hf, k], 0)) for k in range(3)], 1)
    mean /= np.maximum(n_lab, 1e-9)[:, None]
    def strength(n):  # (zero everywhere when there are no such papers, e.g. no unclustered ones)
        if not (n > 1e-6).any():
            return np.zeros_like(n)
        return np.clip(np.log1p(n) / np.log1p(np.percentile(n[n > 1e-6], 90)), 0, 1) * (n > 1e-6)
    a_lab, a_unl = strength(n_lab) * MAX_ALPHA, strength(n_unl) * MAX_ALPHA
    grey = rgb[0]
    img = 255 * (1 - a_unl[:, None]) + grey * a_unl[:, None]          # grey layer over white
    img = img * (1 - a_lab[:, None]) + mean * a_lab[:, None]           # field colours on top
    alpha = np.maximum(a_lab, a_unl) > 0.02
    rgba = np.concatenate([img, np.where(alpha, 255, 0)[:, None]], 1).reshape(height, WIDTH, 4)
    Image.fromarray(rgba.round().astype(np.uint8), "RGBA").save(out / "background.png", optimize=True)

    # Field names sit at the median position of the field's human-gene papers.
    meta_fields = []
    order = np.argsort(hf, kind="stable"); sh = hf[order]
    start, stop = np.searchsorted(sh, np.arange(len(fields))), np.searchsorted(sh, np.arange(len(fields)), "right")
    for i, ((fname, colour), c) in enumerate(zip(fields, rgb)):
        sel = order[start[i]:stop[i]]
        x, y = (np.median(hx[sel], 0) if len(sel) else (None, None))
        meta_fields.append({"id": i, "name": fname, "colour": "#%02x%02x%02x" % tuple(int(v) for v in c),
                            "n": len(sel), "x": None if x is None else float(x), "y": None if y is None else float(y)})
    meta = {"map": name, "fields": meta_fields, "extent": {"x": [float(lo[0]), float(hi[0])], "y": [float(lo[1]), float(hi[1])]},
            "n_papers": len(pmid), "n_human": int(on.sum()), "size": [WIDTH, height], "built": time.strftime("%Y-%m-%d")}
    for fn, a in {"pmid": pmid, "xy": xy, "field": field}.items():
        np.save(out / f"{fn}.npy", a)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out} ({WIDTH}x{height} background) in {time.time() - t0:.0f}s")


class Landscape:
    def __init__(self, dir: Path) -> None:
        self.dir = dir
        self.pmid = np.load(dir / "pmid.npy", mmap_mode="r")
        self.xy = np.load(dir / "xy.npy", mmap_mode="r")
        self.field = np.load(dir / "field.npy", mmap_mode="r")
        self.meta = json.loads((dir / "meta.json").read_text())
        self.meta["version"] = int((dir / "background.png").stat().st_mtime)  # (cache-busts the image URL)

    def lookup(self, pm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For these PMIDs: which are on the landscape (bool), and their rows there (where on)."""
        i = np.searchsorted(self.pmid, pm).clip(0, len(self.pmid) - 1)
        on = self.pmid[i] == pm
        return on, i[on]


_loaded: dict[str, Landscape] = {}


def load(name: str = "pubmedbert") -> Landscape | None:
    """The built map, or None if build(name) has not been run (checked again on each call until it is)."""
    if name not in _loaded and name in MAPS and (MAPS[name] / "meta.json").exists():  # (build() writes it last)
        _loaded[name] = Landscape(MAPS[name])
    return _loaded.get(name)


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "pubmedbert")
