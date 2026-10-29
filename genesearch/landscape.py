"""The PubMedBERT map tab: papers placed on "The landscape of biomedical research"
(González-Márquez, Schmidt, Schmidt, Berens & Kobak, Patterns 2024), a 2-D t-SNE of the
PubMedBERT embeddings of 23.4M PubMed abstracts (Zenodo record 12794883, data v2 2024).
Positions and field labels (40, from journal names; most papers unlabelled) are theirs.

Build once (~2 min): uv run python landscape.py
  reads  LANDSCAPE (their metadata table as parquet: PMID, x, y, Labels, Colors, ...)
         index/gene_pmids.parquet + index/genes.parquet (PubTator3 gene tags of HGNC genes)
  writes index/landscape/
           pmid.npy     int64, sorted, one row per PMID on their map (duplicates: the first)
           xy.npy       float32 (n, 2), their t-SNE position
           field.npy    int8, index into fields.json's fields
           meta.json    fields (name, colour, centre and count over the human-gene papers), extent
           background.png  the human-gene papers drawn in their field colours (unlabelled = grey)
The app loads these (load()) and draws the current list's papers over the background.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "index" / "landscape"
LANDSCAPE = "/data/public/PubMedLandscape/2024_v2/pubmed_landscape_data_2024_v2.parquet"
WIDTH = 2400  # background image width in pixels (height follows the data's aspect)
UNLABELLED = "unlabeled"
SMOOTH = 3  # pixels
MAX_ALPHA = 0.75  # strongest background colour: lighter, so the list's dots stand out on top


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#").ljust(6, "0")  # (their cancer colour is '#00000')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def build() -> None:
    from PIL import Image
    from scipy.ndimage import uniform_filter

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(config={"threads": 32})
    fields = con.sql(f"""SELECT Labels, any_value(Colors) FROM '{LANDSCAPE}' GROUP BY Labels
                         ORDER BY Labels = '{UNLABELLED}' DESC, Labels""").fetchall()  # unlabelled = 0
    names = [f for f, _ in fields]
    t = con.sql(f"""
        SELECT PMID AS pmid, x::FLOAT AS x, y::FLOAT AS y, Labels AS label
        FROM '{LANDSCAPE}' WHERE PMID IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY PMID ORDER BY x, y) = 1
        ORDER BY PMID""").fetchnumpy()
    pmid = t["pmid"].astype(np.int64)
    xy = np.stack([t["x"], t["y"]], 1).astype(np.float32)
    lut = {n: i for i, n in enumerate(names)}
    field = np.array([lut[v] for v in t["label"]], np.int8)
    human = con.sql(f"""
        SELECT DISTINCT g.pmid FROM '{HERE}/index/gene_pmids.parquet' g
        JOIN (SELECT DISTINCT entrez_id::BIGINT AS ncbi FROM '{HERE}/index/genes.parquet' WHERE entrez_id IS NOT NULL) h
        USING (ncbi)""").fetchnumpy()["pmid"].astype(np.int64)
    on = np.isin(pmid, human)
    print(f"{len(pmid):,} papers on the landscape; {on.sum():,} of {len(human):,} human-gene papers", flush=True)

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
    strength = lambda n: np.clip(np.log1p(n) / np.log1p(np.percentile(n[n > 1e-6], 90)), 0, 1) * (n > 1e-6)
    a_lab, a_unl = strength(n_lab) * MAX_ALPHA, strength(n_unl) * MAX_ALPHA
    grey = rgb[0]
    img = 255 * (1 - a_unl[:, None]) + grey * a_unl[:, None]          # grey layer over white
    img = img * (1 - a_lab[:, None]) + mean * a_lab[:, None]           # field colours on top
    alpha = np.maximum(a_lab, a_unl) > 0.02
    rgba = np.concatenate([img, np.where(alpha, 255, 0)[:, None]], 1).reshape(height, WIDTH, 4)
    Image.fromarray(rgba.round().astype(np.uint8), "RGBA").save(OUT / "background.png", optimize=True)

    # Field names sit at the median position of the field's human-gene papers.
    meta_fields = []
    for i, ((name, colour), c) in enumerate(zip(fields, rgb)):
        sel = hf == i
        x, y = (np.median(hx[sel], 0) if sel.any() else (None, None))
        meta_fields.append({"id": i, "name": name, "colour": "#%02x%02x%02x" % tuple(int(v) for v in c),
                            "n": int(sel.sum()), "x": None if x is None else float(x), "y": None if y is None else float(y)})
    meta = {"fields": meta_fields, "extent": {"x": [float(lo[0]), float(hi[0])], "y": [float(lo[1]), float(hi[1])]},
            "n_papers": len(pmid), "n_human": int(on.sum()), "size": [WIDTH, height], "built": time.strftime("%Y-%m-%d")}
    for name, a in {"pmid": pmid, "xy": xy, "field": field}.items():
        np.save(OUT / f"{name}.npy", a)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {OUT} ({WIDTH}x{height} background) in {time.time() - t0:.0f}s")


class Landscape:
    def __init__(self) -> None:
        self.pmid = np.load(OUT / "pmid.npy", mmap_mode="r")
        self.xy = np.load(OUT / "xy.npy", mmap_mode="r")
        self.field = np.load(OUT / "field.npy", mmap_mode="r")
        self.meta = json.loads((OUT / "meta.json").read_text())
        self.meta["version"] = int((OUT / "background.png").stat().st_mtime)  # (cache-busts the image URL)

    def lookup(self, pm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For these PMIDs: which are on the landscape (bool), and their rows there (where on)."""
        i = np.searchsorted(self.pmid, pm).clip(0, len(self.pmid) - 1)
        on = self.pmid[i] == pm
        return on, i[on]


_loaded: Landscape | None = None


def load() -> Landscape | None:
    """The built landscape, or None if build() has not been run (checked again on each call until it is)."""
    global _loaded
    if _loaded is None and (OUT / "meta.json").exists():  # (build() writes meta.json last)
        _loaded = Landscape()
    return _loaded


if __name__ == "__main__":
    build()
