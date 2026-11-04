"""MeSH descriptors, and each human-gene paper's MeSH headings and publication types, from local PubMed XML.

Inputs (read-only): IUK-IPF PubMed baseline XML (data/sources/pubmed/raw/pubmed26n*.xml.gz) and MeSH desc2026.gz.
Writes data/geometry/medembed_small/manifold/:
  mesh_desc.parquet   ui (int, D000001 -> 1), name, scope (scope note), trees (list of tree numbers)
  mesh_papers.parquet row (row in X.npy), pmid, mesh (list<int32> descriptor ui), major (list<bool>),
                      ptypes (list<str> publication types). Later archives override earlier ones (PubMed updates).
CPU, root env (~10 min, 100 processes): uv run python geometry/manifold/parse_mesh.py
"""
import gzip, sys, time
from multiprocessing import Pool
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/geometry/medembed_small"
OUT = D / "manifold"
IUK = Path("/outsee5/home/danbolser/Work/IUK-IPF")
RAW = IUK / "data/sources/pubmed/raw"
DESC = IUK / "TermRelevance/enrich/raw/desc2026.gz"
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
ui = lambda s: int(s[1:])


def descriptors():
    rows = []
    for _, el in ET.iterparse(gzip.open(DESC), events=("end",)):
        if el.tag != "DescriptorRecord": continue
        sn = el.find("ConceptList/Concept[@PreferredConceptYN='Y']/ScopeNote")
        rows.append(dict(ui=ui(el.findtext("DescriptorUI")), name=el.findtext("DescriptorName/String"),
                         scope=(sn.text or "").strip() if sn is not None else "",
                         trees=[t.text for t in el.findall("TreeNumberList/TreeNumber")]))
        el.clear()
    pq.write_table(pa.Table.from_pylist(rows), OUT / "mesh_desc.parquet")
    log(f"descriptors {len(rows):,}")


def one(path):
    """All (pmid, MeSH uis, major flags, publication types) in one archive; main() keeps our papers."""
    out = []
    for _, el in ET.iterparse(gzip.open(path), events=("end",)):
        if el.tag != "PubmedArticle": continue
        mc = el.find("MedlineCitation")
        pmid = int(mc.findtext("PMID"))
        hs = mc.findall("MeshHeadingList/MeshHeading/DescriptorName")
        pts = [p.text for p in mc.findall("Article/PublicationTypeList/PublicationType")]
        out.append((pmid, [ui(h.get("UI")) for h in hs], [h.get("MajorTopicYN") == "Y" for h in hs], pts))
        el.clear()
    return path.name, out


if __name__ == "__main__":
    descriptors()
    pm = np.load(D / "pmids.npy"); order = np.argsort(pm); spm = pm[order]
    files = sorted(RAW.glob("pubmed26n*.xml.gz"))
    best = {}
    with Pool(100) as pool:
        for i, (nm, recs) in enumerate(pool.imap(one, files, chunksize=1)):  # imap keeps file order
            arr = np.fromiter((r[0] for r in recs), np.int64, len(recs))
            j = np.searchsorted(spm, arr).clip(0, len(spm) - 1); hit = spm[j] == arr
            for r, h, jj in zip(recs, hit, j):
                if h: best[int(order[jj])] = r
            if i % 100 == 0: log(f"{i + 1}/{len(files)} archives, {len(best):,} of our papers")
    rows = sorted(best)
    t = pa.table({"row": pa.array(rows, pa.int64()), "pmid": pa.array([best[r][0] for r in rows], pa.int64()),
                  "mesh": pa.array([best[r][1] for r in rows], pa.list_(pa.int32())),
                  "major": pa.array([best[r][2] for r in rows], pa.list_(pa.bool_())),
                  "ptypes": pa.array([best[r][3] for r in rows], pa.list_(pa.string()))})
    pq.write_table(t, OUT / "mesh_papers.parquet")
    nm = np.array([len(best[r][1]) for r in rows])
    log(f"papers found {len(rows):,} of {len(pm):,}; with MeSH {np.mean(nm > 0):.1%}, median {np.median(nm):.0f} headings")
