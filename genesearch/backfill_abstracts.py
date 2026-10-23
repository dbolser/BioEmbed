"""Backfill papers the index refers to (PubTator gene tags, Open Targets links) but that are not in
abstracts.parquet: the local PubMed copy ends at a date, and only holds papers with an abstract.
Fetched from PubMed E-utilities (title, abstract, year, journal; title-only when there is no abstract).

Steps (default: all, in this order; each resumes where it stopped):
  fetch   missing PMIDs -> index/backfill/NNNN.parquet, 500 PMIDs per request, 3/s (10/s with NCBI_API_KEY);
          a request takes a few seconds, so several run at once
  merge   abstracts.parquet := abstracts + backfill (sorted by pmid), then build_index.py arrow
  embed   MedEmbed-small vectors for the backfilled papers on CPU, as one more shard of
          index/emb_medembed_small/ (pmids.npy gets them appended: the app sorts on load)

Usage: uv run python backfill_abstracts.py [fetch|merge|embed ...]
Rerunning build_index.py abstracts drops the backfill: run merge (and nothing else) again after it.
"""

import os
import threading
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

IDX = Path(__file__).resolve().parent / "index"
BF = IDX / "backfill"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
KEY = os.environ.get("NCBI_API_KEY")
BATCH, RATE, INFLIGHT = 500, 10 if KEY else 3, 12
SCHEMA = pa.schema([("pmid", pa.int64()), ("title", pa.string()), ("abstract", pa.string()),
                    ("year", pa.int16()), ("journal", pa.string())])


def missing() -> np.ndarray:
    return duckdb.sql(f"""
        WITH want AS (SELECT DISTINCT pmid FROM '{IDX}/gene_pmids.parquet'
                      UNION SELECT DISTINCT pmid FROM '{IDX}/epmc.parquet'
                      UNION SELECT DISTINCT pmid FROM '{IDX}/curated.parquet')
        SELECT pmid FROM want
        WHERE pmid BETWEEN 1 AND 999999999 AND pmid NOT IN (SELECT pmid FROM '{IDX}/abstracts.parquet')
        ORDER BY pmid""").fetchnumpy()["pmid"].astype(np.int64)


def parse(xml: bytes) -> list[dict]:
    out = []
    for a in ET.fromstring(xml).iter("PubmedArticle"):
        m = a.find("MedlineCitation")
        art = m.find("Article")
        t = art.find("ArticleTitle")
        parts = []
        for s in art.iterfind("Abstract/AbstractText"):
            text = "".join(s.itertext()).strip()
            label = s.get("Label")
            if text:
                parts.append(f"{label}: {text}" if label else text)
        date = art.find("Journal/JournalIssue/PubDate")
        y = (date.findtext("Year") or date.findtext("MedlineDate") or "")[:4] if date is not None else ""
        out.append({"pmid": int(m.findtext("PMID")), "title": "".join(t.itertext()).strip() or None if t is not None else None,
                    "abstract": "\n".join(parts) or None,  # sections one per line, as the local copy
                    "year": int(y) if y.isdigit() else None,
                    "journal": art.findtext("Journal/Title")})
    return out


_last, _tick = 0.0, threading.Lock()


def throttle():  # at most RATE requests start per second, across threads
    global _last
    with _tick:
        now = time.time()
        wait = _last + 1 / RATE - now
        if wait > 0:
            time.sleep(wait)
        _last = max(now, _last + 1 / RATE)


def efetch(pmids: np.ndarray) -> list[dict]:
    body = {"db": "pubmed", "retmode": "xml", "id": ",".join(map(str, pmids))}
    if KEY:
        body["api_key"] = KEY
    data = urllib.parse.urlencode(body).encode()
    for attempt in range(6):
        throttle()
        try:
            with urllib.request.urlopen(urllib.request.Request(EUTILS, data=data), timeout=120) as r:
                return parse(r.read())
        except Exception as e:  # 429, 5xx, timeouts, truncated XML: wait and retry
            print(f"  retry {attempt + 1}: {e}", flush=True)
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"efetch failed for PMIDs {pmids[0]}..{pmids[-1]}")


def fetch():
    BF.mkdir(exist_ok=True)
    todo = missing()
    n = len(todo)
    print(f"fetch: {n:,} PMIDs missing from abstracts.parquet, {-(-n // BATCH):,} requests at {RATE}/s", flush=True)
    t0 = time.time()

    def one(b: int) -> int:
        f = BF / f"{b:04d}.parquet"
        if f.exists():
            return 0
        rows = efetch(todo[b * BATCH:(b + 1) * BATCH])
        table = pa.Table.from_pylist(rows, schema=SCHEMA) if rows else SCHEMA.empty_table()
        pq.write_table(table, f.with_suffix(".tmp"), compression="zstd")
        f.with_suffix(".tmp").rename(f)
        return len(rows)

    with ThreadPoolExecutor(INFLIGHT) as pool:
        for b, got in enumerate(pool.map(one, range(-(-n // BATCH)))):
            if b % 50 == 0:
                print(f"  batch {b}: {min(n, (b + 1) * BATCH):,}/{n:,} PMIDs ({time.time() - t0:.0f}s)", flush=True)
    stats = duckdb.sql(f"""SELECT count(*) AS papers, count(abstract) AS with_abstract, count(title) AS with_title,
                           min(year), max(year) FROM '{BF}/*.parquet'""").fetchall()[0]
    print(f"fetch done: {stats[0]:,} papers found ({stats[1]:,} with an abstract, {stats[2]:,} with a title), "
          f"years {stats[3]}-{stats[4]}; {n - stats[0]:,} PMIDs unknown to PubMed")


def merge():
    con = duckdb.connect(config={"threads": 64, "memory_limit": "300GB", "temp_directory": str(IDX / ".tmp"),
                                 "preserve_insertion_order": True})
    (IDX / ".tmp").mkdir(exist_ok=True)
    before = con.sql(f"SELECT count(*) FROM '{IDX}/abstracts.parquet'").fetchall()[0][0]
    con.sql(f"""
        COPY (
          SELECT * FROM '{IDX}/abstracts.parquet'
          UNION ALL
          SELECT pmid, title, abstract, year, journal FROM '{BF}/*.parquet'
          WHERE pmid NOT IN (SELECT pmid FROM '{IDX}/abstracts.parquet') AND (title IS NOT NULL OR abstract IS NOT NULL)
          ORDER BY pmid
        ) TO '{IDX}/abstracts.parquet.tmp' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 100000)""")
    after = con.sql(f"SELECT count(*), count(DISTINCT pmid) FROM read_parquet('{IDX}/abstracts.parquet.tmp')").fetchall()[0]
    assert after[0] == after[1], "duplicate PMIDs after merge"
    (IDX / "abstracts.parquet.tmp").rename(IDX / "abstracts.parquet")
    con.close()
    print(f"merge: {before:,} -> {after[0]:,} papers", flush=True)
    import build_index
    build_index.arrow(build_index.connect())  # its own memory limit: the arrow step needs it
    print("merge: abstracts.arrow rebuilt", flush=True)


def embed():
    import multiprocessing as mp
    import semantic
    full = IDX / "emb_medembed_small"
    have = np.load(full / "pmids.npy")  # (not memory-mapped: the file is replaced below)
    shards = sorted(f for f in full.glob("[0-9]*.npy"))
    rows = sum(np.load(f, mmap_mode="r").shape[0] for f in shards)
    assert rows == len(have), f"{full}: {rows:,} vectors for {len(have):,} PMIDs"
    tbl = duckdb.sql(f"""SELECT pmid, title, abstract FROM '{IDX}/abstracts.parquet'
                         WHERE pmid IN (SELECT pmid FROM '{BF}/*.parquet') ORDER BY pmid""").to_arrow_table()
    pmids = tbl.column("pmid").to_numpy()
    new = ~np.isin(pmids, np.asarray(have))
    pmids = pmids[new]
    texts = [semantic.doc_text(t, a) for t, a in zip(np.asarray(tbl.column("title"))[new], np.asarray(tbl.column("abstract"))[new])]
    del tbl
    n = len(texts)
    print(f"embed: {n:,} papers in abstracts.parquet have no vector; {semantic.WORKERS} workers x {semantic.THREADS} threads", flush=True)
    if not n:
        return
    order = np.argsort([len(t) for t in texts], kind="stable")
    chunks = [order[i:i + 256] for i in range(0, n, 256)]
    vecs = np.zeros((n, semantic.DIM), np.float16)
    t0, done = time.time(), 0
    with mp.get_context("spawn").Pool(semantic.WORKERS, semantic._worker_init, (semantic.THREADS,)) as p:
        for idx, v in zip(chunks, p.imap(semantic._worker_encode, ([texts[i] for i in c] for c in chunks))):
            vecs[idx] = v
            done += len(idx)
            if done % 25600 < 256:
                print(f"  {done:,}/{n:,} ({done / (time.time() - t0):.0f}/s)", flush=True)
    f = full / f"{len(shards):04d}.npy"
    with open(f.with_suffix(".npy.tmp"), "wb") as fh:
        np.save(fh, vecs)
        fh.flush()
        os.fsync(fh.fileno())
    # pmids.npy is memory-mapped by the running app: a new file, then rename, leaves its copy intact.
    with open(full / "pmids.npy.tmp", "wb") as fh:
        np.save(fh, np.concatenate([np.asarray(have, dtype=np.int64), pmids]))
        fh.flush()
        os.fsync(fh.fileno())
    f.with_suffix(".npy.tmp").rename(f)
    (full / "pmids.npy.tmp").rename(full / "pmids.npy")
    print(f"embed: {f.name} ({n:,} vectors, {time.time() - t0:.0f}s); pmids.npy now {len(have) + n:,}", flush=True)


STEPS = {"fetch": fetch, "merge": merge, "embed": embed}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(STEPS):
        STEPS[name]()
