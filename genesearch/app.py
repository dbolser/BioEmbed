"""genesearch: type a gene, see its literature.

Run:  uv run python app.py --port 8766 --host 0.0.0.0
Needs ./index/ (build once with: uv run python build_index.py).
"""

import argparse
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import bm25s
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv
import Stemmer
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response

HERE = Path(__file__).resolve().parent
IDX = HERE / "index"
PAGE = 50
K1, B = 1.5, 0.75  # BM25 parameters (bm25s defaults)

STEMMER = Stemmer.Stemmer("english")
WORD = re.compile(r"(?u)\b\w\w+\b")  # bm25s's default token pattern
POOL = ThreadPoolExecutor(32)  # Arrow compute kernels release the GIL

# ---------------------------------------------------------------- data (loaded once)
GENES = pd.read_parquet(IDX / "genes.parquet")
GENES["row"] = range(len(GENES))
CURATED = pd.read_parquet(IDX / "curated.parquet")
_names = pd.read_parquet(IDX / "disease_names.parquet")
DISEASE_NAME = dict(zip(_names.disease_id, _names.name))
del _names

# Abstracts: uncompressed Arrow IPC, memory-mapped (zero-copy); rows sorted by pmid.
_ipc = pa.ipc.open_file(pa.memory_map(str(IDX / "abstracts.arrow")))
BATCHES = [_ipc.get_batch(i) for i in range(_ipc.num_record_batches)]
STARTS = np.cumsum([0] + [b.num_rows for b in BATCHES])
PMIDS = np.concatenate([b.column("pmid").to_numpy() for b in BATCHES])
YEARS = np.concatenate([b.column("year").fill_null(0).to_numpy(zero_copy_only=False) for b in BATCHES])


def take(rows: np.ndarray, cols: list[str]) -> pa.Table:
    """Abstract rows by row number, in the given order. Takes batch by batch: a whole-table
    take would concatenate 45 GB of strings and overflow Arrow's 32-bit string offsets."""
    rows = np.asarray(rows, dtype=np.int64)
    batch = np.searchsorted(STARTS, rows, side="right") - 1
    order = np.argsort(batch, kind="stable")
    parts = list(POOL.map(lambda b: BATCHES[b].select(cols).take(pa.array(rows[order][batch[order] == b] - STARTS[b])),
                          np.unique(batch)))
    back = np.empty_like(order)
    back[order] = np.arange(len(order))
    table = pa.Table.from_batches(parts, schema=BATCHES[0].select(cols).schema)
    return table.combine_chunks().take(pa.array(back))


_local = threading.local()


def db() -> duckdb.DuckDBPyConnection:
    if not hasattr(_local, "con"):
        _local.con = duckdb.connect(config={"threads": 16})
    return _local.con


def _split(v) -> list[str]:
    return [x for x in v.split("|") if x] if isinstance(v, str) else []


# Gene lookup maps (upper-cased keys -> GENES row numbers).
SYMBOL, ALIAS, PREV, NAME, IDS = {}, {}, {}, {}, {}
for g in GENES.itertuples():
    SYMBOL.setdefault(g.symbol.upper(), []).append(g.row)
    for a in _split(g.alias_symbol):
        ALIAS.setdefault(a.upper(), []).append(g.row)
    for p in _split(g.prev_symbol):
        PREV.setdefault(p.upper(), []).append(g.row)
    for n in [g.name, *_split(g.alias_name), *_split(g.prev_name)]:
        NAME.setdefault(n.upper(), []).append(g.row)
    for i in (g.hgnc_id, g.ensembl_gene_id, None if pd.isna(g.entrez_id) else str(int(g.entrez_id))):
        if isinstance(i, str):
            IDS[i.upper()] = [g.row]
NAME_TEXT = (GENES.name.fillna("") + " | " + GENES.alias_name.fillna("") + " | " + GENES.prev_name.fillna("")).str.upper()


def gene_info(row: int) -> dict:
    g = GENES.iloc[row].to_dict()  # (Series.name would be the row label, not the gene name)
    return {
        "hgnc_id": g["hgnc_id"], "symbol": g["symbol"], "name": g["name"],
        "aliases": _split(g["alias_symbol"]), "prev": _split(g["prev_symbol"]),
        "entrez": None if pd.isna(g["entrez_id"]) else int(g["entrez_id"]),
        "ensembl": g["ensembl_gene_id"] if isinstance(g["ensembl_gene_id"], str) else None,
        "location": g["location"], "locus_group": g["locus_group"], "n_papers": int(g["n_papers"]),
    }


def gene_row(hgnc_id: str) -> int:
    rows = IDS.get(hgnc_id.upper())
    if not rows:
        raise HTTPException(404, f"unknown gene {hgnc_id}")
    return rows[0]


# ---------------------------------------------------------------- gene search
def resolve(q: str) -> dict:
    """Exact symbol wins; else alias/previous symbol; else full name; else prefix/word matches.
    Prefix/word matches are flagged partial: the page shows them as a pick list, never auto-loads."""
    u = q.strip().upper()
    if not u:
        return {"matches": []}

    def tag(maps_whys, skip=(), limit=None, by_reason=False):
        hits = {}  # row -> (reason rank, reason); most-cited genes first within a rank
        for rank, (m, why) in enumerate(maps_whys):
            for r in (m if isinstance(m, list) else []):
                if r not in skip:
                    hits.setdefault(r, (rank if by_reason else 0, why))
        rows = sorted(hits, key=lambda r: (hits[r][0], -GENES.n_papers.iat[r]))
        return [{**gene_info(r), "why": hits[r][1]} for r in rows[:limit]], len(rows)

    if u in SYMBOL or u in IDS:
        rows = SYMBOL.get(u) or IDS[u]
        return {"matches": tag([(rows, "symbol")])[0],
                "also": tag([(ALIAS.get(u), "alias"), (PREV.get(u), "previous symbol")], skip=rows)[0]}
    for found, _ in (tag([(ALIAS.get(u), "alias"), (PREV.get(u), "previous symbol")]),
                     tag([(NAME.get(u), "name")])):
        if found:
            return {"matches": found}
    # Partial: symbol/alias prefix, or every query word starts a word of a name ('AD' must
    # not hit 'cADherin').
    mask = np.ones(len(GENES), bool)
    for w in u.split():
        mask &= NAME_TEXT.str.contains(r"(?<!\w)" + re.escape(w), regex=True).to_numpy()
    found, n = tag([([r for s, rs in SYMBOL.items() if s.startswith(u) for r in rs], "symbol starts with"),
                    ([r for s, rs in ALIAS.items() if s.startswith(u) for r in rs], "alias starts with"),
                    (np.flatnonzero(mask).tolist(), "name contains")], limit=40, by_reason=True)
    return {"matches": found, "n_more": max(0, n - 40), "partial": True}


# ---------------------------------------------------------------- paper sets
def rows_for(pmids: np.ndarray) -> np.ndarray:
    """Row numbers in the abstracts file for each pmid (-1 if not in the local PubMed copy)."""
    idx = np.searchsorted(PMIDS, pmids).clip(0, len(PMIDS) - 1)
    return np.where(PMIDS[idx] == pmids, idx, -1)


@lru_cache(maxsize=128)
def gene_tags(hgnc_id: str) -> tuple[dict[int, str], pd.DataFrame]:
    """({pmid: gene mentions} for PubTator-tagged papers, Open Targets curated links) for a gene."""
    g = gene_info(gene_row(hgnc_id))
    tagged = db().execute(
        f"SELECT pmid, mentions FROM '{IDX}/gene_pmids.parquet' WHERE ncbi = ?", [g["entrez"] or -1]).fetchnumpy()
    cur = CURATED[CURATED.ensembl == g["ensembl"]] if g["ensembl"] else CURATED.iloc[:0]
    return dict(zip(tagged["pmid"].tolist(), tagged["mentions"].tolist())), cur


@lru_cache(maxsize=128)
def paper_set(hgnc_id: str, src: str = "", disease: str = "") -> pd.DataFrame:
    """The gene's papers: PubTator-tagged plus OT-curated, or only those linked to one disease."""
    tagged, cur = gene_tags(hgnc_id)
    if src == "curated":
        pmids = cur.loc[cur.disease_id == disease, "pmid"].unique()
    elif src == "epmc":
        ensembl = gene_info(gene_row(hgnc_id))["ensembl"]
        pmids = db().execute(f"SELECT pmid FROM '{IDX}/epmc.parquet' WHERE ensembl = ? AND disease_id = ?",
                             [ensembl, disease]).df().pmid.to_numpy()
    else:
        pmids = np.union1d(np.fromiter(tagged, np.int64, len(tagged)), cur.pmid.to_numpy())
    df = pd.DataFrame({"pmid": np.unique(np.asarray(pmids, dtype=np.int64))})
    df["row"] = rows_for(df.pmid.to_numpy())
    df["year"] = np.where(df.row >= 0, YEARS[df.row.clip(0)], 0)
    df["no_text"] = df.row < 0  # no abstract in the local copy: no title or year either
    df["undated"] = df.year == 0  # listed last in either year order
    return df.sort_values(["undated", "year", "pmid"], ascending=[True, False, False], ignore_index=True)


def query_words(term: str) -> dict[str, str]:
    """Query word -> stem, with bm25s's tokenizer and English stopwords."""
    words = bm25s.tokenize(term, stopwords="en", return_ids=False, show_progress=False)[0]
    return {w: STEMMER.stemWord(w) for w in dict.fromkeys(words)}


def pmap(fn, text: pa.ChunkedArray) -> list:
    return list(POOL.map(fn, text.chunks))


def count(text: pa.ChunkedArray, pattern: str) -> np.ndarray:
    """Regex matches per paper (RE2: ASCII \\w and \\b), over 32 threads."""
    return np.concatenate([c.to_numpy() for c in pmap(lambda a: pc.count_substring_regex(a, pattern), text)])


@lru_cache(maxsize=8)
def corpus(hgnc_id: str, src: str, disease: str) -> tuple[pa.ChunkedArray, np.ndarray]:
    """Lower-cased title + abstract of the papers in paper_set that have text, and their positions."""
    df = paper_set(hgnc_id, src, disease)
    have = np.flatnonzero(df.row.to_numpy() >= 0)
    t = take(df.row.to_numpy()[have], ["title", "abstract"])
    step = max(1, -(-len(t) // 32))

    def prep(i):
        s = t.slice(i, step)
        joined = pc.binary_join_element_wise(s["title"].fill_null(""), s["abstract"].fill_null(""), " \n")
        return pc.utf8_lower(joined).combine_chunks()

    return pa.chunked_array(list(POOL.map(prep, range(0, len(t), step))), type=pa.string()), have


@lru_cache(maxsize=64)
def bm25(hgnc_id: str, src: str, disease: str, term: str) -> tuple[np.ndarray, np.ndarray]:
    """BM25 (Lucene IDF, k1=1.5, b=0.75 as bm25s) over the gene's papers, built on the fly,
    plus how many of the query's stems each paper contains.

    Vectorised in Arrow: for each query stem, find every word form in these papers that has
    that stem, then count those forms per paper with one regex. Exact term frequencies; doc
    length is measured in characters. ~20x faster than tokenising with bm25s (8.5 s for CFTR).
    """
    df = paper_set(hgnc_id, src, disease)
    text, have = corpus(hgnc_id, src, disease)
    scores, hits = np.zeros(len(df)), np.zeros(len(df), np.int64)
    if not len(have):
        return scores, hits
    dl = pc.utf8_length(text).to_numpy()
    norm = K1 * (1 - B + B * dl / dl.mean())
    for stem, forms in stem_forms(hgnc_id, src, disease, term).items():
        if not forms:
            continue
        tf = count(text, words_re(forms))
        n_with = np.count_nonzero(tf)
        idf = np.log(1 + (len(have) - n_with + 0.5) / (n_with + 0.5))
        scores[have] += idf * tf * (K1 + 1) / (tf + norm)
        hits[have] += tf > 0
    return scores, hits


@lru_cache(maxsize=64)
def stem_forms(hgnc_id: str, src: str, disease: str, term: str) -> dict[str, set[str]]:
    """Query stem -> word forms in these papers with that stem (e.g. fibrosi -> fibrosis, fibroses)."""
    text, _ = corpus(hgnc_id, src, disease)
    out: dict[str, set[str]] = {}
    for word, stem in query_words(term).items():
        # Scan words sharing the stem minus its last letter, which Snowball may rewrite
        # ('studies' -> 'studi' must still find 'study'); keep those whose stem matches exactly.
        prefix = os.path.commonprefix([word, stem[:-1]])
        if len(prefix) < 3:
            prefix = os.path.commonprefix([word, stem]) or word
        pattern = re.compile(r"\b" + re.escape(prefix) + r"\w*", re.ASCII)
        n_all = count(text, pattern.pattern)
        docs = np.flatnonzero(n_all)

        def scan(idx):
            return set(pattern.findall("\n".join(text.take(pa.array(idx)).to_pylist())))

        # Word forms from a sample, then scan only the papers where those forms miss a match.
        seen = scan(docs[:2000])
        if seen and len(docs) > 2000:
            seen |= scan(np.flatnonzero(count(text, words_re(seen)) != n_all))
        out.setdefault(stem, set()).update(f for f in seen if STEMMER.stemWord(f) == stem)
    return out


def words_re(words) -> str:
    return r"\b(?:" + "|".join(map(re.escape, sorted(words))) + r")\b"


def select(hgnc_id, term, src, disease, ymin, ymax, sort, match="all") -> tuple[pd.DataFrame, dict]:
    """The filtered, sorted paper list, and how the words matched: papers with all the query
    words by default (any word if none has all, or if asked)."""
    df = paper_set(hgnc_id, src, disease)
    info = {}
    if term := term.strip().lower():
        scores, hits = bm25(hgnc_id, src, disease, term)
        n_words = len(stem_forms(hgnc_id, src, disease, term))
        has_all, has_any = (hits == n_words) & (n_words > 0), scores > 0
        info = {"n_words": n_words, "n_all": int(has_all.sum()), "n_any": int(has_any.sum())}
        info["match"] = "any" if match == "any" or not info["n_all"] else "all"
        df = df.assign(score=scores)[has_all if info["match"] == "all" else has_any]
    else:
        df = df.assign(score=np.nan)
    if ymin:
        df = df[df.year >= ymin]
    if ymax:
        df = df[(df.year <= ymax) & (df.year > 0)]
    if sort == "oldest":
        df = df.sort_values(["undated", "year", "pmid"], ascending=[True, True, True])
    elif sort == "relevance" and term:
        df = df.sort_values(["score", "year"], ascending=False)
    return df, info  # else already newest first


def describe(df: pd.DataFrame, hgnc_id: str, term: str) -> list[dict]:
    """Full display records for a (small) slice of the paper list."""
    mentions, cur = gene_tags(hgnc_id)
    cur = cur[cur.pmid.isin(df.pmid)]
    cur_by = {p: g for p, g in cur.groupby("pmid")}
    stems = set(query_words(term).values()) if term.strip() else set()
    rows = df.row.to_numpy()
    t = take(rows.clip(0), ["title", "abstract", "year", "journal", "diseases"]).to_pylist()
    out = []
    for (pmid, row, score), a in zip(df[["pmid", "row", "score"]].itertuples(index=False), t):
        if row < 0:
            a = {"title": None, "abstract": None, "year": None, "journal": None, "diseases": None}
        c = cur_by.get(pmid)
        text = f"{a['title'] or ''} {a['abstract'] or ''}"
        out.append({
            "pmid": int(pmid), "title": a["title"], "abstract": a["abstract"], "year": a["year"],
            "journal": a["journal"], "score": None if np.isnan(score) else round(float(score), 2),
            "gene_mentions": _split(mentions.get(pmid)),
            "term_words": sorted({w for w in set(WORD.findall(text.lower())) if STEMMER.stemWord(w) in stems}),
            "curated": [] if c is None else [
                {"disease_id": d, "disease": n, "sources": sorted({s for x in g.sources for s in x})}
                for (d, n), g in c.groupby(["disease_id", "disease_name"])],
            "diseases": disease_tags(a["diseases"], text),
        })
    return out


def disease_tags(tags: list[dict] | None, text: str) -> list[dict]:
    """PubTator disease tags of one paper; those mentioned in the title/abstract first.
    Others come from the full text (PMC) or MeSH indexing."""
    out, seen = [], set()
    for t in tags or []:
        mentions = _split(t["mentions"])
        canon = DISEASE_NAME.get(t["id"], t["id"])
        # Label with the paper's own words: PubTator's id (and so its canonical name) is
        # sometimes wrong, e.g. 'TR insufficiency' -> MESH:D000309 adrenal insufficiency.
        name = canon if not mentions or canon.lower() in map(str.lower, mentions) else max(mentions, key=len)
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append({"id": t["id"], "name": name, "canonical": canon,
                    "in_abstract": any(mentioned(m, text) for m in mentions)})
    return sorted(out, key=lambda d: not d["in_abstract"])


def mentioned(mention: str, text: str) -> bool:
    """Whole-word match; case-sensitive for short mentions ('TR', 'AD') so they don't hit 'tr'."""
    flags = 0 if len(mention) <= 4 else re.IGNORECASE
    return re.search(r"(?<!\w)" + re.escape(mention) + r"(?!\w)", text, flags) is not None


# ---------------------------------------------------------------- API
app = FastAPI(title="genesearch")


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/resolve")
def api_resolve(q: str):
    t0 = time.perf_counter()
    res = resolve(q)
    res["elapsed_ms"] = round(1000 * (time.perf_counter() - t0))
    return res


@app.get("/api/gene/{hgnc_id}")
def api_gene(hgnc_id: str):
    t0 = time.perf_counter()
    g = gene_info(gene_row(hgnc_id))
    tagged, cur = gene_tags(hgnc_id)
    curated = [
        {"disease_id": d, "disease": n, "n_papers": int(x.pmid.nunique()),
         "sources": sorted({s for v in x.sources for s in v})}
        for (d, n), x in cur.groupby(["disease_id", "disease_name"])]
    curated.sort(key=lambda r: (-r["n_papers"], r["disease"]))
    epmc = db().execute(
        f"""SELECT disease_id, count(*) AS n FROM '{IDX}/epmc.parquet' WHERE ensembl = ?
            GROUP BY 1 ORDER BY n DESC, disease_id LIMIT 25""", [g["ensembl"] or ""]).fetchall()
    df = paper_set(hgnc_id)
    return {
        **g, "n_tagged": len(tagged), "n_curated_papers": int(cur.pmid.nunique()),
        "n_set": len(df), "n_local": int((df.row >= 0).sum()),
        "year_min": int(df.year[df.year > 0].min()) if (df.year > 0).any() else None,
        "year_max": int(df.year.max()) if len(df) else None,
        "curated": curated,
        "epmc": [{"disease_id": d, "disease": DISEASE_NAME.get(d, d), "n_papers": n} for d, n in epmc],
        "elapsed_ms": round(1000 * (time.perf_counter() - t0)),
    }


@app.get("/api/papers")
def api_papers(gene: str, term: str = "", src: str = "", disease: str = "",
               ymin: int = 0, ymax: int = 0, sort: str = "", match: str = "all", page: int = 1):
    t0 = time.perf_counter()
    sort = sort or ("relevance" if term.strip() else "newest")
    df, info = select(gene, term, src, disease, ymin, ymax, sort, match)
    n_pages = max(1, -(-len(df) // PAGE))
    page = min(max(1, page), n_pages)
    papers = describe(df.iloc[(page - 1) * PAGE: page * PAGE], gene, term)
    return {"n_matched": len(df), "n_set": len(paper_set(gene, src, disease)), **info,
            "n_no_text": int(df.no_text.sum()), "page": page,
            "n_pages": n_pages, "sort": sort, "papers": papers,
            "elapsed_ms": round(1000 * (time.perf_counter() - t0))}


@app.get("/api/papers.csv")
def api_papers_csv(gene: str, term: str = "", src: str = "", disease: str = "",
                   ymin: int = 0, ymax: int = 0, sort: str = "", match: str = "all"):
    sort = sort or ("relevance" if term.strip() else "newest")
    df, _ = select(gene, term, src, disease, ymin, ymax, sort, match)
    _, cur = gene_tags(gene)
    have = pa.array(df.row.to_numpy() >= 0)
    t = take(df.row.to_numpy().clip(0), ["title", "journal", "diseases"]).combine_chunks()
    # PubTator tag names per paper, all in Arrow: name each distinct id once, keep the list
    # offsets, join each list. (A pandas groupby-join took 12 s for TP53's 4.8M tags.)
    tags = t.column("diseases").combine_chunks()
    ids = pc.dictionary_encode(pc.struct_field(tags.values, "id"))
    names = pa.array([DISEASE_NAME.get(i, i) for i in ids.dictionary.to_pylist()], pa.string())
    named = pa.ListArray.from_arrays(tags.offsets, names.take(ids.indices), mask=tags.is_null())
    cur_names = cur.groupby("pmid").disease_name.agg(lambda s: "; ".join(sorted(set(s))))
    out = pa.table({
        "pmid": df.pmid.to_numpy(),
        "year": pa.array(df.year.to_numpy(), mask=df.year.to_numpy() == 0),
        "journal": pc.if_else(have, t["journal"], None), "title": pc.if_else(have, t["title"], None),
        "bm25_score": pa.array(df.score.round(3).to_numpy(), from_pandas=True),
        "curated_diseases": df.pmid.map(cur_names).fillna("").to_numpy(),
        "pubtator_diseases": pc.if_else(have, pc.binary_join(named, "; ").fill_null(""), ""),
        "url": pc.binary_join_element_wise("https://pubmed.ncbi.nlm.nih.gov/",
                                           pc.cast(pa.array(df.pmid.to_numpy()), pa.string()), "/", ""),
    })
    buf = pa.BufferOutputStream()
    pa_csv.write_csv(out, buf)
    sym = gene_info(gene_row(gene))["symbol"]
    name = re.sub(r"\W+", "_", f"{sym} {term} {DISEASE_NAME.get(disease, disease)}".strip())
    return Response(buf.getvalue().to_pybytes(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port)
