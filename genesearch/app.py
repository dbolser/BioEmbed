"""genesearch: type a gene, see its literature.

Run:  uv run python app.py --port 8766 --host 0.0.0.0
Needs ./index/ (build once with: uv run python build_index.py).
Meaning (AI embedding) ranking and the literature map: semantic.py, litmap.py.
"""

import argparse
import json
import os
import re
import threading
import time
import unicodedata
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, reduce
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
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response

import litmap
import semantic
from semantic import Pending

HERE = Path(__file__).resolve().parent
IDX = HERE / "index"
PAGE = 50
K1, B = 1.5, 0.75  # BM25 parameters (bm25s defaults)
RRF_K = 60  # reciprocal-rank fusion constant for hybrid ranking
SEM_CAP = 30_000  # most papers given meaning ranking while vectors are computed here on CPU
MAP_CAP = 20_000  # most papers on a map (a fixed sample beyond that)
MAP_VERSION = 7  # bump to recompute cached maps
EPMC_TOP = 50  # text-mined diseases listed per gene

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
        "location": _str(g["location"]), "locus_group": _str(g["locus_group"]), "n_papers": int(g["n_papers"]),
    }


def _str(v) -> str | None:
    return v if isinstance(v, str) else None


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
def gene_epmc(hgnc_id: str) -> pd.DataFrame:
    """Europe PMC text-mined gene-disease links for a gene (Open Targets evidence): pmid, disease_id."""
    ensembl = gene_info(gene_row(hgnc_id))["ensembl"]
    return db().execute(f"SELECT pmid, disease_id FROM '{IDX}/epmc.parquet' WHERE ensembl = ?", [ensembl or ""]).df()


@lru_cache(maxsize=128)
def epmc_counts(hgnc_id: str) -> pd.Series:
    """Papers per text-mined disease for a gene, most first (ties by id)."""
    n = gene_epmc(hgnc_id).groupby("disease_id").size()
    return n.iloc[np.lexsort((n.index.to_numpy(), -n.to_numpy()))]


@lru_cache(maxsize=128)
def paper_set(hgnc_id: str, ot: str = "", ep: str = "") -> pd.DataFrame:
    """The gene's papers: PubTator-tagged plus OT-curated. With a disease facet, only the papers that
    Open Targets curators (ot) and Europe PMC text mining (ep) link to the gene and that disease
    ('any' = any disease); with both, the papers in both."""
    tagged, cur = gene_tags(hgnc_id)
    sets = []
    if ot:
        sets.append(cur.pmid.to_numpy() if ot == "any" else cur.loc[cur.disease_id == ot, "pmid"].to_numpy())
    if ep:
        e = gene_epmc(hgnc_id)
        sets.append(e.pmid.to_numpy() if ep == "any" else e.loc[e.disease_id == ep, "pmid"].to_numpy())
    if sets:
        pmids = reduce(np.intersect1d, sets)
    else:
        pmids = np.union1d(np.fromiter(tagged, np.int64, len(tagged)), cur.pmid.to_numpy())
    df = pd.DataFrame({"pmid": np.unique(np.asarray(pmids, dtype=np.int64))})
    df["row"] = rows_for(df.pmid.to_numpy())
    df["year"] = np.where(df.row >= 0, YEARS[df.row.clip(0)], 0)
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
    return np.concatenate([np.zeros(0, np.int64), *[c.to_numpy() for c in pmap(lambda a: pc.count_substring_regex(a, pattern), text)]])


@lru_cache(maxsize=8)
def corpus(hgnc_id: str, ot: str, ep: str) -> tuple[pa.ChunkedArray, np.ndarray]:
    """Lower-cased title + abstract of the papers in paper_set that have text, and their positions."""
    df = paper_set(hgnc_id, ot, ep)
    have = np.flatnonzero(df.row.to_numpy() >= 0)
    t = take(df.row.to_numpy()[have], ["title", "abstract"])
    step = max(1, -(-len(t) // 32))

    def prep(i):
        s = t.slice(i, step)
        joined = pc.binary_join_element_wise(s["title"].fill_null(""), s["abstract"].fill_null(""), " \n")
        return pc.utf8_lower(joined).combine_chunks()

    return pa.chunked_array(list(POOL.map(prep, range(0, len(t), step))), type=pa.string()), have


@lru_cache(maxsize=64)
def bm25(hgnc_id: str, ot: str, ep: str, term: str) -> tuple[np.ndarray, np.ndarray]:
    """BM25 (Lucene IDF, k1=1.5, b=0.75 as bm25s) over the gene's papers, built on the fly,
    plus how many of the query's stems each paper contains.

    Vectorised in Arrow: for each query stem, find every word form in these papers that has
    that stem, then count those forms per paper with one regex. Exact term frequencies; doc
    length is measured in characters. ~20x faster than tokenising with bm25s (8.5 s for CFTR).
    """
    df = paper_set(hgnc_id, ot, ep)
    text, have = corpus(hgnc_id, ot, ep)
    scores, hits = np.zeros(len(df)), np.zeros(len(df), np.int64)
    if not len(have):
        return scores, hits
    dl = pc.utf8_length(text).to_numpy()
    norm = K1 * (1 - B + B * dl / dl.mean())
    for stem, forms in stem_forms(hgnc_id, ot, ep, term).items():
        if not forms:
            continue
        tf = count(text, words_re(forms))
        n_with = np.count_nonzero(tf)
        idf = np.log(1 + (len(have) - n_with + 0.5) / (n_with + 0.5))
        scores[have] += idf * tf * (K1 + 1) / (tf + norm)
        hits[have] += tf > 0
    return scores, hits


@lru_cache(maxsize=64)
def stem_forms(hgnc_id: str, ot: str, ep: str, term: str) -> dict[str, set[str]]:
    """Query stem -> word forms in these papers with that stem (e.g. fibrosi -> fibrosis, fibroses)."""
    text, _ = corpus(hgnc_id, ot, ep)
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


@lru_cache(maxsize=128)
def curated_pmids(hgnc_id: str) -> np.ndarray:
    return np.unique(gene_tags(hgnc_id)[1].pmid.to_numpy())


def scope(hgnc_id, term, ot, ep, ymin, ymax, match, rank) -> tuple[pd.DataFrame, dict]:
    """The current list before ranking: the gene's papers (or those the disease facets pick, see
    paper_set), the year filters, and for keyword ranking only the papers with the words: all of
    them by default (any of them if asked, or if no paper has all). Meaning and hybrid ranking
    keep every paper, ordered by how close it is."""
    df = paper_set(hgnc_id, ot, ep)
    keep = np.ones(len(df), bool)
    year = df.year.to_numpy()
    if ymin:
        keep &= year >= ymin
    if ymax:
        keep &= (year <= ymax) & (year > 0)
    info = {"n_scope": int(keep.sum())}
    df = df.assign(bm25=np.nan)
    if term := term.strip().lower():
        scores, hits = bm25(hgnc_id, ot, ep, term)
        n_words = len(stem_forms(hgnc_id, ot, ep, term))
        has_all, has_any = (hits == n_words) & (n_words > 0) & keep, (scores > 0) & keep
        info.update(n_words=n_words, n_all=int(has_all.sum()), n_any=int(has_any.sum()))
        df["bm25"] = scores
        if rank == "keywords":
            info["match"] = "any" if match == "any" or not info["n_all"] else "all"
            keep = has_all if info["match"] == "all" else has_any
    return df[keep], info


def texts_for(pmids: np.ndarray) -> list[str]:
    """'title\n\nabstract' of papers in the local PubMed copy (the text the embeddings are of)."""
    t = take(rows_for(np.asarray(pmids)), ["title", "abstract"])
    return [semantic.doc_text(a, b) for a, b in zip(t["title"].to_pylist(), t["abstract"].to_pylist())]


def similarity(hgnc_id: str, df: pd.DataFrame, term: str) -> tuple[np.ndarray, dict]:
    """Cosine between the words (embedded as a query) and each paper (as a document); NaN where
    a paper has no vector. Raises Pending while vectors are being computed."""
    have = np.flatnonzero(df.row.to_numpy() >= 0)
    info = {"sem_source": semantic.source()}
    if info["sem_source"] == "cpu" and len(have) > SEM_CAP:
        # Until the whole-PubMed index exists: the best keyword matches, then the newest.
        sub = df.iloc[have]
        best = np.lexsort((-sub.pmid.to_numpy(), -sub.year.to_numpy(), -sub.bm25.fillna(0).to_numpy()))
        have = np.sort(have[best[:SEM_CAP]])
        info["sem_capped"] = {"n": SEM_CAP, "of": int((df.row >= 0).sum())}
    vecs, ok = semantic.vectors(hgnc_id, df.pmid.to_numpy()[have], texts_for)
    sim = np.full(len(df), np.nan)
    sim[have[ok]] = vecs[ok] @ semantic.query_vector(term)
    info["n_sem"] = int(ok.sum())
    return sim, info


def ranks(score: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """1 = best; NaN where not ranked. Ties keep the list order (newest first)."""
    r = np.full(len(score), np.nan)
    idx = np.flatnonzero(valid)
    r[idx[np.argsort(-score[idx], kind="stable")]] = np.arange(1, len(idx) + 1)
    return r


def select(hgnc_id, term, ot, ep, ymin, ymax, sort, match="all", rank="keywords") -> tuple[pd.DataFrame, dict]:
    """The filtered, ranked paper list and how the words matched (see scope). Ranking: keywords =
    BM25; meaning = cosine of MedEmbed vectors; hybrid = reciprocal-rank fusion of the two (k=60).
    Raises Pending while meaning vectors are being computed."""
    df, info = scope(hgnc_id, term, ot, ep, ymin, ymax, match, rank)
    df = df.assign(score=df.bm25, sim=np.nan, kw_rank=np.nan, sem_rank=np.nan)
    if (term := term.strip().lower()) and rank in ("meaning", "hybrid"):
        sim, sem = similarity(hgnc_id, df, term)
        info.update(sem)
        kw, mn = ranks(df.bm25.to_numpy(), df.bm25.to_numpy() > 0), ranks(sim, ~np.isnan(sim))
        df = df.assign(sim=sim, kw_rank=kw, sem_rank=mn)
        df["score"] = sim if rank == "meaning" else np.nan_to_num(1 / (RRF_K + kw)) + np.nan_to_num(1 / (RRF_K + mn))
    if sort == "oldest":
        df = df.sort_values(["undated", "year", "pmid"], ascending=[True, True, True])
    elif sort == "relevance" and term:
        df = df.sort_values(["score", "year"], ascending=False, na_position="last")
    return df, info  # else already newest first


def describe(df: pd.DataFrame, hgnc_id: str, term: str) -> list[dict]:
    """Full display records for a (small) slice of the paper list."""
    mentions, cur = gene_tags(hgnc_id)
    cur = cur[cur.pmid.isin(df.pmid)]
    cur_by = {p: g for p, g in cur.groupby("pmid")}
    ep = gene_epmc(hgnc_id)
    ep_by = {p: g.disease_id.tolist() for p, g in ep[ep.pmid.isin(df.pmid)].groupby("pmid")}
    ep_n = epmc_counts(hgnc_id)
    stems = set(query_words(term).values()) if term.strip() else set()
    rows = df.row.to_numpy()
    t = take(rows.clip(0), ["title", "abstract", "year", "journal", "diseases"]).to_pylist()
    out = []
    for (pmid, row, bm, sim, kw, mn), a in zip(
            df[["pmid", "row", "bm25", "sim", "kw_rank", "sem_rank"]].itertuples(index=False), t):
        if row < 0:
            a = {"title": None, "abstract": None, "year": None, "journal": None, "diseases": None}
        c = cur_by.get(pmid)
        text = f"{a['title'] or ''} {a['abstract'] or ''}"
        out.append({
            "pmid": int(pmid), "title": a["title"], "abstract": a["abstract"], "year": a["year"],
            "journal": a["journal"], "bm25": _num(bm, 2) if bm > 0 else None, "sim": _num(sim, 3),
            "kw_rank": _num(kw, 0), "sem_rank": _num(mn, 0),
            "gene_mentions": _split(mentions.get(pmid)),
            "term_words": sorted({w for w in set(WORD.findall(text.lower())) if STEMMER.stemWord(w) in stems}),
            "curated": [] if c is None else [
                {"disease_id": d, "disease": n, "sources": sorted({s for x in g.sources for s in x})}
                for (d, n), g in c.groupby(["disease_id", "disease_name"])],
            "epmc": [{"disease_id": d, "disease": DISEASE_NAME.get(d, d)}
                     for d in sorted(ep_by.get(pmid, []), key=lambda d: (-ep_n.get(d, 0), d))],
            "diseases": disease_tags(a["diseases"], text),
        })
    return out


def _num(v: float, digits: int):
    return None if np.isnan(v) else round(float(v), digits) if digits else int(v)


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
app.add_middleware(GZipMiddleware, minimum_size=2000)


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
    ep = gene_epmc(hgnc_id)
    epmc = epmc_counts(hgnc_id).head(EPMC_TOP)
    df = paper_set(hgnc_id)
    return {
        **g, "n_tagged": len(tagged), "n_curated_papers": int(cur.pmid.nunique()),
        "n_set": len(df), "n_local": int((df.row >= 0).sum()),
        "year_min": int(df.year[df.year > 0].min()) if (df.year > 0).any() else None,
        "year_max": int(df.year.max()) if len(df) else None,
        "curated": curated, "n_epmc_papers": int(ep.pmid.nunique()), "n_epmc_diseases": int(ep.disease_id.nunique()),
        "epmc": [{"disease_id": d, "disease": DISEASE_NAME.get(d, d), "n_papers": int(n)} for d, n in epmc.items()],
        "elapsed_ms": round(1000 * (time.perf_counter() - t0)),
    }


@app.get("/api/facets")
def api_facets(gene: str, ot: str = "", ep: str = "", term: str = "", ymin: int = 0, ymax: int = 0,
               match: str = "all", rank: str = "keywords"):
    """Papers per disease in each box within the other filters (the other box's pick and the years;
    diseases with none are left out): what clicking a row gives. Clicking a disease clears the
    "and..." words, so they count only for 'any' (that box's 'Any disease' row, which keeps them).
    Each box counts within its own 'any' set, as a click on one of its rows does. Plus the names."""
    _, cur = gene_tags(gene)

    def counts(links: pd.DataFrame, box_ot: str, box_ep: str) -> dict:
        def within(t: str) -> pd.DataFrame:
            return links[links.pmid.isin(scope(gene, t, box_ot, box_ep, ymin, ymax, match, rank)[0].pmid)]
        rows = within("")
        return {**rows.groupby("disease_id").pmid.nunique().astype(int).to_dict(),
                "any": int((within(term) if term.strip() else rows).pmid.nunique())}

    out = {"ot": counts(cur[["pmid", "disease_id"]], "any", ep), "ep": counts(gene_epmc(gene), ot, "any")}
    out["names"] = {i: DISEASE_NAME.get(i, i) for k in ("ot", "ep") for i in out[k] if i != "any"}
    return out


@app.get("/api/papers")
def api_papers(gene: str, term: str = "", ot: str = "", ep: str = "", ymin: int = 0, ymax: int = 0,
               sort: str = "", match: str = "all", rank: str = "keywords", page: int = 1):
    t0 = time.perf_counter()
    sort = sort or ("relevance" if term.strip() else "newest")
    try:
        df, info = select(gene, term, ot, ep, ymin, ymax, sort, match, rank)
    except Pending as p:
        return {"pending": p.job.status()}
    n_pages = max(1, -(-len(df) // PAGE))
    page = min(max(1, page), n_pages)
    papers = describe(df.iloc[(page - 1) * PAGE: page * PAGE], gene, term)
    return {"n_matched": len(df), **info, "rank": rank, "page": page, "n_pages": n_pages, "sort": sort,
            "papers": papers, "names": facet_names(ot, ep), "elapsed_ms": round(1000 * (time.perf_counter() - t0))}


def facet_names(*ids: str) -> dict[str, str]:
    """Names of the diseases picked in the facets (the page may only have their ids, from a URL)."""
    return {i: DISEASE_NAME.get(i, i) for i in ids if i and i != "any"}


@app.get("/api/paper")
def api_paper(gene: str, pmid: int, term: str = ""):
    """One paper as in the list (the map's click-through)."""
    df = pd.DataFrame({"pmid": [pmid], "row": rows_for(np.array([pmid])), "bm25": np.nan, "sim": np.nan,
                       "kw_rank": np.nan, "sem_rank": np.nan})
    return describe(df, gene, term)[0]


@app.get("/api/map")
def api_map(gene: str, term: str = "", ot: str = "", ep: str = "", ymin: int = 0, ymax: int = 0,
            match: str = "all", rank: str = "keywords", topic: str = ""):
    """Literature map of the current list (its papers with an abstract; a fixed sample of MAP_CAP
    beyond that). topic = a drill-in path ('3', '3.2'): a new map of only topic 3's papers, then of
    its sub-topic 2. Returns {"pending": progress, "n": papers} until it is ready."""
    df, _ = scope(gene, term, ot, ep, ymin, ymax, match, rank)
    pm = np.sort(df.pmid.to_numpy()[df.row.to_numpy() >= 0])
    about = {"n_list": len(df), "n_text": len(pm), "sampled": len(pm) > MAP_CAP}
    if len(pm) > MAP_CAP:  # pseudo-random by PMID: the same sample each time, overlapping across filters
        pm = np.sort(pm[np.argsort((pm * 2654435761) % 2**32, kind="stable")[:MAP_CAP]])
    about["n_shown"] = len(pm)
    path, within = [], ""
    for step in [s for s in topic.split(".") if s]:
        if len(pm) < MIN_MAP:
            break
        try:
            m, _ = get_map(gene, pm, within)
        except Pending as p:
            return {"pending": p.job.status(), "n": len(pm)}
        cl = next((c for c in m["clusters"] if str(c["id"]) == step), None)
        if cl is None:
            raise HTTPException(404, f"no topic {step} in this map")
        pm = np.asarray(m["pmid"], np.int64)[np.asarray(m["c"]) == cl["id"]]
        within = cl["name"]
        path.append({"id": cl["id"], "name": cl["name"], "n": len(pm)})
    if len(pm) < MIN_MAP:
        return {**about, "too_few": True, "path": path, "names": facet_names(ot, ep)}
    try:
        m, key = get_map(gene, pm, within)
    except Pending as p:
        return {"pending": p.job.status(), "n": len(pm)}
    return {**m, **about, "key": key, "path": path, "names": facet_names(ot, ep)}


MIN_MAP = 20  # fewest papers for a map


def get_map(gene: str, pm: np.ndarray, within: str = "") -> tuple[dict, str]:
    """The map of these papers (cached) and its key; raises Pending while it is made. within = the
    name of the topic they all come from (a drill-in), so their sub-topics are named as such."""
    key = semantic.digest(gene, pm, semantic.source(), MAP_VERSION,  # gene: curated rings, stop words, names
                          *(["drill"] if within else []))  # not the name: an AI rename must not lose the map
    if (m := cached_map(key)) is None:
        j = semantic.job(("map", key), lambda jb: make_map(gene, pm, key, jb, within))
        if not j.finished or j.error:
            raise Pending(j)
        m = j.result
    elif m["named_by"] == "terms" and litmap.name_clusters(gene_info(gene_row(gene))["symbol"], m["clusters"], within):
        m["named_by"] = "llm"  # the LLM answers now (e.g. credit added): keep its names
        save_map(key, m)
    return m, key


@app.get("/api/axes")
def api_axes(gene: str, key: str, a: str, b: str):
    """Map Explorer: the papers of a map placed by how close each is in meaning to two things (the
    axes): words, embedded as a query, or a topic's centre ('topic:3', the mean of its papers'
    vectors). Closeness = cosine similarity, i.e. the paper vectors projected onto each axis."""
    if not re.fullmatch(r"[0-9a-f]{20}", key) or (m := cached_map(key)) is None:
        raise HTTPException(404, "That map is gone: open the map again.")
    pm = np.asarray(m["pmid"], np.int64)
    try:
        vecs, ok = semantic.vectors(gene, pm, texts_for)
    except Pending as p:
        return {"pending": p.job.status(), "n": len(pm)}
    labels = np.asarray(m["c"])

    def axis(spec: str) -> np.ndarray:
        if t := re.fullmatch(r"topic:(-?\d+)", spec.strip()):
            members = vecs[(labels == int(t[1])) & ok]
            if not len(members):
                raise HTTPException(400, f"no topic {t[1]} in this map")
            v = members.mean(0)
            v /= np.linalg.norm(v)
        elif spec.strip():
            v = semantic.query_vector(spec.strip())
        else:
            raise HTTPException(400, "Give both axes some words or a topic.")
        return np.where(ok, vecs @ v, np.nan)

    x, y = axis(a), axis(b)
    r = np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 2 else np.nan
    as_list = lambda v: [None if np.isnan(s) else round(float(s), 4) for s in v]
    return {"x": as_list(x), "y": as_list(y), "r": _num(r, 2)}


MAPS = semantic.CACHE_DIR / "maps"


@lru_cache(maxsize=16)
def _read_map(path: Path, mtime: float) -> dict:
    return json.loads(path.read_text())


def cached_map(key: str) -> dict | None:
    f = MAPS / f"{key}.json"
    return _read_map(f, f.stat().st_mtime) if f.exists() else None


def make_map(gene: str, pm: np.ndarray, key: str, job: semantic.Job, within: str = "") -> dict:
    t0 = time.perf_counter()
    vecs, ok = job.wait_for(lambda: semantic.vectors(gene, pm, texts_for))
    pm, vecs = pm[ok], vecs[ok]
    t = take(rows_for(pm), ["title", "abstract", "year"])
    titles, abstracts = t["title"].to_pylist(), t["abstract"].to_pylist()
    g = gene_info(gene_row(gene))
    m = litmap.build(vecs, [f"{a or ''}\n{b or ''}" for a, b in zip(titles, abstracts)], titles,
                     g["symbol"], {g["symbol"], *g["aliases"], *g["prev"]}, job, within)
    m.update(pmid=pm.tolist(), year=t["year"].fill_null(0).to_pylist(),
             title=[(x or "")[:220] for x in titles], curated=np.isin(pm, curated_pmids(gene)).astype(int).tolist(),
             source=semantic.source(), seconds=round(time.perf_counter() - t0, 1))
    save_map(key, m)
    return m


def save_map(key: str, m: dict):
    MAPS.mkdir(parents=True, exist_ok=True)
    tmp = MAPS / f"{key}.{os.getpid()}.{threading.get_ident()}.tmp"
    tmp.write_text(json.dumps(m))
    os.replace(tmp, MAPS / f"{key}.json")


@app.get("/api/papers.csv")
def api_papers_csv(gene: str, term: str = "", ot: str = "", ep: str = "", ymin: int = 0, ymax: int = 0,
                   sort: str = "", match: str = "all", rank: str = "keywords"):
    sort = sort or ("relevance" if term.strip() else "newest")
    try:
        df, _ = select(gene, term, ot, ep, ymin, ymax, sort, match, rank)
    except Pending:
        raise HTTPException(409, "The meaning ranking is still being worked out. Wait for the list, then download.")
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
    links = gene_epmc(gene)
    links = links[links.pmid.isin(df.pmid)]
    ep_names = links.groupby("pmid").disease_id.agg(lambda s: "; ".join(sorted({DISEASE_NAME.get(d, d) for d in s})))
    cols = {
        "pmid": df.pmid.to_numpy(),
        "year": pa.array(df.year.to_numpy(), mask=df.year.to_numpy() == 0),
        "journal": pc.if_else(have, t["journal"], None), "title": pc.if_else(have, t["title"], None),
        "bm25_score": pa.array(df.bm25.where(df.bm25 > 0).round(3).to_numpy(), from_pandas=True),
    }
    if df.sim.notna().any():
        cols["meaning_similarity"] = pa.array(df.sim.round(4).to_numpy(), from_pandas=True)
    cols.update({
        "curated_diseases": df.pmid.map(cur_names).fillna("").to_numpy(),
        "textmined_diseases": df.pmid.map(ep_names).fillna("").to_numpy(),
        "pubtator_diseases": pc.if_else(have, pc.binary_join(named, "; ").fill_null(""), ""),
        "url": pc.binary_join_element_wise("https://pubmed.ncbi.nlm.nih.gov/",
                                           pc.cast(pa.array(df.pmid.to_numpy()), pa.string()), "/", ""),
    })
    buf = pa.BufferOutputStream()
    pa_csv.write_csv(pa.table(cols), buf)
    sym = gene_info(gene_row(gene))["symbol"]
    facets = [f"{'curated' if k == 'ot' else 'textmined'} {'' if v == 'any' else DISEASE_NAME.get(v, v)}"
              for k, v in (("ot", ot), ("ep", ep)) if v]
    name = re.sub(r"\W+", "_", " ".join([sym, term, *facets]).strip()).strip("_")
    # HTTP headers are Latin-1: an ASCII name ('TGF-β' -> 'TGF'), plus the real one for browsers that read it
    ascii_name = re.sub(r"[^A-Za-z0-9]+", "_", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()).strip("_")
    return Response(buf.getvalue().to_pybytes(), media_type="text/csv", headers={"Content-Disposition":
                    f"attachment; filename=\"{ascii_name or 'papers'}.csv\"; filename*=UTF-8''{urllib.parse.quote(name)}.csv"})


def warm_up():
    """Load the query model and compile UMAP before anyone waits for them."""
    semantic.full_index()
    semantic.query_vector("warm-up")
    litmap.warm_up()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    threading.Thread(target=warm_up, daemon=True).start()
    uvicorn.run(app, host=a.host, port=a.port)
