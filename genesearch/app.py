"""genesearch: type a gene (or a disease), see its literature.

Run:  uv run python app.py --port 8766 --host 0.0.0.0
Needs ./index/ (build once with: uv run python build_index.py).
Semantic (AI embedding) ranking and the literature map: semantic.py, litmap.py.

The search is about one subject: a gene (HGNC id, 'HGNC:1884') or a disease (an Open Targets id,
'MONDO_0009061', or a PubTator id no Open Targets disease covers, 'MESH:D001289'). Its papers are
those PubTator3 tagged with it plus those Open Targets curators linked to it; the facets are the
other side of Open Targets' gene-disease links (diseases for a gene, genes for a disease).
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
from functools import cache, lru_cache, reduce
from pathlib import Path

import bm25s
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
import Stemmer
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response

import build_index
import litmap
import semantic
from semantic import Pending

HERE = Path(__file__).resolve().parent
IDX = HERE / "index"
PAGE = 50
K1, B = 1.5, 0.75  # BM25 parameters (bm25s defaults)
RRF_K = 60  # reciprocal-rank fusion constant for hybrid ranking
SEM_CAP = 30_000  # most papers given semantic ranking while vectors are computed here on CPU
MAP_CAP = 20_000  # most papers on a map (a fixed sample beyond that)
MAP_VERSION = 7  # bump to recompute cached maps
EPMC_TOP = 50  # text-mined facets listed per subject

STEMMER = Stemmer.Stemmer("english")
WORD = re.compile(r"(?u)\b\w\w+\b")  # bm25s's default token pattern
POOL = ThreadPoolExecutor(32)  # Arrow compute kernels release the GIL

# ---------------------------------------------------------------- data (loaded once)
GENES = pd.read_parquet(IDX / "genes.parquet")
GENES["row"] = range(len(GENES))
DISEASES = pd.read_parquet(IDX / "diseases.parquet")
DISEASES["row"] = range(len(DISEASES))
CURATED = pd.read_parquet(IDX / "curated.parquet")
_names = pd.read_parquet(IDX / "disease_names.parquet")
DISEASE_NAME = dict(zip(_names.disease_id, _names.name))
del _names
ENSEMBL_SYMBOL = {**dict(zip(CURATED.ensembl, CURATED.symbol)),  # (OT's own, for ids HGNC lacks, e.g. SHOX's ENSG00000292354)
                  **dict(zip(GENES.ensembl_gene_id.dropna(), GENES.symbol[GENES.ensembl_gene_id.notna()]))}

# Abstracts: uncompressed Arrow IPC, memory-mapped (zero-copy); rows sorted by pmid.
_ipc = pa.ipc.open_file(pa.memory_map(str(IDX / "abstracts.arrow")))
BATCHES = [_ipc.get_batch(i) for i in range(_ipc.num_record_batches)]
STARTS = np.cumsum([0] + [b.num_rows for b in BATCHES])
PMIDS = np.concatenate([b.column("pmid").to_numpy() for b in BATCHES])
YEARS = np.concatenate([b.column("year").fill_null(0).to_numpy(zero_copy_only=False) for b in BATCHES])


def take(rows: np.ndarray, cols: list[str]) -> pa.Table:
    """Abstract rows by row number, in the given order. Takes batch by batch: a whole-table
    take would concatenate 45 GB of strings and overflow Arrow's 32-bit string offsets. Text
    columns come back as large_string (64-bit offsets): a big disease's abstracts pass 2 GB."""
    rows = np.asarray(rows, dtype=np.int64)
    batch = np.searchsorted(STARTS, rows, side="right") - 1
    order = np.argsort(batch, kind="stable")
    schema = BATCHES[0].select(cols).schema
    schema = pa.schema([f.with_type(pa.large_string()) if f.type == pa.string() else f for f in schema])
    parts = list(POOL.map(lambda b: BATCHES[b].select(cols).take(pa.array(rows[order][batch[order] == b] - STARTS[b])).cast(schema),
                          np.unique(batch)))
    back = np.empty_like(order)
    back[order] = np.arange(len(order))
    table = pa.Table.from_batches(parts, schema=schema)
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

# Disease lookup maps (upper-cased keys -> DISEASES row numbers): id (also the PubTator ids it
# covers), name, Open Targets exact synonyms, and the mentions PubTator3 most uses for it ('CF', 'T2D').
D_ID, D_PUBTATOR, D_NAME, D_SYNONYM, D_MENTION = {}, {}, {}, {}, {}
for d in DISEASES.itertuples():
    D_ID[d.disease_id.upper()] = [d.row]
    for p in d.pubtator:
        D_PUBTATOR.setdefault(p.upper(), []).append(d.row)
    D_NAME.setdefault(d.name.upper(), []).append(d.row)
    for s in d.synonyms:
        D_SYNONYM.setdefault(s.upper(), []).append(d.row)
    for m in d.mentions:
        D_MENTION.setdefault(m.upper(), []).append(d.row)
D_TEXT = (DISEASES.name + " | " + DISEASES.synonyms.str.join(" | ") + " | " + DISEASES.mentions.str.join(" | ")).str.upper()


def is_gene(subject: str) -> bool:
    return subject.upper().startswith("HGNC:")


def gene_info(row: int) -> dict:
    g = GENES.iloc[row].to_dict()  # (Series.name would be the row label, not the gene name)
    return {
        "hgnc_id": g["hgnc_id"], "symbol": g["symbol"], "name": g["name"],
        "aliases": _split(g["alias_symbol"]), "prev": _split(g["prev_symbol"]),
        "entrez": None if pd.isna(g["entrez_id"]) else int(g["entrez_id"]),
        "ensembl": g["ensembl_gene_id"] if isinstance(g["ensembl_gene_id"], str) else None,
        "location": _str(g["location"]), "locus_group": _str(g["locus_group"]), "n_papers": int(g["n_papers"]),
        "id": g["hgnc_id"], "kind": "gene", "label": g["symbol"],
    }


def disease_info(row: int) -> dict:
    d = DISEASES.iloc[row]
    return {"disease_id": d.disease_id, "name": d["name"], "synonyms": list(d.synonyms), "pubtator": list(d.pubtator),
            "n_papers": int(d.n_papers), "id": d.disease_id, "kind": "disease", "label": d["name"]}


def _str(v) -> str | None:
    return v if isinstance(v, str) else None


def info(subject: str) -> dict:
    """A gene's or a disease's record; 404 if unknown."""
    rows = (IDS if is_gene(subject) else D_ID).get(subject.upper())
    if not rows:
        raise HTTPException(404, f"unknown {'gene' if is_gene(subject) else 'disease'} {subject}")
    return (gene_info if is_gene(subject) else disease_info)(rows[0])


def facet_name(fid: str) -> str:
    """Name of a facet item: a disease id's name, or an Ensembl gene id's symbol."""
    return DISEASE_NAME.get(fid) or ENSEMBL_SYMBOL.get(fid) or fid


# ---------------------------------------------------------------- subject search
def _tag(maps_whys, describe, n_papers, skip=(), limit=None, by_reason=False):
    """Records of the rows in the maps, each with why it matched; most-cited first within a reason."""
    hits = {}  # row -> (reason rank, reason)
    for rank, (m, why) in enumerate(maps_whys):
        for r in (m if isinstance(m, list) else []):
            if r not in skip:
                hits.setdefault(r, (rank if by_reason else 0, why))
    rows = sorted(hits, key=lambda r: (hits[r][0], -n_papers[r]))
    return [{**describe(r), "why": hits[r][1]} for r in rows[:limit]], len(rows)


def resolve(q: str) -> dict:
    """Exact symbol wins; else alias/previous symbol; else full name; else prefix/word matches.
    Prefix/word matches are flagged partial: the page shows them as a pick list, never auto-loads."""
    u = q.strip().upper()
    if not u:
        return {"matches": []}
    n = GENES.n_papers.to_numpy()
    tag = lambda maps_whys, **kw: _tag(maps_whys, gene_info, n, **kw)
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
    found, n_found = tag([([r for s, rs in SYMBOL.items() if s.startswith(u) for r in rs], "symbol starts with"),
                          ([r for s, rs in ALIAS.items() if s.startswith(u) for r in rs], "alias starts with"),
                          (np.flatnonzero(words_start(NAME_TEXT, u)).tolist(), "name contains")], limit=40, by_reason=True)
    return {"matches": found, "n_more": max(0, n_found - 40), "partial": True}


def resolve_disease(q: str) -> dict:
    """Id or exact name wins; else an Open Targets synonym or a PubTator mention (several = a pick
    list); else diseases whose name, synonyms or mentions have words starting with each query word."""
    u = q.strip().upper()
    if not u:
        return {"matches": []}
    n = DISEASES.n_papers.to_numpy()
    tag = lambda maps_whys, **kw: _tag(maps_whys, disease_info, n, **kw)
    if rows := D_ID.get(u) or D_PUBTATOR.get(u) or D_NAME.get(u):
        return {"matches": tag([(rows, "name")])[0],
                "also": tag([(D_SYNONYM.get(u), "synonym"), (D_MENTION.get(u), "PubTator3 mention")], skip=rows)[0]}
    found, _ = tag([(D_SYNONYM.get(u), "synonym"), (D_MENTION.get(u), "PubTator3 mention")])
    if found:
        return {"matches": found}
    found, n_found = tag([(np.flatnonzero(words_start(DISEASES.name.str.upper(), u)).tolist(), "name contains"),
                          (np.flatnonzero(words_start(D_TEXT, u)).tolist(), "synonym contains")], limit=40, by_reason=True)
    return {"matches": found, "n_more": max(0, n_found - 40), "partial": True}


def words_start(text: pd.Series, q: str) -> np.ndarray:
    """Rows where every word of q starts a word of the text."""
    mask = np.ones(len(text), bool)
    for w in q.split():
        mask &= text.str.contains(r"(?<!\w)" + re.escape(w), regex=True).to_numpy()
    return mask


# ---------------------------------------------------------------- paper sets
def rows_for(pmids: np.ndarray) -> np.ndarray:
    """Row numbers in the abstracts file for each pmid (-1 if not in the local PubMed copy)."""
    idx = np.searchsorted(PMIDS, pmids).clip(0, len(PMIDS) - 1)
    return np.where(PMIDS[idx] == pmids, idx, -1)


@lru_cache(maxsize=128)
def subject_tags(subject: str) -> tuple[dict[int, str], pd.DataFrame]:
    """({pmid: the subject's mentions} for PubTator-tagged papers, Open Targets curated links) for a
    gene or disease. Curated links: pmid, fid and fname (the other side: a disease for a gene, an
    Ensembl gene id for a disease, and its name), sources."""
    s = info(subject)
    if s["kind"] == "gene":
        tagged = db().execute(
            f"SELECT pmid, mentions FROM '{IDX}/gene_pmids.parquet' WHERE ncbi = ?", [s["entrez"] or -1]).fetchnumpy()
        cur = CURATED[CURATED.ensembl == s["ensembl"]] if s["ensembl"] else CURATED.iloc[:0]
        cur = cur.rename(columns={"disease_id": "fid", "disease_name": "fname"})
    else:
        tagged = db().execute(
            f"""SELECT pmid, string_agg(mentions, '|') AS mentions FROM '{IDX}/disease_pmids.parquet'
                WHERE list_contains(?, disease_id) GROUP BY pmid""", [s["pubtator"]]).fetchnumpy()
        cur = CURATED[CURATED.disease_id == s["id"]]
        cur = cur.assign(fid=cur.ensembl, fname=[ENSEMBL_SYMBOL.get(e, g) for e, g in zip(cur.ensembl, cur.symbol)])
    return dict(zip(tagged["pmid"].tolist(), tagged["mentions"].tolist())), cur[["pmid", "fid", "fname", "sources"]]


@lru_cache(maxsize=128)
def epmc_links(subject: str) -> pd.DataFrame:
    """Europe PMC text-mined gene-disease links (Open Targets evidence) of a gene or disease: pmid,
    fid (the other side, as in subject_tags)."""
    s = info(subject)
    if s["kind"] == "gene":
        return db().execute(f"SELECT pmid, disease_id AS fid FROM '{IDX}/epmc.parquet' WHERE ensembl = ?",
                            [s["ensembl"] or ""]).df()
    return db().execute(f"SELECT pmid, ensembl AS fid FROM '{IDX}/epmc.parquet' WHERE disease_id = ?", [s["id"]]).df()


@lru_cache(maxsize=128)
def epmc_counts(subject: str) -> pd.Series:
    """Papers per text-mined facet item, most first (ties by id)."""
    n = epmc_links(subject).groupby("fid").size()
    return n.iloc[np.lexsort((n.index.to_numpy(), -n.to_numpy()))]


@lru_cache(maxsize=128)
def paper_set(subject: str, ot: str = "", ep: str = "") -> pd.DataFrame:
    """The subject's papers: PubTator-tagged plus OT-curated. With a facet, only the papers that
    Open Targets curators (ot) and Europe PMC text mining (ep) link to the subject and that item
    ('any' = any item); with both, the papers in both."""
    tagged, cur = subject_tags(subject)
    sets = []
    if ot:
        sets.append(cur.pmid.to_numpy() if ot == "any" else cur.loc[cur.fid == ot, "pmid"].to_numpy())
    if ep:
        e = epmc_links(subject)
        sets.append(e.pmid.to_numpy() if ep == "any" else e.loc[e.fid == ep, "pmid"].to_numpy())
    if sets:
        pmids = reduce(np.intersect1d, sets)
    else:
        pmids = np.union1d(np.fromiter(tagged, np.int64, len(tagged)), cur.pmid.to_numpy())
    df = pd.DataFrame({"pmid": np.unique(np.asarray(pmids, dtype=np.int64))})
    df["row"] = rows_for(df.pmid.to_numpy())
    df["year"] = np.where(df.row >= 0, YEARS[df.row.clip(0)], 0)
    df["undated"] = df.year == 0  # listed last in either year order
    return df.sort_values(["undated", "year", "pmid"], ascending=[True, False, False], ignore_index=True)


def subject_names(s: dict) -> list[str]:
    """Names looked for in every paper of a gene: its symbol, HGNC aliases and previous symbols (a
    disease: its name and Open Targets synonyms). Of 3 letters or more but the symbol: shorter ones hit
    other things (CF, CFTR's old symbol, is everywhere in cystic fibrosis papers)."""
    more = [*s["aliases"], *s["prev"]] if s["kind"] == "gene" else s["synonyms"]
    return [w for w in dict.fromkeys([s["label"], *(m for m in more if len(m) >= 3)]) if len(w) > 1]


def subject_words(s: dict, mentions) -> list[str]:
    """The other names to look for in one paper: its PubTator mentions of the subject. For a gene,
    lower-case digit-free spellings of the symbol are left out: PubTator sometimes tags plain words
    ('was' for WAS). The page highlights subject_names and these."""
    words = _split(mentions)
    if s["kind"] == "gene":
        words = [m for m in words if not (m == m.lower() and not re.search(r"\d", m) and m.upper() == s["label"].upper())]
    return [w for w in dict.fromkeys(words) if len(w) > 1]


def any_case(word: str) -> bool:
    """A long mixed-case name matches in any case; others only in their own (WAS must not match 'was')."""
    return len(word) > 5 and word != word.upper()


def names_re(words) -> str:
    """RE2 pattern: any of the words, whole (no letter or digit either side), each in its case rule."""
    alts = "|".join(f"(?i:{re.escape(w)})" if any_case(w) else re.escape(w) for w in sorted(words, key=len, reverse=True))
    return rf"(?:^|[^\pL\pN])(?:{alts})(?:[^\pL\pN]|$)"


def has_word(text: str, low: str, word: str) -> bool:
    """Whether the text (low: lower-cased) has the word whole: no letter or digit either side."""
    if any_case(word):
        text, word = low, word.lower()
    i = text.find(word)
    while i >= 0:
        j = i + len(word)
        if not (i and text[i - 1].isalnum()) and not (j < len(text) and text[j].isalnum()):
            return True
        i = text.find(word, i + 1)
    return False


@lru_cache(maxsize=32)
def named_pmids(subject: str) -> np.ndarray:
    """The subject's papers (also the text-mined ones) whose title or abstract names it, sorted:
    subject_names in all of them (one RE2 scan), each paper's own PubTator mentions (subject_words)
    in that paper. Mentions go commonest first, in the papers not yet matched. While a mention is in
    many papers: one RE2 scan of all the titles and abstracts (32 threads). The long tail of rare
    spellings (a disease has thousands) in Python: RE2 takes ~6 ms to compile a pattern with Unicode
    classes. (Title and abstract are matched apart: joining them took 4 s for 875k papers.)"""
    s = info(subject)
    tagged, _ = subject_tags(subject)
    pm = np.union1d(paper_set(subject).pmid.to_numpy(), epmc_links(subject).pmid.to_numpy())
    pm = pm[rows_for(pm) >= 0]
    if not len(pm):
        return pm
    t = take(rows_for(pm), ["title", "abstract"])
    title, abstract = t["title"].fill_null("").combine_chunks(), t["abstract"].fill_null("").combine_chunks()
    pairs = pd.DataFrame([(p, w) for p, m in tagged.items() for w in subject_words(s, m)], columns=["pmid", "word"])
    pairs["i"] = np.searchsorted(pm, pairs.pmid.to_numpy())
    names = subject_names(s)
    pairs = pairs[(pairs.i < len(pm)) & (pm[pairs.i.clip(0, len(pm) - 1)] == pairs.pmid) & ~pairs.word.isin(names)]
    code, words = pd.factorize(pairs.word)
    order = np.argsort(code, kind="stable")
    by_word = np.split(pairs.i.to_numpy()[order], np.cumsum(np.bincount(code, minlength=len(words)))[:-1])
    pattern = names_re(names)
    named = matches(title, pattern) | matches(abstract, pattern)
    groups = sorted(zip(words, by_word), key=lambda x: -len(x[1]))
    for n, (word, idx) in enumerate(groups):
        idx = idx[~named[idx]]
        if len(idx) < 500:
            break
        pattern = names_re([word])
        named[idx[(matches(title, pattern) | matches(abstract, pattern))[idx]]] = True
    else:
        return pm[named]
    rest = [(word, idx[~named[idx]]) for word, idx in groups[n:]]
    todo = np.unique(np.concatenate([idx for _, idx in rest]))
    texts = {i: f"{a}\n{b}" for i, a, b in zip(todo.tolist(), title.take(todo).to_pylist(), abstract.take(todo).to_pylist())}
    lows = {i: x.lower() for i, x in texts.items()}
    for word, idx in rest:
        named[[i for i in idx.tolist() if not named[i] and has_word(texts[i], lows[i], word)]] = True
    return pm[named]


def matches(text: pa.Array, pattern: str) -> np.ndarray:
    """Whether each string matches the RE2 pattern, over 32 threads."""
    step = max(1000, -(-len(text) // 32))
    parts = POOL.map(lambda i: pc.match_substring_regex(text.slice(i, step), pattern).to_numpy(zero_copy_only=False),
                     range(0, len(text), step))
    return np.concatenate([np.zeros(0, bool), *parts])


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
def corpus(subject: str, ot: str, ep: str) -> tuple[pa.ChunkedArray, np.ndarray]:
    """Lower-cased title + abstract of the papers in paper_set that have text, and their positions."""
    df = paper_set(subject, ot, ep)
    have = np.flatnonzero(df.row.to_numpy() >= 0)
    t = take(df.row.to_numpy()[have], ["title", "abstract"])
    step = max(1, -(-len(t) // 32))

    def prep(i):
        s = t.slice(i, step)
        joined = pc.binary_join_element_wise(s["title"].fill_null(""), s["abstract"].fill_null(""), pa.scalar(" \n", pa.large_string()))
        return pc.utf8_lower(joined).combine_chunks()

    return pa.chunked_array(list(POOL.map(prep, range(0, len(t), step))), type=pa.large_string()), have


@lru_cache(maxsize=64)
def bm25(subject: str, ot: str, ep: str, term: str) -> tuple[np.ndarray, np.ndarray]:
    """BM25 (Lucene IDF, k1=1.5, b=0.75 as bm25s) over the subject's papers, built on the fly,
    plus how many of the query's stems each paper contains.

    Vectorised in Arrow: for each query stem, find every word form in these papers that has
    that stem, then count those forms per paper with one regex. Exact term frequencies; doc
    length is measured in characters. ~20x faster than tokenising with bm25s (8.5 s for CFTR).
    """
    df = paper_set(subject, ot, ep)
    text, have = corpus(subject, ot, ep)
    scores, hits = np.zeros(len(df)), np.zeros(len(df), np.int64)
    if not len(have):
        return scores, hits
    dl = pc.utf8_length(text).to_numpy()
    norm = K1 * (1 - B + B * dl / dl.mean())
    for stem, forms in stem_forms(subject, ot, ep, term).items():
        if not forms:
            continue
        tf = count(text, words_re(forms))
        n_with = np.count_nonzero(tf)
        idf = np.log(1 + (len(have) - n_with + 0.5) / (n_with + 0.5))
        scores[have] += idf * tf * (K1 + 1) / (tf + norm)
        hits[have] += tf > 0
    return scores, hits


@lru_cache(maxsize=64)
def stem_forms(subject: str, ot: str, ep: str, term: str) -> dict[str, set[str]]:
    """Query stem -> word forms in these papers with that stem (e.g. fibrosi -> fibrosis, fibroses)."""
    text, _ = corpus(subject, ot, ep)
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
def curated_pmids(subject: str) -> np.ndarray:
    return np.unique(subject_tags(subject)[1].pmid.to_numpy())


def scope(subject, term, ot, ep, ymin, ymax, match, rank, named=False) -> tuple[pd.DataFrame, dict]:
    """The current list before ranking: the subject's papers (or those the facets pick, see
    paper_set), the year filters, only those naming the subject in the title or abstract if asked,
    and for keyword ranking only the papers with the words: all of them by default (any of them if
    asked, or if no paper has all). Semantic and hybrid ranking keep every paper, ordered by how
    close it is."""
    df = paper_set(subject, ot, ep)
    keep = np.ones(len(df), bool)
    year = df.year.to_numpy()
    if ymin:
        keep &= year >= ymin
    if ymax:
        keep &= (year <= ymax) & (year > 0)
    if named:
        keep &= np.isin(df.pmid.to_numpy(), named_pmids(subject))
    info = {"n_scope": int(keep.sum())}
    df = df.assign(bm25=np.nan)
    if term := term.strip().lower():
        scores, hits = bm25(subject, ot, ep, term)
        n_words = len(stem_forms(subject, ot, ep, term))
        has_all, has_any = (hits == n_words) & (n_words > 0) & keep, (scores > 0) & keep
        info.update(n_words=n_words, n_all=int(has_all.sum()), n_any=int(has_any.sum()))
        df["bm25"] = scores
        if rank == "keyword":
            info["match"] = "any" if match == "any" or not info["n_all"] else "all"
            keep = has_all if info["match"] == "all" else has_any
    return df[keep], info


def texts_for(pmids: np.ndarray) -> list[str]:
    """'title\n\nabstract' of papers in the local PubMed copy (the text the embeddings are of)."""
    t = take(rows_for(np.asarray(pmids)), ["title", "abstract"])
    return [semantic.doc_text(a, b) for a, b in zip(t["title"].to_pylist(), t["abstract"].to_pylist())]


def similarity(subject: str, df: pd.DataFrame, term: str) -> tuple[np.ndarray, dict]:
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
    vecs, ok = semantic.vectors(subject, df.pmid.to_numpy()[have], texts_for)
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


def select(subject, term, ot, ep, ymin, ymax, sort, match="all", rank="semantic", named=False) -> tuple[pd.DataFrame, dict]:
    """The filtered, ranked paper list and how the words matched (see scope). Ranking: keyword =
    BM25; semantic = cosine of MedEmbed vectors; hybrid = reciprocal-rank fusion of the two (k=60).
    Raises Pending while semantic vectors are being computed."""
    df, info = scope(subject, term, ot, ep, ymin, ymax, match, rank, named)
    df = df.assign(score=df.bm25, sim=np.nan, kw_rank=np.nan, sem_rank=np.nan)
    if (term := term.strip().lower()) and rank != "keyword":
        sim, sem = similarity(subject, df, term)
        info.update(sem)
        kw, mn = ranks(df.bm25.to_numpy(), df.bm25.to_numpy() > 0), ranks(sim, ~np.isnan(sim))
        df = df.assign(sim=sim, kw_rank=kw, sem_rank=mn)
        df["score"] = np.nan_to_num(1 / (RRF_K + kw)) + np.nan_to_num(1 / (RRF_K + mn)) if rank == "hybrid" else sim
    if sort == "oldest":
        df = df.sort_values(["undated", "year", "pmid"], ascending=[True, True, True])
    elif sort == "relevance" and term:
        df = df.sort_values(["score", "year"], ascending=False, na_position="last")
    return df, info  # else already newest first


def describe(df: pd.DataFrame, subject: str, term: str) -> list[dict]:
    """Full display records for a (small) slice of the paper list."""
    s = info(subject)
    mentions, cur = subject_tags(subject)
    cur = cur[cur.pmid.isin(df.pmid)]
    cur_by = {p: g for p, g in cur.groupby("pmid")}
    ep = epmc_links(subject)
    ep_by = {p: g.fid.tolist() for p, g in ep[ep.pmid.isin(df.pmid)].groupby("pmid")}
    ep_n = epmc_counts(subject)
    own = set(s.get("pubtator", []))  # a disease's own PubTator tags are on every paper: not shown
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
            "mentions": _split(mentions.get(pmid)),
            "term_words": sorted({w for w in set(WORD.findall(text.lower())) if STEMMER.stemWord(w) in stems}),
            "curated": [] if c is None else [
                {"id": f, "name": n, "sources": sorted({x for v in g.sources for x in v})}
                for (f, n), g in c.groupby(["fid", "fname"])],
            "epmc": [{"id": f, "name": facet_name(f)} for f in sorted(ep_by.get(pmid, []), key=lambda f: (-ep_n.get(f, 0), f))],
            "diseases": disease_tags([t for t in a["diseases"] or [] if t["id"] not in own], text),
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


@app.get("/about")
def about():
    return FileResponse(HERE / "about.html")


@app.get("/api/resolve")
def api_resolve(q: str, kind: str = "gene"):
    t0 = time.perf_counter()
    res = resolve_disease(q) if kind == "disease" else resolve(q)
    res["elapsed_ms"] = round(1000 * (time.perf_counter() - t0))
    return res


@app.get("/api/subject/{subject}")
def api_subject(subject: str):
    """A gene or disease: its record, paper counts, and its facets (Open Targets curated and the
    top Europe PMC text-mined items: diseases for a gene, genes for a disease)."""
    t0 = time.perf_counter()
    s = info(subject)
    tagged, cur = subject_tags(subject)
    curated = [{"id": f, "name": n, "n_papers": int(x.pmid.nunique()), "sources": sorted({v for vs in x.sources for v in vs})}
               for (f, n), x in cur.groupby(["fid", "fname"])]
    curated.sort(key=lambda r: (-r["n_papers"], r["name"]))
    ep = epmc_links(subject)
    df = paper_set(subject)
    return {
        **s, "names": subject_names(s), "n_tagged": len(tagged), "n_curated_papers": int(cur.pmid.nunique()),
        "n_set": len(df), "n_local": int((df.row >= 0).sum()),
        "year_min": int(df.year[df.year > 0].min()) if (df.year > 0).any() else None,
        "year_max": int(df.year.max()) if len(df) else None,
        "curated": curated, "n_epmc_papers": int(ep.pmid.nunique()), "n_epmc_items": int(ep.fid.nunique()),
        "epmc": [{"id": f, "name": facet_name(f), "n_papers": int(n)} for f, n in epmc_counts(subject).head(EPMC_TOP).items()],
        "elapsed_ms": round(1000 * (time.perf_counter() - t0)),
    }


@app.get("/api/facets")
def api_facets(subject: str, ot: str = "", ep: str = "", term: str = "", ymin: int = 0, ymax: int = 0,
               match: str = "all", rank: str = "semantic", named: int = 0):
    """Papers per item in each box within the other filters (the other box's pick, the years and the
    named filter; items with none are left out): what clicking a row gives. Clicking an item clears
    the keyword words, so they count only for 'any' (that box's 'Any' row, which keeps them). Each
    box counts within its own 'any' set, as a click on one of its rows does. Plus how many papers
    name the subject within all the other filters, and the names."""
    _, cur = subject_tags(subject)

    def counts(links: pd.DataFrame, box_ot: str, box_ep: str) -> dict:
        def within(t: str) -> pd.DataFrame:
            return links[links.pmid.isin(scope(subject, t, box_ot, box_ep, ymin, ymax, match, rank, named)[0].pmid)]
        rows = within("")
        return {**rows.groupby("fid").pmid.nunique().astype(int).to_dict(),
                "any": int((within(term) if term.strip() else rows).pmid.nunique())}

    out = {"ot": counts(cur[["pmid", "fid"]], "any", ep), "ep": counts(epmc_links(subject), ot, "any")}
    out["names"] = {i: facet_name(i) for k in ("ot", "ep") for i in out[k] if i != "any"}
    out["named"] = int(np.isin(scope(subject, term, ot, ep, ymin, ymax, match, rank)[0].pmid, named_pmids(subject)).sum())
    return out


@app.get("/api/papers")
def api_papers(subject: str, term: str = "", ot: str = "", ep: str = "", ymin: int = 0, ymax: int = 0,
               sort: str = "", match: str = "all", rank: str = "semantic", named: int = 0, page: int = 1):
    t0 = time.perf_counter()
    sort = sort or ("relevance" if term.strip() else "newest")
    try:
        df, info = select(subject, term, ot, ep, ymin, ymax, sort, match, rank, bool(named))
    except Pending as p:
        return {"pending": p.job.status()}
    n_pages = max(1, -(-len(df) // PAGE))
    page = min(max(1, page), n_pages)
    papers = describe(df.iloc[(page - 1) * PAGE: page * PAGE], subject, term)
    return {"n_matched": len(df), **info, "rank": rank, "page": page, "n_pages": n_pages, "sort": sort,
            "papers": papers, "names": facet_names(ot, ep), "elapsed_ms": round(1000 * (time.perf_counter() - t0))}


def facet_names(*ids: str) -> dict[str, str]:
    """Names of the items picked in the facets (the page may only have their ids, from a URL)."""
    return {i: facet_name(i) for i in ids if i and i != "any"}


@app.get("/api/paper")
def api_paper(subject: str, pmid: int, term: str = ""):
    """One paper as in the list (the map's click-through)."""
    df = pd.DataFrame({"pmid": [pmid], "row": rows_for(np.array([pmid])), "bm25": np.nan, "sim": np.nan,
                       "kw_rank": np.nan, "sem_rank": np.nan})
    return describe(df, subject, term)[0]


@app.get("/api/map")
def api_map(subject: str, term: str = "", ymin: int = 0, ymax: int = 0, match: str = "all", rank: str = "semantic",
            topic: str = ""):
    """Literature map of the current list without its facets (its papers with an abstract; a fixed
    sample of MAP_CAP beyond that): the page greys out the papers the facets leave out (see
    api_map_facets) instead of making a new map. topic = a drill-in path ('3', '3.2'): a new map of
    only topic 3's papers, then of its sub-topic 2. Returns {"pending": progress, "n": papers} until
    it is ready."""
    df, _ = scope(subject, term, "", "", ymin, ymax, match, rank)
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
            m, _ = get_map(subject, pm, within)
        except Pending as p:
            return {"pending": p.job.status(), "n": len(pm)}
        cl = next((c for c in m["clusters"] if str(c["id"]) == step), None)
        if cl is None:
            raise HTTPException(404, f"no topic {step} in this map")
        pm = np.asarray(m["pmid"], np.int64)[np.asarray(m["c"]) == cl["id"]]
        within = cl["name"]
        path.append({"id": cl["id"], "name": cl["name"], "n": len(pm)})
    if len(pm) < MIN_MAP:
        return {**about, "too_few": True, "path": path}
    try:
        m, key = get_map(subject, pm, within)
    except Pending as p:
        return {"pending": p.job.status(), "n": len(pm)}
    return {**m, **about, "key": key, "path": path}


@app.get("/api/map_facets")
def api_map_facets(subject: str, key: str):
    """For each facet item, the map's papers (positions in its pmid list) it links: Open Targets
    curated (ot) and Europe PMC text-mined (ep); plus those naming the subject in the title or
    abstract (named), and the items' names. The page greys out the other papers when a facet is
    picked or hovered over."""
    if not re.fullmatch(r"[0-9a-f]{20}", key) or (m := cached_map(key)) is None:
        raise HTTPException(404, "That map is gone: open the map again.")
    pm = np.asarray(m["pmid"], np.int64)  # (sorted)

    def at(links: pd.DataFrame) -> dict[str, list[int]]:
        links = links[links.pmid.isin(pm)].drop_duplicates(["pmid", "fid"])
        return {f: np.searchsorted(pm, g.pmid.to_numpy()).tolist() for f, g in links.groupby("fid")}

    out = {"ot": at(subject_tags(subject)[1]), "ep": at(epmc_links(subject)),
           "named": np.flatnonzero(np.isin(pm, named_pmids(subject))).tolist()}
    out["names"] = {f: facet_name(f) for k in ("ot", "ep") for f in out[k]}
    return out


MIN_MAP = 20  # fewest papers for a map


def map_subject(subject: str) -> tuple[str, str, set[str]]:
    """For naming a map's topics: what the papers are about, what the names must leave out, and the
    words left out of the topic terms (on every paper, so they don't tell topics apart)."""
    s = info(subject)
    if s["kind"] == "gene":
        return f"the human gene {s['symbol']}", "the gene symbol", {s["symbol"], *s["aliases"], *s["prev"]}
    text = " ".join([s["name"], *DISEASES.mentions.iat[D_ID[subject.upper()][0]]]).lower()
    return f"the disease {s['name']}", "the disease name", {*WORD.findall(text), *litmap.TOKEN.findall(text)}  # (TOKEN: 'coffin-lowry')


def get_map(subject: str, pm: np.ndarray, within: str = "") -> tuple[dict, str]:
    """The map of these papers (cached) and its key; raises Pending while it is made. within = the
    name of the topic they all come from (a drill-in), so their sub-topics are named as such."""
    key = semantic.digest(subject, pm, semantic.source(), MAP_VERSION,  # subject: curated rings, stop words, names
                          *(["drill"] if within else []))  # not the name: an AI rename must not lose the map
    if (m := cached_map(key)) is None:
        j = semantic.job(("map", key), lambda jb: make_map(subject, pm, key, jb, within))
        if not j.finished or j.error:
            raise Pending(j)
        m = j.result
    elif m["named_by"] == "terms" and litmap.name_clusters(*map_subject(subject)[:2], m["clusters"], within):
        m["named_by"] = "llm"  # the LLM answers now (e.g. credit added): keep its names
        save_map(key, m)
    return m, key


@app.get("/api/axes")
def api_axes(subject: str, key: str, a: str, b: str):
    """Map Explorer: the papers of a map placed by how close each is in meaning to two things (the
    axes): words, embedded as a query, or a topic's centre ('topic:3', the mean of its papers'
    vectors). Closeness = cosine similarity, i.e. the paper vectors projected onto each axis."""
    if not re.fullmatch(r"[0-9a-f]{20}", key) or (m := cached_map(key)) is None:
        raise HTTPException(404, "That map is gone: open the map again.")
    pm = np.asarray(m["pmid"], np.int64)
    try:
        vecs, ok = semantic.vectors(subject, pm, texts_for)
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


def make_map(subject: str, pm: np.ndarray, key: str, job: semantic.Job, within: str = "") -> dict:
    t0 = time.perf_counter()
    vecs, ok = job.wait_for(lambda: semantic.vectors(subject, pm, texts_for))
    pm, vecs = pm[ok], vecs[ok]
    t = take(rows_for(pm), ["title", "abstract", "year"])
    titles, abstracts = t["title"].to_pylist(), t["abstract"].to_pylist()
    about, avoid, skip = map_subject(subject)
    m = litmap.build(vecs, [f"{a or ''}\n{b or ''}" for a, b in zip(titles, abstracts)], titles,
                     about, avoid, skip, job, within)
    m.update(pmid=pm.tolist(), year=t["year"].fill_null(0).to_pylist(),
             title=[(x or "")[:220] for x in titles], curated=np.isin(pm, curated_pmids(subject)).astype(int).tolist(),
             source=semantic.source(), seconds=round(time.perf_counter() - t0, 1))
    save_map(key, m)
    return m


def save_map(key: str, m: dict):
    MAPS.mkdir(parents=True, exist_ok=True)
    tmp = MAPS / f"{key}.{os.getpid()}.{threading.get_ident()}.tmp"
    tmp.write_text(json.dumps(m))
    os.replace(tmp, MAPS / f"{key}.json")


@app.get("/api/papers.csv")
def api_papers_csv(subject: str, term: str = "", ot: str = "", ep: str = "", ymin: int = 0, ymax: int = 0,
                   sort: str = "", match: str = "all", rank: str = "semantic", named: int = 0):
    sort = sort or ("relevance" if term.strip() else "newest")
    try:
        df, _ = select(subject, term, ot, ep, ymin, ymax, sort, match, rank, bool(named))
    except Pending:
        raise HTTPException(409, "The semantic ranking is still being worked out. Wait for the list, then download.")
    s = info(subject)
    _, cur = subject_tags(subject)
    have = pa.array(df.row.to_numpy() >= 0)
    t = take(df.row.to_numpy().clip(0), ["title", "journal", "diseases"]).combine_chunks()
    # PubTator tag names per paper, all in Arrow: name each distinct id once, keep the list
    # offsets, join each list. (A pandas groupby-join took 12 s for TP53's 4.8M tags.)
    tags = t.column("diseases").combine_chunks()
    ids = pc.dictionary_encode(pc.struct_field(tags.values, "id"))
    names = pa.array([DISEASE_NAME.get(i, i) for i in ids.dictionary.to_pylist()], pa.string())
    named_tags = pa.ListArray.from_arrays(tags.offsets, names.take(ids.indices), mask=tags.is_null())
    cur_names = cur.groupby("pmid").fname.agg(lambda v: "; ".join(sorted(set(v))))
    links = epmc_links(subject)
    links = links[links.pmid.isin(df.pmid)]
    ep_names = links.groupby("pmid").fid.agg(lambda v: "; ".join(sorted({facet_name(f) for f in v})))
    cols = {
        "pmid": df.pmid.to_numpy(),
        "year": pa.array(df.year.to_numpy(), mask=df.year.to_numpy() == 0),
        "journal": pc.if_else(have, t["journal"], None), "title": pc.if_else(have, t["title"], None),
        "bm25_score": pa.array(df.bm25.where(df.bm25 > 0).round(3).to_numpy(), from_pandas=True),
    }
    if df.sim.notna().any():
        cols["semantic_similarity"] = pa.array(df.sim.round(4).to_numpy(), from_pandas=True)
    other = "diseases" if s["kind"] == "gene" else "genes"
    cols.update({
        f"curated_{other}": df.pmid.map(cur_names).fillna("").to_numpy(),
        f"textmined_{other}": df.pmid.map(ep_names).fillna("").to_numpy(),
        "pubtator_diseases": pc.if_else(have, pc.binary_join(named_tags, "; ").fill_null(""), ""),
        "url": pc.binary_join_element_wise("https://pubmed.ncbi.nlm.nih.gov/",
                                           pc.cast(pa.array(df.pmid.to_numpy()), pa.string()), "/", ""),
    })
    buf = pa.BufferOutputStream()
    pa_csv.write_csv(pa.table(cols), buf)
    facets = [f"{'curated' if k == 'ot' else 'textmined'} {'' if v == 'any' else facet_name(v)}"
              for k, v in (("ot", ot), ("ep", ep)) if v]
    name = re.sub(r"\W+", "_", " ".join([s["label"], term, *facets]).strip()).strip("_")
    # HTTP headers are Latin-1: an ASCII name ('TGF-β' -> 'TGF'), plus the real one for browsers that read it
    ascii_name = re.sub(r"[^A-Za-z0-9]+", "_", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()).strip("_")
    return Response(buf.getvalue().to_pybytes(), media_type="text/csv", headers={"Content-Disposition":
                    f"attachment; filename=\"{ascii_name or 'papers'}.csv\"; filename*=UTF-8''{urllib.parse.quote(name)}.csv"})


@app.get("/api/about")
@cache
def api_about():
    """The data behind the search, for the About page: sizes from the index, versions from the
    source paths build_index.py read."""
    rows = lambda f: pq.ParquetFile(IDX / f).metadata.num_rows
    full = semantic.full_index()
    llm = litmap._llm_config()
    return {
        "pubmed": {"n": len(PMIDS)},
        "pubtator": {"version": Path(build_index.PUBTATOR).name, "gene_links": int(GENES.n_papers.sum()),  # (HGNC genes only)
                     "disease_links": rows("disease_pmids.parquet"), "n_disease_ids": int(DISEASES.pubtator.explode().nunique())},
        "hgnc": {"version": Path(build_index.HGNC).parent.name, "n": len(GENES), "n_with_papers": int((GENES.n_papers > 0).sum())},
        "opentargets": {"version": Path(build_index.OT).name, "curated_links": len(CURATED),
                        "curated_papers": int(CURATED.pmid.nunique()),
                        "curated_pairs": int(CURATED[["ensembl", "disease_id"]].drop_duplicates().shape[0]),
                        "epmc_links": rows("epmc.parquet")},
        "diseases": {"n": len(DISEASES), "n_pubtator_only": int(DISEASES.disease_id.str.match(r"MESH:|OMIM:").sum())},
        "embeddings": {"model": semantic.MODEL, "revision": semantic.REVISION, "dim": semantic.DIM,
                       "n": None if full is None else len(full.pmids)},
        "topic_names": llm.get("model", "google/gemini-3.1-flash-lite") if llm else None,
        "map": {"cap": MAP_CAP, "max_topics": litmap.MAX_TOPICS, "neighbours": litmap.K},
    }


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
