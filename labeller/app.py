"""Blind relevance labeller for gene-disease abstracts (gold labels for judge calibration).

Serve:  uv run python app.py --items items_demo.parquet --port 8765 --host 0.0.0.0
Export: uv run python app.py --export      # -> export/labels.csv and export/labels.parquet

Blindness: the browser never receives item_id, pmid, round_id, stratum or is_repeat, only opaque
handles that differ between a first look and a repeat.

Repeats: a round with is_repeat items is shown in two parts of equal size. Part 1 holds most of the
fresh abstracts (including the repeat items' first look). Part 2 holds the rest of the fresh
abstracts plus the repeat items again, shuffled, so repeats hide among new abstracts. Part 2 is
placed >= --repeat-gap rounds later and opens --repeat-hours after part 1 is finished (a later
session). Rounds without repeat items are shown whole.
"""

import argparse
import hashlib
import os
import random
import re
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
REQUIRED = ["item_id", "round_id", "gene_symbol", "gene_name", "gene_aliases", "disease_name",
            "disease_synonyms", "pmid", "title", "abstract", "stratum", "is_repeat"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS labels (
    labeller   TEXT NOT NULL,
    item_id    TEXT NOT NULL,
    pass       INTEGER NOT NULL,          -- 0 = first look, 1 = intra-rater repeat
    round_id   TEXT NOT NULL,
    pmid       TEXT NOT NULL,
    grade      INTEGER CHECK (grade IN (0, 1, 2)),
    unsure     INTEGER NOT NULL,          -- 1 => grade IS NULL
    flag       INTEGER NOT NULL,
    note       TEXT NOT NULL,
    seconds    REAL NOT NULL,             -- active time on this item (capped per visit), summed
    n_changes  INTEGER NOT NULL DEFAULT 0,
    first_at   TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (labeller, item_id, pass),
    CHECK ((grade IS NULL) = (unsure = 1))
);
CREATE TABLE IF NOT EXISTS log (          -- append-only: every submission, for audit
    at TEXT NOT NULL, labeller TEXT NOT NULL, item_id TEXT NOT NULL, pass INTEGER NOT NULL,
    grade INTEGER, unsure INTEGER, flag INTEGER, note TEXT, seconds REAL
);
"""

UPSERT = """
INSERT INTO labels (labeller, item_id, pass, round_id, pmid, grade, unsure, flag, note, seconds,
                    first_at, updated_at)
VALUES (:labeller, :item_id, :pass, :round_id, :pmid, :grade, :unsure, :flag, :note, :seconds,
        :at, :at)
ON CONFLICT (labeller, item_id, pass) DO UPDATE SET
    n_changes  = n_changes + (grade IS NOT excluded.grade),
    grade      = excluded.grade,
    unsure     = excluded.unsure,
    flag       = excluded.flag,
    note       = excluded.note,
    seconds    = seconds + excluded.seconds,
    updated_at = excluded.updated_at
"""


def now() -> datetime:
    return datetime.now(timezone.utc)


def seeded(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest(), 16))


def handle(*parts) -> str:
    """Opaque id sent to the browser in place of item/round ids."""
    return hashlib.sha256(("labeller\x1f" + "\x1f".join(map(str, parts))).encode()).hexdigest()[:12]


def as_list(v) -> list[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, str):
        v = v.split("|")
    return list(dict.fromkeys(s.strip() for s in v if s and s.strip()))


# --- highlighting ---------------------------------------------------------------------------
# Kinds: 'g' gene, 'd' disease (name or synonym), 'dp' part of a disease name (a broader or
# related name, shown lighter so it does not suggest "same disease").

GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "kappa": "κ"}
STOP = {"a", "an", "the", "of", "and", "or", "and/or", "with", "to", "in", "due", "for", "by", "on",
        "type", "form", "onset", "associated", "related", "induced"}
GENERIC = {"disease", "diseases", "disorder", "disorders", "syndrome", "syndromes", "type", "form",
           "deficiency", "cancer", "carcinoma", "tumor", "tumour", "neoplasm", "infection", "defect",
           "abnormality", "familial", "hereditary", "congenital", "inherited", "autosomal", "dominant",
           "recessive", "x-linked", "early", "late", "onset", "juvenile", "adult", "childhood", "primary",
           "chronic", "acute", "severe", "mild", "generalized", "generalised", "localized", "isolated",
           "progressive", "syndromic", "non-syndromic", "nonsyndromic", "rare"}
TRAIL = re.compile(r"\s+(?:(?i:type|form|group)\s+)?(?:\d+[A-Za-z]?|[IVX]+)$")  # '... type 2A', '... 5'
ABBR = re.compile(r"\s*\(([A-Z][A-Za-z0-9αβγδκ-]{1,11})\)")                    # '... fibrosis (IPF)'
PRIORITY = {"d": 0, "dp": 1, "g": 2}  # on equal spans, disease wins (gene aliases like ALS, CF)


def sym_pat(s: str) -> str:
    """Symbol, case-sensitive, optional hyphen before a trailing number (SOD-1, IL-6)."""
    return re.sub(r"(?<=[A-Za-z])(?=\d)", "-?", re.escape(s))


def name_pat(n: str) -> str:
    """Name, case-insensitive unless an acronym; space and hyphen interchangeable; beta == β."""
    toks = [t for t in re.split(r"[\s-]+", n.strip()) if t]
    body = r"[\s-]?".join(f"(?:{re.escape(t)}|{GREEK[t.lower()]})" if t.lower() in GREEK else re.escape(t)
                          for t in toks)
    return body if n.isupper() else f"(?i:{body})"


def compile_terms(pats) -> re.Pattern | None:
    pats = sorted(set(pats), key=len, reverse=True)
    return re.compile(rf"(?<![A-Za-z0-9])(?:{'|'.join(pats)})(?![A-Za-z0-9])") if pats else None


def disease_partials(terms: list[str]) -> list[str]:
    """Broader names hidden in long disease names: the part before the first comma, a part after
    'and/or', trailing numbers/types dropped, and head phrases with leading qualifiers dropped
    ('Generalized junctional epidermolysis bullosa' -> 'junctional epidermolysis bullosa',
    'epidermolysis bullosa')."""
    exact, out = {t.casefold() for t in terms}, set()
    for t in terms:
        segs = [s.strip() for s in t.split(",")]
        heads = [segs[0]] + [re.sub(r"^(?:and/or|and|or)\s+", "", s) for s in segs[1:]
                             if re.match(r"(?:and/or|and|or)\s", s)]
        for head in heads:
            head = TRAIL.sub("", head)
            w = head.split()
            for c in [head] + [" ".join(w[k:]) for k in range(1, len(w) - 1)]:
                cw = c.split()
                if (c.casefold() in exact or not cw or cw[0].lower() in STOP or not cw[0][0].isalpha()
                        or (len(cw) == 1 and len(c) < 5)
                        or not any(len(x) >= 4 and x.lower() not in GENERIC for x in cw)):
                    continue
                out.add(c)
    return sorted(out)


def matchers(gene: dict, disease: dict) -> list[tuple[str, re.Pattern, bool]]:
    """[(kind, regex, mid_sentence_only)]. Gene symbols are case-sensitive (CAT the gene, not the
    cat). Upper-case symbols also match the zebrafish form (zmynd10) and mouse form (Cfh); for
    symbols without a digit the mouse form only counts mid-sentence ('Cat models' is not CAT).
    Aliases of 2 characters or fewer, or equal to a disease term, are not highlighted."""
    dterms = [disease["name"], *disease["synonyms"]]
    dfold = {t.casefold() for t in dterms}
    strict, mid = [], []
    for s in [gene["symbol"], *(a for a in gene["aliases"] if len(a) > 2 and a.casefold() not in dfold)]:
        strict.append(sym_pat(s))
        if s.isupper() and len(s) > 1:
            (strict if any(c.isdigit() for c in s) else mid).append(sym_pat(s[0] + s[1:].lower()))
            if any(c.isdigit() for c in s):
                strict.append(sym_pat(s.lower()))
    if len(gene["name"]) > 3:
        strict.append(name_pat(gene["name"]))
    out = [("d", compile_terms(name_pat(t) for t in dterms if len(t) > 1), False),
           ("dp", compile_terms(name_pat(t) for t in disease_partials(dterms)), False),
           ("g", compile_terms(strict), False), ("g", compile_terms(mid), True)]
    return [m for m in out if m[1]]


def sentence_start(text: str, at: int) -> bool:
    before = text[:at].rstrip()
    return not before or before[-1] in ".!?:"


def find_spans(text: str, ms: list[tuple[str, re.Pattern, bool]]) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), k) for k, r, mid in ms for m in r.finditer(text)
            if not (mid and sentence_start(text, m.start()))]


def segments(text: str, ms) -> list[list[str]]:
    """Split text into [chunk, kind] with kind 'g', 'd', 'dp' or '' (plain). Longest span wins."""
    spans = sorted(find_spans(text, ms), key=lambda s: (s[0], s[0] - s[1], PRIORITY[s[2]]))
    out, pos = [], 0
    for a, b, k in spans:
        if a < pos:
            continue
        if a > pos:
            out.append([text[pos:a], ""])
        out.append([text[a:b], k])
        pos = b
    if pos < len(text):
        out.append([text[pos:], ""])
    return out


def highlight(title: str, abstract: str, ms) -> dict:
    """Also picks up abbreviations the text defines after a multi-word match: 'pulmonary fibrosis
    (IPF)' makes IPF a disease-part term for this abstract."""
    text, extra = f"{title}\n{abstract}", {}
    for a, b, k in sorted(find_spans(text, ms), key=lambda s: PRIORITY[s[2]]):
        if (m := ABBR.match(text, b)) and re.search(r"[\s-]", text[a:b]) and \
                sum(c.isupper() for c in m[1]) >= 2:
            extra.setdefault(m[1], k)
    kinds = {}
    for abbr, k in extra.items():
        kinds.setdefault(k, []).append(re.escape(abbr))
    ms = ms + [(k, compile_terms(p), False) for k, p in kinds.items()]
    return {"title": segments(title, ms),
            "abstract": [segments(p, ms) for p in abstract.splitlines() if p.strip()]}


# --- items and scheduling -------------------------------------------------------------------

class Items:
    def __init__(self, path: Path, gap: int, hours: float):
        df = pd.read_parquet(path)
        if missing := sorted(set(REQUIRED) - set(df.columns)):
            raise SystemExit(f"{path}: missing columns {missing}")
        for c in ["item_id", "round_id", "pmid", "gene_name", "title", "abstract"]:
            df[c] = df[c].fillna("").astype(str)
        df["is_repeat"] = df.is_repeat.fillna(False).astype(bool)
        if df.item_id.duplicated().any():
            raise SystemExit(f"{path}: duplicate item_id")
        self.gap, self.hours = gap, hours
        self.items, self.rounds = {}, {}
        for rid, g in df.groupby("round_id", sort=False):
            if g[["gene_symbol", "disease_name"]].drop_duplicates().shape[0] != 1:
                raise SystemExit(f"{path}: round {rid} has more than one gene-disease pair")
            r = g.iloc[0]
            gene = {"symbol": r.gene_symbol, "name": r.gene_name, "aliases": as_list(r.gene_aliases)}
            disease = {"name": r.disease_name, "synonyms": as_list(r.disease_synonyms)}
            ms = matchers(gene, disease)
            self.rounds[rid] = {"gene": gene, "disease": disease,
                                "ids": sorted(g.item_id), "repeat": sorted(g.item_id[g.is_repeat])}
            for it in g.itertuples():
                self.items[it.item_id] = {"round_id": rid, "pmid": it.pmid,
                                          **highlight(it.title, it.abstract, ms)}
        self.round_by_handle = {handle(rid, k): (rid, k) for rid, r in self.rounds.items()
                                for k in ([0, 1] if r["repeat"] else [0])}

    def order(self, who: str) -> list[tuple[str, int]]:
        """(round_id, part): first parts in file order; each second part inserted at a seeded spot
        at least `gap` rounds after its first part."""
        rng, order = seeded(who, "order"), [(rid, 0) for rid in self.rounds]
        for rid, r in self.rounds.items():
            if r["repeat"]:
                lo = min(order.index((rid, 0)) + 1 + self.gap, len(order))
                order.insert(rng.randint(lo, len(order)), (rid, 1))
        return order

    def part(self, who: str, rid: str, k: int) -> list[tuple[str, int]]:
        """Shuffled (item_id, pass) list for part k of a round."""
        r = self.rounds[rid]
        if not r["repeat"]:
            parts = [[(i, 0) for i in r["ids"]]]
        else:
            fresh = [i for i in r["ids"] if i not in r["repeat"]]
            seeded(who, rid, "hold").shuffle(fresh)
            held = set(fresh[: (len(r["ids"]) - len(r["repeat"])) // 2])
            parts = [[(i, 0) for i in r["ids"] if i not in held],
                     [(i, 0) for i in sorted(held)] + [(i, 1) for i in r["repeat"]]]
        out = parts[k]
        seeded(who, rid, k).shuffle(out)
        return out


# --- app ------------------------------------------------------------------------------------

class LabelIn(BaseModel):
    labeller: str = Field(min_length=1, max_length=64)
    round: str
    item: str
    grade: int | None = Field(None, ge=0, le=2)
    unsure: bool = False
    flag: bool = False
    note: str = Field("", max_length=4000)
    seconds: float = Field(0, ge=0, le=86400)
    expect: str | None = None  # updated_at the client last saw (None = not yet rated)


def create_app(items: Items, db_path: Path) -> FastAPI:
    app = FastAPI(title="labeller", docs_url=None, redoc_url=None)

    @contextmanager
    def db():
        with closing(sqlite3.connect(db_path, timeout=10)) as con, con:
            con.row_factory = sqlite3.Row
            yield con

    with db() as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(SCHEMA)

    def name(s: str) -> str:
        if not (s := s.strip().lower()):
            raise HTTPException(422, "empty labeller name")
        return s

    def progress(who: str) -> dict:
        with db() as con:
            labs = {(r["item_id"], r["pass"]): r for r in
                    con.execute("SELECT * FROM labels WHERE labeller = ?", (who,))}
        order = items.order(who)
        parts = {k: items.part(who, *k) for k in order}
        n_done = {k: sum(x in labs for x in parts[k]) for k in order}
        done = [k for k in order if n_done[k] == len(parts[k])]
        started = [k for k in order if 0 < n_done[k] < len(parts[k])]
        finished = {k: max(labs[x]["first_at"] for x in parts[k]) for k in done}

        pending = [k for k in order if k not in done]
        locked = {k: datetime.fromisoformat(finished[(k[0], 0)]) + timedelta(hours=items.hours)
                  for k in pending if k[1] == 1 and (k[0], 0) in done}
        ready = [k for k in pending if k[1] == 0 or (k in locked and locked[k] <= now())]
        # Alternate first and second parts so second parts (which hold repeats) don't bunch up.
        last = max(finished, key=finished.get, default=None)
        pick = next((k for k in ready if last and k[1] != last[1]), ready[0] if ready else None)
        return {"labs": labs, "parts": parts, "done": done,
                "current": started[0] if started else pick,
                "next_at": min((t for k, t in locked.items() if k not in ready), default=None),
                "stats": {"rounds_done": len(done), "n_rounds": len(order), "rated": len(labs),
                          "level": len(done) // 2 + 1}}

    def shown(lab: sqlite3.Row | None) -> dict | None:
        return lab and {"grade": lab["grade"], "unsure": bool(lab["unsure"]), "flag": bool(lab["flag"]),
                        "note": lab["note"], "updated_at": lab["updated_at"]}

    @app.get("/")
    def index():
        return FileResponse(HERE / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    def state(labeller: str):
        who = name(labeller)
        pr = progress(who)
        out = {"labeller": who, **pr["stats"], "round": None, "waiting": None}
        if pr["current"]:
            rid, k = pr["current"]
            r = items.rounds[rid]
            out["round"] = {
                "key": handle(rid, k), "number": len(pr["done"]) + 1, "gene": r["gene"], "disease": r["disease"],
                "items": [{"key": handle(i, p), "title": items.items[i]["title"],
                           "abstract": items.items[i]["abstract"], "label": shown(pr["labs"].get((i, p)))}
                          for i, p in pr["parts"][pr["current"]]]}
        elif pr["next_at"]:
            out["waiting"] = {"next_at": pr["next_at"].isoformat(timespec="seconds")}
        return out

    @app.post("/api/label")
    def label(x: LabelIn):
        who = name(x.labeller)
        if (rk := items.round_by_handle.get(x.round)) is None:
            raise HTTPException(404, "unknown round")
        if (ip := {handle(i, p): (i, p) for i, p in items.part(who, *rk)}.get(x.item)) is None:
            raise HTTPException(404, "item not in round")
        if (x.grade is None) != x.unsure:
            raise HTTPException(422, "give a grade 0/1/2, or unsure with no grade")
        item_id, p = ip
        row = {"labeller": who, "item_id": item_id, "pass": p, "round_id": rk[0],
               "pmid": items.items[item_id]["pmid"], "grade": x.grade, "unsure": int(x.unsure),
               "flag": int(x.flag), "note": x.note.strip(), "seconds": round(x.seconds, 2),
               "at": now().isoformat(timespec="milliseconds")}
        with db() as con:
            con.execute("BEGIN IMMEDIATE")
            cur = con.execute("SELECT updated_at FROM labels WHERE labeller = ? AND item_id = ? AND pass = ?",
                              (who, item_id, p)).fetchone()
            if (cur and cur[0]) != x.expect:  # another tab changed it since this one loaded
                raise HTTPException(409, "changed elsewhere; reload")
            con.execute(UPSERT, row)
            con.execute("INSERT INTO log VALUES (:at, :labeller, :item_id, :pass, :grade, :unsure, :flag, "
                        ":note, :seconds)", row)
        return {**progress(who)["stats"], "updated_at": row["at"]}

    @app.get("/api/export")
    def export_endpoint():
        return export(db_path)

    return app


def export(db_path: Path) -> dict:
    out = HERE / "export"
    out.mkdir(exist_ok=True)
    with closing(sqlite3.connect(db_path)) as con:
        df = pd.read_sql("SELECT * FROM labels ORDER BY labeller, first_at", con)
    df = df.astype({"grade": "Int64", "unsure": bool, "flag": bool})
    df.to_csv(out / "labels.csv", index=False)
    df.to_parquet(out / "labels.parquet", index=False)
    return {"rows": len(df), "csv": str(out / "labels.csv"), "parquet": str(out / "labels.parquet")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default=os.environ.get("LABELLER_ITEMS"), help="items parquet (or $LABELLER_ITEMS)")
    ap.add_argument("--db", type=Path, default=HERE / "labels.sqlite")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--repeat-gap", type=int, default=2, help="min rounds between a round's two parts")
    ap.add_argument("--repeat-hours", type=float, default=8, help="min hours before a round's second part opens")
    ap.add_argument("--export", action="store_true", help="write export/labels.{csv,parquet} and exit")
    a = ap.parse_args()
    if a.export:
        print(export(a.db))
        return
    if not a.items:
        ap.error("--items (or $LABELLER_ITEMS) is required")
    import uvicorn
    uvicorn.run(create_app(Items(Path(a.items), a.repeat_gap, a.repeat_hours), a.db),
                host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
