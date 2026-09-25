#!/usr/bin/env python
"""Build a contrastive fine-tuning set for embedders from human-curated GO
annotations. No synthetic text, no LLM calls.

Pair types (anchor -> positive):
  go_abstract  GO term "name: definition" -> PubMed abstract cited as
               experimental evidence for that term (GAF goa_annotation_reference
               rows, experimental codes only)
  go_protein   GO term "name: definition" -> UniProt function text of a protein
               annotated with that term (same GAF rows, object_id = accession)

Leakage exclusion (strict): a pair is dropped if its GO id, PMID or UniProt
accession appears anywhere in data/tasks/GOPubMedRetrieval (queries, corpus,
qrels) or data/tasks/GOProteinPairCls (sentence texts mapped back to ids, and
matched by exact text as a belt-and-braces check), or if its GO id is an is_a
parent or child of a test GO term.

Output: data/finetune/pairs.parquet (anchor, positive, pair_type) and
        data/finetune/manifest.json.

Run:  uv run python scripts/build_finetune_pairs.py
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

import duckdb
from datasets import load_from_disk

SEED = 42
CAP_PER_TYPE = 100_000  # ~200k total, balanced
EXP_CODES = ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"]

GOA_JSONL = "/outsee5/home/danbolser/Work/IUK-IPF/data/sources/goa/pmid-links.jsonl"
RECORDS = "/outsee5/home/danbolser/Work/IUK-IPF/data/embeddings-catalogue/records/*.parquet"
ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "data" / "tasks"
OUT_DIR = ROOT / "data" / "finetune"

ABSTRACT_MARK = "\nAbstract: "
# Two definition parsers exist in the test builders; both are reproduced so the
# text-match exclusion catches either form.
DEF_RE = re.compile(r"\nDefinition: (.*?)(?:\nSynonyms:|\nComment:|$)", re.DOTALL)  # retrieval
DEF_LINE_RE = re.compile(r"^Definition: (.*)$", re.M)  # pair-cls
ECO_RE = re.compile(r"\s*\{ECO:[^}]*\}")
FUNC_LINE_RE = re.compile(r"^Function: (.*)$", re.M)


def clean_function_text(raw: str) -> str:
    """Same cleaning as build_tier_c_pairs.py."""
    text = raw.replace("FUNCTION: ", "")
    text = ECO_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+\.", ".", text)


def load_source_tables(con: duckdb.DuckDBPyConnection) -> None:
    exp = ", ".join(f"'{c}'" for c in EXP_CODES)
    con.sql(f"""
        CREATE TEMP TABLE links AS
        SELECT DISTINCT go_id, pmid, object_id AS accession, evidence_code, species
        FROM read_json('{GOA_JSONL}', format='newline_delimited',
            columns={{'link_type':'VARCHAR','go_id':'VARCHAR','pmid':'VARCHAR',
                      'object_id':'VARCHAR','evidence_code':'VARCHAR','species':'VARCHAR'}})
        WHERE link_type = 'goa_annotation_reference'
          AND evidence_code IN ({exp})
    """)
    con.sql(f"""
        CREATE TEMP TABLE pubmed AS
        SELECT pmid, title, abstract FROM (
            SELECT replace(record_id, 'PMID:', '') AS pmid,
                   coalesce(trim(title), '') AS title,
                   trim(substr(text, position('{ABSTRACT_MARK}' IN text)
                                     + {len(ABSTRACT_MARK)})) AS abstract,
                   row_number() OVER (PARTITION BY record_id ORDER BY record_hash) AS rn
            FROM read_parquet('{RECORDS}')
            WHERE record_type = 'pubmed_abstract'
              AND position('{ABSTRACT_MARK}' IN text) > 0
              AND replace(record_id, 'PMID:', '') IN (SELECT pmid FROM links)
        ) WHERE rn = 1 AND abstract <> ''
    """)
    con.sql(f"""
        CREATE TEMP TABLE go_terms AS
        SELECT go_id, name, text, metadata FROM (
            SELECT record_id AS go_id, coalesce(trim(title), '') AS name, text, metadata,
                   row_number() OVER (PARTITION BY record_id ORDER BY record_hash) AS rn
            FROM read_parquet('{RECORDS}')
            WHERE record_type = 'go_term'
        ) WHERE rn = 1
    """)
    con.sql(f"""
        CREATE TEMP TABLE proteins AS
        SELECT accession, text FROM (
            SELECT record_id AS accession, text,
                   row_number() OVER (PARTITION BY record_id ORDER BY record_hash) AS rn
            FROM read_parquet('{RECORDS}')
            WHERE record_type IN ('uniprot_human', 'uniprot_protein')
              AND text LIKE '%' || chr(10) || 'Function: %'
        ) WHERE rn = 1
    """)


def main() -> None:
    con = duckdb.connect()
    load_source_tables(con)

    # ---- GO terms: both text forms, alt ids, is_a edges ----
    terms: dict[str, dict] = {}
    alt_map: dict[str, str] = {}
    for go_id, name, text, meta_json in con.sql("SELECT * FROM go_terms").fetchall():
        meta = json.loads(meta_json)
        m1, m2 = DEF_RE.search(text), DEF_LINE_RE.search(text)
        d_ret = " ".join(m1.group(1).split()) if m1 else ""
        d_cls = m2.group(1).strip() if m2 else ""
        terms[go_id] = {
            "name": name,
            "text_ret": f"{name}: {d_ret}" if name and d_ret else "",
            "text_cls": f"{name}: {d_cls}" if name and d_cls else "",
            "is_a": meta.get("is_a") or [],
        }
        for alt in meta.get("alt_ids") or []:
            alt_map[alt] = go_id
    children: dict[str, set[str]] = {}
    for g, t in terms.items():
        for p in t["is_a"]:
            children.setdefault(p, set()).add(g)

    proteins = {
        acc: clean_function_text(m.group(1))
        for acc, text in con.sql("SELECT accession, text FROM proteins").fetchall()
        if (m := FUNC_LINE_RE.search(text))
    }
    proteins = {a: f for a, f in proteins.items() if f}
    pubmed = {
        pmid: f"{title}\n\n{abstract}" if title else abstract
        for pmid, title, abstract in con.sql("SELECT * FROM pubmed").fetchall()
    }

    # ---- Test-set ids and texts ----
    ret = load_from_disk(str(TASKS / "GOPubMedRetrieval"))
    cls = load_from_disk(str(TASKS / "GOProteinPairCls"))["test"]

    test_go = set(ret["queries"]["_id"]) | set(ret["qrels"]["query-id"])
    test_pmid = set(ret["corpus"]["_id"]) | set(ret["qrels"]["corpus-id"])
    test_text = set(ret["queries"]["text"]) | set(ret["corpus"]["text"])
    test_text |= set(cls["sentence1"]) | set(cls["sentence2"])

    go_by_text = {}
    for g, t in terms.items():
        for k in ("text_ret", "text_cls"):
            if t[k]:
                go_by_text.setdefault(t[k], set()).add(g)
    acc_by_text: dict[str, set[str]] = {}
    for a, f in proteins.items():
        acc_by_text.setdefault(f, set()).add(a)

    cls_go_unmapped = cls_acc_unmapped = 0
    for s in set(cls["sentence1"]):
        if s in go_by_text:
            test_go |= go_by_text[s]
        else:
            cls_go_unmapped += 1
    test_acc: set[str] = set()
    for s in set(cls["sentence2"]):
        if s in acc_by_text:
            test_acc |= acc_by_text[s]
        else:
            cls_acc_unmapped += 1  # falls back to exact-text exclusion
    test_go = {alt_map.get(g, g) for g in test_go}
    neighbours = set()
    for g in test_go:
        neighbours |= set(terms.get(g, {}).get("is_a", []))
        neighbours |= children.get(g, set())
    neighbours -= test_go
    excluded_go = test_go | neighbours

    print(f"test GO ids: {len(test_go)} (+{len(neighbours)} is_a neighbours), "
          f"test PMIDs: {len(test_pmid)}, test accessions: {len(test_acc)}, "
          f"test texts: {len(test_text)}")
    print(f"unmapped PairCls sentences: GO {cls_go_unmapped}, protein {cls_acc_unmapped} "
          f"(covered by exact-text exclusion)")

    def go_text(go_id: str) -> str:
        return terms.get(alt_map.get(go_id, go_id), {}).get("text_ret", "")

    def is_leaky(go_id: str, anchor: str, positive: str, pmid=None, accs=()) -> str | None:
        g = alt_map.get(go_id, go_id)
        if g in test_go:
            return "go_id"
        if g in neighbours:
            return "go_neighbour"
        if pmid is not None and pmid in test_pmid:
            return "pmid"
        if any(a in test_acc for a in accs):
            return "accession"
        if anchor in test_text or positive in test_text:
            return "text"
        return None

    stats = {"go_abstract": Counter(), "go_protein": Counter()}
    kept: dict[str, dict[tuple[str, str], tuple]] = {"go_abstract": {}, "go_protein": {}}

    # ---- go_abstract: (go_id, pmid) evidence pairs ----
    rows = con.sql(
        "SELECT l.go_id, l.pmid, list(DISTINCT l.accession) FROM links l "
        "JOIN pubmed USING (pmid) GROUP BY 1, 2 ORDER BY 1, 2"
    ).fetchall()
    seen_pairs = set()
    for go_id, pmid, accs in rows:
        st = stats["go_abstract"]
        anchor, positive = go_text(go_id), pubmed[pmid]
        if not anchor:
            st["no_term_text"] += 1
            continue
        why = is_leaky(go_id, anchor, positive, pmid=pmid, accs=accs)
        if why:
            st[f"excluded_{why}"] += 1
            continue
        if (anchor, positive) in seen_pairs:
            st["duplicate"] += 1
            continue
        seen_pairs.add((anchor, positive))
        kept["go_abstract"][(anchor, positive)] = (go_id, pmid)

    # ---- go_protein: (go_id, accession) annotation pairs ----
    rows = con.sql(
        "SELECT DISTINCT l.go_id, l.accession FROM links l "
        "JOIN proteins p USING (accession) ORDER BY 1, 2"
    ).fetchall()
    seen_pairs = set()
    for go_id, acc in rows:
        st = stats["go_protein"]
        anchor, positive = go_text(go_id), proteins.get(acc, "")
        if not anchor or not positive:
            st["no_text"] += 1
            continue
        why = is_leaky(go_id, anchor, positive, accs=(acc,))
        if why:
            st[f"excluded_{why}"] += 1
            continue
        if (anchor, positive) in seen_pairs:
            st["duplicate"] += 1
            continue
        seen_pairs.add((anchor, positive))
        kept["go_protein"][(anchor, positive)] = (go_id, acc)

    # ---- Cap, seeded, balanced ----
    out = []
    for ptype in ("go_abstract", "go_protein"):
        keys = sorted(kept[ptype])
        rng = random.Random(SEED)
        rng.shuffle(keys)
        keys = keys[:CAP_PER_TYPE]
        stats[ptype]["available"] = len(kept[ptype])
        stats[ptype]["kept"] = len(keys)
        out += [(a, p, ptype) for a, p in keys]
    random.Random(SEED).shuffle(out)
    assert len({(a, p) for a, p, _ in out}) == len(out)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con.sql("CREATE TEMP TABLE pairs (anchor VARCHAR, positive VARCHAR, pair_type VARCHAR)")
    con.executemany("INSERT INTO pairs VALUES (?, ?, ?)", out)
    con.sql(f"COPY pairs TO '{OUT_DIR / 'pairs.parquet'}' (FORMAT PARQUET)")

    counts = Counter(t for _, _, t in out)
    manifest = {
        "name": "GO contrastive fine-tuning pairs",
        "columns": ["anchor", "positive", "pair_type"],
        "sizes": {"total": len(out), **dict(counts)},
        "distinct_go_terms": len({kept[t][(a, p)][0] for a, p, t in out}),
        "exclusion": {
            "test_tasks": ["GOPubMedRetrieval", "GOProteinPairCls"],
            "test_go_ids": len(test_go),
            "test_go_is_a_neighbours": len(neighbours),
            "test_pmids": len(test_pmid),
            "test_accessions": len(test_acc),
            "test_texts": len(test_text),
            "paircls_unmapped_sentences": {"go": cls_go_unmapped, "protein": cls_acc_unmapped},
            "rule": "drop pair if GO id (alt ids resolved) in test or is_a parent/child of a "
                    "test term, PMID in test corpus, UniProt accession in test, or either "
                    "text exactly equals a test query/doc/sentence",
        },
        "stats": {t: dict(sorted(c.items())) for t, c in stats.items()},
        "provenance": {
            "links": GOA_JSONL,
            "link_type": "goa_annotation_reference",
            "evidence_codes": EXP_CODES,
            "texts": RECORDS,
            "anchor": "GO term 'name: definition'",
            "positive": {
                "go_abstract": "'{title}\\n\\n{abstract}' of a PMID cited as evidence",
                "go_protein": "UniProt Function text (FUNCTION:/ECO tags stripped) of "
                              "the annotated protein (GAF object_id)",
            },
            "seed": SEED,
            "cap_per_type": CAP_PER_TYPE,
            "dedup": "(anchor, positive)",
            "build_script": "scripts/build_finetune_pairs.py",
            "synthetic": False,
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nSaved {OUT_DIR / 'pairs.parquet'}  rows={len(out)}  {dict(counts)}")
    for t, c in stats.items():
        print(f"{t}: {dict(sorted(c.items()))}")
    for a, p, t in out[:3]:
        print(f"\n[{t}]\n  anchor:   {a[:200]}\n  positive: {p[:200]}")


if __name__ == "__main__":
    main()
