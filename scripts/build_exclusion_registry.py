#!/usr/bin/env python
"""Build the exclusion registry: every id that appears in an eval set, so
fine-tune builders can drop it (docs/PREREG.md, "Exclusion registry").

One row per (kind, id, source); `parts` lists where in the source it appears.

kinds
  pmid          PubMed id of a task doc, a gold/qrels doc or a haystack silver doc
  gene          Ensembl gene of a gene-disease query
  pair          '{Ensembl gene}|{disease id}' of a gene-disease query
  disease       OT disease id of a gene-disease query
  go_id         GO term of a GO task (+ its is_a parents/children, part
                'is_a_neighbour', as build_finetune_pairs.py excludes them)
  uniprot       UniProt accession behind a GOProteinPairCls protein text
  haystack_doc  PMID in a haystack's PubTator gene-filtered candidates

sources
  data/tasks/*  read from the saved datasets. GOProteinPairCls texts are mapped
                back to GO ids / accessions by exact text (build_finetune_pairs
                parsers); BioMeSHClassification docs to PMIDs by exact title.
                R2MEDBiologyPooled and PubChemSynonymPC500 carry none of these
                ids (Wikipedia/StackExchange passages, chemical names).
  data/haystack/{GeneDiseaseHaystack,GeneDiseaseHaystackDev}
                queries (gene, pair, disease), qrels and silver (pmid),
                gene_pmids (haystack_doc).

Re-run after any eval set is built or changed.
Run:    uv run python scripts/build_exclusion_registry.py
Output: data/exclusion/registry.parquet, data/exclusion/manifest.json (untracked).
"""

import json
import re
import sys
from pathlib import Path

import duckdb
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_finetune_pairs import DEF_LINE_RE, DEF_RE, FUNC_LINE_RE, clean_function_text  # noqa: E402

RECORDS = "/outsee5/home/danbolser/Work/IUK-IPF/data/embeddings-catalogue/records/*.parquet"
ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "data" / "tasks"
HAYSTACKS = ["GeneDiseaseHaystack", "GeneDiseaseHaystackDev"]
OUT_DIR = ROOT / "data" / "exclusion"

PAIR_RE = re.compile(r"^ENSG\d+\|\S+$")

Row = tuple[str, str, str, str]  # kind, id, source, part


def go_terms(con: duckdb.DuckDBPyConnection) -> tuple[dict, dict]:
    """GO id -> (texts in both test-builder forms, is_a parents); alt id -> id."""
    terms, alt_map = {}, {}
    for go_id, name, text, meta_json in con.sql(f"""
        SELECT go_id, name, text, metadata FROM (
            SELECT record_id AS go_id, coalesce(trim(title), '') AS name, text, metadata,
                   row_number() OVER (PARTITION BY record_id ORDER BY record_hash) AS rn
            FROM read_parquet('{RECORDS}') WHERE record_type = 'go_term'
        ) WHERE rn = 1
    """).fetchall():
        meta = json.loads(meta_json)
        m1, m2 = DEF_RE.search(text), DEF_LINE_RE.search(text)
        texts = {f"{name}: {' '.join(m1.group(1).split())}" if m1 else "",
                 f"{name}: {m2.group(1).strip()}" if m2 else ""}
        terms[go_id] = ({t for t in texts if name and not t.endswith(": ")},
                        set(meta.get("is_a") or []))
        for alt in meta.get("alt_ids") or []:
            alt_map[alt] = go_id
    return terms, alt_map


def protein_texts(con: duckdb.DuckDBPyConnection) -> dict[str, set[str]]:
    """Cleaned UniProt Function text -> accessions (as build_tier_c_pairs.py)."""
    by_text: dict[str, set[str]] = {}
    for acc, text in con.sql(f"""
        SELECT accession, text FROM (
            SELECT record_id AS accession, text,
                   row_number() OVER (PARTITION BY record_id ORDER BY record_hash) AS rn
            FROM read_parquet('{RECORDS}')
            WHERE record_type IN ('uniprot_human', 'uniprot_protein')
              AND text LIKE '%' || chr(10) || 'Function: %'
        ) WHERE rn = 1
    """).fetchall():
        if (m := FUNC_LINE_RE.search(text)) and (f := clean_function_text(m.group(1))):
            by_text.setdefault(f, set()).add(acc)
    return by_text


def norm_title(t: str) -> str:
    return " ".join(t.lower().split()).rstrip(".")


def title_pmids(con: duckdb.DuckDBPyConnection, titles: set[str]) -> dict[str, set[str]]:
    """Normalised title -> every local PMID with that title (all kept: conservative)."""
    con.sql("CREATE OR REPLACE TEMP TABLE want (t VARCHAR)")
    con.executemany("INSERT INTO want VALUES (?)", [(t,) for t in titles])
    out: dict[str, set[str]] = {}
    for t, pmid in con.sql(f"""
        SELECT w.t, replace(r.record_id, 'PMID:', '')
        FROM read_parquet('{RECORDS}') r
        JOIN want w ON rtrim(lower(trim(regexp_replace(r.title, '\\s+', ' ', 'g'))), '.') = w.t
        WHERE r.record_type = 'pubmed_abstract'
    """).fetchall():
        out.setdefault(t, set()).add(pmid)
    return out


def gene_disease_rows(qids, part: str, source: str) -> list[Row]:
    rows = []
    for q in qids:
        gene, disease = q.split("|", 1)
        rows += [("pair", q, source, part), ("gene", gene, source, part),
                 ("disease", disease, source, part)]
    return rows


def main() -> None:
    con = duckdb.connect()
    con.sql("SET threads=48")
    rows: list[Row] = []
    notes: dict[str, str] = {}
    terms, alt_map = go_terms(con)

    for task_dir in sorted(p for p in TASKS.iterdir() if (p / "dataset_dict.json").exists()):
        name = task_dir.name
        dd = load_from_disk(str(task_dir))
        n0 = len(rows)
        if "queries" in dd and all(PAIR_RE.match(q) for q in dd["queries"]["_id"]):
            rows += gene_disease_rows(dd["queries"]["_id"], "query", name)
            rows += [("pmid", p, name, "corpus") for p in dd["corpus"]["_id"]]
            rows += [("pmid", p, name, "qrels") for p in dd["qrels"]["corpus-id"]]
        elif name == "GOPubMedRetrieval":
            rows += [("go_id", g, name, "query") for g in dd["queries"]["_id"]]
            rows += [("pmid", p, name, "corpus") for p in dd["corpus"]["_id"]]
            rows += [("pmid", p, name, "qrels") for p in dd["qrels"]["corpus-id"]]
        elif name == "GOProteinPairCls":
            go_by_text: dict[str, set[str]] = {}
            for g, (texts, _) in terms.items():
                for t in texts:
                    go_by_text.setdefault(t, set()).add(g)
            acc_by_text = protein_texts(con)
            s1, s2 = set(dd["test"]["sentence1"]), set(dd["test"]["sentence2"])
            rows += [("go_id", g, name, "sentence1") for s in s1 for g in go_by_text.get(s, ())]
            rows += [("uniprot", a, name, "sentence2") for s in s2 for a in acc_by_text.get(s, ())]
            notes[name] = (f"unmapped texts: GO {len(s1 - go_by_text.keys())}/{len(s1)}, "
                           f"protein {len(s2 - acc_by_text.keys())}/{len(s2)}")
        elif name == "BioMeSHClassification":
            titles = {split: [norm_title(t.split("\n\n", 1)[0]) for t in dd[split]["text"]]
                      for split in dd}
            found = title_pmids(con, {t for ts in titles.values() for t in ts})
            for split, ts in titles.items():
                rows += [("pmid", p, name, split) for t in ts for p in found.get(t, ())]
            n = sum(len(ts) for ts in titles.values())
            hit = sum(t in found for ts in titles.values() for t in ts)
            notes[name] = f"title-matched {hit}/{n} docs to local PMIDs"
        else:
            notes[name] = "no pmid/gene/disease/GO/UniProt ids"
        print(f"{name}: {len(rows) - n0} rows  {notes.get(name, '')}", flush=True)

    # GO is_a parents/children of each source's GO ids (build_finetune_pairs rule).
    children: dict[str, set[str]] = {}
    for g, (_, parents) in terms.items():
        for p in parents:
            children.setdefault(p, set()).add(g)
    go_by_source: dict[str, set[str]] = {}
    for kind, i, source, _ in rows:
        if kind == "go_id":
            go_by_source.setdefault(source, set()).add(alt_map.get(i, i))
    for source, gs in sorted(go_by_source.items()):
        for g in sorted(gs):
            for n in sorted((terms.get(g, (set(), set()))[1] | children.get(g, set())) - gs):
                rows.append(("go_id", n, source, "is_a_neighbour"))

    for name in HAYSTACKS:
        h = ROOT / "data" / "haystack" / name
        rows += gene_disease_rows(
            [r[0] for r in con.sql(f"SELECT qid FROM '{h}/queries.parquet'").fetchall()],
            "query", name)
        for part in ("qrels", "silver"):
            rows += [("pmid", r[0], name, part) for r in
                     con.sql(f"SELECT DISTINCT pmid FROM '{h}/{part}.parquet'").fetchall()]
        print(f"{name}: queries, qrels, silver", flush=True)

    con.sql("CREATE TEMP TABLE raw (kind VARCHAR, id VARCHAR, source VARCHAR, part VARCHAR)")
    con.executemany("INSERT INTO raw VALUES (?, ?, ?, ?)", rows)
    docs = " UNION ALL ".join(
        f"SELECT DISTINCT 'haystack_doc', pmid, '{n}', 'gene_pmids' "
        f"FROM '{ROOT / 'data' / 'haystack' / n / 'gene_pmids.parquet'}'" for n in HAYSTACKS)
    con.sql(f"INSERT INTO raw {docs}")
    con.sql("""
        CREATE TEMP TABLE registry AS
        SELECT kind, id, source, list(DISTINCT part ORDER BY part) AS parts
        FROM raw GROUP BY ALL
    """)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con.sql(f"""COPY (SELECT * FROM registry ORDER BY kind, source, id)
                TO '{OUT_DIR / 'registry.parquet'}' (FORMAT parquet)""")

    counts = con.sql("""
        SELECT source, kind, count(*) FROM registry GROUP BY ALL ORDER BY source, kind
    """).fetchall()
    distinct = dict(con.sql("SELECT kind, count(DISTINCT id) FROM registry GROUP BY 1 ORDER BY 1").fetchall())
    by_source: dict[str, dict[str, int]] = {}
    for s, k, n in counts:
        by_source.setdefault(s, {})[k] = n
    manifest = {
        "name": "exclusion registry",
        "columns": ["kind", "id", "source", "parts"],
        "rows": sum(n for _, _, n in counts),
        "distinct_ids_by_kind": distinct,
        "rows_by_source": by_source,
        "notes": notes,
        "id_formats": {
            "pmid / haystack_doc": "PubMed id, digits only",
            "gene": "Ensembl gene (OT targetId)",
            "disease": "OT diseaseId (MONDO_/Orphanet_/HP_/EFO_...)",
            "pair": "'{gene}|{disease}'",
            "go_id": "GO:nnnnnnn (primary id)",
            "uniprot": "UniProt accession",
        },
        "build_script": "scripts/build_exclusion_registry.py",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "distinct_ids_by_kind", "rows_by_source")},
                     indent=2))


if __name__ == "__main__":
    main()
