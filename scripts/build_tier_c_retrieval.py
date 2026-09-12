#!/usr/bin/env python
"""Build GOPubMedRetrieval (Tier C): GO-term -> PubMed-evidence retrieval with
human-curated gold.

Query  = GO term "name: definition".
Corpus = PubMed abstracts.
Gold   = abstracts cited by GO Consortium curators as experimental evidence
         (GAF-derived goa_annotation_reference links, experimental codes only).

Deterministic: SEED=42 everywhere; all pools sorted before sampling.
Re-runnable: output directory is rebuilt from scratch on each run.

Run:  uv run python scripts/build_tier_c_retrieval.py
"""

import json
import random
import re
import shutil
from pathlib import Path

import duckdb
from datasets import Dataset, DatasetDict

SEED = 42
N_QUERIES = 75
CORPUS_SIZE = 300
MIN_GOLD, MAX_GOLD = 2, 15
# All golds must fit in the corpus alongside distractors, so terms are accepted
# greedily (in seeded shuffle order) while the distinct-gold union stays within
# this budget. 240 leaves >= 60 distractor slots in a 300-doc corpus.
GOLD_BUDGET = 240

EXP_CODES = ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"]

GOA_JSONL = "/outsee5/home/danbolser/Work/IUK-IPF/data/sources/goa/pmid-links.jsonl"
RECORDS = "/outsee5/home/danbolser/Work/IUK-IPF/data/embeddings-catalogue/records/*.parquet"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "tasks" / "GOPubMedRetrieval"

ABSTRACT_MARK = "\nAbstract: "
DEF_RE = re.compile(r"\nDefinition: (.*?)(?:\nSynonyms:|\nComment:|$)", re.DOTALL)


def load_source_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Curated evidence links (experimental codes only) and the PubMed/GO
    records they touch. The jsonl mixes two row shapes, so columns are given
    explicitly; uniprot_entry_reference rows (entry-level, not per-PMID
    evidence) are excluded."""
    exp = ", ".join(f"'{c}'" for c in EXP_CODES)
    con.sql(f"""
        CREATE TEMP TABLE links AS
        SELECT DISTINCT go_id, pmid, evidence_code, species
        FROM read_json('{GOA_JSONL}', format='newline_delimited',
            columns={{'link_type':'VARCHAR','go_id':'VARCHAR','pmid':'VARCHAR',
                      'evidence_code':'VARCHAR','species':'VARCHAR'}})
        WHERE link_type = 'goa_annotation_reference'
          AND evidence_code IN ({exp})
    """)
    # PubMed abstracts for linked PMIDs. Duplicate record_ids exist (same
    # abstract ingested twice); keep one per PMID by record_hash.
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
        SELECT go_id, name, text FROM (
            SELECT record_id AS go_id, coalesce(trim(title), '') AS name, text,
                   row_number() OVER (PARTITION BY record_id ORDER BY record_hash) AS rn
            FROM read_parquet('{RECORDS}')
            WHERE record_type = 'go_term'
              AND record_id IN (SELECT DISTINCT go_id FROM links)
        ) WHERE rn = 1
    """)


def parse_definition(record_text: str) -> str:
    m = DEF_RE.search(record_text)
    return " ".join(m.group(1).split()) if m else ""


def main() -> None:
    con = duckdb.connect()
    load_source_tables(con)

    # (go_id, pmid) evidence pairs whose abstract exists in the records table.
    pairs = con.sql(
        "SELECT DISTINCT l.go_id, l.pmid FROM links l JOIN pubmed USING (pmid)"
    ).fetchall()
    golds_by_term: dict[str, set[str]] = {}
    for go_id, pmid in pairs:
        golds_by_term.setdefault(go_id, set()).add(pmid)

    terms = {
        go_id: (name, parse_definition(text))
        for go_id, name, text in con.sql("SELECT go_id, name, text FROM go_terms").fetchall()
    }

    # 1. Candidate terms: non-empty name+definition, 2-15 gold PMIDs in records.
    candidates = sorted(
        go_id
        for go_id, pmids in golds_by_term.items()
        if MIN_GOLD <= len(pmids) <= MAX_GOLD
        and go_id in terms
        and terms[go_id][0]
        and terms[go_id][1]
    )
    print(f"candidate terms: {len(candidates)}")

    # Seeded shuffle, then greedy accept with a paced budget: after k accepts
    # the gold union may use at most GOLD_BUDGET * (k+1)/N_QUERIES slots, so
    # spending averages GOLD_BUDGET/N_QUERIES new golds per term throughout
    # instead of exhausting the budget on the first large terms.
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    selected: list[str] = []
    gold_pmids: set[str] = set()
    for go_id in candidates:
        if len(selected) == N_QUERIES:
            break
        union = gold_pmids | golds_by_term[go_id]
        if len(union) <= GOLD_BUDGET * (len(selected) + 1) / N_QUERIES:
            selected.append(go_id)
            gold_pmids = union
    if len(selected) < N_QUERIES:
        raise SystemExit(f"only {len(selected)} terms fit GOLD_BUDGET={GOLD_BUDGET}")
    selected.sort()

    # 2. Corpus: all golds + distractors from other GO-evidence PMIDs.
    distractor_pool = sorted(
        {pmid for pmids in golds_by_term.values() for pmid in pmids} - gold_pmids
    )
    distractors = random.Random(SEED).sample(distractor_pool, CORPUS_SIZE - len(gold_pmids))
    corpus_pmids = sorted(gold_pmids | set(distractors), key=int)

    docs = {
        pmid: (title, abstract)
        for pmid, title, abstract in con.sql("SELECT pmid, title, abstract FROM pubmed").fetchall()
    }
    corpus = Dataset.from_dict({
        "_id": corpus_pmids,
        "title": [docs[p][0] for p in corpus_pmids],
        "text": [
            f"{docs[p][0]}\n\n{docs[p][1]}" if docs[p][0] else docs[p][1]
            for p in corpus_pmids
        ],
    })
    queries = Dataset.from_dict({
        "_id": selected,
        "text": [f"{terms[t][0]}: {terms[t][1]}" for t in selected],
    })
    qrel_rows = [(t, p) for t in selected for p in sorted(golds_by_term[t], key=int)]
    qrels = Dataset.from_dict({
        "query-id": [q for q, _ in qrel_rows],
        "corpus-id": [p for _, p in qrel_rows],
        "score": [1] * len(qrel_rows),
    })

    # 3. Validate.
    corpus_ids = set(corpus["_id"])
    assert len(corpus) == CORPUS_SIZE, len(corpus)
    assert len(queries) == N_QUERIES
    assert all(p in corpus_ids for _, p in qrel_rows), "gold doc missing from corpus"
    assert all(MIN_GOLD <= len(golds_by_term[t]) <= MAX_GOLD for t in selected)
    assert all(corpus["text"]) and all(queries["text"]), "empty text field"

    # 4. Provenance stats for the manifest (gold pairs only).
    gold_sql = " OR ".join(
        f"(go_id='{t}' AND pmid IN ({', '.join(repr(p) for p in golds_by_term[t])}))"
        for t in selected
    )
    species_cov = dict(con.sql(
        f"SELECT species, count(DISTINCT (go_id, pmid)) FROM links WHERE {gold_sql} "
        "GROUP BY 1 ORDER BY 2 DESC, 1"
    ).fetchall())
    code_cov = dict(con.sql(
        f"SELECT evidence_code, count(DISTINCT (go_id, pmid)) FROM links WHERE {gold_sql} "
        "GROUP BY 1 ORDER BY 2 DESC, 1"
    ).fetchall())
    mean_golds = len(qrel_rows) / len(selected)

    # 5. Save.
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    DatasetDict({"corpus": corpus, "queries": queries, "qrels": qrels}).save_to_disk(OUT_DIR)
    manifest = {
        "name": "GOPubMedRetrieval",
        "category": "Retrieval",
        "sizes": {"corpus": len(corpus), "queries": len(queries), "qrels": len(qrels)},
        "provenance": {
            "gold_links": GOA_JSONL,
            "gold_link_type": "goa_annotation_reference (GAF-derived; "
                              "uniprot_entry_reference rows excluded: entry-level, not per-PMID evidence)",
            "texts": RECORDS,
            "evidence_codes_used": EXP_CODES,
            "evidence_code_counts_gold_pairs": code_cov,
            "species_coverage_gold_pairs": species_cov,
            "species_note": "a (term, PMID) pair annotated in several species counts once per species",
        },
        "protocol": {
            "seed": SEED,
            "query_text": "'{GO name}: {GO definition}'",
            "doc_text": "'{title}\\n\\n{abstract}' if title else abstract; text is "
                        "self-contained -- do NOT re-prepend the title column",
            "candidate_terms": f"non-empty name+definition, {MIN_GOLD}-{MAX_GOLD} distinct "
                               "experimental-evidence PMIDs with abstracts in the records table",
            "selection": f"seeded shuffle of {len(candidates)} candidates, greedily accept "
                         f"terms under a paced gold budget (union <= {GOLD_BUDGET} * "
                         f"(accepted+1)/{N_QUERIES}), stop at {N_QUERIES}",
            "distractors": "sampled (seed 42) from other experimental GO-evidence PMIDs "
                           f"present in records, to {CORPUS_SIZE} docs total",
            "qrels": "score=1 for every curated (term, PMID) evidence pair; "
                     "all gold docs are in the corpus",
            "mean_golds_per_query": round(mean_golds, 3),
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nSaved to {OUT_DIR}")
    print(f"corpus:  {len(corpus)} docs ({len(gold_pmids)} gold, {len(distractors)} distractor)")
    print(f"queries: {len(queries)}")
    print(f"qrels:   {len(qrels)} (mean golds/query {mean_golds:.2f}, "
          f"min {min(len(golds_by_term[t]) for t in selected)}, "
          f"max {max(len(golds_by_term[t]) for t in selected)})")
    print(f"species coverage (gold pairs): {species_cov}")
    print(f"evidence codes (gold pairs):   {code_cov}")


if __name__ == "__main__":
    main()
