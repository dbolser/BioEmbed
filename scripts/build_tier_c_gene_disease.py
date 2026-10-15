#!/usr/bin/env python
"""Build GeneDiseaseRetrieval (Tier C): gene-disease -> PubMed-evidence retrieval
with expert-curated gold (needle-in-a-haystack, 300-doc subset).

Query  = "{SYMBOL} ({gene name}) in {disease name}".
Corpus = PubMed abstracts.
Gold   = PMIDs cited by curators for the gene-disease link in Open Targets
         evidence (orphanet, gene2phenotype, genomics_england, uniprot_literature).
Hard negatives = per query, abstracts linked (curated or Europe PMC) to the
         same gene with an unrelated disease, so the gene symbol alone cannot
         find the gold: the model must match the disease too, as in the full
         PubMed haystack.
Silver guard = no corpus doc is linked to a query (curated or Europe PMC
         text-mining) unless it is in that query's qrels, so incomplete gold
         never penalises a correct hit.

Also writes the full curated gold table (data/gold/, untracked) for the
full-haystack version of the benchmark.

Deterministic: SEED=42 everywhere; all pools sorted before sampling.
Re-runnable: output directory is rebuilt from scratch on each run.

Run:  uv run python scripts/build_tier_c_gene_disease.py
"""

import json
import random
import shutil
from pathlib import Path

import duckdb
from datasets import Dataset, DatasetDict

SEED = 42
N_QUERIES = 75
CORPUS_SIZE = 300
MIN_GOLD, MAX_GOLD = 1, 15
# Paced gold budget as in GOPubMedRetrieval. One new gold per query keeps
# recall@1 a clean hit@1 (ceiling 1.0) and leaves 225 slots for
# HARD_PER_QUERY same-gene negatives per query. With 2+ golds and 2 negatives
# per query, small embedders and BM25 sat at 80-90% of the recall@1 ceiling.
GOLD_BUDGET = 75
HARD_PER_QUERY = 3
# Candidate genes need this many same-gene, other-disease links (before the
# abstract check) so every query can get its hard negatives.
MIN_HARD_POOL = 10

CURATED_SOURCES = ["orphanet", "gene2phenotype", "genomics_england", "uniprot_literature"]

IUK = "/outsee5/home/danbolser/Work/IUK-IPF/data"
OT = f"{IUK}/sources/opentargets/26.06"
RECORDS = f"{IUK}/embeddings-catalogue/records/*.parquet"
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "tasks" / "GeneDiseaseRetrieval"
GOLD_OUT = ROOT / "data" / "gold" / "ot_curated_gene_disease_pmid.parquet"

ABSTRACT_MARK = "\nAbstract: "


def load_source_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Curated links, Europe PMC silver links for the same genes, names,
    disease hierarchy, and abstracts for every curated PMID."""
    union = " UNION ALL ".join(
        f"SELECT targetId, diseaseId, unnest(literature) AS pmid, datasourceId "
        f"FROM '{OT}/evidence_{s}/*.parquet' WHERE literature IS NOT NULL"
        for s in CURATED_SOURCES
    )
    con.sql(f"""
        CREATE TEMP TABLE curated AS
        SELECT targetId, diseaseId, pmid, list(DISTINCT datasourceId ORDER BY datasourceId) AS sources
        FROM ({union}) WHERE regexp_matches(pmid, '^[0-9]+$')
        GROUP BY ALL
    """)
    con.sql(f"""
        CREATE TEMP TABLE silver AS
        SELECT DISTINCT targetId, diseaseId, unnest(literature) AS pmid
        FROM '{OT}/evidence_europepmc/*.parquet'
        WHERE targetId IN (SELECT targetId FROM curated)
           OR diseaseId IN (SELECT diseaseId FROM curated)
    """)
    con.sql(f"""
        CREATE TEMP TABLE genes AS
        SELECT id AS targetId, approvedSymbol AS symbol, approvedName AS gene_name
        FROM '{OT}/target/*.parquet' WHERE id IN (SELECT targetId FROM curated)
    """)
    con.sql(f"""
        CREATE TEMP TABLE diseases AS
        SELECT id AS diseaseId, name AS disease_name,
               list_concat(coalesce(ancestors, []), coalesce(descendants, [])) AS related
        FROM '{OT}/disease/*.parquet' WHERE id IN (SELECT diseaseId FROM curated)
    """)
    load_abstracts(con, "pubmed", "SELECT pmid FROM curated")


def load_abstracts(con: duckdb.DuckDBPyConnection, table: str, pmid_sql: str) -> None:
    """Title + abstract for the PMIDs selected by pmid_sql. Duplicate
    record_ids exist (same abstract ingested twice); keep one per PMID by
    record_hash."""
    con.sql(f"""
        CREATE TEMP TABLE {table} AS
        SELECT pmid, title, abstract FROM (
            SELECT replace(record_id, 'PMID:', '') AS pmid,
                   coalesce(trim(title), '') AS title,
                   trim(substr(text, position('{ABSTRACT_MARK}' IN text)
                                     + {len(ABSTRACT_MARK)})) AS abstract,
                   row_number() OVER (PARTITION BY record_id ORDER BY record_hash) AS rn
            FROM read_parquet('{RECORDS}')
            WHERE record_type = 'pubmed_abstract'
              AND position('{ABSTRACT_MARK}' IN text) > 0
              AND replace(record_id, 'PMID:', '') IN ({pmid_sql})
        ) WHERE rn = 1 AND abstract <> ''
    """)


def write_gold_table(con: duckdb.DuckDBPyConnection) -> None:
    GOLD_OUT.parent.mkdir(parents=True, exist_ok=True)
    con.sql(f"""
        COPY (
            SELECT c.targetId, g.symbol, c.diseaseId, d.disease_name, c.pmid, c.sources,
                   c.pmid IN (SELECT pmid FROM pubmed) AS has_local_abstract
            FROM curated c JOIN genes g USING (targetId) JOIN diseases d USING (diseaseId)
            ORDER BY c.targetId, c.diseaseId, c.pmid::BIGINT
        ) TO '{GOLD_OUT}' (FORMAT parquet)
    """)


def main() -> None:
    con = duckdb.connect()
    load_source_tables(con)
    write_gold_table(con)

    Pair = tuple[str, str]
    gold: dict[Pair, set[str]] = {}
    for t, d, p in con.sql(
        "SELECT DISTINCT targetId, diseaseId, pmid FROM curated JOIN pubmed USING (pmid)"
    ).fetchall():
        gold.setdefault((t, d), set()).add(p)
    # Any link (curated, with or without local abstract, or silver) per pair.
    linked: dict[Pair, set[str]] = {}
    for t, d, p in con.sql(
        "SELECT targetId, diseaseId, pmid FROM curated UNION SELECT * FROM silver"
    ).fetchall():
        linked.setdefault((t, d), set()).add(p)

    genes = {t: (s, n) for t, s, n in con.sql("SELECT * FROM genes").fetchall()}
    diseases = {d: (n, set(r)) for d, n, r in con.sql("SELECT * FROM diseases").fetchall()}

    by_gene: dict[str, list[Pair]] = {}
    for t, d in linked:
        by_gene.setdefault(t, []).append((t, d))

    def hard_pool(q: Pair) -> set[str]:
        """PMIDs linked to q's gene with a disease that is not q's disease or
        its ancestor/descendant, and not linked to q itself."""
        t, d = q
        related = diseases[d][1] | {d}
        pool = {p for o in by_gene[t] if o[1] not in related for p in linked[o]}
        return pool - linked[q]

    # 1. Candidate pairs: named gene and disease, 2-15 gold PMIDs with
    # abstracts, enough same-gene other-disease links for hard negatives.
    candidates = sorted(
        pair for pair, pmids in gold.items()
        if MIN_GOLD <= len(pmids) <= MAX_GOLD
        and pair[0] in genes and genes[pair[0]][0]
        and pair[1] in diseases and diseases[pair[1]][0]
        and len(hard_pool(pair)) >= MIN_HARD_POOL
    )
    print(f"candidate pairs: {len(candidates)}")

    def clashes(a: Pair, b: Pair) -> bool:
        """True if a doc gold for one pair is linked to the other but not its gold."""
        return bool((gold[a] & linked.get(b, set())) - gold[b]) or bool(
            (gold[b] & linked.get(a, set())) - gold[a]
        )

    # Seeded shuffle, greedy accept under a paced gold budget. At most one
    # query per gene and per disease (diversity; avoids nested-disease overlap).
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    selected: list[Pair] = []
    gold_pmids: set[str] = set()
    for pair in candidates:
        if len(selected) == N_QUERIES:
            break
        if any(pair[0] == s[0] or pair[1] == s[1] for s in selected):
            continue
        union = gold_pmids | gold[pair]
        if len(union) > GOLD_BUDGET * (len(selected) + 1) / N_QUERIES:
            continue
        if any(clashes(pair, s) for s in selected):
            continue
        selected.append(pair)
        gold_pmids = union
    if len(selected) < N_QUERIES:
        raise SystemExit(f"only {len(selected)} pairs fit GOLD_BUDGET={GOLD_BUDGET}")
    selected.sort()

    # 2. Corpus: all golds + HARD_PER_QUERY same-gene negatives per query,
    # topped up with random curated abstracts. A doc is admissible only if it
    # is not linked to any query.
    forbidden = set().union(*(linked[q] for q in selected))
    pools = {q: hard_pool(q) - forbidden for q in selected}
    con.sql("CREATE TEMP TABLE hard_pmids (pmid VARCHAR)")
    con.executemany("INSERT INTO hard_pmids VALUES (?)", [(p,) for p in set().union(*pools.values())])
    load_abstracts(con, "hard_abstracts", "SELECT pmid FROM hard_pmids")
    docs = {
        pmid: (title, abstract)
        for table in ("pubmed", "hard_abstracts")
        for pmid, title, abstract in con.sql(f"SELECT pmid, title, abstract FROM {table}").fetchall()
    }

    rng = random.Random(SEED)
    hard: set[str] = set()
    for q in selected:
        pool = sorted((pools[q] & docs.keys()) - hard, key=int)
        if len(pool) < HARD_PER_QUERY:
            raise SystemExit(f"{q}: only {len(pool)} same-gene negatives with abstracts")
        hard |= set(rng.sample(pool, HARD_PER_QUERY))
    rest_pool = sorted(
        {p for pmids in gold.values() for p in pmids} - forbidden - gold_pmids - hard, key=int
    )
    filler = rng.sample(rest_pool, CORPUS_SIZE - len(gold_pmids) - len(hard))
    corpus_pmids = sorted(gold_pmids | hard | set(filler), key=int)
    corpus = Dataset.from_dict({
        "_id": corpus_pmids,
        "title": [docs[p][0] for p in corpus_pmids],
        "text": [
            f"{docs[p][0]}\n\n{docs[p][1]}" if docs[p][0] else docs[p][1]
            for p in corpus_pmids
        ],
    })
    qid = {q: f"{q[0]}|{q[1]}" for q in selected}
    queries = Dataset.from_dict({
        "_id": [qid[q] for q in selected],
        "text": [
            f"{genes[t][0]} ({genes[t][1]}) in {diseases[d][0]}" for t, d in selected
        ],
    })
    qrel_rows = [(qid[q], p) for q in selected for p in sorted(gold[q], key=int)]
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
    assert all(MIN_GOLD <= len(gold[q]) <= MAX_GOLD for q in selected)
    assert all(corpus["text"]) and all(queries["text"]), "empty text field"
    for q in selected:  # silver guard
        assert not (corpus_ids & linked[q]) - gold[q], f"unlabelled linked doc for {q}"

    # 4. Provenance stats for the manifest (gold pairs only).
    con.sql("CREATE TEMP TABLE sel (targetId VARCHAR, diseaseId VARCHAR)")
    con.executemany("INSERT INTO sel VALUES (?, ?)", selected)
    source_cov = dict(con.sql(
        "SELECT s, count(*) FROM (SELECT unnest(sources) AS s FROM curated "
        "JOIN sel USING (targetId, diseaseId) JOIN pubmed USING (pmid)) "
        "GROUP BY 1 ORDER BY 2 DESC, 1"
    ).fetchall())
    mean_golds = len(qrel_rows) / len(selected)

    # 5. Save.
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    DatasetDict({"corpus": corpus, "queries": queries, "qrels": qrels}).save_to_disk(OUT_DIR)
    manifest = {
        "name": "GeneDiseaseRetrieval",
        "category": "Retrieval",
        "sizes": {"corpus": len(corpus), "queries": len(queries), "qrels": len(qrels)},
        "provenance": {
            "gold_links": f"{OT}/evidence_{{{','.join(CURATED_SOURCES)}}}",
            "silver_links": f"{OT}/evidence_europepmc (used only to exclude unlabelled positives)",
            "names": f"{OT}/target, {OT}/disease",
            "texts": RECORDS,
            "source_counts_gold_pairs": source_cov,
            "full_gold_table": str(GOLD_OUT.relative_to(ROOT)) + " (untracked; rebuilt by this script)",
        },
        "protocol": {
            "seed": SEED,
            "query_text": "'{SYMBOL} ({approved gene name}) in {OT disease name}'",
            "query_id": "'{Ensembl gene}|{disease id}'",
            "doc_text": "'{title}\\n\\n{abstract}' if title else abstract; text is "
                        "self-contained -- do NOT re-prepend the title column",
            "candidate_pairs": f"{MIN_GOLD}-{MAX_GOLD} distinct curated PMIDs with abstracts "
                               "in the records table",
            "selection": f"seeded shuffle of {len(candidates)} candidates, greedily accept "
                         "pairs with an unused gene and disease, no silver clash with "
                         f"accepted pairs, under a paced gold budget (union <= {GOLD_BUDGET} * "
                         f"(accepted+1)/{N_QUERIES}), stop at {N_QUERIES}",
            "distractors": f"{HARD_PER_QUERY} hard negatives per query ({len(hard)} total): "
                           "abstracts linked (curated or Europe PMC) to the query gene with a "
                           "disease that is not the query disease or its ancestor/descendant; "
                           f"{len(filler)} random curated abstracts; sampled with seed 42",
            "silver_guard": "no corpus doc is linked (curated or Europe PMC) to a query "
                            "unless it is in that query's qrels",
            "qrels": "score=1 for every curated (pair, PMID) link; all gold docs are in the corpus",
            "mean_golds_per_query": round(mean_golds, 3),
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    n_gold_rows, n_pairs = con.sql(
        f"SELECT count(*), count(DISTINCT (targetId, diseaseId)) FROM '{GOLD_OUT}'"
    ).fetchone()
    print(f"\nGold table: {GOLD_OUT} ({n_gold_rows} rows, {n_pairs} pairs)")
    print(f"Saved to {OUT_DIR}")
    print(f"corpus:  {len(corpus)} docs ({len(gold_pmids)} gold, {len(hard)} hard, "
          f"{len(filler)} filler)")
    print(f"queries: {len(queries)}")
    print(f"qrels:   {len(qrels)} (mean golds/query {mean_golds:.2f}, "
          f"min {min(len(gold[q]) for q in selected)}, max {max(len(gold[q]) for q in selected)})")
    print(f"sources (gold pairs): {source_cov}")
    for t in queries["text"][:5]:
        print("  e.g.", t)


if __name__ == "__main__":
    main()
