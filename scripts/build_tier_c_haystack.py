#!/usr/bin/env python
"""Build GeneDiseaseHaystack (Tier C): gene-disease -> PubMed-evidence retrieval
over the full local PubMed baseline (~28M abstracts).

Query  = "{SYMBOL} ({gene name}) in {disease name}" (as GeneDiseaseRetrieval).
Corpus = every PubMed abstract in the IUK-IPF records table (not copied;
         defined by the records table, see manifest).
Gold   = curated Open Targets PMIDs for the pair with a local abstract
         (data/gold/ot_curated_gene_disease_pmid.parquet, from
         build_tier_c_gene_disease.py).
Silver = Europe PMC text-mining PMIDs for the pair (via Open Targets), used
         only for the condensed metric (unlabelled silver dropped from runs).

Also writes the PubTator 3 candidate sets that the $0 database-lookup
baselines and the filter-then-rank pipelines use:
  gene_pmids   PMIDs PubTator tags with the query gene (NCBI Gene via HGNC)
  lookup       gene-tagged PMIDs also tagged with the query disease, its
               descendants, parents or any ancestor (MeSH/OMIM xrefs from
               the OT disease ontology); `tier` is the narrowest that hits
and docs_gene.parquet (pmid, text) for the union of gene_pmids, so models can
rank the gene-filtered haystack without the 50 GB records table.

Deterministic: SEED=42; candidate pairs sorted before sampling.
Output: data/haystack/GeneDiseaseHaystack/ (untracked; rebuilt each run).

Run:  uv run python scripts/build_tier_c_haystack.py
"""

import json
import random
import shutil
from pathlib import Path

import duckdb

SEED = 42
N_QUERIES = 500
MAX_GOLD = 50

IUK = "/outsee5/home/danbolser/Work/IUK-IPF/data"
OT = f"{IUK}/sources/opentargets/26.06"
PT = f"{IUK}/sources/pubtator3/2026-10-06"
HGNC = f"{IUK}/sources/hgnc/2026-10-14/hgnc_complete_set.txt"
RECORDS = f"{IUK}/embeddings-catalogue/records/*.parquet"
ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "gold" / "ot_curated_gene_disease_pmid.parquet"
OUT_DIR = ROOT / "data" / "haystack" / "GeneDiseaseHaystack"

ABSTRACT_MARK = "\nAbstract: "

# Disease expansion tiers, narrowest first.
TIERS = {
    "exact": "[d.id]",
    "descendants": "coalesce(d.descendants, [])",
    "parents": "coalesce(d.parents, [])",
    "ancestors": "coalesce(d.ancestors, [])",
}


def main() -> None:
    con = duckdb.connect()
    con.sql("SET threads=48")
    con.sql(f"""
        CREATE TABLE local AS
        SELECT DISTINCT replace(record_id, 'PMID:', '') AS pmid
        FROM read_parquet('{RECORDS}')
        WHERE record_type = 'pubmed_abstract'
    """)
    n_local = con.sql("SELECT count(*) FROM local").fetchone()[0]

    # 1. Queries: seeded sample of curated pairs with 1..MAX_GOLD local golds
    # and a gene that maps to NCBI Gene.
    con.sql(f"""
        CREATE TABLE gmap AS
        SELECT DISTINCT o.id AS targetId, h.entrez_id AS ncbi
        FROM '{OT}/target/*.parquet' o, unnest(o.dbXrefs) x,
             read_csv('{HGNC}', delim='\t', all_varchar=true) h
        WHERE x.unnest.source = 'HGNC' AND h.hgnc_id = 'HGNC:' || x.unnest.id
          AND h.entrez_id IS NOT NULL
    """)
    con.sql(f"""
        CREATE TABLE gold_all AS
        SELECT targetId, symbol, diseaseId, disease_name, pmid FROM '{GOLD}'
        WHERE has_local_abstract
    """)
    pairs = con.sql(f"""
        SELECT targetId, diseaseId FROM gold_all
        WHERE targetId IN (SELECT targetId FROM gmap)
        GROUP BY ALL HAVING count(*) <= {MAX_GOLD}
        ORDER BY ALL
    """).fetchall()
    selected = sorted(random.Random(SEED).sample(pairs, N_QUERIES))
    con.sql("CREATE TABLE sel (targetId VARCHAR, diseaseId VARCHAR)")
    con.executemany("INSERT INTO sel VALUES (?, ?)", selected)
    con.sql(f"""
        CREATE TABLE queries AS
        SELECT s.targetId || '|' || s.diseaseId AS qid, s.targetId, s.diseaseId,
               t.approvedSymbol || ' (' || t.approvedName || ') in ' || d.name AS text
        FROM sel s
        JOIN '{OT}/target/*.parquet' t ON t.id = s.targetId
        JOIN '{OT}/disease/*.parquet' d ON d.id = s.diseaseId
        ORDER BY qid
    """)
    assert con.sql("SELECT count(*) FROM queries").fetchone()[0] == N_QUERIES

    con.sql("""
        CREATE TABLE qrels AS
        SELECT DISTINCT q.qid, g.pmid FROM queries q JOIN gold_all g USING (targetId, diseaseId)
    """)
    con.sql(f"""
        CREATE TABLE silver AS
        SELECT DISTINCT q.qid, e.pmid
        FROM queries q
        JOIN (SELECT targetId, diseaseId, unnest(literature) AS pmid
              FROM '{OT}/evidence_europepmc/*.parquet'
              WHERE targetId IN (SELECT targetId FROM sel)) e USING (targetId, diseaseId)
        WHERE e.pmid IN (SELECT pmid FROM local)
    """)

    # 2. PubTator candidate sets, restricted to local abstracts.
    con.sql(f"""
        CREATE TABLE gene_pmids AS
        SELECT DISTINCT q.qid, p.pmid
        FROM queries q JOIN gmap USING (targetId)
        JOIN (SELECT column0 AS pmid, unnest(string_split(column2, ';')) AS ncbi
              FROM read_csv('{PT}/gene2pubtator3.gz', delim='\t', header=false,
                            all_varchar=true, quote='')) p USING (ncbi)
        WHERE p.pmid IN (SELECT pmid FROM local)
    """)
    tier_rows = " UNION ALL ".join(
        f"SELECT d.id AS diseaseId, unnest({rel}) AS rel, {i} AS tier "
        f"FROM '{OT}/disease/*.parquet' d WHERE d.id IN (SELECT diseaseId FROM sel)"
        for i, rel in enumerate(TIERS.values())
    )
    con.sql(f"""
        CREATE TABLE dmap AS
        SELECT t.diseaseId, x AS xref, min(t.tier) AS tier
        FROM ({tier_rows}) t
        JOIN '{OT}/disease/*.parquet' r ON r.id = t.rel, unnest(r.dbXRefs) u(x)
        WHERE x LIKE 'MESH:%' OR x LIKE 'OMIM:%'
        GROUP BY ALL
    """)
    tier_names = "[" + ", ".join(f"'{t}'" for t in TIERS) + "]"
    con.sql(f"""
        CREATE TABLE lookup AS
        SELECT g.qid, g.pmid, {tier_names}[min(m.tier) + 1] AS tier
        FROM gene_pmids g JOIN queries q USING (qid) JOIN dmap m USING (diseaseId)
        JOIN (SELECT column0 AS pmid, unnest(string_split(column2, ';')) AS xref
              FROM read_csv('{PT}/disease2pubtator3.gz', delim='\t', header=false,
                            all_varchar=true, quote='')) p
          ON p.pmid = g.pmid AND p.xref = m.xref
        GROUP BY ALL
    """)

    # 3. Save. Doc text as GeneDiseaseRetrieval; one record per PMID by
    # record_hash.
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    for t in ["queries", "qrels", "silver", "gene_pmids", "lookup"]:
        con.sql(f"COPY (SELECT * FROM {t} ORDER BY ALL) TO '{OUT_DIR / t}.parquet' (FORMAT parquet)")
    con.sql(f"""
        COPY (
            SELECT pmid, CASE WHEN title <> '' THEN title || chr(10) || chr(10) || abstract
                              ELSE abstract END AS text
            FROM (
                SELECT replace(record_id, 'PMID:', '') AS pmid,
                       coalesce(trim(title), '') AS title,
                       trim(substr(text, position('{ABSTRACT_MARK}' IN text)
                                         + {len(ABSTRACT_MARK)})) AS abstract,
                       row_number() OVER (PARTITION BY record_id ORDER BY record_hash) AS rn
                FROM read_parquet('{RECORDS}')
                WHERE record_type = 'pubmed_abstract'
                  AND replace(record_id, 'PMID:', '') IN (SELECT DISTINCT pmid FROM gene_pmids)
            ) WHERE rn = 1
            ORDER BY pmid::BIGINT
        ) TO '{OUT_DIR / "docs_gene.parquet"}' (FORMAT parquet, COMPRESSION zstd)
    """)

    # 4. Stats.
    def recall(cands: str) -> float:
        return con.sql(f"""
            SELECT avg(coalesce(h, 0) / n) FROM
              (SELECT qid, count(*) n FROM qrels GROUP BY 1) r
              LEFT JOIN (SELECT qid, count(*) h FROM qrels JOIN ({cands}) USING (qid, pmid)
                         GROUP BY 1) USING (qid)
        """).fetchone()[0]

    def set_size(cands: str) -> float:
        return con.sql(f"SELECT median(n) FROM (SELECT qid, count(*) n FROM ({cands}) GROUP BY 1)").fetchone()[0]

    stats = {"local_abstracts": n_local}
    for name, cands in [("gene_pmids", "SELECT qid, pmid FROM gene_pmids")] + [
        (f"lookup<= {t}", f"SELECT qid, pmid FROM lookup WHERE list_position({tier_names}, tier) <= {i + 1}")
        for i, t in enumerate(TIERS)
    ]:
        stats[name] = {"recall": round(recall(cands), 4), "median_set": set_size(cands)}
    stats["gene_pmids_union"] = con.sql("SELECT count(DISTINCT pmid) FROM gene_pmids").fetchone()[0]
    stats["mean_golds_per_query"] = round(
        con.sql("SELECT count(*) FROM qrels").fetchone()[0] / N_QUERIES, 3)
    stats["mean_silver_per_query"] = round(
        con.sql("SELECT count(*) FROM silver").fetchone()[0] / N_QUERIES, 3)

    manifest = {
        "name": "GeneDiseaseHaystack",
        "category": "Retrieval",
        "provenance": {
            "gold": str(GOLD.relative_to(ROOT)),
            "silver": f"{OT}/evidence_europepmc",
            "corpus": f"{RECORDS} where record_type='pubmed_abstract' (distinct PMID)",
            "pubtator": PT,
            "gene_map": f"OT target HGNC xref -> {HGNC} entrez_id",
            "disease_map": "OT disease dbXRefs MESH:/OMIM:, expanded by tier "
                           "(exact, descendants, parents, ancestors)",
        },
        "protocol": {
            "seed": SEED,
            "query_text": "'{SYMBOL} ({approved gene name}) in {OT disease name}'",
            "query_id": "'{Ensembl gene}|{disease id}'",
            "doc_text": "'{title}\\n\\n{abstract}' as GeneDiseaseRetrieval",
            "selection": f"seeded sample of {N_QUERIES} from {len(pairs)} curated pairs with "
                         f"1-{MAX_GOLD} local golds and an NCBI-mapped gene",
            "metrics": "recall@k over gold; 'condensed' variant drops silver-not-gold "
                       "PMIDs from a run before scoring",
        },
        "stats": stats,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
