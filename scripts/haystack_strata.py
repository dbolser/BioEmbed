#!/usr/bin/env python
"""Strata labels and a clean-gold flag for the frozen GeneDiseaseHaystack test
set (docs/PREREG.md: strata are labels only; queries and qrels unchanged).

strata.parquet, one row per query:
  disease_prefix     ontology of the query disease id (MONDO, Orphanet, HP, EFO)
  main_source        curated source citing the most golds (ties: alphabetical)
  n_gold, gold_band  '1', '2-3', '4-9', '10+'
  gene_set_size, gene_set_quartile
                     PubTator gene-filtered candidates (gene_pmids); 1 = smallest
  clingen            ClinGen gene-disease validity class (strongest if several),
                     matched on HGNC id and MONDO id: OT MONDO_ ids directly,
                     other OT ids via MONDO SSSOM exactMatch; null if none
  max_gold_year      latest publication_year among the golds (records metadata)

qrels_clean.parquet, one row per gold (qid, pmid):
  gene_tag           PubTator 3: 'query_gene', 'other_genes_only' or 'none'
  n_genes_cited      distinct genes the OT curated gold cites this PMID for
  sources            curated sources citing it for the pair
  ge_green           genomics_england cites it for the pair with green
                     confidence (null if genomics_england does not cite it)
  clean              gene_tag = 'query_gene' AND n_genes_cited < 10 AND, if
                     genomics_england cites it for the pair, a row is green
  clean_lenient      as clean, but amber genomics_england only fails a gold
                     that no other curated source cites

Also prints the gene-filter loss (golds outside gene_pmids) split by gene_tag.

Run:    uv run python scripts/haystack_strata.py
Output: data/haystack/GeneDiseaseHaystack/{strata,qrels_clean}.parquet (only these)
"""

from pathlib import Path

import duckdb

MAX_GENES_CITED = 10
CLINGEN_ORDER = ["Definitive", "Strong", "Moderate", "Limited", "Disputed", "Refuted",
                 "No Known Disease Relationship"]
GOLD_BANDS = [(1, 1, "1"), (2, 3, "2-3"), (4, 9, "4-9"), (10, None, "10+")]

IUK = "/outsee5/home/danbolser/Work/IUK-IPF/data"
OT = f"{IUK}/sources/opentargets/26.06"
PT = f"{IUK}/sources/pubtator3/2026-10-06"
HGNC = f"{IUK}/sources/hgnc/2026-10-14/hgnc_complete_set.txt"
CLINGEN = f"{IUK}/sources/clingen/gene-validity-2026-10-06.csv"
MONDO_SSSOM = f"{IUK}/sources/mondo/2026-10-14/mondo.sssom.tsv"
RECORDS = f"{IUK}/embeddings-catalogue/records/*.parquet"
ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "gold" / "ot_curated_gene_disease_pmid.parquet"
HAYSTACK = ROOT / "data" / "haystack" / "GeneDiseaseHaystack"


def main() -> None:
    con = duckdb.connect()
    con.sql("SET threads=48")
    con.sql(f"CREATE TABLE queries AS SELECT qid, targetId, diseaseId FROM '{HAYSTACK}/queries.parquet'")
    con.sql(f"""
        CREATE TABLE qrels AS
        SELECT r.qid, r.pmid, q.targetId, q.diseaseId
        FROM '{HAYSTACK}/qrels.parquet' r JOIN queries q USING (qid)
    """)
    con.sql(f"""
        CREATE TABLE gmap AS
        SELECT DISTINCT o.id AS targetId, h.entrez_id AS ncbi, h.hgnc_id
        FROM '{OT}/target/*.parquet' o, unnest(o.dbXrefs) x,
             read_csv('{HGNC}', delim='\t', all_varchar=true) h
        WHERE x.unnest.source = 'HGNC' AND h.hgnc_id = 'HGNC:' || x.unnest.id
    """)

    # 1. Per-gold flags.
    con.sql(f"""
        CREATE TABLE tags AS
        SELECT DISTINCT column0 AS pmid, unnest(string_split(column2, ';')) AS ncbi
        FROM read_csv('{PT}/gene2pubtator3.gz', delim='\t', header=false,
                      all_varchar=true, quote='')
        WHERE column0 IN (SELECT pmid FROM qrels)
    """)
    con.sql(f"""
        CREATE TABLE ge AS
        SELECT targetId, diseaseId, pmid, bool_or(confidence = 'green') AS ge_green
        FROM (SELECT targetId, diseaseId, confidence, unnest(literature) AS pmid
              FROM '{OT}/evidence_genomics_england/*.parquet' WHERE literature IS NOT NULL)
        GROUP BY ALL
    """)
    con.sql(f"""
        CREATE TABLE qrels_clean AS
        WITH f AS (
            SELECT r.qid, r.pmid,
                   CASE WHEN EXISTS (SELECT 1 FROM tags t JOIN gmap m USING (ncbi)
                                     WHERE t.pmid = r.pmid AND m.targetId = r.targetId)
                        THEN 'query_gene'
                        WHEN EXISTS (SELECT 1 FROM tags t WHERE t.pmid = r.pmid)
                        THEN 'other_genes_only'
                        ELSE 'none' END AS gene_tag,
                   c.n_genes_cited, g.sources,
                   CASE WHEN list_contains(g.sources, 'genomics_england')
                        THEN coalesce(ge.ge_green, false) END AS ge_green
            FROM qrels r
            JOIN (SELECT pmid, count(DISTINCT targetId) AS n_genes_cited
                  FROM '{GOLD}' GROUP BY 1) c USING (pmid)
            JOIN '{GOLD}' g USING (targetId, diseaseId, pmid)
            LEFT JOIN ge USING (targetId, diseaseId, pmid)
        )
        SELECT *, gene_tag = 'query_gene' AND n_genes_cited < {MAX_GENES_CITED}
                  AND coalesce(ge_green, true) AS clean,
               gene_tag = 'query_gene' AND n_genes_cited < {MAX_GENES_CITED}
                  AND (len(sources) > 1 OR coalesce(ge_green, true)) AS clean_lenient
        FROM f
    """)
    assert con.sql("SELECT count(*) FROM qrels_clean").fetchone() == \
        con.sql("SELECT count(*) FROM qrels").fetchone(), "gold rows lost in the join"

    # 2. Per-query strata.
    bands = " ".join(f"WHEN n_gold BETWEEN {lo} AND {hi or 10**6} THEN '{name}'"
                     for lo, hi, name in GOLD_BANDS)
    order = "[" + ", ".join(f"'{c}'" for c in CLINGEN_ORDER) + "]"
    con.sql(f"""
        CREATE TABLE clingen AS
        SELECT "GENE ID (HGNC)" AS hgnc_id, "DISEASE ID (MONDO)" AS mondo, CLASSIFICATION AS cls
        FROM read_csv('{CLINGEN}', skip=4, header=true, all_varchar=true)
        WHERE "GENE ID (HGNC)" LIKE 'HGNC:%'
    """)
    con.sql(f"""
        CREATE TABLE mondo_of AS
        SELECT diseaseId, replace(diseaseId, '_', ':') AS mondo FROM queries
        WHERE diseaseId LIKE 'MONDO\\_%' ESCAPE '\\'
        UNION
        SELECT q.diseaseId, s.subject_id
        FROM queries q
        JOIN read_csv('{MONDO_SSSOM}', delim='\t', comment='#', header=true) s
          ON s.object_id = replace(q.diseaseId, '_', ':') AND s.predicate_id = 'skos:exactMatch'
    """)
    con.sql(f"""
        CREATE TABLE strata AS
        WITH n AS (SELECT qid, count(*) AS n_gold FROM qrels GROUP BY 1),
        src AS (
            SELECT qid, s AS main_source FROM (
                SELECT qid, s, row_number() OVER (PARTITION BY qid ORDER BY count(*) DESC, s) AS rn
                FROM (SELECT qid, unnest(sources) AS s FROM qrels_clean) GROUP BY qid, s
            ) WHERE rn = 1
        ),
        gs AS (
            SELECT q.qid, count(g.pmid) AS gene_set_size
            FROM queries q LEFT JOIN '{HAYSTACK}/gene_pmids.parquet' g USING (qid) GROUP BY 1
        ),
        cg AS (
            SELECT q.qid, {order}[min(list_position({order}, c.cls))] AS clingen
            FROM queries q JOIN gmap m USING (targetId) JOIN mondo_of d USING (diseaseId)
            JOIN clingen c ON c.hgnc_id = m.hgnc_id AND c.mondo = d.mondo
            GROUP BY 1
        ),
        yr AS (
            SELECT r.qid, max(y.year) AS max_gold_year
            FROM qrels r JOIN (
                SELECT replace(record_id, 'PMID:', '') AS pmid,
                       max(try_cast(json_extract_string(metadata, '$.publication_year') AS INTEGER)) AS year
                FROM read_parquet('{RECORDS}')
                WHERE record_type = 'pubmed_abstract'
                  AND replace(record_id, 'PMID:', '') IN (SELECT pmid FROM qrels)
                GROUP BY 1
            ) y USING (pmid)
            GROUP BY 1
        )
        SELECT q.qid, split_part(q.diseaseId, '_', 1) AS disease_prefix, src.main_source,
               n.n_gold, CASE {bands} END AS gold_band,
               gs.gene_set_size, ntile(4) OVER (ORDER BY gs.gene_set_size, q.qid) AS gene_set_quartile,
               cg.clingen, yr.max_gold_year
        FROM queries q JOIN n USING (qid) JOIN src USING (qid) JOIN gs USING (qid)
        LEFT JOIN cg USING (qid) LEFT JOIN yr USING (qid)
    """)
    assert con.sql("SELECT count(*) FROM strata").fetchone()[0] == \
        con.sql("SELECT count(*) FROM queries").fetchone()[0]

    for t in ("strata", "qrels_clean"):
        con.sql(f"COPY (SELECT * FROM {t} ORDER BY ALL) TO '{HAYSTACK / t}.parquet' (FORMAT parquet)")
        print(f"-> {(HAYSTACK / t).relative_to(ROOT)}.parquet")

    # 3. Report.
    for col in ("disease_prefix", "main_source", "gold_band", "gene_set_quartile", "clingen"):
        print(f"\n{col}: " + ", ".join(f"{k}={v}" for k, v in con.sql(
            f"SELECT {col}, count(*) FROM strata GROUP BY 1 ORDER BY 2 DESC").fetchall()))
    print("gene_set_size by quartile (min-max): " + ", ".join(
        f"Q{q} {lo}-{hi}" for q, lo, hi in con.sql(
            "SELECT gene_set_quartile, min(gene_set_size), max(gene_set_size) FROM strata "
            "GROUP BY 1 ORDER BY 1").fetchall()))
    print("max_gold_year: " + ", ".join(f"{k}={v}" for k, v in con.sql("""
        SELECT CASE WHEN max_gold_year IS NULL THEN 'unknown'
                    WHEN max_gold_year < 2000 THEN '<2000'
                    ELSE (max_gold_year // 5 * 5)::VARCHAR || '-' || (max_gold_year // 5 * 5 + 4)::VARCHAR END,
               count(*) FROM strata GROUP BY 1 ORDER BY 1""").fetchall()))

    n = con.sql("SELECT count(*) FROM qrels_clean").fetchone()[0]
    for col in ("clean", "clean_lenient"):
        n_clean, n_q = con.sql(f"SELECT count(*) FILTER (WHERE {col}), "
                               f"count(DISTINCT qid) FILTER (WHERE {col}) FROM qrels_clean").fetchone()
        print(f"\nqrels_clean.{col}: {n_clean}/{n} golds ({n_clean / n:.1%}); "
              f"queries with >=1 such gold: {n_q}")
    print("  failing each test: " + ", ".join(f"{k}={v}" for k, v in con.sql(f"""
        SELECT 'gene_not_tagged', count(*) FILTER (WHERE gene_tag <> 'query_gene') FROM qrels_clean
        UNION ALL SELECT 'cited_for_>={MAX_GENES_CITED}_genes',
               count(*) FILTER (WHERE n_genes_cited >= {MAX_GENES_CITED}) FROM qrels_clean
        UNION ALL SELECT 'ge_not_green (clean)', count(*) FILTER (WHERE NOT ge_green) FROM qrels_clean
        UNION ALL SELECT 'ge_amber_only (clean_lenient)',
               count(*) FILTER (WHERE len(sources) = 1 AND NOT ge_green) FROM qrels_clean
    """).fetchall()))
    lost = con.sql(f"""
        SELECT gene_tag, count(*) FROM qrels_clean c
        WHERE NOT EXISTS (SELECT 1 FROM '{HAYSTACK}/gene_pmids.parquet' g
                          WHERE g.qid = c.qid AND g.pmid = c.pmid)
        GROUP BY 1 ORDER BY 2 DESC
    """).fetchall()
    n_lost = sum(v for _, v in lost)
    print(f"\ngene-filter loss: {n_lost}/{n} golds ({n_lost / n:.1%}) outside gene_pmids: "
          + ", ".join(f"{k}={v} ({v / n_lost:.0%})" for k, v in lost))


if __name__ == "__main__":
    main()
