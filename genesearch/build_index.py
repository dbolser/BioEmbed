"""One-off build of the genesearch index (writes to ./index/).

Outputs (all sorted for fast lookup):
  abstracts.parquet      pmid, title, abstract, year, journal          (sorted by pmid)
  gene_pmids.parquet     ncbi, pmid, mentions                          (sorted by ncbi, pmid)
  disease_pmids.parquet  pmid, disease_id, mentions                    (sorted by pmid)
  disease_names.parquet  disease_id, name, ot_name, n_papers  (PubTator MeSH/OMIM ids + Open Targets ids)
  genes.parquet          HGNC lookup table + n_papers
  curated.parquet        Open Targets curated gene-disease-PMID links (copy of the gold file)
  epmc.parquet           Open Targets Europe PMC text-mined gene-disease-PMID links (sorted by ensembl)
  abstracts.arrow        uncompressed Arrow IPC copy of abstracts + per-paper disease tags, memory-mapped
                         by the app for random access by row (parquet row groups can't do that fast)

Usage: uv run python build_index.py [step ...]   (default: all steps)
"""

import sys
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

RECORDS = "/outsee5/home/danbolser/Work/IUK-IPF/data/embeddings-catalogue/records/*.parquet"
PUBTATOR = "/data/public/PubTator3/2026-10-06"
HGNC = "/data/public/HGNC/2026-10-14/hgnc_complete_set.txt"
CURATED = "/home/danbolser/Work/BioEmbed/data/gold/ot_curated_gene_disease_pmid.parquet"
OT = "/data/public/OpenTargets/26.06"

OUT = Path(__file__).resolve().parent / "index"
PQ = "FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 100000"


def connect() -> duckdb.DuckDBPyConnection:
    (OUT / ".tmp").mkdir(parents=True, exist_ok=True)
    return duckdb.connect(config={
        "threads": 112,
        "memory_limit": "400GB",
        "temp_directory": str(OUT / ".tmp"),
        "preserve_insertion_order": True,
    })


def pubtator_csv(name: str) -> str:
    return (f"read_csv('{PUBTATOR}/{name}', delim='\\t', header=false, quote='', escape='', "
            "columns={'pmid':'BIGINT','type':'VARCHAR','ids':'VARCHAR','mentions':'VARCHAR','resource':'VARCHAR'})")


def abstracts(con):
    # Some PMIDs were ingested twice; keep one per PMID by record_hash (repo convention).
    con.sql(f"""
        COPY (
          SELECT CAST(substr(record_id, 6) AS BIGINT) AS pmid,
                 title,
                 CASE WHEN position(chr(10) || 'Abstract: ' IN text) > 0  -- no marker: no abstract (not the text from char 11)
                      THEN substr(text, position(chr(10) || 'Abstract: ' IN text) + 11) END AS abstract,
                 try_cast(json_extract_string(metadata, '$.publication_year') AS SMALLINT) AS year,
                 json_extract_string(metadata, '$.journal') AS journal
          FROM (SELECT *, row_number() OVER (PARTITION BY record_id ORDER BY record_hash) AS rn
                FROM '{RECORDS}' WHERE record_type = 'pubmed_abstract')
          WHERE rn = 1
          ORDER BY pmid
        ) TO '{OUT}/abstracts.parquet' ({PQ})""")


def gene_pmids(con):
    # One row per (gene, pmid); multi-gene ids like '5595;5594' count for each gene. ~12M rows are
    # curated links (gene2pubmed, GeneRIF, CTD, MeSH) with no text mention: kept, mentions ''.
    con.sql(f"""
        COPY (
          SELECT CAST(ncbi AS BIGINT) AS ncbi, pmid,
                 coalesce(string_agg(DISTINCT m, '|' ORDER BY m) FILTER (WHERE m <> ''), '') AS mentions
          FROM (SELECT ncbi, pmid, unnest(string_split(coalesce(mentions, ''), '|')) AS m
                FROM (SELECT pmid, unnest(string_split(ids, ';')) AS ncbi, mentions
                      FROM {pubtator_csv('gene2pubtator3.gz')}))
          WHERE try_cast(ncbi AS BIGINT) IS NOT NULL
          GROUP BY ALL
          ORDER BY ncbi, pmid
        ) TO '{OUT}/gene_pmids.parquet' ({PQ})""")


def disease_pmids(con):
    con.sql(f"""
        COPY (
          SELECT pmid, disease_id,
                 coalesce(string_agg(DISTINCT m, '|' ORDER BY m) FILTER (WHERE m <> ''), '') AS mentions
          FROM (SELECT disease_id, pmid, unnest(string_split(coalesce(mentions, ''), '|')) AS m
                FROM (SELECT pmid, unnest(string_split(ids, ';')) AS disease_id, mentions
                      FROM {pubtator_csv('disease2pubtator3.gz')}))
          WHERE disease_id NOT IN ('', '-')
          GROUP BY ALL
          ORDER BY pmid, disease_id
        ) TO '{OUT}/disease_pmids.parquet' ({PQ})""")


def disease_names(con):
    # Name PubTator MeSH/OMIM ids by their Open Targets disease name (via xrefs) when PubTator
    # itself uses that name as a mention of the id (xrefs can be loose: C536801 'short telomere
    # syndrome' xrefs to Phelan-McDermid); otherwise by the most frequent mention.
    con.sql(f"""
        COPY (
          WITH dp AS (SELECT * FROM '{OUT}/disease_pmids.parquet'),
          c AS (SELECT disease_id, mention, count(*) AS n
                FROM (SELECT disease_id, lower(unnest(string_split(mentions, '|'))) AS mention
                      FROM dp WHERE mentions <> '') GROUP BY ALL),
          top AS (SELECT disease_id, arg_max(mention, n) AS name FROM c GROUP BY 1),
          ot AS (SELECT x AS disease_id, name FROM (SELECT unnest(dbXRefs) AS x, name FROM '{OT}/disease/*.parquet')
                 WHERE x LIKE 'MESH:%' OR x LIKE 'OMIM:%'),
          ot_ok AS (SELECT ot.disease_id, arg_max(ot.name, c.n) AS name
                    FROM ot JOIN c ON c.disease_id = ot.disease_id AND c.mention = lower(ot.name) GROUP BY 1),
          cnt AS (SELECT disease_id, count(*) AS n_papers FROM dp GROUP BY 1)
          SELECT cnt.disease_id, coalesce(ot_ok.name, top.name, cnt.disease_id) AS name,
                 ot_ok.name IS NOT NULL AS ot_name, n_papers
          FROM cnt LEFT JOIN top USING (disease_id) LEFT JOIN ot_ok USING (disease_id)
          UNION ALL
          SELECT id, name, true, NULL FROM '{OT}/disease/*.parquet'
          ORDER BY disease_id
        ) TO '{OUT}/disease_names.parquet' (FORMAT parquet, COMPRESSION zstd)""")


def genes(con):
    con.sql(f"""
        CREATE TEMP TABLE h AS SELECT * FROM read_csv('{HGNC}', delim='\\t', header=true, all_varchar=true)""")
    con.sql(f"""
        COPY (
          SELECT h.hgnc_id, h.symbol, h.name,
                 trim(h.alias_symbol, '"') AS alias_symbol, trim(h.alias_name, '"') AS alias_name,
                 trim(h.prev_symbol, '"') AS prev_symbol, trim(h.prev_name, '"') AS prev_name,
                 CAST(h.entrez_id AS BIGINT) AS entrez_id, h.ensembl_gene_id, h.locus_group, h.location,
                 coalesce(c.n_papers, 0) AS n_papers
          FROM h LEFT JOIN (SELECT ncbi, count(*) AS n_papers FROM '{OUT}/gene_pmids.parquet' GROUP BY 1) c
            ON c.ncbi = CAST(h.entrez_id AS BIGINT)
          WHERE h.status = 'Approved'
          ORDER BY h.symbol
        ) TO '{OUT}/genes.parquet' (FORMAT parquet, COMPRESSION zstd)""")


def opentargets(con):
    con.sql(f"""
        COPY (SELECT targetId AS ensembl, symbol, diseaseId AS disease_id, disease_name,
                     CAST(pmid AS BIGINT) AS pmid, sources
              FROM '{CURATED}' WHERE try_cast(pmid AS BIGINT) IS NOT NULL ORDER BY ensembl, pmid)
        TO '{OUT}/curated.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    con.sql(f"""
        COPY (SELECT DISTINCT targetId AS ensembl, diseaseId AS disease_id, CAST(p AS BIGINT) AS pmid
              FROM (SELECT targetId, diseaseId, unnest(literature) AS p FROM '{OT}/evidence_europepmc/*.parquet')
              WHERE try_cast(p AS BIGINT) IS NOT NULL
              ORDER BY ensembl, disease_id, pmid)
        TO '{OUT}/epmc.parquet' ({PQ})""")


def arrow(con):
    # Same rows/order as abstracts.parquet, plus the PubTator disease tags per paper, as an
    # uncompressed Arrow IPC file the app memory-maps: fetching N scattered papers touches
    # only their pages instead of decompressing every 100k-row parquet row group.
    rel = con.sql(f"""
        SELECT a.pmid, a.title, a.abstract, a.year, a.journal, d.diseases
        FROM '{OUT}/abstracts.parquet' a
        LEFT JOIN (SELECT pmid, list(struct_pack(id := disease_id, mentions := mentions) ORDER BY disease_id) AS diseases
                   FROM '{OUT}/disease_pmids.parquet' GROUP BY 1) d USING (pmid)
        ORDER BY a.pmid""")
    reader = rel.to_arrow_reader(1_000_000)
    tmp = OUT / "abstracts.arrow.tmp"
    with pa.OSFile(str(tmp), "wb") as sink, pa.ipc.new_file(sink, reader.schema) as w:
        for batch in reader:
            w.write_batch(batch)
    tmp.rename(OUT / "abstracts.arrow")


STEPS = {"abstracts": abstracts, "gene_pmids": gene_pmids, "disease_pmids": disease_pmids,
         "disease_names": disease_names, "genes": genes, "opentargets": opentargets, "arrow": arrow}


def main():
    steps = sys.argv[1:] or list(STEPS)
    con = connect()
    t_all = time.time()
    for name in steps:
        t = time.time()
        STEPS[name](con)
        print(f"{name}: {time.time() - t:.0f}s", flush=True)
    print(f"total: {time.time() - t_all:.0f}s")
    for f in sorted(OUT.glob("*.*")):
        if f.is_file():
            rows = pq.ParquetFile(f).metadata.num_rows if f.suffix == ".parquet" else ""
            print(f"  {f.name:24s} {f.stat().st_size / 1e9:7.2f} GB  {rows}")


if __name__ == "__main__":
    main()
