"""Build items_demo.parquet: 3 gene-disease rounds x 10 abstracts from the frozen test haystack.

Each round mixes up to 3 gold PMIDs (qrels) with the top Qwen3-Embedding-0.6B hits.
`stratum` records where each doc came from (hidden in the UI), e.g. 'gold', 'gold+qwen06@4',
'silver+qwen06@2', 'qwen06@1'. A few items per round are marked is_repeat.

Gene aliases (HGNC alias + previous symbols) are cleaned so highlighting doesn't mark the wrong
thing as the gene. Dropped: aliases of 2 characters or fewer (MT), aliases that are an approved
symbol (AR for AKR1B1) or shared by two or more genes (MCT1), and aliases equal to any Open
Targets disease name or synonym (ALS for SOD1, CF for CFTR). The real builder should do the same.

Run: cd labeller && uv run python make_demo_items.py
"""

import argparse
import random
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent
HAY = HERE.parent / "data/haystack/GeneDiseaseHaystack"
RUN = HAY / "runs/Qwen__Qwen3-Embedding-0.6B__gene.parquet"
HGNC = Path("/outsee5/home/danbolser/Work/IUK-IPF/data/sources/hgnc/2026-10-14/hgnc_complete_set.txt")
OT = Path("/data/public/OpenTargets/26.06")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--per-round", type=int, default=10)
    ap.add_argument("--max-gold", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=2, help="items per round marked is_repeat")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=HERE / "items_demo.parquet")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    con = duckdb.connect()

    qids = [r[0] for r in con.sql(f"""
        SELECT qid FROM '{HAY}/strata.parquet' WHERE n_gold >= 2 ORDER BY qid""").fetchall()]
    qids = sorted(rng.sample(qids, a.rounds))
    con.execute("CREATE TEMP TABLE pick AS SELECT unnest(?::VARCHAR[]) AS qid", [qids])

    gold = con.sql(f"SELECT qid, pmid FROM '{HAY}/qrels.parquet' JOIN pick USING (qid) ORDER BY qid, pmid").fetchall()
    silver = set(con.sql(f"SELECT qid, pmid FROM '{HAY}/silver.parquet' JOIN pick USING (qid)").fetchall())
    run = con.sql(f"SELECT qid, pmid, rank FROM '{RUN}' JOIN pick USING (qid) ORDER BY qid, rank").fetchall()
    rank = {(q, p): r for q, p, r in run}

    chosen = []  # (qid, pmid, stratum)
    for q in qids:
        golds = [p for qq, p in gold if qq == q]
        docs = rng.sample(golds, min(a.max_gold, len(golds)))
        docs += [p for qq, p, _ in run if qq == q and p not in docs][: a.per_round - len(docs)]
        for p in docs:
            tag = "gold" if p in golds else "silver" if (q, p) in silver else ""
            r = rank.get((q, p))
            chosen.append((q, p, "+".join(filter(None, [tag, r and f"qwen06@{r}"]))))
    con.execute("CREATE TEMP TABLE chosen (qid VARCHAR, pmid VARCHAR, stratum VARCHAR)")
    con.executemany("INSERT INTO chosen VALUES (?, ?, ?)", chosen)

    df = con.sql(f"""
        WITH hgnc AS (
            SELECT hgnc_id, alias_symbol, prev_symbol
            FROM read_csv('{HGNC}', delim='\t', header=true, quote='"', all_varchar=true)),
        gene AS (
            SELECT t.id AS targetId, t.approvedSymbol AS gene_symbol, t.approvedName AS gene_name,
                   list_filter(list_distinct(
                       string_split(coalesce(h.alias_symbol, ''), '|')
                       || string_split(coalesce(h.prev_symbol, ''), '|')),
                       s -> s <> '' AND s <> t.approvedSymbol) AS gene_aliases
            FROM '{OT}/target/*.parquet' t
            LEFT JOIN hgnc h ON h.hgnc_id = 'HGNC:' || list_filter(t.dbXrefs, x -> x.source = 'HGNC')[1].id),
        docs AS (
            SELECT pmid, text FROM '{HAY}/docs_gene.parquet' WHERE pmid IN (SELECT pmid FROM chosen))
        SELECT c.qid || '|' || c.pmid AS item_id, c.qid AS round_id,
               g.gene_symbol, g.gene_name, g.gene_aliases,
               d.name AS disease_name, coalesce(d.exactSynonyms, []) AS disease_synonyms,
               c.pmid,
               split_part(x.text, chr(10) || chr(10), 1) AS title,
               substr(x.text, length(split_part(x.text, chr(10) || chr(10), 1)) + 3) AS abstract,
               c.stratum
        FROM chosen c
        JOIN '{HAY}/queries.parquet' q USING (qid)
        JOIN gene g USING (targetId)
        JOIN '{OT}/disease/disease.parquet' d ON d.id = q.diseaseId
        JOIN docs x USING (pmid)
        ORDER BY round_id, pmid""").df()

    other = {r[0] for r in con.sql(f"""
        WITH h AS (SELECT * FROM read_csv('{HGNC}', delim='\t', header=true, quote='"', all_varchar=true)),
        a AS (SELECT DISTINCT hgnc_id, unnest(string_split(coalesce(alias_symbol, '') || '|' ||
                                                           coalesce(prev_symbol, ''), '|')) AS s FROM h)
        SELECT symbol FROM h
        UNION SELECT s FROM a WHERE s <> '' GROUP BY s HAVING count(*) > 1""").fetchall()}
    dterms = {r[0] for r in con.sql(f"""
        SELECT lower(name) FROM '{OT}/disease/disease.parquet'
        UNION SELECT lower(unnest(exactSynonyms)) FROM '{OT}/disease/disease.parquet'""").fetchall()}
    before = df.gene_aliases.map(len).sum()
    df["gene_aliases"] = [[x for x in a if len(x) > 2 and x not in other and x.lower() not in dterms]
                          for a in df.gene_aliases]  # own symbol already excluded above
    print(f"aliases kept: {df.gene_aliases.map(len).sum()} of {before} (over all rows)")

    df["is_repeat"] = False
    for _, g in df.groupby("round_id"):
        df.loc[rng.sample(list(g.index), min(a.repeats, len(g))), "is_repeat"] = True

    assert len(df) == len(chosen), f"lost rows in joins: {len(df)} of {len(chosen)}"
    df.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} items, {df.round_id.nunique()} rounds, {df.is_repeat.sum()} repeats")
    print(df.groupby(["round_id", "gene_symbol", "disease_name"]).stratum.agg(list).to_string())


if __name__ == "__main__":
    main()
