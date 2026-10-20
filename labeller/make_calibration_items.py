"""Build calibration items: ~20 rounds x 10 abstracts from a haystack's judging pool, for blind
0/1/2 grades that calibrate the LLM judges (PREREG "The answer key").

Strata: curation label (gold; silver = Europe PMC-linked, not gold; unlabelled) x best rank over
the pooled runs (1-3, 4-10, 11+ = deeper or unranked, i.e. pooled as gold or a PubTator lookup
hit). stratum = '<label>@<band>', hidden in the UI. Rounds (one gene-disease pair each) are a
seeded sample of queries every judge has fully graded. Each slot in a round goes to the stratum
with the fewest items so far (ties random), so strata come out as even as the rounds allow;
repeats (is_repeat, shown twice by app.py) are spread over rounds and strata the same way. Pool
counts per stratum are printed next to the sample, for reweighting to the pool. Queries already
shown in another items file (--exclude-items, default items_demo.parquet) are left out: they are
not blind any more, and their item_ids (qid|pmid) would clash in labels.sqlite.

Order: any tuning of the rubric, prompt or judge combination uses items built from the practice
haystack (--haystack GeneDiseaseHaystackDev). Build the test items only once those are frozen
in docs/PREREG.md, so no choice is made on test grades.

Inputs (data/haystack/<NAME>/, from llm/judge_pool.py): pool.parquet, judgments.parquet,
pool_context.parquet (the gene and disease names, cleaned aliases included, that the judges saw),
docs_gene.parquet + pool_extra_docs.parquet (texts).

Run: cd labeller && uv run python make_calibration_items.py                  # test haystack
     uv run python make_calibration_items.py --haystack GeneDiseaseHaystackDev \\
         --out /tmp/items_calibration_dev.parquet                            # dry run
Output: items_calibration.parquet (default), the columns app.py requires
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import duckdb
from app import REQUIRED

HERE = Path(__file__).resolve().parent
HAYSTACKS = HERE.parent / "data/haystack"
LABELS = ["gold", "silver", "unlabelled"]
BANDS = ["1-3", "4-10", "11+"]


def pick(avail: dict[str, list], counts: Counter, rng: random.Random) -> tuple[str, object]:
    """Pop an item from the non-empty stratum with the lowest count (ties random)."""
    s = min((s for s in avail if avail[s]), key=lambda s: (counts[s], rng.random()))
    counts[s] += 1
    return s, avail[s].pop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--haystack", default="GeneDiseaseHaystack")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--per-round", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=30, help="items marked is_repeat, over all rounds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-unjudged", action="store_true", help="sample queries not yet fully judged (dry runs)")
    ap.add_argument("--exclude-items", nargs="*", type=Path, default=[HERE / "items_demo.parquet"],
                    help="items files whose rounds (queries) are left out")
    ap.add_argument("--out", type=Path, default=HERE / "items_calibration.parquet")
    a = ap.parse_args()
    hay = HAYSTACKS / a.haystack
    rng = random.Random(a.seed)
    con = duckdb.connect()

    extra = hay / "pool_extra_docs.parquet"
    con.sql(f"""
        CREATE TEMP TABLE docs AS
        SELECT pmid, text FROM '{hay}/docs_gene.parquet' SEMI JOIN '{hay}/pool.parquet' USING (pmid)
        {f"UNION ALL SELECT pmid, text FROM '{extra}'" if extra.exists() else ""}""")
    judg = hay / "judgments.parquet"
    judges = [r[0] for r in con.sql(f"SELECT DISTINCT judge FROM '{judg}' ORDER BY 1").fetchall()] \
        if judg.exists() else []
    con.sql(f"""
        CREATE TEMP TABLE pool AS
        WITH graded AS ({f"SELECT qid, pmid, count(DISTINCT judge) AS n FROM '{judg}' GROUP BY ALL"
                         if judges else "SELECT NULL::VARCHAR AS qid, NULL::VARCHAR AS pmid, 0 AS n"})
        SELECT p.qid, p.pmid,
               CASE WHEN in_gold THEN 'gold' WHEN in_silver THEN 'silver' ELSE 'unlabelled' END || '@' ||
               CASE WHEN best_rank <= 3 THEN '1-3' WHEN best_rank <= 10 THEN '4-10' ELSE '11+' END AS stratum,
               coalesce(g.n, 0) = {len(judges)} AND {len(judges)} > 0 AS judged
        FROM '{hay}/pool.parquet' p SEMI JOIN docs USING (pmid)
        LEFT JOIN graded g USING (qid, pmid)""")
    seen = [str(p) for p in a.exclude_items if p.exists()]
    eligible = [r[0] for r in con.sql(f"""
        SELECT qid FROM pool
        {f"ANTI JOIN (SELECT DISTINCT round_id AS qid FROM read_parquet({seen})) USING (qid)" if seen else ""}
        GROUP BY qid {"" if a.allow_unjudged else "HAVING bool_and(judged)"}
        ORDER BY qid""").fetchall()]
    n_seen = con.sql(f"SELECT count(DISTINCT round_id) FROM read_parquet({seen}) SEMI JOIN pool ON round_id = qid"
                     ).fetchone()[0] if seen else 0
    print(f"judges: {judges or 'none'}; {len(eligible)} eligible queries"
          f"{' (unjudged allowed)' if a.allow_unjudged else ' (fully judged)'}; "
          f"{n_seen} left out as already shown in {[Path(p).name for p in seen]}")
    if not eligible:
        raise SystemExit("no eligible queries: run llm/judge_pool.py first, or --allow-unjudged for a dry run")
    qids = rng.sample(eligible, min(a.rounds, len(eligible)))

    counts, chosen = Counter(), []  # (qid, pmid, stratum)
    for q in qids:
        avail = {}
        for pmid, s in con.execute("SELECT pmid, stratum FROM pool WHERE qid = ? ORDER BY pmid", [q]).fetchall():
            avail.setdefault(s, []).append(pmid)
        for v in avail.values():
            rng.shuffle(v)
        for _ in range(min(a.per_round, sum(map(len, avail.values())))):
            s, p = pick(avail, counts, rng)
            chosen.append((q, p, s))

    n_rep = dict.fromkeys(qids, a.repeats // len(qids))
    for q in rng.sample(qids, a.repeats % len(qids)):
        n_rep[q] += 1
    rep_counts, repeats = Counter(), set()
    for q in qids:
        avail = {}
        for qq, p, s in chosen:
            if qq == q:
                avail.setdefault(s, []).append(p)
        for _ in range(min(n_rep[q], sum(map(len, avail.values())))):
            repeats.add((q, pick(avail, rep_counts, rng)[1]))

    con.execute("CREATE TEMP TABLE chosen (qid VARCHAR, pmid VARCHAR, stratum VARCHAR, is_repeat BOOLEAN)")
    con.executemany("INSERT INTO chosen VALUES (?, ?, ?, ?)", [(q, p, s, (q, p) in repeats) for q, p, s in chosen])
    df = con.sql(f"""
        SELECT c.qid || '|' || c.pmid AS item_id, c.qid AS round_id,
               x.gene_symbol, x.gene_name, x.gene_aliases, x.disease_name, x.disease_synonyms,
               c.pmid,
               split_part(d.text, chr(10) || chr(10), 1) AS title,
               substr(d.text, length(split_part(d.text, chr(10) || chr(10), 1)) + 3) AS abstract,
               c.stratum, c.is_repeat
        FROM chosen c
        JOIN '{hay}/pool_context.parquet' x USING (qid)
        JOIN docs d USING (pmid)
        ORDER BY round_id, pmid::BIGINT""").df()[REQUIRED]
    assert len(df) == len(chosen), f"lost rows in joins: {len(df)} of {len(chosen)}"
    df.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} items, {df.round_id.nunique()} rounds, {df.is_repeat.sum()} repeats")

    tab = con.sql("""
        SELECT stratum, count(*) AS pool, count(*) FILTER (WHERE judged) AS pool_judged FROM pool GROUP BY 1
    """).df().set_index("stratum").reindex([f"{lab}@{b}" for lab in LABELS for b in BANDS], fill_value=0)
    tab["sample"] = df.stratum.value_counts()
    tab["repeats"] = df[df.is_repeat].stratum.value_counts()
    tab["queries"] = df.groupby("stratum").round_id.nunique()
    print(tab.fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
