#!/usr/bin/env python
"""Build the gene-disease fine-tuning set (docs/PREREG.md C5, C6, C8): curated
Open Targets links -> (anchor, positive abstract, up to 3 hard negatives).
No synthetic text, no LLM calls.

Positives. Curated = OT evidence from orphanet, gene2phenotype,
genomics_england and uniprot_literature (the rows of
data/gold/ot_curated_gene_disease_pmid.parquet; loaded from the evidence files
by build_tier_c_gene_disease_v2.load_source_tables and checked equal). A
positive is a curated PMID for the pair with a local abstract, PubTator-tagged
with the gene (the haystack only ranks gene-tagged abstracts), cited by
curators for < 10 genes (no panels or guidelines); at most 10 per pair.
Text '{title}\\n\\n{abstract}', as the haystack docs.

Exclusion (strict; data/exclusion/registry.parquet). Dropped: every gene of a
test or practice set (haystack test + dev, v1, v2 batches 0-4), every
registry pair, every registry PMID (any task), every PMID in the test or
practice haystack docs (registry kind 'haystack_doc', checked against both
docs_gene.parquet files; any missing docs are added), every PMID PubTator tags
with an excluded gene (the haystack-doc rule for the small-task genes too),
every PMID curated for an excluded gene, and every PMID any OT literature
evidence (curated, Europe PMC, ...) links to a registry pair. Negatives obey the same rules. By default query diseases
are not excluded (a test disease with another gene is allowed; counted in the
manifest). --exclude-test-diseases also drops every pair whose disease is a
registry 'disease' (any test or practice query disease) or its twin, and
writes to data/finetune_gd/no_test_disease/.

Anchors, one per pair (so a no-duplicates batch sampler never puts two of a
pair's positives in one batch): 50% the test template '{SYMBOL} ({gene name})
in {disease}', 25% '{SYMBOL} {disease}', 25% the test template with an HGNC
alias/previous symbol for SYMBOL (never another gene's approved symbol) or an
OT exact synonym for the disease (the test template if the pair has neither).

Hard negatives, up to 3 per row, curated first: 2 same-gene slots (same_gene,
else same_gene_bm25) + 1 same_disease, topped up from same_gene, then
same_disease; at most 1 same_gene_bm25 per row. Sampled per row; columns
ordered same_gene, same_gene_bm25, same_disease.
  same_gene       curated PMID for the gene with a disease not related to the
                  anchor's (v2 small-task rule, build_tier_c_gene_disease_v2.py),
                  PubTator-tagged with the gene, not naming the anchor disease
  same_gene_bm25  BM25 '{SYMBOL} {disease}' over a seeded sample of 500 of the
                  gene's tagged abstracts; the top 10 passing the same-gene
                  rules and, if every curated disease of the gene is related to
                  the anchor's ("solo"), not titled with the gene and clinical
                  words: likely a report of its disease ('Compound Heterozygous
                  MRPS14 Variants Associated With Leigh Syndrome' for MRPS14
                  combined OXPHOS deficiency)
  same_disease    (v2 rule) curated PMID for the disease with another gene, not
                  tagged with the anchor gene, not linked to it by any OT evidence
Same-gene rules (both kinds):
  - not in the guard: PMIDs any OT literature evidence (curated, Europe PMC,
    EVA, ...) links to the gene with a disease related to the anchor's
    (ontology, category or name relation, as v2);
  - not naming the disease or a twin: a name/synonym as a phrase, or all
    stemmed core words of one ('congenital disorders of glycosylation' names
    'RFT1-congenital disorder of glycosylation'), also with hyphens joined
    ('carpotarsal' names 'carpo-tarsal'); accent-folded (Kohlschütter-Tönz =
    Kohlschutter-Tonz). Stricter than v2's phrase match, which let true
    positives through (MORC2 'A case of Charcot-Marie-Tooth disease type 2Z'
    curated by G2P to DIGFAN, a negative for 'Charcot-Marie-Tooth disease
    axonal type 2Z');
  - not PubTator-tagged with it: no disease2pubtator3 tag (MeSH/OMIM, mapped
    by OT and MONDO SSSOM exact xrefs) of the disease, a twin, a descendant,
    or an ancestor with <= 100 descendants ('Generalized gangliosidosis:
    beta-galactosidase deficiency' tagged GM1 gangliosidosis, for GLB1 GM1
    gangliosidosis type 2; hereditary disease is too general), and no tag
    mention (mapped or not) whose core words are a subset or superset of a
    name/synonym's of the disease, a twin or such an ancestor ('dyskinesia'
    for ADCY5 neurodevelopmental disorder with hyperkinetic movements and
    dyskinesia; 'nonketotic hyperglycinemia', a synonym of the parent, for
    GLDC glycine encephalopathy 1).
Not used: diseases co-curated in one paper for the gene as the same disease;
75% of co-curations are Genomics England panels giving one paper set to all of
a gene's diseases (all of COL2A1's skeletal dysplasias), so it dropped 74% of
curated same-gene candidates, the allelic ones included, and caught no extra
judged positive in the audit. No same-gene negatives for phenotype (HP)
anchors: the gene's other diseases may show the phenotype. The manifest's
filter_audit_practice applies the v2 rule and these rules to the LLM-judged
non-gold docs of the practice haystack (practice data, as PREREG allows for
choosing settings; no test data).

Split: 5% of genes (seeded) -> valid. Train rows never use a valid positive
PMID (as positive or negative).

In-batch guard: guard_pmids lists the file's PMIDs (any row's positive or
negative) in the row's pair guard. With in-batch negatives another row's doc
can be a known positive of this row's pair (e.g. a same-gene negative of
(g, d1) curated for (g, d2)); finetune.py batches rows so that never happens.

Silver variant (train_silver.parquet = train + europepmc rows): Europe PMC
text-mined PMIDs for the train pairs (train genes only), co-mention in the
title or abstract, the same positive rules and exclusions, not curated for the
pair, top 5 per pair by resourceScore, negatives as above. More positives for
the curated pairs only: no new (Europe PMC only) pairs.

Run:  uv run python scripts/build_finetune_gene_disease.py [--exclude-test-diseases]
Output: data/finetune_gd/{train,valid,train_silver}.parquet (qid, targetId,
        diseaseId, pmid, source, template, anchor, positive, negative_1..3
        (null if fewer), neg_pmids, neg_types, guard_pmids) and manifest.json;
        with --exclude-test-diseases under data/finetune_gd/no_test_disease/.
"""

import argparse
import functools
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import bm25s
import duckdb
import numpy as np
import pandas as pd
import Stemmer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tier_c_gene_disease_v2 import (  # noqa: E402
    ABSTRACT_MARK, BM25_B, BM25_K1, CATEGORY_XREFS, CONCEPT_XREFS, CURATED_SOURCES, HGNC, IUK,
    LINK_SOURCES, MAX_GENES_CITED, MIN_TERM_LEN, NAME_MODIFIERS, NAME_NOISE, OT, PT,
    QUERY_DISEASE_PREFIXES, RECORDS, load_source_tables)

SEED = 42
MAX_POS_PER_PAIR = 10
SILVER_PER_PAIR = 5
SILVER_SECTIONS = ["title", "abstract"]
SAME_GENE, N_NEG = 2, 3  # preferred same_gene count; total negatives per row
MAX_MINED = 1  # same_gene_bm25 per row
BM25_SAMPLE = 500  # gene-tagged abstracts per gene to mine from
BM25_KEEP = 10  # mined negatives kept per pair
TAG_ANCESTOR_MAX = 100  # a tag of an ancestor with more descendants is too general to ban
CLINICAL = re.compile(r"\b(?:mutat|variant|patient|case|famil|pedigree|proband|homozyg|heterozyg|"
                      r"de novo|novel|deficien|syndrom|disease|disorder|related|associated)\w*")
VALID_FRAC = 0.05
TEMPLATES = [("full", 0.5), ("short", 0.25), ("variant", 0.25)]

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "gold" / "ot_curated_gene_disease_pmid.parquet"
REGISTRY = ROOT / "data" / "exclusion" / "registry.parquet"
HAYSTACKS = ["GeneDiseaseHaystack", "GeneDiseaseHaystackDev"]
EXCLUDE_GENES_FROM = HAYSTACKS + ["GeneDiseaseRetrieval"] + [f"GeneDiseaseRetrieval_v2_{i}" for i in range(5)]
PRACTICE = ROOT / "data" / "haystack" / "GeneDiseaseHaystackDev"  # judgments.parquet: filter audit
MONDO_SSSOM = f"{IUK}/sources/mondo/2026-10-14/mondo.sssom.tsv"
OUT_DIR = ROOT / "data" / "finetune_gd"

Pair = tuple[str, str]
Row = tuple[str, str, str]  # gene, disease, pmid


def fold(s: str) -> str:
    """Lower case, accents dropped (NFKD to ASCII)."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def dehyphen(s: str) -> str:
    return re.sub(r"(?<=[a-z0-9])-(?=[a-z0-9])", "", s)


def add_texts_and_tags(con: duckdb.DuckDBPyConnection, want: str) -> None:
    """Abstracts and PubTator gene tags for PMIDs in table `want` (pmid) not
    loaded yet: the load_source_tables SQL for another PMID set."""
    con.sql(f"""
        INSERT INTO pubmed SELECT pmid, title, abstract FROM (
            SELECT replace(record_id, 'PMID:', '') AS pmid,
                   coalesce(trim(title), '') AS title,
                   trim(substr(text, position('{ABSTRACT_MARK}' IN text)
                                     + {len(ABSTRACT_MARK)})) AS abstract,
                   row_number() OVER (PARTITION BY record_id ORDER BY record_hash) AS rn
            FROM read_parquet('{RECORDS}')
            WHERE record_type = 'pubmed_abstract'
              AND position('{ABSTRACT_MARK}' IN text) > 0
              AND replace(record_id, 'PMID:', '') IN (SELECT pmid FROM {want})
              AND replace(record_id, 'PMID:', '') NOT IN (SELECT pmid FROM pubmed)
        ) WHERE rn = 1 AND abstract <> ''
    """)
    con.sql(f"""
        INSERT INTO tags SELECT DISTINCT p.pmid, m.targetId
        FROM (SELECT column0 AS pmid, unnest(string_split(column2, ';')) AS ncbi
              FROM read_csv('{PT}/gene2pubtator3.gz', delim='\t', header=false,
                            all_varchar=true, quote='')
              WHERE column0 IN (SELECT pmid FROM {want})
                AND column0 NOT IN (SELECT DISTINCT pmid FROM tags)) p
        JOIN entrez m USING (ncbi)
    """)


def size(rows) -> dict[str, int]:
    return {"rows": len(rows), "pairs": len({r[:2] for r in rows}),
            "genes": len({r[0] for r in rows}), "pmids": len({r[2] for r in rows})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-test-diseases", action="store_true",
                    help="also drop pairs of any test/practice query disease (or twin)")
    args = ap.parse_args()
    out_dir = OUT_DIR / "no_test_disease" if args.exclude_test_diseases else OUT_DIR
    con = duckdb.connect()
    con.sql("SET threads=48")
    load_source_tables(con)
    con.sql(f"""
        CREATE TEMP TABLE entrez AS
        SELECT DISTINCT o.id AS targetId, h.entrez_id AS ncbi
        FROM '{OT}/target/*.parquet' o, unnest(o.dbXrefs) x,
             read_csv('{HGNC}', delim='\t', all_varchar=true) h
        WHERE x.unnest.source = 'HGNC' AND h.hgnc_id = 'HGNC:' || x.unnest.id
          AND h.entrez_id IS NOT NULL
    """)
    manifest_counts: dict = {}
    n_diff = con.sql(f"""
        SELECT count(*) FROM (SELECT targetId, diseaseId, pmid FROM curated) a
        FULL JOIN (SELECT targetId, diseaseId, pmid FROM '{GOLD}') b USING (targetId, diseaseId, pmid)
        WHERE a.targetId IS NULL OR b.targetId IS NULL
    """).fetchone()[0]
    assert n_diff == 0, f"curated evidence differs from {GOLD.name} in {n_diff} rows"

    # Exclusion sets.
    reg: dict[str, set[str]] = {}
    for kind, i in con.sql(f"""SELECT DISTINCT kind, id FROM '{REGISTRY}'
                               WHERE kind IN ('gene', 'pair', 'pmid', 'haystack_doc', 'disease')""").fetchall():
        reg.setdefault(kind, set()).add(i)
    gene_sources = {s for (s,) in con.sql(
        f"SELECT DISTINCT source FROM '{REGISTRY}' WHERE kind = 'gene'").fetchall()}
    assert set(EXCLUDE_GENES_FROM) <= gene_sources, set(EXCLUDE_GENES_FROM) - gene_sources
    doc_check = {}
    for h in HAYSTACKS:
        docs = {p for (p,) in con.sql(
            f"SELECT DISTINCT pmid FROM '{ROOT / 'data' / 'haystack' / h / 'docs_gene.parquet'}'").fetchall()}
        missing = docs - reg["haystack_doc"]
        doc_check[h] = {"docs_gene_pmids": len(docs), "missing_from_registry_added": len(missing)}
        reg["haystack_doc"] |= missing
    ex_gene_pmids = {p for (p,) in con.execute(
        "SELECT DISTINCT pmid FROM curated WHERE targetId IN (SELECT unnest(?))", [sorted(reg["gene"])]
    ).fetchall()}
    ex_gene_tagged = {p for (p,) in con.execute(f"""
        SELECT DISTINCT p.pmid
        FROM (SELECT column0 AS pmid, unnest(string_split(column2, ';')) AS ncbi
              FROM read_csv('{PT}/gene2pubtator3.gz', delim='\t', header=false,
                            all_varchar=true, quote='')) p
        JOIN entrez USING (ncbi) WHERE targetId IN (SELECT unnest(?))
    """, [sorted(reg["gene"])]).fetchall()}
    ex_pair_linked = {p for (p,) in con.execute("""
        SELECT DISTINCT pmid FROM (SELECT targetId, diseaseId, pmid FROM curated UNION SELECT * FROM silver)
        WHERE targetId || '|' || diseaseId IN (SELECT unnest(?))
    """, [sorted(reg["pair"])]).fetchall()}
    banned = reg["pmid"] | reg["haystack_doc"] | ex_gene_pmids | ex_gene_tagged | ex_pair_linked
    con.sql("CREATE TEMP TABLE banned (pmid VARCHAR)")
    con.append("banned", pd.DataFrame({"pmid": sorted(banned)}))
    manifest_counts["registry"] = {k: len(v) for k, v in sorted(reg.items())}
    manifest_counts["registry_haystack_doc_check"] = doc_check
    manifest_counts["pmids_curated_for_excluded_genes"] = len(ex_gene_pmids)
    manifest_counts["pmids_tagged_with_excluded_genes"] = len(ex_gene_tagged)
    manifest_counts["pmids_tagged_with_excluded_genes_not_haystack_docs"] = len(ex_gene_tagged - reg["haystack_doc"])
    manifest_counts["pmids_ot_linked_to_registry_pairs"] = len(ex_pair_linked)
    manifest_counts["pmids_banned"] = len(banned)

    # Lookups (as build_tier_c_gene_disease_v2.build).
    local: dict[Pair, set[str]] = {}
    for t, d, p in con.sql(
        "SELECT DISTINCT targetId, diseaseId, pmid FROM curated JOIN pubmed USING (pmid)"
    ).fetchall():
        local.setdefault((t, d), set()).add(p)
    linked: dict[Pair, set[str]] = {}
    for t, d, p in con.sql(
        "SELECT targetId, diseaseId, pmid FROM curated UNION SELECT * FROM silver"
    ).fetchall():
        linked.setdefault((t, d), set()).add(p)
    genes = {t: (s, n) for t, s, n in con.sql("SELECT * FROM genes").fetchall()}
    names, up, down, codes, by_xref, by_code, terms, synonyms = {}, {}, {}, {}, {}, {}, {}, {}
    for d, n, anc, desc, xrefs, syns in con.sql("SELECT * FROM diseases").fetchall():
        names[d], up[d], down[d] = n, anc, desc
        synonyms[d] = sorted({s for s in syns if s.lower() != n.lower()})
        terms[d] = {fold(x) for x in (n, *syns) if len(x) >= MIN_TERM_LEN}
        codes[d] = {x for x in xrefs if x.split(":", 1)[0] in CATEGORY_XREFS}
        for x in codes[d]:
            by_code.setdefault(x, set()).add(d)
        for x in {*xrefs, d.replace("_", ":", 1)}:
            if x.split(":", 1)[0] in CONCEPT_XREFS:
                by_xref.setdefault(x, set()).add(d)
    twins = {d: set() for d in names}
    for ds in by_xref.values():
        for d in ds:
            twins[d] |= ds
    approved = {s.upper() for (s,) in con.sql(
        f"SELECT symbol FROM read_csv('{HGNC}', delim='\t', all_varchar=true)").fetchall()}
    aliases, other_symbol_aliases = {}, 0  # an alias that is another gene's symbol: not an anchor
    for t, al in con.sql(f"""
        SELECT o.id, string_agg(concat_ws('|', h.alias_symbol, h.prev_symbol), '|')
        FROM '{OT}/target/*.parquet' o, unnest(o.dbXrefs) x,
             read_csv('{HGNC}', delim='\t', all_varchar=true) h
        WHERE x.unnest.source = 'HGNC' AND h.hgnc_id = 'HGNC:' || x.unnest.id
          AND o.id IN (SELECT targetId FROM genes)
        GROUP BY 1
    """).fetchall():
        al = {a for a in (al or "").split("|") if a and a != genes[t][0]}
        aliases[t] = sorted(a for a in al if a.upper() not in approved)
        other_symbol_aliases += len(al) - len(aliases[t])
    manifest_counts["aliases_dropped_as_another_genes_symbol"] = other_symbol_aliases
    n_genes_cited = dict(con.sql(
        "SELECT pmid, count(DISTINCT targetId) FROM curated GROUP BY 1").fetchall())
    gene_diseases: dict[str, set[str]] = {}
    for t, d in linked:
        gene_diseases.setdefault(t, set()).add(d)
    curated_diseases: dict[str, set[str]] = {}
    for t, d in con.sql("SELECT DISTINCT targetId, diseaseId FROM curated").fetchall():
        curated_diseases.setdefault(t, set()).add(d)
    local_by_gene: dict[str, set[Pair]] = {}
    local_by_disease: dict[str, set[Pair]] = {}
    for (t, d), ps in local.items():
        for p in ps:
            local_by_gene.setdefault(t, set()).add((d, p))
            local_by_disease.setdefault(d, set()).add((t, p))

    # Silver candidates for every non-excluded curated pair; abstracts and tags
    # loaded now so the guard, mention and tag checks see them.
    con.sql(f"""
        CREATE TEMP TABLE epmc AS
        SELECT targetId, diseaseId, pmid, max(resourceScore) AS score FROM (
            SELECT targetId, diseaseId, unnest(literature) AS pmid, resourceScore,
                   [s.section FOR s IN textMiningSentences] AS secs
            FROM '{OT}/evidence_europepmc/*.parquet'
            WHERE targetId IN (SELECT DISTINCT targetId FROM curated)
        ) e SEMI JOIN (SELECT DISTINCT targetId, diseaseId FROM curated) c USING (targetId, diseaseId)
        WHERE list_has_any(secs, {SILVER_SECTIONS}) AND regexp_matches(pmid, '^[0-9]+$')
        GROUP BY ALL
    """)
    con.sql("CREATE TEMP TABLE epmc_pmids AS SELECT DISTINCT pmid FROM epmc")
    add_texts_and_tags(con, "epmc_pmids")
    # BM25 mining pool: a seeded sample of each candidate gene's PubTator-tagged,
    # non-excluded abstracts.
    con.execute(f"""
        CREATE TEMP TABLE gene_docs AS
        SELECT targetId, pmid FROM (
            SELECT m.targetId, p.pmid, row_number() OVER (
                       PARTITION BY m.targetId ORDER BY md5(p.pmid || '{SEED}')) AS rn
            FROM (SELECT column0 AS pmid, unnest(string_split(column2, ';')) AS ncbi
                  FROM read_csv('{PT}/gene2pubtator3.gz', delim='\t', header=false,
                                all_varchar=true, quote='')) p
            JOIN (SELECT * FROM entrez WHERE targetId IN (SELECT targetId FROM curated)
                    AND targetId NOT IN (SELECT unnest($genes))) m USING (ncbi)
            WHERE p.pmid NOT IN (SELECT pmid FROM banned)
        ) WHERE rn <= {BM25_SAMPLE}
    """, {"genes": sorted(reg["gene"])})
    con.sql("CREATE TEMP TABLE gene_doc_pmids AS SELECT DISTINCT pmid FROM gene_docs")
    add_texts_and_tags(con, "gene_doc_pmids")
    # Practice haystack docs LLM-judged and not curator gold: the filter audit
    # only (banned, so never in the output).
    con.sql(f"""
        CREATE TEMP TABLE judged AS
        SELECT qid, pmid, count(*) AS n_judges, count(*) FILTER (WHERE grade = 2) AS n_grade2
        FROM '{PRACTICE / 'judgments.parquet'}'
        ANTI JOIN '{PRACTICE / 'qrels.parquet'}' USING (qid, pmid)
        GROUP BY ALL
    """)
    con.sql("CREATE TEMP TABLE judged_pmids AS SELECT DISTINCT pmid FROM judged")
    add_texts_and_tags(con, "judged_pmids")
    tagged: dict[str, set[str]] = {}
    for p, t in con.sql("SELECT pmid, targetId FROM tags").fetchall():
        tagged.setdefault(p, set()).add(t)
    # PubTator disease tags (MeSH/OMIM) as OT diseases: OT xrefs + MONDO SSSOM exact.
    pt_ot: dict[str, set[str]] = {}
    for x, ds in by_xref.items():
        if x.split(":", 1)[0] in ("MESH", "OMIM"):
            pt_ot.setdefault(x, set()).update(ds)
    for s, o in con.sql(f"""
        SELECT subject_id, object_id FROM read_csv('{MONDO_SSSOM}', delim='\t', comment='#',
                                                   header=true, all_varchar=true)
        WHERE predicate_id = 'skos:exactMatch' AND regexp_matches(object_id, '^(mesh|OMIM):')
    """).fetchall():
        if (d := s.replace(":", "_", 1)) in names:
            pt_ot.setdefault(o.replace("mesh:", "MESH:", 1), set()).add(d)
    disease_tags: dict[str, set[str]] = {}  # PMID -> OT diseases
    disease_mentions: dict[str, set[str]] = {}  # PMID -> mention texts (mapped or not)
    for p, i, m in con.sql(f"""
        SELECT DISTINCT column0, column2, column3
        FROM read_csv('{PT}/disease2pubtator3.gz', delim='\t', header=false, all_varchar=true, quote='')
        WHERE column0 IN (SELECT pmid FROM pubmed)
    """).fetchall():
        if i in pt_ot:
            disease_tags.setdefault(p, set()).update(pt_ot[i])
        disease_mentions.setdefault(p, set()).update(
            x for x in fold(m or "").split("|") if len(x) >= MIN_TERM_LEN)
    doc = {p: (ti, ab) for p, ti, ab in con.sql("SELECT * FROM pubmed").fetchall()}
    text = {p: fold(f"{ti} {ab}") for p, (ti, ab) in doc.items()}

    # Relations and negative pools, as build_tier_c_gene_disease_v2.build.
    def closure(d: str, step: dict[str, list[str]]) -> set[str]:
        seen, todo = set(), [d]
        while todo:
            x = todo.pop()
            if x not in seen:
                seen.add(x)
                todo += twins.get(x, ())
                todo += step.get(x, ())
        return seen

    @functools.cache
    def onto_related(d: str) -> frozenset[str]:
        return frozenset(closure(d, up) | closure(d, down))

    @functools.cache
    def category_related(d: str) -> frozenset[str]:
        same = set().union(*(by_code[x] for y in twins.get(d, set()) | {d} for x in codes[y]))
        fam = set().union(*({x, *up[x], *down[x]} for x in same))
        return frozenset().union(fam, *(twins[x] for x in fam))

    stemmer = Stemmer.Stemmer("english")

    @functools.cache
    def core_words(name: str) -> frozenset[str]:
        words = re.findall(r"[a-z0-9]+", fold(name))
        return frozenset(stemmer.stemWords(
            [w for w in words if w not in NAME_MODIFIERS and not NAME_NOISE.fullmatch(w)]))

    def name_core(d: str) -> frozenset[str]:
        return core_words(names[d])

    def within(a: frozenset[str], b: frozenset[str]) -> bool:
        return all(any(w in v for v in b) for w in a)

    @functools.cache
    def related(d: str, o: str) -> bool:
        if o not in names or o in onto_related(d) or o in category_related(d):
            return True
        a, b = name_core(d), name_core(o)
        return bool(a and b and (within(a, b) or within(b, a)))

    def twins_of(d: str) -> frozenset[str]:
        return frozenset(twins.get(d, set()) | {d})

    @functools.cache
    def mention(ds: frozenset[str]) -> re.Pattern:
        """A name or exact synonym of a disease in ds, as a whole phrase."""
        alts = sorted(set().union(*(terms.get(x, ()) for x in ds)), key=len, reverse=True)
        return re.compile(r"\b(?:" + "|".join(map(re.escape, alts)) + r")\b" if alts else r"(?!)")

    @functools.cache
    def name_cores(ds: frozenset[str]) -> tuple[frozenset[str], ...]:
        return tuple({c for x in ds if x in names for n in (names[x], *synonyms[x])
                      for v in (n, dehyphen(fold(n))) if (c := core_words(v))})

    @functools.cache
    def words(p: str) -> frozenset[str]:
        return frozenset(stemmer.stemWords(re.findall(r"[a-z0-9]+", f"{text[p]} {dehyphen(text[p])}")))

    def names_disease(ds: frozenset[str], p: str) -> bool:
        """p names a disease in ds: phrase, or all stemmed core words of a name/synonym."""
        return bool(mention(ds).search(text[p])) or any(c <= words(p) for c in name_cores(ds))

    @functools.cache
    def near_above(d: str) -> frozenset[str]:
        """d, twins, and ancestors with <= TAG_ANCESTOR_MAX descendants."""
        return frozenset({a for a in closure(d, up) if len(down.get(a, ())) <= TAG_ANCESTOR_MAX} | twins_of(d))

    @functools.cache
    def tag_near(d: str) -> frozenset[str]:
        return near_above(d) | closure(d, down)

    @functools.cache
    def mention_cores(m: str) -> tuple[frozenset[str], ...]:
        return tuple({c for v in (m, dehyphen(m)) if (c := core_words(v))})

    def tags_disease(d: str, p: str) -> bool:
        """PubTator tags p with an OT disease in tag_near(d), or with a mention
        whose core words are a subset or superset of those of a name/synonym of
        near_above(d) (unmapped MeSH ids too: 'myoglobinuria' for 'myoglobinuria,
        acute recurrent, autosomal recessive')."""
        if not disease_tags.get(p, set()).isdisjoint(tag_near(d)):
            return True
        cores = name_cores(near_above(d))
        return any(b <= a or a <= b for m in disease_mentions.get(p, ()) for b in mention_cores(m)
                   for a in cores)

    @functools.cache
    def solo(q: Pair) -> bool:
        """Every curated disease of the gene is related to the anchor's."""
        return all(related(q[1], o) for o in curated_diseases[q[0]])

    @functools.cache
    def gene_word(t: str) -> re.Pattern:
        return re.compile(r"\b(?:" + "|".join(map(re.escape, [genes[t][0], *aliases.get(t, [])])) + r")\b")

    def clinical_title(q: Pair, p: str) -> bool:
        """For a solo gene: the title names the gene with clinical words."""
        ti = doc[p][0]
        return solo(q) and bool(gene_word(q[0]).search(ti)) and bool(CLINICAL.search(fold(ti)))

    def same_gene_reject(q: Pair, p: str, mined_rules: bool) -> str | None:
        """Why p may not be a same-gene negative of q (None: it may)."""
        d = q[1]
        if p in guard(q):
            return "guard"
        if names_disease(twins_of(d), p):
            return "names_disease"
        if tags_disease(d, p):
            return "pubtator_disease_tag"
        if mined_rules and clinical_title(q, p):
            return "solo_gene_clinical_title"
        return None

    @functools.cache
    def guard(q: Pair) -> frozenset[str]:
        """PMIDs linked to the gene with a disease related to the anchor's (v2)."""
        t, d = q
        return frozenset().union(*(linked[(t, o)] for o in gene_diseases.get(t, ()) if related(d, o)))

    @functools.cache
    def gene_linked(t: str) -> frozenset[str]:
        return frozenset().union(*(linked[(t, o)] for o in gene_diseases[t]))

    mined: dict[Pair, list[str]] = {}  # filled below, before the first pools() call
    same_gene_candidates = Counter()  # (pair, PMID): v2 rule, and why this one rejects

    @functools.cache
    def pools(q: Pair) -> tuple[frozenset[str], ...]:
        """(same_gene, same_gene_bm25, same_disease) candidates before split bans."""
        t, d = q
        sg = set()
        if d.startswith(QUERY_DISEASE_PREFIXES):
            v2 = {p for o, p in local_by_gene[t] if not related(d, o) and t in tagged.get(p, ())
                  and not mention(twins_of(d)).search(text[p])} - guard(q) - banned
            why = {p: same_gene_reject(q, p, mined_rules=False) for p in v2}
            sg = {p for p, w in why.items() if w is None}
            same_gene_candidates["v2_rule"] += len(v2)
            same_gene_candidates.update(f"rejected_{w}" for w in why.values() if w)
            same_gene_candidates["kept"] += len(sg)
        sd = {p for o, p in local_by_disease.get(d, ())
              if o != t and t not in tagged.get(p, ()) and p not in gene_linked(t)}
        return frozenset(sg), frozenset(set(mined.get(q, ())) - sg), frozenset(sd - guard(q) - banned)

    def negatives(q: Pair, rng: random.Random, ban: set[str]) -> list[tuple[str, str]]:
        """SAME_GENE same-gene slots (same_gene, else same_gene_bm25) + the rest
        same_disease, topped up from same_gene, then same_disease; at most
        MAX_MINED same_gene_bm25."""
        sg, sm, sd = (rng.sample(x, len(x)) for x in (sorted(s - ban, key=int) for s in pools(q)))
        k_sg = min(SAME_GENE, len(sg))
        k_sm = min(MAX_MINED, len(sm), SAME_GENE - k_sg)
        k_sd = min(N_NEG - SAME_GENE, len(sd))
        k_sg = min(len(sg), N_NEG - k_sm - k_sd)
        k_sd = min(len(sd), N_NEG - k_sg - k_sm)
        k_sm = min(MAX_MINED, len(sm), N_NEG - k_sg - k_sd)
        return ([(p, "same_gene") for p in sg[:k_sg]] + [(p, "same_gene_bm25") for p in sm[:k_sm]]
                + [(p, "same_disease") for p in sd[:k_sd]])

    # 1. Curated positives, funnel.
    funnel = {}
    curated_rows: set[Row] = {tuple(r) for r in con.sql(
        "SELECT DISTINCT targetId, diseaseId, pmid FROM curated").fetchall()}
    rows = curated_rows
    test_diseases = frozenset().union(*map(twins_of, reg["disease"]))  # query diseases and twins
    steps = [
        ("curated", lambda r: True),
        ("local_abstract", lambda r: r[2] in doc and r[2] in n_genes_cited),
        ("drop_excluded_gene", lambda r: r[0] not in reg["gene"]),
        ("drop_registry_pair", lambda r: f"{r[0]}|{r[1]}" not in reg["pair"]),
        *([("drop_test_or_practice_disease", lambda r: r[1] not in test_diseases)]
          if args.exclude_test_diseases else []),
        ("drop_registry_pmid", lambda r: r[2] not in reg["pmid"]),
        ("drop_haystack_doc", lambda r: r[2] not in reg["haystack_doc"]),
        ("drop_pmid_tagged_with_excluded_gene", lambda r: r[2] not in ex_gene_tagged),
        ("drop_pmid_curated_for_excluded_gene", lambda r: r[2] not in ex_gene_pmids),
        ("drop_pmid_ot_linked_to_registry_pair", lambda r: r[2] not in ex_pair_linked),
        ("gene_tagged", lambda r: r[0] in tagged.get(r[2], ())),
        (f"cited_for_<{MAX_GENES_CITED}_genes", lambda r: n_genes_cited[r[2]] < MAX_GENES_CITED),
        ("gene_and_disease_named", lambda r: genes.get(r[0], ("", ""))[0] and names.get(r[1])),
    ]
    for name, keep in steps:
        rows = {r for r in rows if keep(r)}
        funnel[name] = size(rows)
        print(f"{name:40s} {funnel[name]}", flush=True)
    positives: dict[Pair, list[str]] = {}
    for t, d, p in sorted(rows):
        positives.setdefault((t, d), []).append(p)

    # 2. Gene split, then per-pair cap and anchors.
    rng = random.Random(SEED)
    all_genes = sorted({t for t, _ in positives})
    valid_genes = set(rng.sample(all_genes, round(VALID_FRAC * len(all_genes))))
    valid_pmids = {p for q, ps in positives.items() if q[0] in valid_genes for p in ps}
    split_pos = {"train": {}, "valid": {}}
    dropped_valid_overlap = 0
    capped = Counter()
    for q, ps in sorted(positives.items()):
        split = "valid" if q[0] in valid_genes else "train"
        if split == "train":
            dropped_valid_overlap += sum(p in valid_pmids for p in ps)
            ps = [p for p in ps if p not in valid_pmids]
        if ps:
            split_pos[split][q] = sorted(rng.sample(ps, min(len(ps), MAX_POS_PER_PAIR)), key=int)
            capped[split] += len(ps) - len(split_pos[split][q])

    @functools.cache
    def anchor(q: Pair) -> tuple[str, str]:
        t, d = q
        sym, gname = genes[t]
        r = random.Random(f"{SEED}|{t}|{d}")
        kind = r.choices([k for k, _ in TEMPLATES], [w for _, w in TEMPLATES])[0]
        if kind == "short":
            return "short", f"{sym} {names[d]}"
        if kind == "variant":
            options = [("gene_alias", a, names[d]) for a in aliases.get(t, [])]
            options += [("disease_synonym", sym, s) for s in synonyms[d]]
            if options:
                k, s, n = r.choice(options)
                return k, f"{s} ({gname}) in {n}"
            kind = "full_no_variant"
        return kind, f"{sym} ({gname}) in {names[d]}"

    def doc_text(p: str) -> str:
        ti, ab = doc[p]
        return f"{ti}\n\n{ab}" if ti else ab

    def make_rows(pos: dict[Pair, list[str]], source: str, ban: set[str], seed: int) -> list[dict]:
        r = random.Random(seed)
        out = []
        for q, ps in sorted(pos.items()):
            kind, a = anchor(q)
            for p in ps:
                negs = negatives(q, r, ban)
                texts = [doc_text(n) for n, _ in negs] + [None] * (N_NEG - len(negs))
                out.append({"qid": f"{q[0]}|{q[1]}", "targetId": q[0], "diseaseId": q[1], "pmid": p,
                            "source": source, "template": kind, "anchor": a, "positive": doc_text(p),
                            **{f"negative_{i + 1}": x for i, x in enumerate(texts)},
                            "neg_pmids": [n for n, _ in negs], "neg_types": [k for _, k in negs]})
        return out

    # 3. BM25-mined same-gene negatives: each pair's '{SYMBOL} {disease}' over
    # its gene's sampled abstracts; top BM25_KEEP passing the same-gene rules
    # (and the solo-gene title rule). Not for phenotype (HP) anchors, as same_gene.
    gene_docs: dict[str, list[str]] = {}
    for t, p in con.sql("SELECT targetId, pmid FROM gene_docs ORDER BY targetId, pmid::BIGINT").fetchall():
        if p in doc:
            gene_docs.setdefault(t, []).append(p)
    by_gene: dict[str, list[Pair]] = {}
    for q in [*split_pos["train"], *split_pos["valid"]]:
        if q[1].startswith(QUERY_DISEASE_PREFIXES) and q[0] in gene_docs:
            by_gene.setdefault(q[0], []).append(q)
    tok = dict(stopwords="en", stemmer=stemmer, show_progress=False)
    mined_rejects = Counter()  # candidates scanned, in BM25 order, until BM25_KEEP kept
    for t, qs in sorted(by_gene.items()):
        ps = gene_docs[t]
        corpus = bm25s.tokenize([doc_text(p) for p in ps], **tok)
        index = bm25s.BM25(k1=BM25_K1, b=BM25_B)
        index.index((corpus.ids, dict(corpus.vocab)), show_progress=False)
        qtok = bm25s.tokenize([f"{genes[t][0]} {names[d]}" for _, d in qs], return_ids=False, **tok)
        for q, toks in zip(qs, qtok):
            s = index.get_scores(toks)
            mined[q] = []
            for j in np.argsort(-s, kind="stable"):
                if s[j] <= 0 or len(mined[q]) == BM25_KEEP:
                    break
                why = same_gene_reject(q, ps[j], mined_rules=True)
                mined_rejects[why or "kept"] += 1
                if why is None:
                    mined[q].append(ps[j])
    mining = {"genes_with_sampled_docs": len(gene_docs),
              "sampled_docs_with_abstract": sum(map(len, gene_docs.values())),
              "pairs_mined": len(mined), "pairs_with_a_mined_negative": sum(map(bool, mined.values())),
              "pairs_solo_gene": sum(map(solo, mined)), "candidates_scanned": dict(mined_rejects)}
    print(f"bm25 mining {mining}", flush=True)

    train = make_rows(split_pos["train"], "curated", valid_pmids, SEED)
    valid = make_rows(split_pos["valid"], "curated", set(), SEED + 1)

    # 4. Silver positives for the train pairs.
    sfunnel = {}
    srows = [tuple(r) for r in con.sql(
        "SELECT * FROM epmc ORDER BY targetId, diseaseId, score DESC, pmid::BIGINT").fetchall()]
    train_pairs = set(split_pos["train"])
    ssteps = [
        ("europepmc_title_or_abstract", lambda r: True),
        ("train_pair", lambda r: r[:2] in train_pairs),
        ("not_curated_for_pair", lambda r: r[:3] not in curated_rows),
        ("local_abstract", lambda r: r[2] in doc),
        ("drop_registry_pmid", lambda r: r[2] not in reg["pmid"]),
        ("drop_haystack_doc", lambda r: r[2] not in reg["haystack_doc"]),
        ("drop_pmid_tagged_with_excluded_gene", lambda r: r[2] not in ex_gene_tagged),
        ("drop_pmid_curated_for_excluded_gene", lambda r: r[2] not in ex_gene_pmids),
        ("drop_pmid_ot_linked_to_registry_pair", lambda r: r[2] not in ex_pair_linked),
        ("drop_valid_pmid", lambda r: r[2] not in valid_pmids),
        ("gene_tagged", lambda r: r[0] in tagged.get(r[2], ())),
        (f"cited_for_<{MAX_GENES_CITED}_genes", lambda r: n_genes_cited.get(r[2], 0) < MAX_GENES_CITED),
    ]
    for name, keep in ssteps:
        srows = [r for r in srows if keep(r)]
        sfunnel[name] = size(srows)
    silver_pos: dict[Pair, list[str]] = {}
    for t, d, p, _ in srows:  # sorted by score
        if len(silver_pos.setdefault((t, d), [])) < SILVER_PER_PAIR:
            silver_pos[(t, d)].append(p)
    sfunnel[f"top_{SILVER_PER_PAIR}_per_pair"] = size([(*q, p) for q, ps in silver_pos.items() for p in ps])
    for name, n in sfunnel.items():
        print(f"silver {name:33s} {n}", flush=True)
    silver = make_rows(silver_pos, "europepmc", valid_pmids, SEED + 2)

    # 5. Filter audit: practice-haystack docs (pooled top 10s) graded by both
    # LLM judges, not curator gold; kept as a same-gene negative by each rule?
    rules = {
        "v2_rule": lambda q, p: p not in guard(q) and not mention(twins_of(q[1])).search(text[p]),
        "this_rule": lambda q, p: same_gene_reject(q, p, mined_rules=True) is None,
    }
    audit = Counter()
    for qid, p, n_judges, n_grade2 in con.sql("SELECT * FROM judged ORDER BY ALL").fetchall():
        q = tuple(qid.split("|", 1))
        if n_judges < 2 or p not in text or not q[1].startswith(QUERY_DISEASE_PREFIXES):
            continue
        for name, keep in [("judged", lambda q, p: True), *rules.items()]:
            if keep(q, p):
                audit[f"{name}_kept"] += 1
                audit[f"{name}_kept_grade2_both"] += n_grade2 == 2
                audit[f"{name}_kept_grade2_either"] += n_grade2 >= 1
    print(f"filter audit {dict(audit)}", flush=True)

    # 6. Checks and output.
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {"train": pd.DataFrame(train), "valid": pd.DataFrame(valid),
              "train_silver": pd.DataFrame(train + silver)}
    for name, df in frames.items():
        used = set(df.pmid) | {p for ps in df.neg_pmids for p in ps}
        assert not used & banned, f"{name}: excluded PMID used"
        assert not set(df.targetId) & reg["gene"], f"{name}: excluded gene used"
        assert not set(df.qid) & reg["pair"], f"{name}: excluded pair used"
        assert all(p not in guard((t, d)) for t, d, ps in zip(df.targetId, df.diseaseId, df.neg_pmids)
                   for p in ps), f"{name}: negative linked to its anchor's pair"
        if name != "valid":
            assert not used & valid_pmids and not set(df.targetId) & valid_genes
        pairs = list(zip(df.targetId, df.diseaseId))
        in_file = {q: sorted(guard(q) & used, key=int) for q in set(pairs)}
        assert not any(names_disease(twins_of(q[1]), p) or tags_disease(q[1], p)
                       for q, ps, ts in zip(pairs, df.neg_pmids, df.neg_types)
                       for p, k in zip(ps, ts) if k != "same_disease"), f"{name}: same-gene negative names its disease"
        assert all(ts.count("same_gene_bm25") <= MAX_MINED for ts in df.neg_types)
        df["guard_pmids"] = [in_file[q] for q in pairs]
        df.to_parquet(out_dir / f"{name}.parquet", index=False)

    def describe(df: pd.DataFrame) -> dict:
        n_negs = df.neg_pmids.map(len)
        return {
            **size(list(zip(df.targetId, df.diseaseId, df.pmid))),
            "rows_by_source": df.source.value_counts().to_dict(),
            "rows_by_n_negatives": {int(k): int(v) for k, v in n_negs.value_counts().sort_index().items()},
            "negatives_by_type": dict(Counter(k for ks in df.neg_types for k in ks)),
            "guard_pmids_per_row_mean": round(float(df.guard_pmids.map(len).mean()), 2),
            "rows_by_template": df.template.value_counts().to_dict(),
            "pairs_by_template": df.drop_duplicates("qid").template.value_counts().to_dict(),
            "pairs_with_phenotype_disease": int((~df.drop_duplicates("qid").diseaseId.str.startswith(
                QUERY_DISEASE_PREFIXES)).sum()),
        }

    sizes = {name: describe(df) for name, df in frames.items()}
    for name, df in frames.items():
        hit = df.diseaseId.isin(test_diseases)
        sizes[name] |= {"rows_with_a_test_or_practice_disease": int(hit.sum()),
                        "pairs_with_a_test_or_practice_disease": int(df[hit].qid.nunique()),
                        "test_or_practice_diseases_used": len(set(df[hit].diseaseId))}
    manifest = {
        "name": "gene-disease fine-tuning set",
        "columns": list(frames["train"].columns),
        "sizes": sizes,
        "funnel_curated": funnel,
        "split": {"genes": len(all_genes), "valid_genes": len(valid_genes),
                  "valid_positive_pmids": len(valid_pmids),
                  "train_positives_dropped_shared_with_valid": dropped_valid_overlap,
                  "positives_dropped_by_cap": dict(capped)},
        "funnel_silver": sfunnel,
        "bm25_mining": mining,
        "same_gene_candidates": dict(same_gene_candidates),
        "filter_audit_practice": dict(audit),
        "exclusion": manifest_counts | {"test_or_practice_diseases_excluded": args.exclude_test_diseases,
                                        "registry_diseases_with_twins": len(test_diseases)},
        "protocol": {
            "positive": "curated PMID for the pair with a local abstract, PubTator-tagged with the "
                        f"gene, cited by curators for <{MAX_GENES_CITED} genes; "
                        f"<= {MAX_POS_PER_PAIR} per pair (seeded sample)",
            "exclusion": f"drop genes of {EXCLUDE_GENES_FROM}, registry pairs, registry PMIDs, "
                         f"haystack docs of {HAYSTACKS}, PMIDs PubTator tags with an excluded gene, "
                         "PMIDs curated for an excluded gene, PMIDs any OT literature evidence links "
                         "to a registry pair; "
                         "applies to positives and negatives"
                         + ("; pairs whose disease is a registry disease or its twin"
                            if args.exclude_test_diseases else ""),
            "anchor": "one per pair: 50% '{SYMBOL} ({gene name}) in {disease}', 25% '{SYMBOL} "
                      "{disease}', 25% the first with an HGNC alias/previous symbol (not another "
                      "gene's approved symbol) or an OT exact disease synonym ('full_no_variant' "
                      "if neither exists)",
            "negatives": f"up to {N_NEG}: {SAME_GENE} same-gene slots (same_gene, else "
                         f"same_gene_bm25) + {N_NEG - SAME_GENE} same_disease, topped up from "
                         f"same_gene, then same_disease; at most {MAX_MINED} same_gene_bm25 per row; "
                         "build_tier_c_gene_disease_v2 rules and guard, but same-gene "
                         "may not name the disease or a twin by phrase or by all stemmed core "
                         "words of a name/synonym (also hyphens joined), accent-folded, nor be "
                         "PubTator-tagged (disease2pubtator3 MeSH/OMIM via OT + MONDO SSSOM exact "
                         "xrefs) with the disease, a twin, a descendant or an ancestor with <= "
                         f"{TAG_ANCESTOR_MAX} descendants, nor carry a PubTator disease mention whose "
                         "stemmed core words are a subset/superset of a name/synonym's of the "
                         "disease, a twin or such an ancestor; "
                         f"no same-gene for phenotype (non-{'/'.join(QUERY_DISEASE_PREFIXES)}) "
                         "diseases; train negatives never a valid positive",
            "same_gene_bm25": f"BM25 (k1 {BM25_K1}, b {BM25_B}, stemmed, stopwords) of "
                              "'{SYMBOL} {disease}' over a seeded sample (md5 order) of "
                              f"{BM25_SAMPLE} non-excluded PubTator-tagged abstracts of the gene; "
                              f"top {BM25_KEEP} with score > 0, same-gene rules as above, and if every "
                              "curated disease of the gene is related to the anchor's (solo), no "
                              "title naming the gene (symbol/alias) with clinical words "
                              f"({CLINICAL.pattern}); sampled per row",
            "split": f"{VALID_FRAC:.0%} of genes to valid (seed {SEED}); train never uses a valid "
                     "positive PMID",
            "silver": f"Europe PMC PMIDs for train pairs, co-mention section in {SILVER_SECTIONS}, "
                      "not curated for the pair, positive rules and exclusions as curated, top "
                      f"{SILVER_PER_PAIR} per pair by resourceScore (ties: PMID); adds positives "
                      "to curated train pairs only, no new pairs",
            "filter_audit_practice": "practice-haystack (qid, pmid) graded by both LLM judges, not "
                                     "curator gold, non-HP disease: how many each same-gene rule "
                                     "keeps as a negative, and of those graded 2 by both / either",
            "doc_text": "'{title}\\n\\n{abstract}' if title else abstract",
        },
        "provenance": {
            "curated": f"{OT}/evidence_{{{','.join(CURATED_SOURCES)}}} (== {GOLD.relative_to(ROOT)})",
            "guard_links": f"{OT}/evidence_{{{','.join(LINK_SOURCES)}}}",
            "silver": f"{OT}/evidence_europepmc",
            "names": f"{OT}/target, {OT}/disease, {HGNC} (alias_symbol, prev_symbol)",
            "gene_tags": f"{PT}/gene2pubtator3.gz",
            "disease_tags": f"{PT}/disease2pubtator3.gz, {MONDO_SSSOM}",
            "texts": RECORDS,
            "exclusion": str(REGISTRY.relative_to(ROOT)),
            "filter_audit": f"{(PRACTICE / 'judgments.parquet').relative_to(ROOT)}, qrels.parquet",
            "build_script": "scripts/build_finetune_gene_disease.py",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"sizes": sizes, "split": manifest["split"]}, indent=2))


if __name__ == "__main__":
    main()
