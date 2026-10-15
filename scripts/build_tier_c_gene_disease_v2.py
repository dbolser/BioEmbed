#!/usr/bin/env python
"""Build GeneDiseaseRetrieval v2 (Tier C): 5 gene-disjoint 300-doc instances of
gene-disease -> PubMed-evidence retrieval (docs/PREREG.md). Supersedes v1
(build_tier_c_gene_disease.py, left untouched).

Query = "{SYMBOL} ({gene name}) in {disease name}", as v1. Four docs per
query, so ranking the gold first needs both the gene and the disease:
  gold          the pair's only curated PMID with a local abstract (pairs with
                2+ are skipped, so hit@1 needs no tie-break); PubTator tags
                the query gene in it; curators cite it for <10 genes (the
                qrels_clean rule, so no panels or guidelines)
  same_gene x2  PubTator tags the query gene AND curators cite it for that
                gene with an unrelated disease AND its text does not name the
                query disease (an abstract naming both would grade >= 1 under
                the PREREG rubric)
  same_disease  curators cite it for the query disease with another gene;
                PubTator does not tag the query gene in it and no OT evidence
                links it to the query gene
"Curated" = Open Targets evidence from orphanet, gene2phenotype,
genomics_england and uniprot_literature. Query diseases are MONDO, Orphanet
or EFO terms; HP phenotypes are not used as queries (a phenotype can be a
feature of the gene's other diseases with no link to them).

Guard (the v1 fix). v1's hard pool, forbidden set and silver guard dropped
only docs linked to the exact query pair, so a negative could be linked to
the query gene with an ancestor/descendant of the query disease. Here a doc
is barred for query (g, d) if any OT evidence with literature (curated,
Europe PMC, EVA/ClinVar, Cancer Gene Census, gene burden, GWAS credible sets)
links it to g with a disease o where related(d, o):
  ontology  o is reachable from d by same-concept links plus ancestor steps
            only, or same-concept links plus descendant steps only. Two OT
            diseases are the same concept if they share a concept dbXRef
            (OMIM, Orphanet, UMLS, MeSH, ...; not PMIDs or ICD classes) or
            one's id is among the other's dbXRefs (MONDO -> Orphanet twins:
            OT keeps Orphanet and HP nodes outside the MONDO tree).
  category  o is in the family (or a twin of it) of a disease sharing an ICD
            code with d or a twin: same clinical category, e.g. ICD-10-CM
            H35.5 puts Stargardt disease with Leber congenital amaurosis.
  names     each stemmed name word of one, minus modifiers (type, autosomal,
            congenital, numbers, ...), is inside a name word of the other
            ("amelogenesis imperfecta" vs "hypomaturation amelogenesis
            imperfecta"; "amelia" vs "tetraamelia"; numbered siblings such
            as "OI type 3" vs "type 9").
No corpus doc may be barred for any query unless it is that query's gold;
asserted for every (query, doc, linked disease) in each instance.

Instances: seeds 42..46, built in order. Each takes 75 queries whose genes
are unused by earlier instances, by the haystack test/dev queries and by v1
(data/exclusion/registry.parquet); one query per gene and per disease.
Greedy accept in seeded-shuffle order; a pair is skipped if its gold or
negatives would break the guard for an accepted query, or vice versa.
Deterministic: pools sorted before seeded sampling.

Run:  uv run python scripts/build_tier_c_gene_disease_v2.py          # build all 5
      uv run python scripts/build_tier_c_gene_disease_v2.py --bm25   # difficulty check
Output: data/tasks/GeneDiseaseRetrieval_v2_{0..4}: corpus/queries/qrels as v1,
        manifest.json, doc_roles.parquet (pmid, qid, role).
"""

import argparse
import functools
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import duckdb
import Stemmer
from datasets import Dataset, DatasetDict, load_from_disk

SEEDS = [42, 43, 44, 45, 46]
N_QUERIES = 75
SAME_GENE, SAME_DISEASE = 2, 1
CORPUS_SIZE = N_QUERIES * (1 + SAME_GENE + SAME_DISEASE)
MAX_GENES_CITED = 10  # gold must be cited for fewer genes (haystack_strata.py rule)
MIN_TERM_LEN = 5  # disease names/synonyms shorter than this are not matched in text

CURATED_SOURCES = ["orphanet", "gene2phenotype", "genomics_england", "uniprot_literature"]
# Other OT evidence with a literature column; guard only.
LINK_SOURCES = ["europepmc", "eva", "cancer_gene_census", "gene_burden", "gwas_credible_sets"]
QUERY_DISEASE_PREFIXES = ("MONDO_", "Orphanet_", "EFO_")
# dbXRef prefixes that name a disease concept (excludes PMID, ICD classes, genes, ...).
CONCEPT_XREFS = {"OMIM", "OMIMPS", "Orphanet", "UMLS", "MEDGEN", "MESH", "DOID", "GARD",
                 "NCIT", "NCIt", "NANDO", "SCTID", "SNOMEDCT", "SNOMEDCT_US", "MedDRA",
                 "MEDDRA", "icd11.foundation", "MONDO", "EFO", "HP", "MP"}
# Clinical classification codes: a shared code puts two diseases in one category.
CATEGORY_XREFS = {"ICD9", "ICD10", "ICD10CM", "ICD10WHO", "ICD-10", "ICDO"}
NAME_MODIFIERS = frozenset("""
    a an and or of the in to with without due related form forms type types group
    autosomal recessive dominant linked familial hereditary inherited genetic congenital
    isolated primary classic classical early late onset neonatal infantile juvenile
    childhood young adult syndrome syndromes disease diseases disorder disorders
    non syndromic nonsyndromic complementation susceptibility phenotype
""".split())
NAME_NOISE = re.compile(r"\d+[a-z]?|[a-z]|[ivx]+")  # numbers, letters, roman numerals

IUK = "/outsee5/home/danbolser/Work/IUK-IPF/data"
OT = f"{IUK}/sources/opentargets/26.06"
PT = f"{IUK}/sources/pubtator3/2026-10-06"
HGNC = f"{IUK}/sources/hgnc/2026-10-14/hgnc_complete_set.txt"
RECORDS = f"{IUK}/embeddings-catalogue/records/*.parquet"
ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "data" / "tasks"
NAME = "GeneDiseaseRetrieval_v2"
V1 = TASKS / "GeneDiseaseRetrieval"
REGISTRY = ROOT / "data" / "exclusion" / "registry.parquet"
EXCLUDE_GENES_FROM = ["GeneDiseaseHaystack", "GeneDiseaseHaystackDev", "GeneDiseaseRetrieval"]

# BM25 check: bm25_haystack.py tokenizer and its dev-selected k1/b.
BM25_K1, BM25_B = 0.9, 0.75

ABSTRACT_MARK = "\nAbstract: "


def load_source_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Curated links, other OT literature links for the curated genes, names,
    disease hierarchy and xrefs, abstracts and PubTator gene tags for every
    curated PMID."""
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
    union = " UNION ALL ".join(
        f"SELECT targetId, diseaseId, unnest(literature) AS pmid "
        f"FROM '{OT}/evidence_{s}/*.parquet' WHERE literature IS NOT NULL "
        f"AND targetId IN (SELECT targetId FROM curated)"
        for s in LINK_SOURCES
    )
    con.sql(f"CREATE TEMP TABLE silver AS SELECT DISTINCT * FROM ({union})")
    con.sql(f"""
        CREATE TEMP TABLE genes AS
        SELECT id AS targetId, approvedSymbol AS symbol, approvedName AS gene_name
        FROM '{OT}/target/*.parquet' WHERE id IN (SELECT targetId FROM curated)
    """)
    con.sql(f"""
        CREATE TEMP TABLE diseases AS
        SELECT id AS diseaseId, name AS disease_name, coalesce(ancestors, []) AS ancestors,
               coalesce(descendants, []) AS descendants, coalesce(dbXRefs, []) AS xrefs,
               coalesce(exactSynonyms, []) AS synonyms
        FROM '{OT}/disease/*.parquet'
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
              AND replace(record_id, 'PMID:', '') IN (SELECT pmid FROM curated)
        ) WHERE rn = 1 AND abstract <> ''
    """)
    con.sql(f"""
        CREATE TEMP TABLE tags AS
        SELECT DISTINCT p.pmid, m.targetId
        FROM (SELECT column0 AS pmid, unnest(string_split(column2, ';')) AS ncbi
              FROM read_csv('{PT}/gene2pubtator3.gz', delim='\t', header=false,
                            all_varchar=true, quote='')
              WHERE column0 IN (SELECT pmid FROM pubmed)) p
        JOIN (SELECT DISTINCT o.id AS targetId, h.entrez_id AS ncbi
              FROM '{OT}/target/*.parquet' o, unnest(o.dbXrefs) x,
                   read_csv('{HGNC}', delim='\t', all_varchar=true) h
              WHERE x.unnest.source = 'HGNC' AND h.hgnc_id = 'HGNC:' || x.unnest.id
                AND h.entrez_id IS NOT NULL) m USING (ncbi)
    """)


def build(con: duckdb.DuckDBPyConnection) -> None:
    load_source_tables(con)

    Pair = tuple[str, str]
    local: dict[Pair, set[str]] = {}  # curated PMIDs with a local abstract
    for t, d, p in con.sql(
        "SELECT DISTINCT targetId, diseaseId, pmid FROM curated JOIN pubmed USING (pmid)"
    ).fetchall():
        local.setdefault((t, d), set()).add(p)
    linked: dict[Pair, set[str]] = {}  # curated (any) or Europe PMC
    for t, d, p in con.sql(
        "SELECT targetId, diseaseId, pmid FROM curated UNION SELECT * FROM silver"
    ).fetchall():
        linked.setdefault((t, d), set()).add(p)
    genes = {t: (s, n) for t, s, n in con.sql("SELECT * FROM genes").fetchall()}
    names, up, down, codes, by_xref, by_code, terms = {}, {}, {}, {}, {}, {}, {}
    for d, n, anc, desc, xrefs, syns in con.sql("SELECT * FROM diseases").fetchall():
        names[d], up[d], down[d] = n, anc, desc
        terms[d] = {x.lower() for x in (n, *syns) if len(x) >= MIN_TERM_LEN}
        codes[d] = {x for x in xrefs if x.split(":", 1)[0] in CATEGORY_XREFS}
        for x in codes[d]:
            by_code.setdefault(x, set()).add(d)
        for x in {*xrefs, d.replace("_", ":", 1)}:
            if x.split(":", 1)[0] in CONCEPT_XREFS:
                by_xref.setdefault(x, set()).add(d)
    twins = {d: set() for d in names}  # same concept under another id
    for ds in by_xref.values():
        for d in ds:
            twins[d] |= ds
    tagged: dict[str, set[str]] = {}
    for p, t in con.sql("SELECT pmid, targetId FROM tags").fetchall():
        tagged.setdefault(p, set()).add(t)
    text = {p: f"{ti} {ab}".lower() for p, ti, ab in con.sql("SELECT * FROM pubmed").fetchall()}
    n_genes_cited = dict(con.sql(
        "SELECT pmid, count(DISTINCT targetId) FROM curated GROUP BY 1").fetchall())

    gene_diseases: dict[str, set[str]] = {}
    for t, d in linked:
        gene_diseases.setdefault(t, set()).add(d)
    local_by_gene: dict[str, set[Pair]] = {}  # gene -> (disease, pmid)
    local_by_disease: dict[str, set[Pair]] = {}  # disease -> (gene, pmid)
    for (t, d), ps in local.items():
        for p in ps:
            local_by_gene.setdefault(t, set()).add((d, p))
            local_by_disease.setdefault(d, set()).add((t, p))

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
        """Family, and its twins, of every disease sharing a classification code
        with d or a twin."""
        same = set().union(*(by_code[x] for y in twins.get(d, set()) | {d} for x in codes[y]))
        fam = set().union(*({x, *up[x], *down[x]} for x in same))
        return frozenset().union(fam, *(twins[x] for x in fam))

    stemmer = Stemmer.Stemmer("english")

    @functools.cache
    def name_core(d: str) -> frozenset[str]:
        words = re.findall(r"[a-z0-9]+", names[d].lower())
        return frozenset(stemmer.stemWords(
            [w for w in words if w not in NAME_MODIFIERS and not NAME_NOISE.fullmatch(w)]))

    def within(a: frozenset[str], b: frozenset[str]) -> bool:
        """Every word of a is inside some word of b ('amelia' in 'tetraamelia')."""
        return all(any(w in v for v in b) for w in a)

    @functools.cache
    def related(d: str, o: str) -> str | None:
        """Why o counts as related to d ('ontology', 'category', 'names'), or
        None. Diseases missing from the OT index count as related."""
        if o not in names or o in onto_related(d):
            return "ontology"
        if o in category_related(d):
            return "category"
        a, b = name_core(d), name_core(o)
        if a and b and (within(a, b) or within(b, a)):
            return "names"
        return None

    @functools.cache
    def mention(d: str) -> re.Pattern:
        """Name or exact synonym of d or a twin, as a whole phrase."""
        alts = sorted(set().union(*(terms[x] for x in twins[d] | {d})), key=len, reverse=True)
        return re.compile(r"\b(?:" + "|".join(map(re.escape, alts)) + r")\b")

    @functools.cache
    def guard(q: Pair) -> frozenset[str]:
        """PMIDs linked to q's gene with q's disease or a related one."""
        t, d = q
        return frozenset().union(*(linked[(t, o)] for o in gene_diseases[t] if related(d, o)))

    def same_gene_raw(q: Pair) -> set[str]:
        t, d = q
        return {p for o, p in local_by_gene[t] if not related(d, o)
                and t in tagged.get(p, ()) and not mention(d).search(text[p])}

    @functools.cache
    def gene_linked(t: str) -> frozenset[str]:
        return frozenset().union(*(linked[(t, o)] for o in gene_diseases[t]))

    def same_disease_raw(q: Pair) -> set[str]:
        t, d = q
        return {p for o, p in local_by_disease[d]
                if o != t and t not in tagged.get(p, ()) and p not in gene_linked(t)}

    # 1. Candidate pairs.
    registry_genes = {r[0] for r in con.sql(f"""
        SELECT DISTINCT id FROM '{REGISTRY}'
        WHERE kind = 'gene' AND source IN ({', '.join(f"'{h}'" for h in EXCLUDE_GENES_FROM)})
    """).fetchall()}
    funnel = Counter()
    candidates = []
    for q in sorted(local):
        t, d = q
        funnel["curated_pairs_with_local_abstract"] += 1
        if len(local[q]) != 1:
            continue
        funnel["one_local_gold"] += 1
        if not (genes.get(t, ("", ""))[0] and names.get(d)):
            continue
        if not d.startswith(QUERY_DISEASE_PREFIXES):
            funnel["dropped_phenotype_disease"] += 1
            continue
        if t in registry_genes:
            funnel["dropped_excluded_gene"] += 1
            continue
        gold = next(iter(local[q]))
        if t not in tagged.get(gold, ()):
            continue
        funnel["gold_gene_tagged"] += 1
        if n_genes_cited[gold] >= MAX_GENES_CITED:
            continue
        funnel[f"gold_cited_for_<{MAX_GENES_CITED}_genes"] += 1
        g = guard(q)
        if len(same_gene_raw(q) - g) < SAME_GENE:
            continue
        funnel["enough_same_gene"] += 1
        if len(same_disease_raw(q) - g) < SAME_DISEASE:
            continue
        funnel["candidates"] += 1
        candidates.append(q)
    print(f"candidate funnel: {dict(funnel)}", flush=True)

    # 2. Instances, greedy in seeded-shuffle order.
    v1_rule_gap = 0
    used_genes: set[str] = set()
    built = []
    for i, seed in enumerate(SEEDS):
        rng = random.Random(seed)
        order = candidates[:]
        rng.shuffle(order)
        selected: list[Pair] = []
        roles: dict[str, tuple[Pair, str]] = {}  # pmid -> (query, role)
        forbidden: set[str] = set()
        diseases_used: set[str] = set()
        rejects = Counter()
        guard_drops = Counter()
        for q in order:
            if len(selected) == N_QUERIES:
                break
            t, d = q
            if t in used_genes:
                rejects["gene_used"] += 1
                continue
            if d in diseases_used:
                rejects["disease_used"] += 1
                continue
            gold = next(iter(local[q]))
            gq = guard(q)
            if gold in forbidden or gold in roles:
                rejects["gold_breaks_guard_of_accepted"] += 1
                continue
            if gq & roles.keys():
                rejects["accepted_doc_breaks_guard"] += 1
                continue
            sg = sorted(same_gene_raw(q) - gq - forbidden - roles.keys(), key=int)
            sd = sorted(same_disease_raw(q) - gq - forbidden - roles.keys(), key=int)
            if len(sg) < SAME_GENE or len(sd) < SAME_DISEASE:
                rejects["negatives_exhausted"] += 1
                continue
            negs = [(p, "same_gene") for p in rng.sample(sg, SAME_GENE)]
            negs += [(p, "same_disease") for p in rng.sample(sd, SAME_DISEASE)]
            selected.append(q)
            used_genes.add(t)
            diseases_used.add(d)
            roles[gold] = (q, "gold")
            roles.update({p: (q, r) for p, r in negs})
            forbidden |= gq
            # What the fix removes: same-gene candidates v1's exact-pair rule
            # would keep, by first reason (link to a related disease, else the
            # text names the query disease).
            v1_kept = {p for o, p in local_by_gene[t]
                       if o != d and t in tagged.get(p, ())} - linked[q]
            removed = v1_kept - (same_gene_raw(q) - gq)
            guard_drops["same_gene_v1_rule"] += len(v1_kept)
            guard_drops["same_gene_removed"] += len(removed)
            for p in removed:
                why = {related(d, o) for o in gene_diseases[t] if p in linked[(t, o)]}
                guard_drops["removed_by_" + next(
                    (r for r in ("ontology", "category", "names") if r in why), "mention")] += 1
            raw = {p for o, p in local_by_disease[d] if o != t and t not in tagged.get(p, ())}
            guard_drops["same_disease_candidates"] += len(raw)
            guard_drops["same_disease_removed_linked_to_gene"] += len(raw & gene_linked(t))
        if len(selected) < N_QUERIES:
            raise SystemExit(f"instance {i}: only {len(selected)} queries fit")
        selected.sort()
        built.append(save_instance(con, i, seed, selected, roles, local, linked, guard,
                                   related, gene_diseases, n_genes_cited, genes, names,
                                   funnel, rejects, guard_drops,
                                   len(registry_genes), len(used_genes) - N_QUERIES))
        v1_rule_gap += guard_drops["same_gene_removed"]

    # 3. Cross-instance checks and the v1 bug, measured.
    all_genes = [q[0] for b in built for q in b]
    assert len(all_genes) == len(set(all_genes)) and not set(all_genes) & registry_genes
    v1 = load_from_disk(str(V1))
    v1_docs = set(v1["corpus"]["_id"])
    v1_gold = {}
    for r in v1["qrels"]:
        v1_gold.setdefault(r["query-id"], set()).add(r["corpus-id"])
    v1_bad = {(qid, p) for qid in v1["queries"]["_id"]
              for p in (guard(tuple(qid.split("|", 1))) & v1_docs) - v1_gold[qid]}
    print(f"\nv1 docs linked to a query gene with a related disease (not gold): "
          f"{len(v1_bad)} (query, doc) pairs, {len({p for _, p in v1_bad})} docs, "
          f"{len({q for q, _ in v1_bad})} queries")
    print(f"v2: the guard removed {v1_rule_gap} same-gene candidates that v1's "
          "exact-pair rule would have kept (selected queries)")


def save_instance(con, i, seed, selected, roles, local, linked, guard, related,
                  gene_diseases, n_genes_cited, genes, names, funnel, rejects, guard_drops,
                  n_registry_genes, n_earlier_genes):
    out_dir = TASKS / f"{NAME}_{i}"
    qid = {q: f"{q[0]}|{q[1]}" for q in selected}
    corpus_pmids = sorted(roles, key=int)
    con.sql("CREATE OR REPLACE TEMP TABLE want (pmid VARCHAR)")
    con.executemany("INSERT INTO want VALUES (?)", [(p,) for p in corpus_pmids])
    docs = {p: (ti, ab) for p, ti, ab in
            con.sql("SELECT pmid, title, abstract FROM pubmed JOIN want USING (pmid)").fetchall()}
    corpus = Dataset.from_dict({
        "_id": corpus_pmids,
        "title": [docs[p][0] for p in corpus_pmids],
        "text": [f"{docs[p][0]}\n\n{docs[p][1]}" if docs[p][0] else docs[p][1]
                 for p in corpus_pmids],
    })
    queries = Dataset.from_dict({
        "_id": [qid[q] for q in selected],
        "text": [f"{genes[t][0]} ({genes[t][1]}) in {names[d]}" for t, d in selected],
    })
    gold = {q: next(iter(local[q])) for q in selected}
    qrels = Dataset.from_dict({
        "query-id": [qid[q] for q in selected],
        "corpus-id": [gold[q] for q in selected],
        "score": [1] * len(selected),
    })

    # Validate. Every link from a corpus doc to a query's gene is checked.
    corpus_ids = set(corpus_pmids)
    assert len(corpus) == CORPUS_SIZE and len(queries) == N_QUERIES, (len(corpus), len(queries))
    assert all(corpus["text"]) and all(queries["text"]), "empty text field"
    role_counts = Counter(r for _, r in roles.values())
    assert role_counts == {"gold": N_QUERIES, "same_gene": SAME_GENE * N_QUERIES,
                           "same_disease": SAME_DISEASE * N_QUERIES}, role_counts
    for q in selected:
        t, d = q
        assert gold[q] in corpus_ids and n_genes_cited[gold[q]] < MAX_GENES_CITED
        for o in gene_diseases[t]:
            for p in (linked[(t, o)] & corpus_ids) - {gold[q]}:
                assert not related(d, o), f"{q}: doc {p} is linked to {t} with {o} ({related(d, o)})"
        assert not (guard(q) & corpus_ids) - {gold[q]}

    con.sql("CREATE OR REPLACE TEMP TABLE sel (targetId VARCHAR, diseaseId VARCHAR, pmid VARCHAR)")
    con.executemany("INSERT INTO sel VALUES (?, ?, ?)", [(*q, gold[q]) for q in selected])
    source_cov = dict(con.sql(
        "SELECT s, count(*) FROM (SELECT unnest(sources) AS s FROM curated "
        "JOIN sel USING (targetId, diseaseId, pmid)) GROUP BY 1 ORDER BY 2 DESC, 1"
    ).fetchall())

    if out_dir.exists():
        shutil.rmtree(out_dir)
    DatasetDict({"corpus": corpus, "queries": queries, "qrels": qrels}).save_to_disk(out_dir)
    con.sql("CREATE OR REPLACE TEMP TABLE doc_roles "
            "(pmid VARCHAR, qid VARCHAR, role VARCHAR, n_genes_cited INTEGER)")
    con.executemany("INSERT INTO doc_roles VALUES (?, ?, ?, ?)",
                    [(p, qid[q], r, n_genes_cited[p]) for p, (q, r) in roles.items()])
    con.sql(f"COPY (SELECT * FROM doc_roles ORDER BY qid, role, pmid::BIGINT) "
            f"TO '{out_dir / 'doc_roles.parquet'}' (FORMAT parquet)")
    manifest = {
        "name": f"{NAME}_{i}",
        "category": "Retrieval",
        "instance": {"index": i, "of": len(SEEDS), "seed": seed},
        "sizes": {"corpus": len(corpus), "queries": len(queries), "qrels": len(qrels)},
        "provenance": {
            "gold_links": f"{OT}/evidence_{{{','.join(CURATED_SOURCES)}}}",
            "silver_links": f"{OT}/evidence_{{{','.join(LINK_SOURCES)}}} (guard only)",
            "names": f"{OT}/target, {OT}/disease",
            "disease_relations": f"{OT}/disease ancestors, descendants, dbXRefs, names",
            "gene_tags": f"{PT}/gene2pubtator3.gz via {HGNC} (Ensembl -> NCBI Gene)",
            "texts": RECORDS,
            "excluded_genes": str(REGISTRY.relative_to(ROOT)),
            "source_counts_gold": source_cov,
            "build_script": "scripts/build_tier_c_gene_disease_v2.py",
        },
        "protocol": {
            "query_text": "'{SYMBOL} ({approved gene name}) in {OT disease name}'",
            "query_id": "'{Ensembl gene}|{disease id}'",
            "doc_text": "'{title}\\n\\n{abstract}' if title else abstract; text is "
                        "self-contained -- do NOT re-prepend the title column",
            "candidate_pairs": "exactly one curated PMID with a local abstract (the gold), "
                               "PubTator-tagged with the query gene and cited by curators "
                               f"for <{MAX_GENES_CITED} genes; disease id prefix in "
                               f"{list(QUERY_DISEASE_PREFIXES)} (no HP phenotypes); gene not "
                               f"used by {EXCLUDE_GENES_FROM}",
            "docs_per_query": {"gold": 1, "same_gene": SAME_GENE, "same_disease": SAME_DISEASE},
            "same_gene": "curated PMID for the query gene with a disease not related to d, "
                         "PubTator-tagged with the query gene; its title/abstract does not "
                         "contain the name or an OT exact synonym (>= "
                         f"{MIN_TERM_LEN} chars) of d or a twin of d",
            "same_disease": "curated PMID for the query disease with another gene, not "
                            "PubTator-tagged with the query gene and not linked to it by "
                            "any OT evidence (any disease)",
            "related": "related(d, o) if (ontology) o is reachable from d by same-concept "
                       "links plus ancestor steps only, or plus descendant steps only (same "
                       f"concept = a shared dbXRef with prefix in {sorted(CONCEPT_XREFS)}, "
                       "or one's id among the other's dbXRefs); or (category) o is in the "
                       "OT family, or a twin of it, of a disease sharing a classification "
                       f"code {sorted(CATEGORY_XREFS)} with d or a twin; or (names) each stemmed "
                       "name word of one, minus modifiers and numbers, is inside a name word "
                       "of the other; or o is not in the OT disease index",
            "guard": "no corpus doc is linked (curated or silver_links) to any query's gene "
                     "with a disease related to the query disease unless it is that "
                     "query's gold; asserted for every (query, doc, linked disease)",
            "selection": f"seeded shuffle (seed {seed}) of {funnel['candidates']} candidates; "
                         "greedily accept pairs with a gene unused by this and earlier "
                         "instances, an unused disease, and no guard clash with accepted "
                         f"queries; negatives sampled from sorted pools; stop at {N_QUERIES}",
            "qrels": "score=1 for the single gold per query",
            "metric": "hit@1 = recall@1 over all 300 docs (random ranking: 1/300); the 4 "
                      "docs sharing the query's gene or disease give within-group chance 0.25",
        },
        "exclusion_counts": {
            "excluded_genes": n_registry_genes,
            "earlier_instance_genes": n_earlier_genes,
            "candidate_funnel": dict(funnel),
            "greedy_rejections": dict(rejects),
            "guard_drops_selected_queries": dict(guard_drops),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n{out_dir.relative_to(ROOT)}: {len(corpus)} docs, {len(queries)} queries; "
          f"rejections {dict(rejects)}; guard drops {dict(guard_drops)}; "
          f"gold sources {source_cov}", flush=True)
    for t in queries["text"][:3]:
        print("  e.g.", t)
    return selected


def bm25_check() -> None:
    """hit@1 and MRR of BM25 on each built instance, three query formats, and
    what the top doc is when it is not the gold."""
    import bm25s
    import numpy as np

    stemmer = Stemmer.Stemmer("english")
    tok = dict(stopwords="en", stemmer=stemmer, show_progress=False)
    formats = ["symbol_disease", "symbol", "disease"]
    hits = {f: [] for f in formats}
    rr = {f: [] for f in formats}
    misses = {f: Counter() for f in formats}
    for i in range(len(SEEDS)):
        dd = load_from_disk(str(TASKS / f"{NAME}_{i}"))
        ids = dd["corpus"]["_id"]
        roles = {p: (q, r) for p, q, r in duckdb.sql(
            f"SELECT pmid, qid, role FROM '{TASKS / f'{NAME}_{i}' / 'doc_roles.parquet'}'").fetchall()}
        gold = {r["query-id"]: ids.index(r["corpus-id"]) for r in dd["qrels"]}
        qids = dd["queries"]["_id"]
        con = duckdb.connect()
        con.sql("CREATE TEMP TABLE q (qid VARCHAR)")
        con.executemany("INSERT INTO q VALUES (?)", [(x,) for x in qids])
        meta = {qid: (s, n) for qid, s, n in con.sql(f"""
            SELECT q.qid, t.approvedSymbol, d.name FROM q
            JOIN '{OT}/target/*.parquet' t ON t.id = split_part(q.qid, '|', 1)
            JOIN '{OT}/disease/*.parquet' d ON d.id = split_part(q.qid, '|', 2)
        """).fetchall()}
        index = bm25s.BM25(k1=BM25_K1, b=BM25_B)
        index.index(bm25s.tokenize(dd["corpus"]["text"], **tok), show_progress=False)
        for f in formats:
            texts = [{"symbol_disease": f"{s} {n}", "symbol": s, "disease": n}[f]
                     for s, n in (meta[x] for x in qids)]
            qtok = bm25s.tokenize(texts, return_ids=False, **tok)
            for x, toks in zip(qids, qtok):
                order = np.argsort(-index.get_scores(toks), kind="stable")  # ties: PMID order
                rank = int(np.flatnonzero(order == gold[x])[0])
                hits[f].append(rank == 0)
                rr[f].append(1 / (rank + 1))
                if rank:
                    q, r = roles[ids[order[0]]]
                    misses[f][r if q == x else "other_query"] += 1
            n = len(qids)
            print(f"  v2_{i} {f:15s} hit@1 {np.mean(hits[f][-n:]):.3f}  MRR {np.mean(rr[f][-n:]):.3f}")
    for f in formats:
        print(f"pooled {f:15s} hit@1 {np.mean(hits[f]):.3f}  MRR {np.mean(rr[f]):.3f}  "
              f"(n={len(hits[f])}; chance 1/{CORPUS_SIZE}, within the 4 docs sharing gene or "
              f"disease 0.25); "
              f"top doc on a miss: {dict(misses[f])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25", action="store_true",
                    help="only run the BM25 difficulty check on the built instances")
    args = ap.parse_args()
    if args.bm25:
        bm25_check()
        return
    con = duckdb.connect()
    con.sql("SET threads=48")
    build(con)


if __name__ == "__main__":
    main()
