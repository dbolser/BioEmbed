#!/usr/bin/env python
"""Build GOProteinPairCls: PairClassification -- does this GO term describe this protein?

sentence1 = GO term "name: definition"
sentence2 = UniProt protein function text (cleaned)
label     = 1 if the protein is annotated with the term, 0 otherwise
            (negatives are hierarchy-aware: never an annotated term nor an
             is_a ancestor of one, which would be a true positive under
             annotation propagation)

Source (read-only): /outsee5/.../embeddings-catalogue/records/ parquet.
Deterministic: SEED=42, all pools sorted before sampling.
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import duckdb
from datasets import Dataset, DatasetDict

SEED = 42
RECORDS = "/outsee5/home/danbolser/Work/IUK-IPF/data/embeddings-catalogue/records/*.parquet"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "tasks" / "GOProteinPairCls"

N_POS = 250
N_NEG = 250
MIN_FUNC_CHARS = 200
MIN_ANNOTATIONS = 3
MAX_PER_PROTEIN_POS = 2  # + 2 negatives => <= 4 appearances total per protein

NAMESPACE_ROOTS = {
    "GO:0008150",  # biological_process
    "GO:0003674",  # molecular_function
    "GO:0005575",  # cellular_component
}

ECO_RE = re.compile(r"\s*\{ECO:[^}]*\}")
FUNC_LINE_RE = re.compile(r"^Function: (.*)$", re.M)
DEF_LINE_RE = re.compile(r"^Definition: (.*)$", re.M)


def clean_function_text(raw: str) -> str:
    """Strip FUNCTION: markers and {ECO:...} evidence tags; normalise whitespace."""
    text = raw.replace("FUNCTION: ", "")
    text = ECO_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+\.", ".", text)  # tidy " ." left by tag removal
    return text


def load_go(con):
    """GO terms: id -> {name, namespace, definition, is_a}; plus alt_id -> id."""
    rows = con.execute(
        f"""
        SELECT record_id, title, text, metadata
        FROM '{RECORDS}'
        WHERE record_type = 'go_term'
        ORDER BY record_id
        """
    ).fetchall()
    terms, alt_map = {}, {}
    for go_id, name, text, meta_json in rows:
        meta = json.loads(meta_json)
        m = DEF_LINE_RE.search(text)
        definition = m.group(1).strip() if m else ""
        if not definition:
            continue
        terms[go_id] = {
            "name": name,
            "namespace": meta["namespace"],
            "definition": definition,
            "is_a": sorted(meta.get("is_a") or []),
        }
        for alt in meta.get("alt_ids") or []:
            alt_map[alt] = go_id
    return terms, alt_map


def build_ancestors(terms):
    """Transitive is_a closure (ancestors, term itself excluded), memoised."""
    cache = {}

    def ancestors(go_id):
        if go_id in cache:
            return cache[go_id]
        cache[go_id] = frozenset()  # break any cycle
        out = set()
        for parent in terms.get(go_id, {}).get("is_a", []):
            out.add(parent)
            out |= ancestors(parent)
        cache[go_id] = frozenset(out)
        return cache[go_id]

    for go_id in terms:
        ancestors(go_id)
    return cache


def load_proteins(con, terms, alt_map):
    """Reviewed proteins with cleaned function text >= MIN_FUNC_CHARS and >= MIN_ANNOTATIONS known GO ids."""
    rows = con.execute(
        f"""
        SELECT record_id, text, metadata
        FROM '{RECORDS}'
        WHERE record_type IN ('uniprot_human', 'uniprot_protein')
          AND text LIKE '%' || chr(10) || 'Function: %'
        ORDER BY record_id
        """
    ).fetchall()
    proteins = {}
    for acc, text, meta_json in rows:
        m = FUNC_LINE_RE.search(text)
        if not m:
            continue
        func = clean_function_text(m.group(1))
        if len(func) < MIN_FUNC_CHARS:
            continue
        raw_ids = json.loads(meta_json).get("go_ids") or []
        annots = sorted({alt_map.get(g, g) for g in raw_ids} & terms.keys())
        if len(annots) < MIN_ANNOTATIONS:
            continue
        proteins[acc] = {"function": func, "annotations": annots}
    return proteins


def main():
    rng = random.Random(SEED)
    con = duckdb.connect()

    terms, alt_map = load_go(con)
    print(f"GO terms with definitions: {len(terms)} (+{len(alt_map)} alt ids)")

    ancestors = build_ancestors(terms)

    # Terms biased away from the top: >= 1 is_a parent that is not a namespace root.
    eligible = sorted(
        g for g, t in terms.items()
        if any(p not in NAMESPACE_ROOTS for p in t["is_a"])
    )
    eligible_set = set(eligible)
    eligible_by_ns = {}
    for g in eligible:
        eligible_by_ns.setdefault(terms[g]["namespace"], []).append(g)
    print(f"Eligible (non-top-level) terms: {len(eligible)} "
          f"{ {ns: len(v) for ns, v in sorted(eligible_by_ns.items())} }")

    proteins = load_proteins(con, terms, alt_map)
    print(f"Protein pool (reviewed, function >= {MIN_FUNC_CHARS} chars, "
          f">= {MIN_ANNOTATIONS} GO annotations): {len(proteins)}")

    # ---- Positives: sample (protein, eligible annotated term) pairs ----
    pos_pool = [
        (acc, g)
        for acc in sorted(proteins)
        for g in proteins[acc]["annotations"]
        if g in eligible_set
    ]
    rng.shuffle(pos_pool)
    per_protein = Counter()
    used_pairs = set()
    positives = []
    for acc, g in pos_pool:
        if len(positives) == N_POS:
            break
        if per_protein[acc] >= MAX_PER_PROTEIN_POS or (acc, g) in used_pairs:
            continue
        positives.append((acc, g))
        used_pairs.add((acc, g))
        per_protein[acc] += 1
    assert len(positives) == N_POS, f"only {len(positives)} positives"

    ns_target = Counter(terms[g]["namespace"] for _, g in positives)
    print(f"Positive namespace distribution: {dict(sorted(ns_target.items()))}")

    # ---- Negatives: same proteins, hierarchy-aware exclusion, namespace-matched ----
    excluded = {
        acc: set(proteins[acc]["annotations"]).union(
            *(ancestors[g] for g in proteins[acc]["annotations"])
        )
        for acc, _ in positives
    }
    slots = sorted({acc for acc, _ in positives}) * 2  # up to 2 negatives each
    rng.shuffle(slots)
    ns_needed = Counter(ns_target)
    negatives = []
    for acc in slots:
        if len(negatives) == N_NEG:
            break
        namespaces = sorted(
            (ns for ns in ns_needed if ns_needed[ns] > 0),
            key=lambda ns: (-ns_needed[ns], ns),
        )
        for ns in namespaces:
            g = next(
                (c for c in (rng.choice(eligible_by_ns[ns]) for _ in range(200))
                 if c not in excluded[acc] and (acc, c) not in used_pairs),
                None,
            )
            if g is not None:
                negatives.append((acc, g))
                used_pairs.add((acc, g))
                ns_needed[ns] -= 1
                break
    assert len(negatives) == N_NEG, f"only {len(negatives)} negatives"

    # ---- Assemble, validate, save ----
    pairs = [(acc, g, 1) for acc, g in positives] + [(acc, g, 0) for acc, g in negatives]
    rng.shuffle(pairs)

    counts = Counter(acc for acc, _, _ in pairs)
    assert max(counts.values()) <= 4, "protein appears > 4 times"
    assert len({(acc, g) for acc, g, _ in pairs}) == len(pairs), "duplicate (protein, term) pair"
    for acc, g, label in pairs:
        in_closure = g in proteins[acc]["annotations"] or g in excluded[acc]
        assert (label == 1) == (g in proteins[acc]["annotations"])
        assert label == 1 or not in_closure, "negative inside annotation/ancestor closure"

    ds = Dataset.from_dict({
        "sentence1": [f"{terms[g]['name']}: {terms[g]['definition']}" for _, g, _ in pairs],
        "sentence2": [proteins[acc]["function"] for acc, _, _ in pairs],
        "label": [label for _, _, label in pairs],
    })
    dsd = DatasetDict({"test": ds})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dsd.save_to_disk(str(OUT_DIR))

    neg_ns = Counter(terms[g]["namespace"] for _, g in negatives)
    manifest = {
        "name": "GOProteinPairCls",
        "category": "PairClassification",
        "sizes": {"test": len(ds), "positives": N_POS, "negatives": N_NEG},
        "provenance": {
            "source": RECORDS,
            "record_types": {
                "proteins": ["uniprot_human", "uniprot_protein"],
                "go_terms": ["go_term"],
            },
            "go_hierarchy": "is_a edges from go_term record metadata (no .obo parsing needed)",
            "seed": SEED,
            "build_script": "scripts/build_tier_c_pairs.py",
        },
        "protocol": {
            "sentence1": "GO term 'name: definition'",
            "sentence2": "UniProt Function section, FUNCTION: markers and {ECO:...} evidence tags stripped",
            "protein_pool": f"reviewed proteins, cleaned function text >= {MIN_FUNC_CHARS} chars, "
                            f">= {MIN_ANNOTATIONS} GO annotations resolvable to defined GO terms "
                            f"(alt_ids mapped to primary)",
            "term_pool": "GO terms with a definition and >= 1 is_a parent that is not a namespace "
                         "root (biases sampling away from top-level terms); applied to positives "
                         "and negatives alike",
            "positives": f"{N_POS} (protein, term) pairs sampled uniformly from actual annotations, "
                         f"<= {MAX_PER_PROTEIN_POS} positives per protein",
            "negatives": f"{N_NEG} terms sampled for the same proteins, excluding each protein's "
                         f"annotated terms and their full is_a ancestor closure (hierarchy-aware, "
                         f"avoids false negatives), namespace distribution matched to positives; "
                         f"<= 2 negatives per protein",
            "constraints": "no protein > 4 appearances total; no duplicate (protein, term) pairs",
            "ancestor_closure": "transitive closure over is_a edges only, memoised DFS, "
                                "cycle-safe; term itself plus all is_a ancestors excluded "
                                "when drawing negatives",
            "namespace_distribution": {
                "positives": dict(sorted(ns_target.items())),
                "negatives": dict(sorted(neg_ns.items())),
            },
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # ---- Final stats ----
    print(f"\nSaved DatasetDict{{test}} -> {OUT_DIR}")
    print(f"rows: {len(ds)}  positives: {N_POS}  negatives: {N_NEG}")
    print(f"distinct proteins: {len(counts)}  max appearances: {max(counts.values())}")
    print(f"namespaces  pos: {dict(sorted(ns_target.items()))}  neg: {dict(sorted(neg_ns.items()))}")
    s1 = [len(x) for x in ds["sentence1"]]
    s2 = [len(x) for x in ds["sentence2"]]
    print(f"sentence1 chars: min {min(s1)} / mean {sum(s1)//len(s1)} / max {max(s1)}")
    print(f"sentence2 chars: min {min(s2)} / mean {sum(s2)//len(s2)} / max {max(s2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
