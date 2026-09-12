"""BioMTEB(LLM) task suite definition (critic-revised).

Principle: any task already in the Embedder's Dilemma paper has published
LLM + embedding results (vendor/embedders-dilemma) on public seed-42 subsets
(HF mteb/llm-eval-*). Those cost nothing to reuse — the suite maximizes
overlap and spends LLM budget only on genuinely new bio tasks.

Tiers:
  A. Lifted: paper tasks (bio + two general anchors). LLM results free;
     we run only OUR additional embedders on the same public subsets.
  B. New-public: bio tasks sampled seed-42 from the MTEB registry with the
     paper's recipe (queries AND corpus subsampled so corpus-in-context
     fits ~128k; difficulty change documented).
  C. New-custom: molecular-biology tasks from human-curated GAF/UniProt/GO
     cross-references (IUK-IPF data, read-only).
"""

SEED = 42

# Tier A: paper task name -> (category, HF subset dataset)
TIER_A = {
    "LLMBIOSSES":                 ("STS", "mteb/llm-eval-biosses"),
    "LLMSTSBenchmark":            ("STS", "mteb/llm-eval-stsbenchmark"),          # general anchor
    "LLMBiorxivClusteringP2PV2":  ("Clustering", "mteb/llm-eval-biorxiv_clustering_p2p_v2"),
    "LLMMedrxivClusteringP2PV2":  ("Clustering", "mteb/llm-eval-medrxiv_clustering_p2p_v2"),
    "LLMMedrxivClusteringS2SV2":  ("Clustering", "mteb/llm-eval-medrxiv_clustering_s2s_v2"),
    "LLMToxicConversationsClassification": ("Classification", "mteb/llm-eval-toxic_conversations"),  # binary anchor
    "LLMPublicHealthQA":          ("Retrieval", "mteb/llm-eval-public-health-qa"),
}

# Tier B: new bio tasks from public MTEB sources.
TIER_B = {
    # fine-grained many-class biomedical classification (the Banking77
    # analogue that carries the paper's biggest embedder win).
    # kNN needs a train split - verified before building.
    "BioMeSHClassification": {
        "category": "Classification",
        "source": "SciRepEvalMeSHDescriptorsClassification",
        "max_test": 500, "max_train": 2000,
    },
    # bio-adjacent pair classification (synonym pairs)
    "PubChemSynonymPC500": {
        "category": "PairClassification",
        "source": "PubChemSynonymPC",
        "max_pairs": 500,
    },
    # reasoning-heavy biology retrieval; corpus pooled to gold + distractors
    # (~300 docs) so the LLM corpus-in-context protocol is feasible.
    # NOTE: pooling reduces difficulty vs full-corpus R2MED - documented.
    "R2MEDBiologyPooled": {
        "category": "Retrieval",
        "source": "R2MEDBiologyRetrieval",
        "max_queries": 100, "corpus_budget": 300,
    },
}

# Tier C: built from IUK-IPF curated data (scripts/build_tier_c.py).
TIER_C = {
    # query = GO term name + definition; corpus = PubMed abstracts cited in
    # GO annotations; gold = GO-Consortium GAF links (human-curated).
    "GOPubMedRetrieval": {
        "category": "Retrieval",
        "max_queries": 75, "corpus_budget": 300,
    },
    # (GO term text, UniProt function text): annotated-or-not, with
    # hierarchy-aware negatives (GO parents excluded from negatives).
    "GOProteinPairCls": {
        "category": "PairClassification",
        "max_pairs": 500,
    },
}

CATEGORIES = ["Retrieval", "Classification", "Clustering", "STS", "PairClassification"]
