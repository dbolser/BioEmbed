"""Build the three Tier B datasets.

Deterministic (SEED=42). Re-runnable: overwrites data/tasks/<Name>/.

  1. BioMeSHClassification   - Classification, 30 real MeSH descriptor labels
  2. PubChemSynonymPC500     - PairClassification, 500 pairs, original balance
  3. R2MEDBiologyPooled      - Retrieval, corpus pooled to ~300 docs (LOFT layout)

Run: uv run python scripts/build_tier_b.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "tasks"


def save(name: str, dd: DatasetDict, manifest: dict) -> None:
    path = OUT / name
    dd.save_to_disk(str(path))
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# 1. BioMeSHClassification
# ---------------------------------------------------------------------------
def build_mesh() -> None:
    from sklearn.model_selection import train_test_split

    ds = load_dataset(
        "allenai/scirepeval",
        "mesh_descriptors",
        split="evaluation",
        revision="781d35d1bf87253b3dcd0fadcb82bfbee9c244f1",
    )
    df = ds.to_pandas()
    n_raw = len(df)

    # Deduplicate by doc_id (mteb dedupes by corpus_id; identical values here).
    df = df.drop_duplicates(subset="doc_id", keep="first")
    n_dedup = len(df)

    # Text construction matches mteb 2.20 SciRepEvalMeSHDescriptorsClassification
    # dataset_transform: f"{title}\n\n{abstract}".strip()
    df["text"] = (
        df["title"].fillna("") + "\n\n" + df["abstract"].fillna("")
    ).str.strip()
    df = df[df["text"] != ""]
    # Safety: drop exact-duplicate texts so train/test cannot share a document
    # that appears under two doc_ids.
    df = df.drop_duplicates(subset="text", keep="first")

    n_desc_total = df["descriptor"].nunique()
    top30 = df["descriptor"].value_counts().head(30).index.tolist()
    df = df[df["descriptor"].isin(top30)][["text", "descriptor"]].rename(
        columns={"descriptor": "label"}
    )
    pool = len(df)

    # Seed-42 stratified sample: 2500 total, then 2000 train / 500 test.
    sampled, _ = train_test_split(
        df, train_size=2500, stratify=df["label"], random_state=SEED
    )
    train, test = train_test_split(
        sampled, train_size=2000, stratify=sampled["label"], random_state=SEED
    )

    dd = DatasetDict(
        {
            "train": Dataset.from_pandas(train, preserve_index=False),
            "test": Dataset.from_pandas(test, preserve_index=False),
        }
    )
    manifest = {
        "name": "BioMeSHClassification",
        "category": "Classification",
        "sizes": {"train": len(train), "test": len(test), "n_classes": 30},
        "provenance": {
            "dataset": "allenai/scirepeval",
            "config": "mesh_descriptors",
            "split": "evaluation",
            "revision": "781d35d1bf87253b3dcd0fadcb82bfbee9c244f1",
            "rows_raw": n_raw,
            "rows_after_doc_id_dedup": n_dedup,
            "rows_in_top30_pool": pool,
            "descriptors_in_source_after_dedup": n_desc_total,
        },
        "protocol": (
            "Deduplicated by doc_id (then by exact text as leakage safety). "
            "text = title + '\\n\\n' + abstract, matching mteb 2.20 "
            "SciRepEvalMeSHDescriptorsClassification.dataset_transform. "
            "Labels are the 30 most frequent descriptor NAME strings. "
            f"Seed-{SEED} stratified sample via sklearn train_test_split: "
            "2500 rows, split 2000 train / 500 test (no doc overlap; train "
            "feeds the embedder kNN protocol)."
        ),
    }
    save("BioMeSHClassification", dd, manifest)

    print("-- BioMeSHClassification --")
    print(f"source rows {n_raw} -> dedup {n_dedup} -> top-30 pool {pool}")
    print(f"train {len(train)}  test {len(test)}  classes {sampled['label'].nunique()}")
    tc = test["label"].value_counts()
    print(f"test per-class: min {tc.min()} max {tc.max()}")
    print("example labels:", sorted(top30)[:5])


# ---------------------------------------------------------------------------
# 2. PubChemSynonymPC500
# ---------------------------------------------------------------------------
def build_pubchem() -> None:
    import mteb

    task = mteb.get_task("PubChemSynonymPC")
    task.load_data()
    d = task.dataset["test"][0]  # dict of parallel arrays
    s1, s2, labels = d["sentence1"], d["sentence2"], d["labels"]
    n = len(labels)
    counts = Counter(labels)

    # Seed-42 sample of 500 preserving the original label balance.
    rng = np.random.default_rng(SEED)
    n_pos = round(500 * counts[1] / n)
    take = {1: n_pos, 0: 500 - n_pos}
    idx: list[int] = []
    for lab, k in take.items():
        cand = [i for i, l in enumerate(labels) if l == lab]
        idx.extend(rng.choice(cand, size=k, replace=False).tolist())
    idx.sort()

    dd = DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "sentence1": [str(s1[i]).strip() for i in idx],
                    "sentence2": [str(s2[i]).strip() for i in idx],
                    "label": [int(labels[i]) for i in idx],
                }
            )
        }
    )
    manifest = {
        "name": "PubChemSynonymPC500",
        "category": "PairClassification",
        "sizes": {"test": 500, "positives": take[1], "negatives": take[0]},
        "provenance": {
            "dataset": "BASF-AI/PubChemSynonymPC",
            "revision": "5037d69d177c9628fb79cb57eea1299178b28c1b",
            "via": "mteb.get_task('PubChemSynonymPC').load_data()",
            "source_pairs": n,
            "source_label_counts": {str(k): v for k, v in sorted(counts.items())},
        },
        "protocol": (
            f"Seed-{SEED} per-class random sample of 500 pairs preserving the "
            f"original label balance ({counts[1]}/{n} positive -> {take[1]}/500). "
            "Kept in original dataset order. label: 1 = synonym pair."
        ),
    }
    save("PubChemSynonymPC500", dd, manifest)

    print("-- PubChemSynonymPC500 --")
    print(f"source {n} pairs {dict(counts)} -> 500 pairs "
          f"(pos {take[1]}, neg {take[0]})")


# ---------------------------------------------------------------------------
# 3. R2MEDBiologyPooled
# ---------------------------------------------------------------------------
def build_r2med() -> None:
    import mteb

    task = mteb.get_task("R2MEDBiologyRetrieval")
    task.load_data()
    corpus = task.corpus["test"]          # doc_id -> {"text": ..., ("title")}
    queries = task.queries["test"]        # qid -> str
    qrels = task.relevant_docs["test"]    # qid -> {doc_id: score}
    n_corpus = len(corpus)

    rng = np.random.default_rng(SEED)
    qids = sorted(queries)
    if len(qids) > 100:
        qids = sorted(rng.choice(qids, size=100, replace=False).tolist())

    gold = sorted({doc for q in qids for doc in qrels.get(q, {})})
    pool = list(gold)
    n_distractors = 0
    if len(pool) < 300:
        non_gold = sorted(set(corpus) - set(gold))
        n_distractors = 300 - len(pool)
        pool += rng.choice(non_gold, size=n_distractors, replace=False).tolist()
    pool = sorted(pool)

    def doc_text(did: str) -> dict:
        doc = corpus[did]
        return {
            "_id": did,
            "title": str(doc.get("title") or "").strip(),
            "text": str(doc.get("text") or "").strip(),
        }

    corpus_ds = Dataset.from_list([doc_text(d) for d in pool])
    queries_ds = Dataset.from_list(
        [{"_id": q, "text": str(queries[q]).strip()} for q in qids]
    )
    qrel_rows = [
        {"query-id": q, "corpus-id": d, "score": int(s)}
        for q in qids
        for d, s in sorted(qrels.get(q, {}).items())
        if d in set(pool)
    ]
    qrels_ds = Dataset.from_list(qrel_rows)

    dd = DatasetDict(
        {"corpus": corpus_ds, "queries": queries_ds, "qrels": qrels_ds}
    )
    manifest = {
        "name": "R2MEDBiologyPooled",
        "category": "Retrieval",
        "sizes": {
            "queries": len(queries_ds),
            "corpus": len(corpus_ds),
            "qrels": len(qrels_ds),
            "gold_docs": len(gold),
            "distractor_docs": n_distractors,
        },
        "provenance": {
            "dataset": "R2MED/Biology",
            "revision": "8b9fec2db9eda4b5742d03732213fbaee8169556",
            "via": "mteb.get_task('R2MEDBiologyRetrieval').load_data()",
            "original_corpus_size": n_corpus,
            "original_queries": len(queries),
        },
        "protocol": (
            f"Seed-{SEED} sample of 100 of {len(queries)} queries. Corpus pool = "
            "every gold doc of the selected queries + random distractors from "
            f"the full corpus up to 300 total (seed {SEED}). Qrels = all original "
            "judgments restricted to selected queries (all gold docs are in the "
            "pool). LOFT layout: corpus(_id,title,text), queries(_id,text), "
            "qrels(query-id,corpus-id,score). NOTE: pooling the corpus from "
            f"{n_corpus} to {len(pool)} docs reduces difficulty vs full-corpus "
            "R2MED-Biology; scores are not comparable to published R2MED numbers."
            + (
                " The gold docs alone exceeded the 300-doc cap, so the pool "
                "contains no distractors: every corpus doc is relevant to "
                "some selected query."
                if n_distractors == 0
                else ""
            )
        ),
    }
    save("R2MEDBiologyPooled", dd, manifest)

    print("-- R2MEDBiologyPooled --")
    print(f"queries {len(queries_ds)}/{len(queries)}  "
          f"corpus {len(corpus_ds)} (gold {len(gold)}, distractors {n_distractors}) "
          f"of original {n_corpus}  qrels {len(qrel_rows)}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build_mesh()
    build_pubchem()
    build_r2med()
    print("done")
