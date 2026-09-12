"""Offline verification: instantiate the five bio LLM tasks, load their local
data, and estimate per-model-pass input tokens (len(text)/4 proxy). No API calls.

    uv run python verify_bio_tasks.py     # from llm/ (llm/.env dummy values suffice)
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# tasks_bio puts the vendor repo on sys.path before importing llm_judge.
from tasks_bio import (
    BIO_TASKS,
    BioMeSHClassification,
    GOProteinPairCls,
    GOPubMedRetrieval,
    PubChemSynonymPC500,
    R2MEDBiologyPooled,
)
from llm_judge.evaluators.llm_retrieval_evaluator import SYSTEM_PROMPT, _format_doc


def toks(chars: int) -> int:
    return round(chars / 4)


def retrieval_estimate(task) -> tuple[dict, int]:
    """Rebuild the per-query LOFT prompt (system + full corpus + query wrapper)."""
    doc_lines = [
        _format_doc(cid, doc["title"], doc["text"], seq)
        for seq, cid in enumerate(sorted(task._corpus))
        for doc in [task._corpus[cid]]
    ]
    docs_chars = len("\n\n".join(doc_lines))
    total_chars = 0
    for qtext in task._queries.values():
        # query wrapper text from LLMRetrievalEvaluator._score_query (~330 chars)
        total_chars += len(SYSTEM_PROMPT) + docs_chars + 330 + len(qtext)
    info = {
        "queries": len(task._queries),
        "corpus docs": len(task._corpus),
        "qrels": sum(len(v) for v in task._qrels.values()),
        "corpus tokens/query": toks(docs_chars),
    }
    return info, toks(total_chars)


def classification_estimate(task) -> tuple[dict, int]:
    instruction = task.evaluator_model.instruction
    test = task.dataset["test"]
    total_chars = sum(len(instruction) + len(t) for t in test["text"])
    info = {"test docs": len(test), "train docs (unused by LLM)": len(task.dataset["train"])}
    return info, toks(total_chars)


def pair_estimate(task) -> tuple[dict, int]:
    test = task.dataset["test"]
    total_chars = sum(
        len(task.instruction) + len(f"Sentence 1: {s1}\nSentence 2: {s2}")
        for s1, s2 in zip(test["sentence1"], test["sentence2"])
    )
    info = {"pairs": len(test), "positives": sum(int(x) for x in test["label"])}
    return info, toks(total_chars)


ESTIMATORS = {
    GOPubMedRetrieval: retrieval_estimate,
    R2MEDBiologyPooled: retrieval_estimate,
    BioMeSHClassification: classification_estimate,
    PubChemSynonymPC500: pair_estimate,
    GOProteinPairCls: pair_estimate,
}


def main() -> None:
    grand_total = 0
    for cls in BIO_TASKS:
        task = cls()
        task.load_data()
        info, est = ESTIMATORS[cls](task)
        grand_total += est
        detail = ", ".join(f"{k}: {v:,}" for k, v in info.items())
        print(f"{task.metadata.name:24s} {detail}")
        print(f"{'':24s} est. input tokens/pass: {est:,}")
    print(f"\nTOTAL estimated input tokens per model pass: {grand_total:,}")


if __name__ == "__main__":
    main()
