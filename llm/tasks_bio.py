"""LLM task classes for the five locally built BioEmbed benchmark datasets.

Follows the patterns in vendor/embedders-dilemma/llm_judge/tasks/
(retrieval.py, classification.py, pair_cls.py) under pinned mteb==2.6.5,
but loads every dataset from data/tasks/<name> (datasets.DatasetDict saved
with save_to_disk) instead of the HF hub.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = REPO_ROOT / "vendor" / "embedders-dilemma"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

DATA_DIR = REPO_ROOT / "data" / "tasks"

from datasets import Features, Value, load_from_disk
from mteb.abstasks.task_metadata import TaskMetadata
from pydantic import Field

from llm_judge.evaluators.llm_classification_evaluator import LLMClassificationEvaluator
from llm_judge.tasks.classification import (
    AbsTaskLLMClassification,
    BaseResponse,
    ClassificationEvaluator,
)
from llm_judge.tasks.pair_cls import AbsTaskLLMPairClassification
from llm_judge.tasks.retrieval import AbsTaskLLMRetrieval


# ---------------------------------------------------------------------------
# Retrieval (LOFT corpus-in-context) — local variant of retrieval._make_metadata
# ---------------------------------------------------------------------------

def _make_local_metadata(
    name: str,
    description: str,
    task_type: str,
    main_score: str,
    domains: list[str],
    task_subtypes: list[str],
    category: str = "t2t",
) -> TaskMetadata:
    """Mimics retrieval._make_metadata/_make_metadata_v2 with a local path marker."""
    return TaskMetadata(
        name=name,
        dataset={"path": f"local/{name}", "revision": "local"},
        description=description,
        reference=None,  # locally built dataset — see data/tasks/<name>/manifest.json
        category=category,
        modalities=["text"],
        type=task_type,
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score=main_score,
        date=("2000-01-01", "2025-12-31"),
        domains=domains,
        task_subtypes=task_subtypes,
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )


class AbsTaskLocalLLMRetrieval(AbsTaskLLMRetrieval):
    """AbsTaskLLMRetrieval that loads corpus/queries/qrels from a local DatasetDict.

    Splits: corpus (_id, title, text), queries (_id, text),
    qrels (query-id, corpus-id, score) — same LOFT layout the vendor loads
    from the HF hub in AbsTaskLLMRetrieval.load_data.
    """

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        dd = load_from_disk(str(DATA_DIR / self.metadata.name))

        # NOTE: doc "text" already contains the title where available (see the
        # dataset manifests). Title is set to "" so the LOFT doc formatting
        # (_format_doc: "ID | TITLE | CONTENT") does not duplicate it.
        self._corpus = {row["_id"]: {"title": "", "text": row["text"]} for row in dd["corpus"]}
        self._queries = {row["_id"]: row["text"] for row in dd["queries"]}
        self._qrels = defaultdict(dict)
        for row in dd["qrels"]:
            self._qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])

        self.data_loaded = True


class GOPubMedRetrieval(AbsTaskLocalLLMRetrieval):
    """GO term (name: definition) -> PubMed abstracts with experimental GO evidence.
    Corpus: 300 docs. 75 queries, ~3.2 gold docs each."""

    metadata = _make_local_metadata(
        name="GOPubMedRetrieval",
        description=(
            "Retrieve the PubMed title+abstracts that provide experimental evidence "
            "for a Gene Ontology term, given the term's name and definition."
        ),
        task_type="Retrieval",
        main_score="recall@1",
        domains=["Medical", "Academic", "Written"],
        task_subtypes=["Article retrieval"],
    )


class R2MEDBiologyPooled(AbsTaskLocalLLMRetrieval):
    """R2MED Biology retrieval pooled to a small corpus. Corpus: 363 docs
    (all gold — the gold docs of the 100 selected queries exceeded the pool cap)."""

    metadata = _make_local_metadata(
        name="R2MEDBiologyPooled",
        description=(
            "R2MED Biology: retrieve the reference passages that answer a biology "
            "question (100-query sample, corpus pooled from 57359 to 363 docs)."
        ),
        task_type="Retrieval",
        main_score="recall@1",
        domains=["Medical", "Academic", "Written"],
        task_subtypes=["Question answering"],
    )


# ---------------------------------------------------------------------------
# Classification — pattern of classification.LLMBanking77Classification
# ---------------------------------------------------------------------------

# The 30 MeSH descriptor label strings: sorted set of the dataset's 'label' column.
MESH_LABELS = [
    "Adenocarcinoma", "Aging", "Anti-Bacterial Agents",
    "Antineoplastic Agents", "Asthma", "Bacterial Proteins",
    "Brain", "Breast Neoplasms", "Calcium",
    "Carcinoma, Squamous Cell", "Cardiovascular Diseases", "DNA",
    "Diabetes Mellitus, Type 2", "Escherichia coli", "HIV Infections",
    "Hypertension", "Kidney", "Liver",
    "Liver Neoplasms", "Lung", "Lung Neoplasms",
    "Mental Disorders", "Myocardial Infarction", "Neoplasms",
    "Neurons", "Obesity", "Postoperative Complications",
    "Prostatic Neoplasms", "Proteins", "Skin Neoplasms",
]

_MESH_LABEL2IDX = {label: i for i, label in enumerate(MESH_LABELS)}


class BioMeSHClassification(AbsTaskLLMClassification):
    metadata = _make_local_metadata(
        name="BioMeSHClassification",
        description=(
            "Assign one of the 30 most frequent MeSH descriptors to a PubMed "
            "title+abstract (30-way, from allenai/scirepeval mesh_descriptors)."
        ),
        task_type="Classification",
        main_score="accuracy",
        domains=["Medical", "Academic", "Written"],
        task_subtypes=["Topic classification"],
        category="t2c",
    )
    train_split = "train"

    class MeSHResponse(BaseResponse):
        output: str = Field(description="The MeSH descriptor name, exactly as listed.")

        @property
        def idx2label(self) -> dict[int, str]:
            return {i: label for i, label in enumerate(MESH_LABELS)}

    evaluator_model = ClassificationEvaluator(
        instruction=(
            "You are given the title and abstract of a PubMed article. "
            "Assign the article to exactly one of the 30 MeSH (Medical Subject Headings) "
            "descriptors below. Output the exact descriptor name. "
            "Output json with fields 'reasoning' and 'output'.\n\n"
            "Descriptors:\n" + ", ".join(MESH_LABELS)
        ),
        response_model=MeSHResponse,
    )
    evaluator = LLMClassificationEvaluator

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        dd = load_from_disk(str(DATA_DIR / self.metadata.name))
        # mteb's classification stack (and the LLM evaluator's idx2label mapping)
        # expects integer labels; the local dataset stores descriptor names.
        # features= is needed because .map alone keeps the original string dtype;
        # keep_in_memory avoids writing cache files into the tracked data dir.
        self.dataset = dd.map(
            lambda row: {"label": _MESH_LABEL2IDX[row["label"]]},
            features=Features({"text": dd["test"].features["text"], "label": Value("int64")}),
            keep_in_memory=True,
        )
        self.dataset_transform()
        self.data_loaded = True


# ---------------------------------------------------------------------------
# Pair classification — pattern of pair_cls.LLMSprintDuplicateQuestionsPC
# (binary judgment with reasoning; local DatasetDict split 'test' with
# columns sentence1/sentence2/label already in mteb layout)
# ---------------------------------------------------------------------------

class AbsTaskLocalLLMPairClassification(AbsTaskLLMPairClassification):
    label_column_name = "label"

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = load_from_disk(str(DATA_DIR / self.metadata.name))
        self.dataset_transform()
        self.data_loaded = True


class PubChemSynonymPC500(AbsTaskLocalLLMPairClassification):
    metadata = _make_local_metadata(
        name="PubChemSynonymPC500",
        description=(
            "Decide whether two chemical identifiers/names are synonyms for the "
            "same compound (500-pair sample of BASF-AI/PubChemSynonymPC)."
        ),
        task_type="PairClassification",
        main_score="max_ap",
        domains=["Chemistry", "Written"],
        task_subtypes=["Duplicate Detection"],
    )
    instruction = (
        "Sentence 1 and Sentence 2 are chemical identifiers or names. "
        "Determine whether they are synonyms for the same chemical compound. "
        "Output json with fields 'reasoning' and 'output' (1 if they name the same compound, 0 if not)."
    )


class GOProteinPairCls(AbsTaskLocalLLMPairClassification):
    metadata = _make_local_metadata(
        name="GOProteinPairCls",
        description=(
            "Decide whether a GO term (name: definition) describes a function, "
            "process, or component of a protein given its UniProt Function text."
        ),
        task_type="PairClassification",
        main_score="max_ap",
        domains=["Medical", "Academic", "Written"],
        task_subtypes=["Textual Entailment"],
    )
    instruction = (
        "Sentence 1 is a Gene Ontology (GO) term, given as 'name: definition'. "
        "Sentence 2 is the UniProt Function description of a protein. "
        "Determine whether the GO term describes a function, process, or component of this protein. "
        "Output json with fields 'reasoning' and 'output' (1 if the GO term applies to the protein, 0 if not)."
    )


BIO_TASKS = [
    GOPubMedRetrieval,
    R2MEDBiologyPooled,
    BioMeSHClassification,
    PubChemSynonymPC500,
    GOProteinPairCls,
]
