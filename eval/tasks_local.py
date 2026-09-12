"""Tier B and Tier C tasks: mteb 2.11.6 task classes over the locally built
datasets in data/tasks/ (see scripts/build_tier_b.py / build_tier_c_*.py).

Both paradigms read exactly these files; the LLM pipeline consumes the same
DatasetDicts via its own task classes.
"""

from pathlib import Path

from datasets import DatasetDict, load_from_disk
from mteb.abstasks import AbsTaskPairClassification, AbsTaskRetrieval
from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.tasks import ToxicConversationsClassification
from mteb.tasks.retrieval.multilingual.public_health_qa_retrieval import (
    PublicHealthQARetrieval,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "tasks"

# metadata templates: copied from same-type built-ins, then renamed. The
# provenance fields (citation, reference...) intentionally stay generic; the
# authoritative record is each dataset's manifest.json.
_CLS_TMPL = ToxicConversationsClassification.metadata
_RET_TMPL = PublicHealthQARetrieval.metadata


def _meta(template, name, description, **extra):
    return template.model_copy(update={
        "name": name,
        "description": description,
        "dataset": {"path": f"local/{name}", "revision": "local"},
        "eval_splits": ["test"],
        "eval_langs": ["eng-Latn"],
        **extra,
    })


class BioMeSHClassification(AbsTaskClassification):
    metadata = _meta(_CLS_TMPL, "BioMeSHClassification",
                     "30-class MeSH descriptor assignment for PubMed papers "
                     "(title+abstract), from allenai/scirepeval mesh_descriptors.")

    def load_data(self, num_proc=None, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = load_from_disk(str(DATA / "BioMeSHClassification"))
        self.data_loaded = True


class _LocalPairTask(AbsTaskPairClassification):
    local_name: str

    def load_data(self, num_proc=None, **kwargs) -> None:
        if self.data_loaded:
            return
        dd = load_from_disk(str(DATA / self.local_name))
        if "label" in dd["test"].column_names:
            dd = DatasetDict({k: v.rename_column("label", "labels") for k, v in dd.items()})
        self.dataset = dd
        self.data_loaded = True


# template for pair tasks: borrow from a built-in pair classification task
from mteb.tasks import SprintDuplicateQuestionsPC  # noqa: E402

_PAIR_TMPL = SprintDuplicateQuestionsPC.metadata


class PubChemSynonymPC500(_LocalPairTask):
    local_name = "PubChemSynonymPC500"
    metadata = _meta(_PAIR_TMPL, "PubChemSynonymPC500",
                     "Are these two chemical names synonyms? 500-pair seed-42 "
                     "sample of PubChemSynonymPC.")


class GOProteinPairCls(_LocalPairTask):
    local_name = "GOProteinPairCls"
    metadata = _meta(_PAIR_TMPL, "GOProteinPairCls",
                     "Does this GO term (name: definition) describe this "
                     "protein (UniProt function text)? Hierarchy-aware "
                     "negatives; gold from curated GO annotations.")


class _LocalRetrievalTask(AbsTaskRetrieval):
    local_name: str

    def load_data(self, num_proc=None, **kwargs) -> None:
        if self.data_loaded:
            return
        dd = load_from_disk(str(DATA / self.local_name))
        corpus = dd["corpus"].rename_column("_id", "id")
        queries = dd["queries"].rename_column("_id", "id")
        qrels: dict[str, dict[str, int]] = {}
        for row in dd["qrels"]:
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
        self.dataset = {"default": {"test": RetrievalSplitData(
            corpus=corpus, queries=queries, relevant_docs=qrels, top_ranked=None,
        )}}
        self.data_loaded = True


class R2MEDBiologyPooled(_LocalRetrievalTask):
    local_name = "R2MEDBiologyPooled"
    metadata = _meta(_RET_TMPL, "R2MEDBiologyPooled",
                     "Reasoning-heavy biology retrieval (R2MED Biology), "
                     "100 queries, corpus pooled to 363 docs for the LLM "
                     "corpus-in-context protocol.")


class GOPubMedRetrieval(_LocalRetrievalTask):
    local_name = "GOPubMedRetrieval"
    metadata = _meta(_RET_TMPL, "GOPubMedRetrieval",
                     "Retrieve PubMed abstracts cited by GO curators as "
                     "experimental evidence for a GO term (name: definition). "
                     "75 queries, 300 docs, GAF gold.")


LOCAL_TASKS = [
    BioMeSHClassification(),
    PubChemSynonymPC500(),
    GOProteinPairCls(),
    R2MEDBiologyPooled(),
    GOPubMedRetrieval(),
]
