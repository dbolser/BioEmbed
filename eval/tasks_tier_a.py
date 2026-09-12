"""Tier A: the paper's biomedical tasks (+ two general anchors) as standard
mteb tasks pointing at the paper's public seed-42 subsets (mteb/llm-eval-*).

Embedding models evaluated on these see exactly the data the paper's LLMs
read; the paper's published LLM and embedding results are directly
comparable. Revisions pinned to the ones in vendor/embedders-dilemma.
"""

from mteb.tasks import (
    BiorxivClusteringP2P,
    MedrxivClusteringP2P,
    MedrxivClusteringS2S,
    ToxicConversationsClassification,
)
from mteb.tasks import BiossesSTS, STSBenchmarkSTS
from mteb.tasks.retrieval.multilingual.public_health_qa_retrieval import (
    PublicHealthQARetrieval as _PHQA_base,  # multilingual original
)
from mteb.abstasks import AbsTaskRetrieval


def _sub(base, name, path, revision, **extra):
    update = {"name": name, "dataset": {"path": path, "revision": revision},
              "eval_splits": ["test"], **extra}
    return type(name, (base,), {"metadata": base.metadata.model_copy(update=update)})


LLMBIOSSES = _sub(BiossesSTS, "LLMBIOSSES",
                  "mteb/llm-eval-biosses", "cf968edb41fa17a96392f6b373819efac1c2d6d6")
LLMSTSBenchmark = _sub(STSBenchmarkSTS, "LLMSTSBenchmark",
                       "mteb/llm-eval-stsbenchmark", "86bbaf4470f501ee381411836b3a22f112bfe42a")
LLMBiorxivClusteringP2PV2 = _sub(BiorxivClusteringP2P, "LLMBiorxivClusteringP2PV2",
                                 "mteb/llm-eval-biorxiv_clustering_p2p_v2", "9e11c95384ef78952ba754f9d8942084ddbb61a7")
LLMMedrxivClusteringP2PV2 = _sub(MedrxivClusteringP2P, "LLMMedrxivClusteringP2PV2",
                                 "mteb/llm-eval-medrxiv_clustering_p2p_v2", "63c8e6cfbcab3f986291141799fe646c60bb441c")
LLMMedrxivClusteringS2SV2 = _sub(MedrxivClusteringS2S, "LLMMedrxivClusteringS2SV2",
                                 "mteb/llm-eval-medrxiv_clustering_s2s_v2", "c565eea82f4a8728b8fe5181388b407d305f2647")
LLMToxicConversationsClassification = _sub(
    ToxicConversationsClassification, "LLMToxicConversationsClassification",
    "mteb/llm-eval-toxic_conversations", "34eeb6105ca217433c04b207ea66810a2ff42625")


# PublicHealthQA has no monolingual standard base task; build metadata from the
# multilingual original, restricted to English, on the llm-eval subset.
LLMPublicHealthQA = type(
    "LLMPublicHealthQA",
    (AbsTaskRetrieval,),
    {"metadata": _PHQA_base.metadata.model_copy(update={
        "name": "LLMPublicHealthQA",
        "dataset": {"path": "mteb/llm-eval-public-health-qa",
                    "revision": "b05938525381b7aebc079f88fc3ed8f572a80bb9"},
        "eval_splits": ["test"],
        "eval_langs": ["eng-Latn"],
    })},
)


TIER_A_TASKS = [
    LLMBIOSSES(),
    LLMSTSBenchmark(),
    LLMBiorxivClusteringP2PV2(),
    LLMMedrxivClusteringP2PV2(),
    LLMMedrxivClusteringS2SV2(),
    LLMToxicConversationsClassification(),
    LLMPublicHealthQA(),
]
