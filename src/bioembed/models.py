"""Embedding model registry: the Pareto ladder.

Two axes the paper doesn't have:
  - bio-specialised vs general models at matched sizes
  - (later) our own cheaply fine-tuned models

`mteb_name` is the name mteb.get_model() resolves (usually the HF id).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EmbModel:
    hf_id: str
    params_m: float          # millions of parameters
    family: str              # "general" | "bio" | "ours"
    notes: str = ""
    encode_kwargs: dict = field(default_factory=dict)
    # Paper models: their published H100 throughput lets us reuse their $/token
    in_paper: bool = False


GENERAL_LADDER = [
    EmbModel("sentence-transformers/all-MiniLM-L6-v2", 22, "general", "floor baseline"),
    EmbModel("BAAI/bge-small-en-v1.5", 33, "general", "MedEmbed-small's base"),
    EmbModel("BAAI/bge-base-en-v1.5", 109, "general", "MedEmbed-base's base"),
    EmbModel("BAAI/bge-large-en-v1.5", 335, "general", "MedEmbed-large's base; IUK-IPF corpus model"),
    EmbModel("intfloat/multilingual-e5-small", 118, "general", "paper's cheapest model", in_paper=True),
    EmbModel("google/embeddinggemma-300m", 308, "general", in_paper=True),
    EmbModel("Snowflake/snowflake-arctic-embed-l-v2.0", 568, "general", in_paper=True),
    EmbModel("Qwen/Qwen3-Embedding-0.6B", 596, "general", "open-SOTA family", in_paper=True),
    EmbModel("Qwen/Qwen3-Embedding-4B", 4000, "general", in_paper=True),
    EmbModel("Qwen/Qwen3-Embedding-8B", 7600, "general", in_paper=True),
]

BIO_SPECIALISTS = [
    EmbModel("NeuML/pubmedbert-base-embeddings", 109, "bio", "PubMedBERT sentence embedder"),
    EmbModel("ncbi/MedCPT-Query-Encoder", 109, "bio",
             "asymmetric bi-encoder (255M PubMed click pairs); article encoder swapped in for corpus side"),
    EmbModel("abhinand/MedEmbed-small-v0.1", 33, "bio", "bge-small + synthetic bio triplets"),
    EmbModel("abhinand/MedEmbed-base-v0.1", 109, "bio", "bge-base + synthetic bio triplets"),
    EmbModel("abhinand/MedEmbed-large-v0.1", 335, "bio", "bge-large + synthetic bio triplets"),
    EmbModel("FremyCompany/BioLORD-2023", 109, "bio", "definition/KG-grounded, concept-level"),
    EmbModel("BMRetriever/BMRetriever-410M", 410, "bio", "bio decoder-embedder ladder"),
    EmbModel("BMRetriever/BMRetriever-1B", 1000, "bio"),
]

ALL_MODELS = GENERAL_LADDER + BIO_SPECIALISTS


def by_id(hf_id: str) -> EmbModel:
    for m in ALL_MODELS:
        if m.hf_id == hf_id:
            return m
    raise KeyError(hf_id)
