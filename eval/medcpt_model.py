"""MedCPT asymmetric bi-encoder for mteb 2.11: queries through the
Query Encoder, documents (and any non-query text) through the Article
Encoder — NCBI's intended usage (both produce 768-d compatible vectors).

As on the model card: [CLS] pooling, query max length 64, article max
length 512, inner-product similarity. The checkpoints ship no
sentence-transformers config, so SentenceTransformer(name) would silently
mean-pool; each encoder is built explicitly. Articles are encoded as one
segment ('title\\n\\nabstract'), not the card's [title, abstract] pair.

MedCPT(cosine=True) is reported as ncbi/MedCPT-cosine.
"""

import numpy as np
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling, Transformer

QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"
QUERY_MAX_LEN = 64
ARTICLE_MAX_LEN = 512

# Written out (as mteb's own registry does) rather than ModelMeta._from_hub,
# which needs the network; values are what _from_hub returned.
META = ModelMeta(
    loader=None, name="ncbi/MedCPT", revision="d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc",
    release_date="2023-10-24", languages=None, n_parameters=109_482_240,
    n_embedding_parameters=23_440_896, memory_usage_mb=418, max_tokens=512, embed_dim=768,
    license=None, open_weights=None, public_training_code=None, public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference=f"https://huggingface.co/{QUERY_ENCODER}", similarity_fn_name=None,
    use_instructions=None, training_datasets=None,
)


def _encoder(name: str, max_len: int, sim: str) -> SentenceTransformer:
    word = Transformer(name, max_seq_length=max_len)
    pool = Pooling(word.get_embedding_dimension(), pooling_mode="cls")
    return SentenceTransformer(modules=[word, pool], similarity_fn_name=sim)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class MedCPT:
    def __init__(self, cosine: bool = False) -> None:
        self.cosine = cosine
        sim = ScoringFunction.COSINE if cosine else ScoringFunction.DOT_PRODUCT
        self.query_enc = _encoder(QUERY_ENCODER, QUERY_MAX_LEN, sim.value)
        self.article_enc = _encoder(ARTICLE_ENCODER, ARTICLE_MAX_LEN, sim.value)
        self.mteb_model_meta = META.model_copy(update=dict(
            name="ncbi/MedCPT-cosine" if cosine else "ncbi/MedCPT",
            similarity_fn_name=sim, loader=lambda *a, **k: self))

    def _texts(self, inputs) -> list[str]:
        return [t for batch in inputs for t in batch["text"]]

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
               prompt_type=None, **kwargs) -> np.ndarray:
        enc = self.query_enc if prompt_type == PromptType.query else self.article_enc
        return enc.encode(self._texts(inputs),
                          batch_size=kwargs.get("batch_size", 32),
                          convert_to_numpy=True, show_progress_bar=False)

    def similarity(self, a, b) -> np.ndarray:
        a, b = np.asarray(a), np.asarray(b)
        if self.cosine:
            a, b = _unit(a), _unit(b)
        return a @ b.T

    def similarity_pairwise(self, a, b) -> np.ndarray:
        a, b = np.asarray(a), np.asarray(b)
        if self.cosine:
            a, b = _unit(a), _unit(b)
        return (a * b).sum(axis=1)
