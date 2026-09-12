"""MedCPT asymmetric bi-encoder for mteb 2.11: queries through the
Query Encoder, documents (and any non-query text) through the Article
Encoder — NCBI's intended usage (both produce 768-d compatible vectors).
"""

import numpy as np
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType
from sentence_transformers import SentenceTransformer


class MedCPT:
    def __init__(self) -> None:
        self.query_enc = SentenceTransformer("ncbi/MedCPT-Query-Encoder")
        self.article_enc = SentenceTransformer("ncbi/MedCPT-Article-Encoder")
        meta = ModelMeta._from_hub("ncbi/MedCPT-Query-Encoder")
        meta.name = "ncbi/MedCPT"
        meta.loader = lambda *a, **k: self
        self.mteb_model_meta = meta

    def _texts(self, inputs) -> list[str]:
        return [t for batch in inputs for t in batch["text"]]

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
               prompt_type=None, **kwargs) -> np.ndarray:
        enc = self.query_enc if prompt_type == PromptType.query else self.article_enc
        return enc.encode(self._texts(inputs),
                          batch_size=kwargs.get("batch_size", 32),
                          convert_to_numpy=True, show_progress_bar=False)

    def similarity(self, a, b) -> np.ndarray:
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        return a @ b.T

    def similarity_pairwise(self, a, b) -> np.ndarray:
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        return (a * b).sum(axis=1)
