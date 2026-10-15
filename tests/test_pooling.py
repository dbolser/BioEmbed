"""Pooling and similarity as loaded by eval/run_embedders.get_model.

Run (models must be in the local HF cache):
    cd eval && uv run --with pytest pytest ../tests
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from mteb.models.model_meta import ScoringFunction  # noqa: E402
from mteb.types import PromptType  # noqa: E402
from sentence_transformers.sentence_transformer.modules import Pooling  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from run_embedders import get_model  # noqa: E402

POOLING = {
    "BAAI/bge-small-en-v1.5": "cls",
    "abhinand/MedEmbed-small-v0.1": "cls",
    "intfloat/multilingual-e5-small": "mean",
    "sentence-transformers/all-MiniLM-L6-v2": "mean",
    "Qwen/Qwen3-Embedding-0.6B": "lasttoken",
    "codefuse-ai/F2LLM-v2-0.6B": "lasttoken",  # mteb's instruct loader, sdpa without flash-attn
}
QUERY = "HFE (homeostatic iron regulator) in hereditary hemochromatosis"
DOC = ("Hereditary haemochromatosis and HFE.\n\nMost patients with hereditary "
       "haemochromatosis are homozygous for the C282Y mutation in HFE.")


def pooling_modes(st) -> list[str]:
    return [m.pooling_mode for m in st.modules() if isinstance(m, Pooling)]


def encode(model, text: str, prompt_type: PromptType) -> np.ndarray:
    return model.encode([{"text": [text]}], prompt_type=prompt_type)


@pytest.mark.parametrize("name,mode", POOLING.items())
def test_pooling(name, mode):
    assert pooling_modes(get_model(name).model) == [mode]


@pytest.fixture(scope="module")
def medcpt():
    return get_model("ncbi/MedCPT")


def test_medcpt_cls_pooling(medcpt):
    assert pooling_modes(medcpt.query_enc) == ["cls"]
    assert pooling_modes(medcpt.article_enc) == ["cls"]
    assert medcpt.query_enc.max_seq_length == 64
    assert medcpt.article_enc.max_seq_length == 512


def test_medcpt_matches_model_card(medcpt):
    """Query vector == last_hidden_state[:, 0, :] as in the model card code."""
    tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    hf = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").eval()
    with torch.no_grad():
        ref = hf(**tok([QUERY], truncation=True, padding=True, return_tensors="pt",
                       max_length=64)).last_hidden_state[:, 0, :].numpy()
    np.testing.assert_allclose(encode(medcpt, QUERY, PromptType.query), ref, atol=1e-4)


def test_medcpt_dot_product(medcpt):
    q = encode(medcpt, QUERY, PromptType.query)
    d = encode(medcpt, DOC, PromptType.document)
    assert q.shape == (1, 768) and d.shape == (1, 768)
    assert medcpt.mteb_model_meta.similarity_fn_name == ScoringFunction.DOT_PRODUCT
    assert medcpt.query_enc.similarity_fn_name == "dot"
    np.testing.assert_allclose(medcpt.similarity(q, d), q @ d.T, rtol=1e-5)
    assert abs(float(np.linalg.norm(q)) - 1) > 0.1  # unnormalised, so dot != cosine


def test_medcpt_cosine_variant():
    m = get_model("ncbi/MedCPT-cosine")
    assert m.mteb_model_meta.name == "ncbi/MedCPT-cosine"
    assert m.mteb_model_meta.similarity_fn_name == ScoringFunction.COSINE
    q = encode(m, QUERY, PromptType.query)
    np.testing.assert_allclose(m.similarity(q, q), [[1.0]], rtol=1e-5)
