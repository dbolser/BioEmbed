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

from run_embedders import GEMMA, GEMMA_MIRROR, get_model  # noqa: E402
from tasks_local import GENE_DISEASE_V2  # noqa: E402

POOLING = {
    "BAAI/bge-small-en-v1.5": "cls",
    "abhinand/MedEmbed-small-v0.1": "cls",
    "intfloat/multilingual-e5-small": "mean",
    "sentence-transformers/all-MiniLM-L6-v2": "mean",
    "Qwen/Qwen3-Embedding-0.6B": "lasttoken",
    "codefuse-ai/F2LLM-v2-0.6B": "lasttoken",  # mteb's instruct loader, sdpa without flash-attn
    "bisectgroup/BiCA-small": "mean",  # not in mteb's registry; its own ST config
    "bisectgroup/BiCA-base": "mean",
    GEMMA: "mean",  # then Dense, Dense, Normalize
}
TASK = GENE_DISEASE_V2[0]().metadata
QUERY = "HFE (homeostatic iron regulator) in hereditary hemochromatosis"
DOC = ("Hereditary haemochromatosis and HFE.\n\nMost patients with hereditary "
       "haemochromatosis are homozygous for the C282Y mutation in HFE.")
# Card checks: 1 query + 2 documents of different length, so batches are padded
DOCS = [DOC, ("Iron overload in a mouse model.\n\nHfe knockout mice accumulate hepatic iron "
              "and show raised transferrin saturation, as in human hereditary haemochromatosis, "
              "while hepcidin expression stays inappropriately low for the iron load.")]


def pooling_modes(st) -> list[str]:
    return [m.pooling_mode for m in st.modules() if isinstance(m, Pooling)]


def encode(model, texts: str | list[str], prompt_type: PromptType) -> np.ndarray:
    texts = [texts] if isinstance(texts, str) else texts
    return np.asarray(model.encode([{"text": texts}], task_metadata=TASK, hf_split="test",
                                   hf_subset="default", prompt_type=prompt_type))


def cosine(a, b) -> float:
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def row_cosines(ours, ref) -> list[float]:
    """Cosine of each of our vectors with the reference vector for the same text."""
    return [cosine(a, b) for a, b in zip(np.asarray(ours), np.asarray(ref), strict=True)]


def ours(model) -> np.ndarray:
    """QUERY then DOCS, through mteb's query / document prompts."""
    return np.vstack([encode(model, QUERY, PromptType.query), encode(model, DOCS, PromptType.document)])


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


def test_no_silent_mean_pooling():
    """A repo with no ST config and no mteb entry is refused, not mean-pooled."""
    with pytest.raises(ValueError, match="pooling is unknown"):
        get_model("ncbi/MedCPT-Query-Encoder")


@pytest.fixture(scope="module", params=["bisectgroup/BiCA-small", "bisectgroup/BiCA-base"])
def bica(request):
    return request.param, get_model(request.param)


def test_bica_no_prompts_cosine(bica):
    """Card: plain SentenceTransformer, empty query/document prompts, 512 tokens,
    mean pooling + Normalize, cosine."""
    _, m = bica
    assert not any(m.model.prompts.values())
    assert m.model.max_seq_length == 512
    assert m.mteb_model_meta.similarity_fn_name == ScoringFunction.COSINE
    np.testing.assert_allclose(np.linalg.norm(ours(m), axis=1), 1, atol=1e-5)


def test_bica_matches_model_card(bica):
    """The card's code, SentenceTransformer(name).encode(sentences), and an
    independent transformers mean pool (one text at a time, so no padding)."""
    from sentence_transformers import SentenceTransformer
    name, m = bica
    texts = [QUERY, *DOCS]
    assert min(row_cosines(ours(m), SentenceTransformer(name).encode(texts))) > 0.999
    tok, hf = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()
    with torch.no_grad():
        ref = [hf(**tok([t], return_tensors="pt")).last_hidden_state.mean(1)[0].numpy() for t in texts]
    assert min(row_cosines(ours(m), ref)) > 0.999


@pytest.fixture(scope="module")
def gemma():
    return get_model(GEMMA)


def test_gemma_prompts(gemma):
    """Card prompts: 'task: search result | query: ' and 'title: none | text: '."""
    def prompt(pt):
        return gemma.model_prompts[gemma.get_prompt_name(TASK, pt)]
    assert prompt(PromptType.query) == "task: search result | query: "
    assert prompt(PromptType.document) == "title: none | text: "
    assert gemma.model.max_seq_length == 2048
    assert gemma.mteb_model_meta.name == GEMMA
    assert gemma.mteb_model_meta.similarity_fn_name == ScoringFunction.COSINE
    assert GEMMA_MIRROR[1] in gemma.mteb_model_meta.reference  # weights' source is recorded


def test_gemma_matches_model_card(gemma):
    """Same vectors as the card's encode_query / encode_document."""
    from sentence_transformers import SentenceTransformer
    card = SentenceTransformer(GEMMA_MIRROR[0], revision=GEMMA_MIRROR[1])
    ref = np.vstack([card.encode_query([QUERY]), card.encode_document(DOCS)])
    assert min(row_cosines(ours(gemma), ref)) > 0.999
    assert cosine(encode(gemma, QUERY, PromptType.query), card.encode([QUERY])) < 0.99  # prompt matters


def test_gemma_mirror_is_identical():
    """Every file of the mirror equals mteb's pinned (gated) Google revision."""
    import mteb
    from huggingface_hub import HfApi

    def files(repo, rev):
        info = HfApi().model_info(repo, revision=rev, files_metadata=True)
        return {s.rfilename: s.lfs.sha256 if s.lfs else s.blob_id for s in info.siblings
                if s.rfilename not in ("README.md", "notebook.ipynb")}
    assert files(GEMMA, mteb.get_model_meta(GEMMA).revision) == files(*GEMMA_MIRROR)


BMR = [f"BMRetriever/BMRetriever-{size}" for size in ("410M", "1B", "2B", "7B")]
BMR_INSTRUCTION = "Retrieve text based on user query."  # mteb's prompt for our v2 tasks


def card_last_token_pool(last_hidden_states, attention_mask):
    """Copied from the BMRetriever model card."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden[torch.arange(last_hidden.shape[0]), sequence_lengths]


def bmr_card(hf, tok, input_texts):
    """The card's encoding of one padded batch. The card appends EOS to rows of a
    'pt' tensor, which fails (Tensor + list), so EOS goes on the token lists as in
    the authors' eval.py."""
    batch = tok(input_texts, max_length=511, truncation=True, return_attention_mask=False)
    batch["input_ids"] = [ids + [tok.eos_token_id] for ids in batch["input_ids"]]
    batch = tok.pad(batch, padding=True, return_attention_mask=True, return_tensors="pt")
    with torch.no_grad():
        return card_last_token_pool(hf(**batch).last_hidden_state, batch["attention_mask"]).numpy()


@pytest.fixture(scope="module", params=BMR)
def bmr(request):
    name = request.param
    return (get_model(name), AutoTokenizer.from_pretrained(name),
            AutoModel.from_pretrained(name, dtype=torch.float32).eval())


def test_bmretriever_prompts(bmr):
    """eval.py templates, with mteb's task instruction; 512 tokens, dot product."""
    m, _, _ = bmr
    assert m.get_task_instruction(TASK, PromptType.query) == f"{BMR_INSTRUCTION}\nQuery: "
    assert m.get_task_instruction(TASK, PromptType.document) == "Represent this passage\npassage: "
    assert m.mteb_model_meta.similarity_fn_name == ScoringFunction.DOT_PRODUCT
    assert m.mteb_model_meta.max_tokens == 512


def test_bmretriever_matches_model_card(bmr):
    """Same vectors as the card's code (query + 2 passages in one padded batch);
    unnormalised, scored by dot product."""
    m, tok, hf = bmr
    ref = bmr_card(hf, tok, [f"{BMR_INSTRUCTION}\nQuery: {QUERY}"]
                   + [f"Represent this passage\npassage: {d}" for d in DOCS])
    got = ours(m)
    assert min(row_cosines(got, ref)) > 0.999
    assert abs(float(np.linalg.norm(got[0])) - 1) > 0.1
    np.testing.assert_allclose(np.asarray(m.similarity(got[:1], got[1:])), got[:1] @ got[1:].T, rtol=1e-5)


def test_bmretriever_device(bmr):
    """On the GPU when there is one (haystack.py runs it there), fp32 on CPU."""
    m, _, _ = bmr
    want = ("cuda", torch.bfloat16) if torch.cuda.is_available() else ("cpu", torch.float32)
    assert (m.model.device.type, m.model.dtype) == want


def test_bmretriever_eos_last_token(bmr):
    """A long passage: 511 tokens + appended EOS, and the vector is the EOS state."""
    m, tok, hf = bmr
    long_doc = DOC + " More text." * 600
    ids = tok([f"Represent this passage\npassage: {long_doc}"], max_length=511, truncation=True)["input_ids"]
    ids = torch.tensor([ids[0] + [tok.eos_token_id]])
    assert ids.shape[1] == 512
    with torch.no_grad():
        ref = hf(input_ids=ids).last_hidden_state[:, -1].numpy()
    assert cosine(encode(m, long_doc, PromptType.document), ref) > 0.999
