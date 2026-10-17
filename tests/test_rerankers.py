"""Cross-encoder rerankers (eval/rerank_haystack.py) against their model cards,
plus the rerank, cache and RRF plumbing.

Model-card tests skip a model that is not in the local HF cache (no ~10 GB
download); they run fp32 on CPU, the 4B peaks at ~32 GB. The RRF test needs
duckdb, which the eval env lacks, so add it:
    cd eval && uv run --with pytest --with duckdb pytest ../tests/test_rerankers.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from huggingface_hub import try_to_load_from_cache

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))

from rerank_haystack import (  # noqa: E402
    FINETUNED, QWEN_INSTRUCTION, PairReranker, QwenReranker, fill_cache, load_cache, load_reranker,
    model_path, pending, rerank, score_pairs,
)

CPU = torch.device("cpu")


def needs(name: str):
    """Skip unless the model is in the local HF cache."""
    cached = isinstance(try_to_load_from_cache(name, "config.json"), str)
    return pytest.mark.skipif(not cached, reason=f"{name} not in the local HF cache")

# ncbi/MedCPT-Cross-Encoder card: query, its first 3 articles, printed logits
MEDCPT_QUERY = "diabetes treatment"
MEDCPT_ARTICLES = [
    ("Type 1 and 2 diabetes mellitus: A review on current treatment approach and gene therapy as "
     "potential intervention. Type 1 and type 2 diabetes mellitus is a serious and lifelong "
     "condition commonly characterised by abnormally elevated blood glucose levels due to a "
     "failure in insulin production or a decrease in insulin sensitivity and function. [...]"),
    ("Diabetes mellitus and its chronic complications. Diabetes mellitus is a major cause of "
     "morbidity and mortality, and it is a major risk factor for early onset of coronary heart "
     "disease. Complications of diabetes are retinopathy, nephropathy, and peripheral neuropathy. "
     "[...]"),
    ("Diagnosis and Management of Central Diabetes Insipidus in Adults. Central diabetes insipidus "
     "(CDI) is a clinical syndrome which results from loss or impaired function of "
     "vasopressinergic neurons in the hypothalamus/posterior pituitary, resulting in impaired "
     "synthesis and/or secretion of arginine vasopressin (AVP). [...]"),
]
MEDCPT_LOGITS = [6.9363, -8.2063, -8.7692]

# BAAI/bge-reranker-v2-m3 card (FlagReranker, fp16, normalize=False)
PANDA = ("The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply "
         "panda, is a bear species endemic to China.")
BGE_PAIRS = [("query", "passage"), ("what is panda?", "hi"), ("what is panda?", PANDA)]
BGE_SCORES = [-5.65234375, -8.1875, 5.26171875]

# Qwen3-Reranker cards: sentence-transformers example, printed scores (logit(yes) - logit(no))
QWEN_QUERY = "What is the capital of China?"
QWEN_DOCS = ["The capital of China is Beijing.",
             ("Gravity is a force that attracts two bodies towards each other. It gives weight to "
              "physical objects and is responsible for the movement of planets around the sun.")]
QWEN_CARD = {"Qwen/Qwen3-Reranker-0.6B": [7.625, -11.375],
             "Qwen/Qwen3-Reranker-4B": [6.4375, -14.375]}


def card_qwen_p_yes(name: str, queries: list[str], docs: list[str]) -> list[float]:
    """The Qwen3-Reranker card's transformers code, verbatim but for the function wrapper
    and dtype: transformers 5 loads the checkpoint's bf16 by default, 4.x (the card's) fp32."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def format_instruction(instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
        return output

    def process_inputs(pairs):
        inputs = tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(inputs, **kwargs):
        batch_scores = model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    tokenizer = AutoTokenizer.from_pretrained(name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    max_length = 8192
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    pairs = [format_instruction(task, query, doc) for query, doc in zip(queries, docs)]
    return compute_logits(process_inputs(pairs))


@needs("ncbi/MedCPT-Cross-Encoder")
def test_medcpt_ce_matches_card():
    m = PairReranker("ncbi/MedCPT-Cross-Encoder", CPU, torch.float32)
    s = m([MEDCPT_QUERY] * 3, MEDCPT_ARTICLES)
    np.testing.assert_allclose(s, MEDCPT_LOGITS, atol=1e-3)


@needs("ncbi/MedCPT-Cross-Encoder")
def test_bf16_body_keeps_fp32_head():
    m = PairReranker("ncbi/MedCPT-Cross-Encoder", CPU, torch.bfloat16)
    assert m.model.bert.dtype == torch.bfloat16
    assert m.model.classifier.weight.dtype == torch.float32
    s = m([MEDCPT_QUERY] * 3, MEDCPT_ARTICLES)
    np.testing.assert_allclose(s, MEDCPT_LOGITS, atol=0.3)
    assert len(set(s)) == 3


@needs("BAAI/bge-reranker-v2-m3")
def test_bge_reranker_matches_card():
    m = PairReranker("BAAI/bge-reranker-v2-m3", CPU, torch.float32)
    s = m(*map(list, zip(*BGE_PAIRS)))
    np.testing.assert_allclose(s, BGE_SCORES, atol=0.01)  # card is fp16


def needs_finetune(name: str, base: str):
    """Skip unless the fine-tune's dir (models/bioembed-<x>) and its base's tokenizer are local."""
    here = Path(model_path(name), "model.safetensors").exists()
    tok = isinstance(try_to_load_from_cache(base, "tokenizer_config.json"), str)
    return pytest.mark.skipif(not (here and tok), reason=f"{name} dir or {base} tokenizer not local")


@pytest.mark.parametrize("name,base", [pytest.param(n, b, marks=needs_finetune(n, b))
                                       for n, b in FINETUNED.items()])
def test_finetuned_scored_as_base(name, base):
    """Our fine-tunes go through their base's pipeline: the base's tokens for each pair
    (truncated at 512), the base's architecture with the fp32 head, and the raw logit that
    dev_eval.py ranked them by (sentence-transformers CrossEncoder, no saved Sigmoid)."""
    from sentence_transformers.cross_encoder import CrossEncoder
    from transformers import AutoConfig, AutoTokenizer

    queries = ["MECP2 (methyl-CpG binding protein 2) in Rett syndrome"] * 3
    docs = MEDCPT_ARTICLES[:2] + [" ".join(MEDCPT_ARTICLES * 10)]  # the last runs past 512 tokens
    m = load_reranker(name, CPU, torch.bfloat16, QWEN_INSTRUCTION)
    assert isinstance(m, PairReranker)
    enc = m.tok(queries, docs, truncation=True, padding=True, max_length=m.max_length)
    assert enc == AutoTokenizer.from_pretrained(base)(queries, docs, truncation=True, padding=True,
                                                      max_length=512)
    assert len(enc["input_ids"][2]) == 512
    assert type(m.model).__name__ == AutoConfig.from_pretrained(base).architectures[0]
    assert m.model.base_model.dtype == torch.bfloat16
    assert all(p.dtype == torch.float32 for p in m.model.classifier.parameters())
    bf16 = m(queries, docs)

    s = PairReranker(model_path(name), CPU, torch.float32)(queries, docs)
    dev = CrossEncoder(model_path(name), max_length=512).predict(
        list(zip(queries, docs)), activation_fn=torch.nn.Identity())
    np.testing.assert_allclose(s, dev, atol=1e-4)
    np.testing.assert_allclose(bf16, s, atol=0.3)
    assert len(set(bf16)) == 3


@pytest.mark.parametrize("name", [pytest.param(n, marks=needs(n)) for n in QWEN_CARD])
def test_qwen_reranker_matches_card(name):
    from sentence_transformers import CrossEncoder

    queries = [QWEN_QUERY] * 2
    m = QwenReranker(name, CPU, torch.float32)
    s = m(queries, QWEN_DOCS)
    # the card's transformers code: P(yes) = sigmoid(logit(yes) - logit(no))
    np.testing.assert_allclose(1 / (1 + np.exp(-s)), card_qwen_p_yes(name, queries, QWEN_DOCS),
                               rtol=1e-4, atol=1e-7)
    # the card's sentence-transformers path, same dtype
    ce = CrossEncoder(name, model_kwargs={"dtype": torch.float32})
    np.testing.assert_allclose(s, ce.predict(list(zip(queries, QWEN_DOCS))), atol=1e-3)
    # the card's printed scores (bf16, GPU): same sign, the relevant one close
    card = np.array(QWEN_CARD[name])
    assert (np.sign(s) == np.sign(card)).all()
    assert abs(s[0] - card[0]) < 0.25
    assert m.instruction == QWEN_INSTRUCTION


def test_score_pairs_keeps_input_order():
    docs = ["a" * n for n in (3, 9, 1, 7, 5)]
    s = score_pairs(lambda q, d: np.array([len(x) for x in d], float), ["q"] * 5, docs, 2)
    np.testing.assert_array_equal(s, [3, 9, 1, 7, 5])


def toy_run(n_docs: int = 6) -> pd.DataFrame:
    return pd.DataFrame([(q, f"{q}{i}", float(-i), i) for q in "ab" for i in range(1, n_docs + 1)],
                        columns=["qid", "pmid", "score", "rank"])


def test_rerank_top_depth_then_first_stage_order():
    run = toy_run()
    ce = pd.DataFrame({"qid": ["a"] * 3 + ["b"] * 3, "pmid": ["a1", "a2", "a3", "b1", "b2", "b3"],
                       "score": [0.1, 0.9, 0.5, 2.0, 2.0, 3.0]})
    out = rerank(run, ce, depth=3)
    assert out.groupby("qid").pmid.agg(list).to_dict() == {
        "a": ["a2", "a3", "a1", "a4", "a5", "a6"],
        "b": ["b3", "b1", "b2", "b4", "b5", "b6"],  # tie b1 = b2 keeps first-stage order
    }
    assert out.groupby("qid")["rank"].agg(list).tolist() == [list(range(1, 7))] * 2
    assert out.score.isna().sum() == 6
    with pytest.raises(AssertionError):
        rerank(run, ce.iloc[1:], depth=3)


def test_cache_resumes(tmp_path):
    top = toy_run(3)[["qid", "pmid"]]
    todo = top.assign(query="q", doc=top.pmid)
    calls = []

    def scorer(q, d):
        calls.extend(d)
        return np.arange(len(d), dtype=float)

    fill_cache(scorer, todo[todo.qid == "a"], tmp_path, batch_size=2)
    assert pending(top, tmp_path).pmid.tolist() == ["b1", "b2", "b3"]
    fill_cache(scorer, todo.merge(pending(top, tmp_path)), tmp_path, batch_size=2)
    assert pending(top, tmp_path).empty
    assert sorted(calls) == sorted(top.pmid)  # every pair scored exactly once
    assert len(load_cache(tmp_path)) == 6


def test_rrf_matches_formula(tmp_path):
    pytest.importorskip("duckdb", reason="the eval env has no duckdb: uv run --with duckdb")
    from rrf_haystack import is_dense, rrf

    a = pd.DataFrame({"qid": "q", "pmid": ["x", "y", "z"], "score": 0.0, "rank": [1, 2, 3]})
    b = pd.DataFrame({"qid": "q", "pmid": ["z", "w"], "score": 0.0, "rank": [1, 2]})
    a.to_parquet(tmp_path / "a.parquet")
    b.to_parquet(tmp_path / "b.parquet")
    out = rrf([tmp_path / "a.parquet", tmp_path / "b.parquet"], top_k=3)
    assert out.pmid.tolist() == ["z", "x", "w"]  # 1/63 + 1/61 > 1/61 > 1/62
    np.testing.assert_allclose(out.score, [1 / 63 + 1 / 61, 1 / 61, 1 / 62])
    assert out["rank"].tolist() == [1, 2, 3]
    assert is_dense("BAAI__bge-base-en-v1.5")
    assert not any(map(is_dense, ["bm25-symbol_disease-k0.9-b0.75", "rrf-bm25+BAAI__bge",
                                  "ce-ncbi__MedCPT-Cross-Encoder-top100-of-bm25",
                                  "llm-deepseek-v4-flash-top20-of-bm25"]))
