"""Cross-encoder rerank of a first-stage run's top 100 on a GeneDiseaseHaystack.

PREREG (change log 22:45): cross-encoders re-sort the top 100 of the first
pass; not tuned. Each model is loaded and scored as its model card does it:

  ncbi/MedCPT-Cross-Encoder   [query, article] pairs, max_length 512, the logit
  BAAI/bge-reranker-v2-m3,    [query, passage] pairs, max_length 512, the raw
  BAAI/bge-reranker-base      logit (FlagReranker.compute_score, normalize=False)
  bioembed/medcpt-ce-gd-s42,  our fine-tunes of MedCPT-CE and bge-reranker-base
  bioembed/bge-reranker-      (finetune_ce.py; local models/bioembed-<x>, as
    base-gd-s42               run_embedders), scored exactly as their bases: the
                              tokenizer saved with them, [query, doc] pairs,
                              max_length 512 (as trained), the raw logit (as
                              dev_eval.py; their saved Sigmoid is not applied)
  Qwen/Qwen3-Reranker-0.6B,   the card's yes/no prompt template and instruction,
  Qwen/Qwen3-Reranker-4B      max_length 8192, last-token logits; score =
                              logit(yes) - logit(no), the card's P(yes) in logit
                              space (= its sentence-transformers score), which
                              does not saturate at 1.0 as P(yes) does in floats

Query = queries.parquet text, doc = docs_gene.parquet text (title + abstract).
bf16 on CUDA, fp32 on CPU; in bf16 the score head runs in fp32, as bf16 logits
round into ties. Rerank practice and test on the same device: the dtypes get
separate caches, but nothing makes their scores match. Scores are cached per
(qid, pmid) as parquet parts under
<haystack>/rerank_cache/<model-slug>-<float32|bfloat16-fp32head>[-<sha8 of instruction>]/,
so a preempted run resumes, and first stages that share candidates share work.

Run (from eval/):
    uv run python rerank_haystack.py --haystack GeneDiseaseHaystackDev \\
        --first bm25-symbol_disease-k0.9-b0.75__gene.parquet \\
        --model ncbi/MedCPT-Cross-Encoder [--batch-size 32] [--max-queries 5]
Output: <haystack>/runs/ce-<model-slug>-top100-of-<first stage>__<track>.parquet
(qid, pmid, score, rank): ranks 1-100 by CE score (ties in first-stage order),
101+ in first-stage order with score NaN. --max-queries N writes to runs_smoke/.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
HAYSTACKS = REPO / "data" / "haystack"
DEPTH = 100
CHUNK_QUERIES = 10  # cache part size
PAIR_MODELS = ["ncbi/MedCPT-Cross-Encoder", "BAAI/bge-reranker-v2-m3", "BAAI/bge-reranker-base"]
FINETUNED = {"bioembed/medcpt-ce-gd-s42": "ncbi/MedCPT-Cross-Encoder",  # fine-tune -> base
             "bioembed/bge-reranker-base-gd-s42": "BAAI/bge-reranker-base"}
QWEN_MODELS = ["Qwen/Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker-4B"]
QWEN_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
QWEN_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
               "the Query and the Instruct provided. Note that the answer can only be \"yes\" or "
               "\"no\".<|im_end|>\n<|im_start|>user\n")
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN_MAX_LEN = 8192


def slug(name: str) -> str:
    return name.replace("/", "__")


def model_path(name: str) -> str:
    """Hub name, or our fine-tune's dir: bioembed/<x> -> models/bioembed-<x> (as run_embedders)."""
    if name.startswith("bioembed/"):
        return str(REPO / "models" / f"bioembed-{name.split('/', 1)[1]}")
    return name


class PairReranker:
    """1-logit sequence classifier over [query, doc] pairs (MedCPT-CE, bge-rerankers, our fine-tunes)."""

    max_length = 512

    def __init__(self, name: str, device: torch.device, dtype: torch.dtype) -> None:
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, dtype=dtype)
        self.model.to(device).eval()
        # fp32 head on the bf16 body: bf16 logits round into ties
        self.model.classifier.float().register_forward_pre_hook(
            lambda _, args: tuple(a.float() for a in args))

    @torch.no_grad()
    def __call__(self, queries: list[str], docs: list[str]) -> np.ndarray:
        enc = self.tok(queries, docs, truncation=True, padding=True, return_tensors="pt",
                       max_length=self.max_length).to(self.model.device)
        return self.model(**enc).logits.view(-1).float().cpu().numpy()


class QwenReranker:
    """Qwen3-Reranker as in its model card's transformers code."""

    def __init__(self, name: str, device: torch.device, dtype: torch.dtype,
                 instruction: str = QWEN_INSTRUCTION) -> None:
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        self.model.to(device).eval()
        self.instruction = instruction
        yes, no = self.tok.convert_tokens_to_ids("yes"), self.tok.convert_tokens_to_ids("no")
        # logit(yes) - logit(no) = h . (w_yes - w_no), in fp32 (bf16 logits round into ties;
        # the LM head is tied to the input embeddings, so it can't be cast in place)
        w = self.model.get_output_embeddings().weight.detach()
        self.head = w[yes].float() - w[no].float()
        self.prefix = self.tok.encode(QWEN_PREFIX, add_special_tokens=False)
        self.suffix = self.tok.encode(QWEN_SUFFIX, add_special_tokens=False)

    @torch.no_grad()
    def __call__(self, queries: list[str], docs: list[str]) -> np.ndarray:
        texts = [f"<Instruct>: {self.instruction}\n<Query>: {q}\n<Document>: {d}"
                 for q, d in zip(queries, docs)]
        enc = self.tok(texts, padding=False, truncation="longest_first", return_attention_mask=False,
                       max_length=QWEN_MAX_LEN - len(self.prefix) - len(self.suffix))
        enc["input_ids"] = [self.prefix + ids + self.suffix for ids in enc["input_ids"]]
        enc = self.tok.pad(enc, padding=True, return_tensors="pt")
        # left padding: the last position is every row's next-token slot
        h = self.model.base_model(**enc.to(self.model.device)).last_hidden_state[:, -1].float()
        return (h @ self.head).cpu().numpy()


def load_reranker(name: str, device: torch.device, dtype: torch.dtype, instruction: str):
    if name in QWEN_MODELS:
        return QwenReranker(name, device, dtype, instruction)
    return PairReranker(model_path(name), device, dtype)


def score_pairs(scorer, queries: list[str], docs: list[str], batch_size: int) -> np.ndarray:
    """Batches of similar length (less padding), returned in input order."""
    order = np.argsort([-len(d) for d in docs], kind="stable")
    out = np.empty(len(docs), dtype=np.float32)
    for i in range(0, len(order), batch_size):
        b = order[i:i + batch_size]
        out[b] = scorer([queries[j] for j in b], [docs[j] for j in b])
    return out


def load_cache(cache: Path) -> pd.DataFrame:
    parts = sorted(cache.glob("*.parquet"))
    if not parts:
        return pd.DataFrame({"qid": pd.Series(dtype=str), "pmid": pd.Series(dtype=str),
                             "score": pd.Series(dtype=np.float32)})
    return pd.concat(map(pd.read_parquet, parts)).drop_duplicates(["qid", "pmid"])


def pending(top: pd.DataFrame, cache: Path) -> pd.DataFrame:
    """The (qid, pmid) rows of top with no cached score."""
    done = load_cache(cache)[["qid", "pmid"]]
    return (top.merge(done, how="left", indicator=True)
            .query("_merge == 'left_only'").drop(columns="_merge"))


def fill_cache(scorer, todo: pd.DataFrame, cache: Path, batch_size: int) -> None:
    """Score todo (qid, pmid, query, doc), writing one cache part per CHUNK_QUERIES queries."""
    cache.mkdir(parents=True, exist_ok=True)
    qids = todo.qid.unique()
    for i in range(0, len(qids), CHUNK_QUERIES):
        t0 = time.time()
        part = todo[todo.qid.isin(qids[i:i + CHUNK_QUERIES])]
        s = score_pairs(scorer, part["query"].tolist(), part["doc"].tolist(), batch_size)
        f = cache / f"{time.time_ns()}.parquet"
        tmp = f.with_suffix(".tmp")
        part[["qid", "pmid"]].assign(score=s).to_parquet(tmp, index=False)
        tmp.rename(f)
        print(f"  queries {min(i + CHUNK_QUERIES, len(qids))}/{len(qids)}: {len(part)} pairs "
              f"({len(part) / (time.time() - t0):.1f} pairs/s)", flush=True)


def rerank(run: pd.DataFrame, scores: pd.DataFrame, depth: int = DEPTH) -> pd.DataFrame:
    """Top `depth` of each query re-sorted by CE score, the rest in first-stage order."""
    run = run.sort_values(["qid", "rank"])
    top = run[run["rank"] <= depth].drop(columns="score").merge(scores, on=["qid", "pmid"])
    assert len(top) == (run["rank"] <= depth).sum(), "missing CE scores"
    top = top.sort_values(["qid", "score", "rank"], ascending=[True, False, True])
    rest = run[run["rank"] > depth].assign(score=np.nan)
    out = pd.concat([top, rest])[["qid", "pmid", "score"]]
    out["rank"] = out.groupby("qid", sort=False).cumcount() + 1
    return out.sort_values(["qid", "rank"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--haystack", default="GeneDiseaseHaystack")
    ap.add_argument("--first", required=True, help="first-stage run file in <haystack>/runs/")
    ap.add_argument("--model", choices=PAIR_MODELS + QWEN_MODELS + list(FINETUNED), required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--instruction", default=QWEN_INSTRUCTION, help="Qwen3-Reranker only")
    ap.add_argument("--max-queries", type=int, default=None, help="smoke test on the first N queries")
    args = ap.parse_args()
    haystack = HAYSTACKS / args.haystack
    first, track = args.first.removesuffix(".parquet").rsplit("__", 1)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.bfloat16 if cuda else torch.float32

    run = pd.read_parquet(haystack / "runs" / f"{first}__{track}.parquet")
    if args.max_queries:
        run = run[run.qid.isin(sorted(run.qid.unique())[: args.max_queries])]
    queries = pd.read_parquet(haystack / "queries.parquet", columns=["qid", "text"])
    top = run.loc[run["rank"] <= DEPTH, ["qid", "pmid"]]
    docs = pd.read_parquet(haystack / "docs_gene.parquet",
                           filters=[("pmid", "in", top.pmid.unique().tolist())])

    tag = f"{slug(args.model)}-{'float32' if dtype == torch.float32 else 'bfloat16-fp32head'}"
    if args.model in QWEN_MODELS:
        tag += "-" + hashlib.sha256(args.instruction.encode()).hexdigest()[:8]
    cache = haystack / "rerank_cache" / tag
    if args.model.startswith("bioembed/"):  # our fine-tunes: a retrain under the same name must not reuse old scores
        weights = REPO / "models" / f"bioembed-{args.model.split('/', 1)[1]}" / "model.safetensors"
        cache = cache / hashlib.sha256(weights.read_bytes()).hexdigest()[:12]
    todo = (pending(top, cache)
            .merge(queries.rename(columns={"text": "query"}), on="qid")
            .merge(docs.rename(columns={"text": "doc"}), on="pmid"))
    print(f"=== {args.model} on {args.haystack}/{first} top {DEPTH}: {len(top):,} pairs, "
          f"{len(todo):,} to score ({device}, {dtype})", flush=True)
    if len(todo):
        fill_cache(load_reranker(args.model, device, dtype, args.instruction), todo, cache,
                   args.batch_size)

    out_dir = haystack / ("runs_smoke" if args.max_queries else "runs")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"ce-{slug(args.model)}-top{DEPTH}-of-{first}__{track}.parquet"
    scores = load_cache(cache).merge(top)
    rerank(run, scores).to_parquet(out, index=False)
    tied = scores.duplicated(["qid", "score"], keep=False).mean()
    print(f"  -> {out.relative_to(REPO)} ({tied:.2%} of pairs tie within their query)", flush=True)


if __name__ == "__main__":
    main()
