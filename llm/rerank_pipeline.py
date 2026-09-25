"""Retrieve-then-rerank, LLM stage.

Consumes results/first_stage/<emb-model-slug>/<task>_top20.json (written by
first_stage.py in the eval/ env) and reranks each query's top-20 listwise with
the vendor's LLMListwiseReranker (llm_judge.llm_reranker: prompt, JSON-list
parsing, stage-1-order backfill, rank-coverage audit, usage aggregation).
Both first-stage-only and reranked metrics are scored with pytrec_eval.

Usage (from llm/):
    uv run --env-file .env python rerank_pipeline.py --first-stage Qwen/Qwen3-Embedding-0.6B
    uv run python rerank_pipeline.py --dry-run             # stub returns first-stage order

Output: results/llm_rerank/<model-slug>/<task>.json
    scores.test[0] = {...reranked metrics, main_score=ndcg_at_10,
                      first_stage: {model, metrics}, usage_stats}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

from datasets import Dataset

from ir_metrics import ranked_list_to_scores, retrieval_metrics
from llm_common import (
    REPO_ROOT, RETRIEVAL_TASKS, install_stub, load_env, load_first_stage,
    load_task_data, model_slug, write_result_json,
)


def _stub_responder():
    """Return the candidate ids in the order given (= first-stage order)."""
    def respond(messages):
        k = int(re.search(r"Candidate documents \((\d+)\)", messages[-1]["content"]).group(1))
        return json.dumps(list(range(k)))
    return respond


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=RETRIEVAL_TASKS)
    ap.add_argument("--first-stage", default="Qwen/Qwen3-Embedding-0.6B",
                    help="embedding model whose top-k file to rerank")
    ap.add_argument("--first-stage-file", default=None,
                    help="explicit top-k JSON (overrides --first-stage; single task only)")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-doc-chars", type=int, default=0,
                    help="truncate candidate docs (0 = no truncation; vendor default is 800)")
    ap.add_argument("--output-suffix", default="")
    ap.add_argument("--dry-run", action="store_true", help="stub client; no API calls")
    args = ap.parse_args()

    load_env(dry_run=args.dry_run)
    from llm_judge.settings import Settings
    from llm_judge.llm_reranker import LLMListwiseReranker
    settings = Settings()
    if not args.dry_run and (settings.token in ("", "dummy") or settings.token.endswith("...")):
        sys.exit("rerank_pipeline.py: placeholder token in llm/.env; fill it in or use --dry-run.")

    slug = model_slug(settings, args.output_suffix)
    out_dir = (REPO_ROOT / "llm" / "dryrun_results" / "llm_rerank" if args.dry_run
               else REPO_ROOT / "results" / "llm_rerank") / slug
    print(f"model={settings.model} first_stage={args.first_stage} dry_run={args.dry_run} -> {out_dir}")
    stub = install_stub(_stub_responder()) if args.dry_run else None

    for task in args.tasks:
        if args.first_stage_file:
            fs = json.loads(open(args.first_stage_file).read())
        else:
            fs = load_first_stage(args.first_stage, task, args.top_k)
        if fs is None:
            sys.exit(f"no first-stage file for {args.first_stage}/{task}; run first_stage.py in eval/ first")
        corpus, queries, qrels = load_task_data(task)
        top_ranked = {q: ids[: args.top_k] for q, ids in fs["top_ranked"].items()}
        # Score the first stage from the *order handed to the LLM* (rank scores), so
        # a no-op rerank reproduces it exactly. Tied similarities make this differ
        # slightly from the similarity-scored number in the first-stage file
        # (pytrec_eval breaks ties by doc id); that one is kept alongside.
        fs_metrics = retrieval_metrics({q: ranked_list_to_scores(ids) for q, ids in top_ranked.items()}, qrels)

        reranker = LLMListwiseReranker(
            settings.model, max_doc_chars=args.max_doc_chars or sys.maxsize)
        reranker.index(
            Dataset.from_dict({"id": list(corpus), "text": [d["text"] for d in corpus.values()]}),
            task_metadata=None, hf_split="test", hf_subset="default", encode_kwargs={})
        t0 = time.time()
        n_before = len(stub.calls) if stub else 0
        results = reranker.search(
            Dataset.from_dict({"id": list(queries), "text": list(queries.values())}),
            task_metadata=None, hf_split="test", hf_subset="default", top_k=args.top_k,
            encode_kwargs={}, top_ranked=top_ranked)
        usage = dict(reranker.last_usage)
        usage.pop("model", None)

        scores = retrieval_metrics(results, qrels)
        scores["main_score"] = scores["ndcg_at_10"]
        scores["first_stage"] = {"model": fs["model"], "top_k": args.top_k, "metrics": fs_metrics,
                                 "metrics_sim_scored": fs["metrics"]}
        scores["usage_stats"] = usage
        write_result_json(out_dir / f"{task}.json", task, scores, t0,
                          extra={"protocol": "retrieve_then_rerank_listwise",
                                 "max_doc_chars": args.max_doc_chars, "dry_run": args.dry_run})
        print(f"  {task}: first-stage ndcg@10={fs_metrics['ndcg_at_10']:.4f} r@1={fs_metrics['recall_at_1']:.4f}"
              f" | reranked ndcg@10={scores['ndcg_at_10']:.4f} r@1={scores['recall_at_1']:.4f}"
              f" | in={usage['input_tokens']:,} cached={usage['cached_tokens']:,} out={usage['output_tokens']:,}"
              f" think={usage['thinking_tokens']:,} fully_ranked={usage['n_fully_ranked']}/{usage['n_queries']}"
              + (f" | stub calls={len(stub.calls) - n_before}" if stub else ""))


if __name__ == "__main__":
    main()
