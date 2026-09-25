"""Ranked-list variant of the corpus-in-context LLM retrieval protocol.

Same LOFT document formatting and cached-prefix message layout as the vendor's
LLMRetrievalEvaluator (send_request_multi), but the prompt asks for ALL relevant
documents as a ranked list (up to 10, best first, never empty) and the result
is scored with pytrec_eval (nDCG@10, recall@1/5/10, ...) so it is directly
comparable to results/embedding/.

Usage (from llm/):
    uv run --env-file .env python retrieval_ranked.py                 # both GO tasks
    uv run --env-file .env python retrieval_ranked.py --tasks GOPubMedRetrieval
    uv run python retrieval_ranked.py --dry-run                        # offline stub, no API calls

Output: results/llm_ranked/<model-slug>/<task>.json  ({scores: {test: [{...metrics, usage_stats}]}})
"""

from __future__ import annotations

import argparse
import asyncio
import random
import re
import sys
import time

from llm_common import (
    REPO_ROOT, RETRIEVAL_TASKS, install_stub, load_env, load_first_stage,
    load_task_data, model_slug, write_result_json,
)

MAX_RANKED = 10

RANKED_SYSTEM_PROMPT = (
    "You will be given a list of documents. You need to read carefully and understand all of them. "
    "Then you will be given a query, and your goal is to find all documents from the list that can help answer the query. "
    "Print out the ID and TITLE of each document.\n\n"
    "Your final answer should be a ranked list of IDs, most relevant first, in the following format:\n"
    "Final Answer: [id1, id2, ...]\n"
    f"List every relevant document (at most {MAX_RANKED} IDs). If only one document is relevant, output:\n"
    "Final Answer: [id1]\n\n"
    "If there is no perfect answer output the closest ones. Do not give an empty final answer."
)


def _query_prompt(query_text: str) -> str:
    return (
        "====== Now let's start! ======\n"
        "Which documents are relevant to answer the query? "
        "Print out the TITLE and ID of each relevant document, then format the IDs into a ranked list, "
        f"most relevant first, at most {MAX_RANKED} IDs.\n"
        "If there is no perfect answer output the closest ones. Do not give an empty final answer.\n"
        f"query: {query_text}\n"
        "The following documents can help answer the query:"
    )


def _make_evaluator_class():
    """Deferred so the vendor client (which builds Settings at import) is only
    imported after load_env()."""
    from llm_judge.evaluators.llm_retrieval_evaluator import (
        LLMRetrievalEvaluator, _format_doc, _parse_ids,
    )
    from llm_judge.llm_client import send_request_multi

    from ir_metrics import ranked_list_to_scores, retrieval_metrics

    class RankedListEvaluator(LLMRetrievalEvaluator):
        """Overrides only the prompt and the scoring; evaluate_async (gather,
        usage aggregation, debug samples) is the vendor's."""

        def __init__(self, *a, task_name: str, **kw):
            super().__init__(*a, **kw)
            self.metadata = type("M", (), {"name": task_name})()  # for the vendor's sample dump
            self.raw_responses: dict[str, str] = {}

        async def _score_query(self, query_id, query_text, candidate_ids):
            id_map: dict[int, str] = {}
            doc_lines = []
            for seq_id, cid in enumerate(candidate_ids):
                doc = self._corpus[cid]
                doc_lines.append(_format_doc(cid, doc["title"], doc["text"], seq_id))
                id_map[seq_id] = cid
            response, usage = await send_request_multi(
                instructions=RANKED_SYSTEM_PROMPT,
                context="\n\n".join(doc_lines),
                query=_query_prompt(query_text),
            )
            retrieved = _parse_ids(response or "", id_map)[:MAX_RANKED]
            self.raw_responses[query_id] = response or ""
            return retrieved, usage, response or ""

        def _compute_metrics(self, results):
            run = {qid: ranked_list_to_scores(ids) for qid, ids in results.items()}
            m = retrieval_metrics(run, self._qrels)
            m["n_empty"] = sum(1 for ids in results.values() if not ids)
            m["mean_list_len"] = round(sum(len(v) for v in results.values()) / max(1, len(results)), 3)
            m["main_score"] = m["ndcg_at_10"]
            return m

    return RankedListEvaluator


def _stub_responder(corpus: dict, queries: dict, qrels: dict, first_stage: dict | None):
    """Answer each prompt with a plausible ranked list: the first-stage top-10 if
    available, else gold docs then random fillers. Doc seq ids follow the
    evaluator's sorted(corpus) order."""
    seq_of = {cid: i for i, cid in enumerate(sorted(corpus))}
    text_to_qid = {t: q for q, t in queries.items()}
    rng = random.Random(0)

    def respond(messages):
        q = re.search(r"query: (.*)\nThe following documents", messages[-1]["content"], re.S).group(1)
        qid = text_to_qid[q]
        if first_stage:
            ids = first_stage["top_ranked"][qid][:MAX_RANKED]
        else:
            ids = list(qrels.get(qid, {}))
            ids += rng.sample([c for c in corpus if c not in ids], MAX_RANKED - len(ids))
        return "Some reasoning...\nFinal Answer: [" + ", ".join(str(seq_of[c]) for c in ids) + "]"

    return respond


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=RETRIEVAL_TASKS)
    ap.add_argument("--output-suffix", default="", help="e.g. __nothink")
    ap.add_argument("--dry-run", action="store_true", help="stub client; no API calls")
    ap.add_argument("--stub-first-stage", default="Qwen/Qwen3-Embedding-0.6B",
                    help="dry-run: embedding model whose top-20 seeds the stub's answers")
    args = ap.parse_args()

    load_env(dry_run=args.dry_run)
    from llm_judge.settings import Settings
    settings = Settings()
    if not args.dry_run and (settings.token in ("", "dummy") or settings.token.endswith("...")):
        sys.exit("retrieval_ranked.py: placeholder token in llm/.env; fill it in or use --dry-run.")

    slug = model_slug(settings, args.output_suffix)
    out_dir = (REPO_ROOT / "llm" / "dryrun_results" / "llm_ranked" if args.dry_run
               else REPO_ROOT / "results" / "llm_ranked") / slug
    print(f"model={settings.model} dry_run={args.dry_run} -> {out_dir}")

    Evaluator = _make_evaluator_class()
    for task in args.tasks:
        corpus, queries, qrels = load_task_data(task)
        stub = None
        if args.dry_run:
            stub = install_stub(_stub_responder(
                corpus, queries, qrels, load_first_stage(args.stub_first_stage, task)))
        t0 = time.time()
        ev = Evaluator(corpus=corpus, queries=queries, qrels=qrels, task_name=task)
        scores = asyncio.run(ev.evaluate_async())
        write_result_json(out_dir / f"{task}.json", task, scores, t0,
                          extra={"protocol": "corpus_in_context_ranked", "max_ranked": MAX_RANKED,
                                 "dry_run": args.dry_run})
        u = scores["usage_stats"]
        print(f"  {task}: ndcg@10={scores['ndcg_at_10']:.4f} recall@1={scores['recall_at_1']:.4f} "
              f"recall@10={scores['recall_at_10']:.4f} | in={u['input_tokens']:,} cached={u['cached_tokens']:,} "
              f"out={u['output_tokens']:,} think={u['thinking_tokens']:,}"
              + (f" | stub calls={len(stub.calls)}" if stub else ""))


if __name__ == "__main__":
    main()
