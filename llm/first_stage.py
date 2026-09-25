"""First stage of retrieve-then-rerank: embedding model top-20 per query.

Runs in the eval/ env (mteb 2.11.6 + sentence-transformers) so the ranking is
produced by exactly the pipeline behind results/embedding/ (mteb.get_model gives
Qwen3 its instruction prompts, e5 its prefixes, etc.). Nothing is written under
results/embedding/; mteb's run goes to a scratch folder and only the top-20
file is kept.

Usage (from eval/):
    uv run python ../llm/first_stage.py --models Qwen/Qwen3-Embedding-0.6B \
        sentence-transformers/all-MiniLM-L6-v2 --tasks GOPubMedRetrieval R2MEDBiologyPooled

Output: results/first_stage/<model-slug>/<task>_top20.json
    {
      "model": ..., "task": ..., "top_k": 20,
      "metrics": {ndcg_at_10, recall_at_1, ...},   # computed from the top-20 lists
      "top_ranked": {qid: [doc_id, ...]},           # best first
      "scores":     {qid: {doc_id: sim}},
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EVAL = REPO / "eval"
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(HERE))

import mteb  # noqa: E402
from datasets import load_from_disk  # noqa: E402

from ir_metrics import retrieval_metrics  # noqa: E402
from run_embedders import get_model  # noqa: E402  (registers local tasks in mteb too)
from tasks_local import LOCAL_TASKS  # noqa: E402

DEFAULT_MODELS = ["Qwen/Qwen3-Embedding-0.6B", "sentence-transformers/all-MiniLM-L6-v2"]
DEFAULT_TASKS = ["GOPubMedRetrieval", "R2MEDBiologyPooled"]


def slug(name: str) -> str:
    return name.replace("/", "__")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    tasks = [t for t in LOCAL_TASKS if t.metadata.name in args.tasks]
    out_root = REPO / "results" / "first_stage"
    # mteb's own run (cache + full predictions) goes to a scratch dir, keyed by
    # model, so the metric step can be rerun without re-encoding.
    scratch = Path(os.environ.get("FIRST_STAGE_SCRATCH", REPO / "eval" / ".first_stage_scratch"))

    for name in args.models:
        print(f"=== {name} ===", flush=True)
        tmp = scratch / slug(name)
        todo = [t for t in tasks if not (tmp / "pred" / t.prediction_file_name).exists()]
        if todo:
            model = get_model(name)
            mteb.evaluate(
                model, todo,
                cache=mteb.cache.ResultCache(tmp / "cache"),
                prediction_folder=tmp / "pred",
                encode_kwargs={"batch_size": args.batch_size},
                overwrite_strategy="always",
                co2_tracker=False,
            )
        for task in tasks:
            pred = json.loads((tmp / "pred" / task.prediction_file_name).read_text())
            full = pred["default"]["test"]  # {qid: {doc_id: score}}
            dd = load_from_disk(str(REPO / "data" / "tasks" / task.metadata.name))
            qrels: dict[str, dict[str, int]] = {}
            for row in dd["qrels"]:
                qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
            scores = {
                qid: dict(sorted(d.items(), key=lambda kv: -kv[1])[: args.top_k])
                for qid, d in full.items()
            }
            top_ranked = {qid: list(d) for qid, d in scores.items()}
            metrics = retrieval_metrics(scores, qrels)
            out = out_root / slug(name) / f"{task.metadata.name}_top{args.top_k}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "model": name, "task": task.metadata.name, "top_k": args.top_k,
                "metrics": metrics, "top_ranked": top_ranked, "scores": scores,
            }, indent=1))
            print(f"  {task.metadata.name}: ndcg@10={metrics['ndcg_at_10']:.4f} "
                  f"recall@1={metrics['recall_at_1']:.4f} recall@10={metrics['recall_at_10']:.4f} "
                  f"-> {out.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
