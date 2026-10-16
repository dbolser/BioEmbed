"""Run embedding models over the BioMTEB(LLM) suite.

Usage:
    uv run python run_embedders.py --models intfloat/multilingual-e5-small \
        --tasks LLMBIOSSES LLMPublicHealthQA
    uv run python run_embedders.py            # everything in the registry
    uv run python run_embedders.py --models BAAI/bge-small-en-v1.5 \
        --tasks GeneDiseaseRetrieval_v2_{0,1,2,3,4} --predictions PRED_DIR
    uv run python run_embedders.py --v2-summary PRED_DIR

Results land in ../results/embedding/<model>/<task>.json (mteb layout).
mteb.get_model() is used so known models get their correct prompts/prefixes
(e5 "query:/passage:", Qwen3 instructions, etc.). Models it does not register
load only if their repo has a sentence-transformers config; anything else
needs a wrapper (medcpt_model.py, bmretriever_model.py), since mteb would
otherwise mean-pool it silently.

--v2-summary output: ../results/v2_scores.csv (model, batch, hit@1 =
recall@1, mrr@10, ndcg@10) and ../results/v2_perquery.parquet (model, batch,
qid, rank_of_gold, hit1), from the --predictions of the v2 runs.
"""

import argparse
import sys
from pathlib import Path

import mteb
import torch
from huggingface_hub import file_exists

sys.path.insert(0, str(Path(__file__).parent))
from tasks_tier_a import TIER_A_TASKS
from tasks_local import LOCAL_TASKS

# Register our task classes in mteb's registry so instruct-style models
# (Qwen3-Embedding etc.) can resolve task instructions by task name.
import importlib  # noqa: E402
_gt_mod = importlib.import_module("mteb.get_tasks")
for _t in TIER_A_TASKS + LOCAL_TASKS:
    _gt_mod._TASKS_REGISTRY.setdefault(_t.metadata.name, type(_t))

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "intfloat/multilingual-e5-small",
    "google/embeddinggemma-300m",
    "Snowflake/snowflake-arctic-embed-l-v2.0",
    "Qwen/Qwen3-Embedding-0.6B",
    "NeuML/pubmedbert-base-embeddings",
    "abhinand/MedEmbed-small-v0.1",
    "abhinand/MedEmbed-base-v0.1",
    "abhinand/MedEmbed-large-v0.1",
    "FremyCompany/BioLORD-2023",
]
V2_BATCHES = 5  # GeneDiseaseRetrieval_v2_{0..4}; 0 is the practice batch
GEMMA = "google/embeddinggemma-300m"
# Google's repo is gated and our token is refused; every file in this mirror is
# byte-identical (LFS sha256 / git blob id) to mteb's pinned Google revision.
GEMMA_MIRROR = ("unsloth/embeddinggemma-300m", "bfa3c846ac738e62aa61806ef9112d34acb1dc5a")


def get_model(name: str):
    model = _load(name)
    if not torch.cuda.is_available():
        # transformers 5 keeps a bf16 checkpoint in bf16 (4.x gave fp32); our
        # Xeons have no bf16 units, so it runs ~3x slower than fp32 on them
        for m in (model, getattr(model, "model", None)):
            if isinstance(m, torch.nn.Module):
                m.float()
    return model


def _load(name: str):
    if name in ("ncbi/MedCPT", "ncbi/MedCPT-cosine"):  # CLS + dot product; -cosine variant
        from medcpt_model import MedCPT
        return MedCPT(cosine=name.endswith("-cosine"))
    if name.startswith("BMRetriever/"):  # last token + EOS, 512 tokens, dot product
        from bmretriever_model import BMRetriever
        return BMRetriever(name)
    if name.startswith("bioembed/"):  # our fine-tunes: models/bioembed-<x> -> results/embedding/bioembed__<x>
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(REPO / "models" / f"bioembed-{name.split('/', 1)[1]}"))
        model.model_card_data.model_name = name
        return model
    if name == GEMMA:  # mteb's loader and prompts, weights from the identical mirror
        meta = mteb.get_model_meta(name).model_copy(deep=True)
        model = meta.loader(GEMMA_MIRROR[0], revision=GEMMA_MIRROR[1], **meta.loader_kwargs)
        meta.reference = "https://huggingface.co/{}/tree/{}".format(*GEMMA_MIRROR)  # recorded in model_meta.json
        model.mteb_model_meta = meta
        return model
    try:
        mteb.get_model_meta(name)
    except KeyError:  # mteb would build a mean-pooling SentenceTransformer from config.json
        if not file_exists(name, "modules.json"):
            raise ValueError(f"{name}: not in mteb's registry and no sentence-transformers "
                             "config, so its pooling is unknown; add a wrapper (see medcpt_model.py)")
    try:
        return mteb.get_model(name)
    except Exception as e:
        if "FlashAttention2" not in str(e):
            raise
        meta = mteb.get_model_meta(name).model_copy(deep=True)  # no flash-attn (CPU): keep mteb's loader and prompts
        meta.loader_kwargs.setdefault("model_kwargs", {})["attn_implementation"] = "sdpa"
        return meta.load_model()


def v2_summary(pred_dir: Path, out: Path) -> None:
    """Per-query rank of the single gold doc in each v2 batch, from mteb's saved
    predictions; the per-batch means are checked against mteb's own recall@1 /
    MRR@10 / nDCG@10."""
    import json

    import numpy as np
    import pandas as pd
    from datasets import load_from_disk

    rows = []
    for i in range(V2_BATCHES):
        task = f"GeneDiseaseRetrieval_v2_{i}"
        gold = {r["query-id"]: r["corpus-id"] for r in load_from_disk(str(REPO / "data" / "tasks" / task))["qrels"]}
        for f in sorted(pred_dir.glob(f"*/{task}_predictions.json")):
            run = json.loads(f.read_text())["default"]["test"]
            for qid, g in gold.items():
                scores = run[qid]
                rank = 1 + sum(s > scores[g] for s in scores.values())
                rows.append((f.parent.name, i, qid, rank, rank == 1))
    pq = pd.DataFrame(rows, columns=["model", "batch", "qid", "rank_of_gold", "hit1"])
    top10 = pq.rank_of_gold <= 10
    pq["mrr@10"] = np.where(top10, 1 / pq.rank_of_gold, 0.0)
    pq["ndcg@10"] = np.where(top10, 1 / np.log2(pq.rank_of_gold + 1), 0.0)
    scores = pq.groupby(["model", "batch"]).agg(**{
        "hit@1": ("hit1", "mean"), "mrr@10": ("mrr@10", "mean"), "ndcg@10": ("ndcg@10", "mean"),
        "n_queries": ("qid", "size")}).reset_index()

    for r in scores.to_dict("records"):  # same numbers as mteb's main results?
        res, = out.glob(f"{r['model']}/*/GeneDiseaseRetrieval_v2_{r['batch']}.json")
        m = json.loads(res.read_text())["scores"]["test"][0]
        got = (r["hit@1"], r["mrr@10"], r["ndcg@10"])
        want = (m["recall_at_1"], m["mrr_at_10"], m["ndcg_at_10"])
        if not np.allclose(got, want, atol=1e-4):
            raise ValueError(f"{r['model']} batch {r['batch']}: per-query {got} != mteb {want}")

    scores.round(4).to_csv(REPO / "results" / "v2_scores.csv", index=False)
    pq[["model", "batch", "qid", "rank_of_gold", "hit1"]].to_parquet(REPO / "results" / "v2_perquery.parquet", index=False)
    pooled = pq.assign(split=np.where(pq.batch == 0, "practice (0)", "test (1-4)")).pivot_table(
        index="model", columns="split", values="hit1", aggfunc="mean").sort_values("practice (0)", ascending=False)
    print(pooled.round(3).to_string())
    print(f"wrote results/v2_scores.csv ({len(scores)} rows), results/v2_perquery.parquet ({len(pq)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="task names; default = whole suite")
    ap.add_argument("--suite", choices=["tier_a", "local", "all"], default="all")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "embedding",
                    help="mteb output folder (default: the tracked results)")
    ap.add_argument("--predictions", type=Path, default=None,
                    help="also save mteb's per-query retrieval scores to <dir>/<model>/")
    ap.add_argument("--v2-summary", type=Path, default=None, metavar="PRED_DIR",
                    help="run nothing; write results/v2_scores.csv and v2_perquery.parquet "
                         "from the predictions in PRED_DIR")
    args = ap.parse_args()

    out = args.out
    if args.v2_summary:
        v2_summary(args.v2_summary, out)
        return

    tasks = {"tier_a": TIER_A_TASKS, "local": LOCAL_TASKS,
             "all": TIER_A_TASKS + LOCAL_TASKS}[args.suite]
    if args.tasks:
        tasks = [t for t in tasks if t.metadata.name in args.tasks]

    failures = []
    for name in args.models:
        print(f"=== {name} ===", flush=True)
        try:
            model = get_model(name)
            evaluation = mteb.MTEB(tasks=tasks)
            extra = {"prediction_folder": args.predictions / name.replace("/", "__")} if args.predictions else {}
            evaluation.run(
                model,
                output_folder=str(out),
                encode_kwargs={"batch_size": args.batch_size},
                verbosity=1,
                **extra,
            )
        except Exception as e:
            print(f"!!! FAILED {name}: {e!r}", flush=True)
            failures.append(name)
    if failures:
        print("FAILED MODELS:", failures)


if __name__ == "__main__":
    main()
