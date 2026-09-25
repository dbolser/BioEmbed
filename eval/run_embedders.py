"""Run embedding models over the BioMTEB(LLM) suite.

Usage:
    uv run python run_embedders.py --models intfloat/multilingual-e5-small \
        --tasks LLMBIOSSES LLMPublicHealthQA
    uv run python run_embedders.py            # everything in the registry

Results land in ../results/embedding/<model>/<task>.json (mteb layout).
mteb.get_model() is used so known models get their correct prompts/prefixes
(e5 "query:/passage:", Qwen3 instructions, etc.).
"""

import argparse
import sys
from pathlib import Path

import mteb

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


def get_model(name: str):
    if name == "ncbi/MedCPT":
        from medcpt_model import MedCPT
        return MedCPT()
    if name.startswith("bioembed/"):  # our fine-tunes: models/bioembed-<x> -> results/embedding/bioembed__<x>
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(REPO / "models" / f"bioembed-{name.split('/', 1)[1]}"))
        model.model_card_data.model_name = name
        return model
    try:
        return mteb.get_model(name)
    except Exception as e:
        print(f"  mteb.get_model failed ({e!r}); falling back to SentenceTransformer")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(name)
        model.model_card_data.model_name = name  # results dir = name, not base_model
        return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="task names; default = whole suite")
    ap.add_argument("--suite", choices=["tier_a", "local", "all"], default="all")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    tasks = {"tier_a": TIER_A_TASKS, "local": LOCAL_TASKS,
             "all": TIER_A_TASKS + LOCAL_TASKS}[args.suite]
    if args.tasks:
        tasks = [t for t in tasks if t.metadata.name in args.tasks]

    out = REPO / "results" / "embedding"
    failures = []
    for name in args.models:
        print(f"=== {name} ===", flush=True)
        try:
            model = get_model(name)
            evaluation = mteb.MTEB(tasks=tasks)
            evaluation.run(
                model,
                output_folder=str(out),
                encode_kwargs={"batch_size": args.batch_size},
                verbosity=1,
            )
        except Exception as e:
            print(f"!!! FAILED {name}: {e!r}", flush=True)
            failures.append(name)
    if failures:
        print("FAILED MODELS:", failures)


if __name__ == "__main__":
    main()
