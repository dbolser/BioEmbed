#!/usr/bin/env python
"""Fine-tune recipe selection for the hs-ft GPU queue (docs/PREREG.md: every
setting is chosen on the practice set only).

  pick   Of the bge-small seed-42 grid ({train, train_silver} x {0, 3} hard
         negatives), take the recipe with the best practice curator-only
         nDCG@10 (raw, as the BM25 setting was chosen) in dev_eval.py's CSV;
         ties go to the earlier recipe in GRID. Writes chosen.json and copies
         that model to models/bioembed-bge-small-gd-s42 (same recipe, same
         seed, so it is the seed-42 model of the chosen recipe).
  args   prints '--data PATH --negatives N' of the chosen recipe, for finetune.py
  data   prints the chosen data file, for finetune_ce.py
  check  MODEL_DIR BASE: the saved model's modules after the transformer
         (pooling, normalisation), max_seq_length and similarity function
         match the base model's; exits 1 if not

Run (from eval/, in its env):
    uv run python ../scripts/ft_select.py pick
    A=$(uv run python ../scripts/ft_select.py args) && uv run --with accelerate python finetune.py ... $A
    uv run python ../scripts/ft_select.py check ../models/bioembed-bge-base-gd-s42 BAAI/bge-base-en-v1.5
Output: data/finetune_gd/runs/chosen.json; models/bioembed-bge-small-gd-s42
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
FT = REPO / "data" / "finetune_gd"
RUNS = FT / "runs"
DEV_EVAL = RUNS / "dev_eval.csv"
CHOSEN = RUNS / "chosen.json"
MODELS = REPO / "models"
METRIC = "raw_ndcg@10"
GRID = [  # recipe -> (data file, hard negatives)
    ("train-n0", "train.parquet", 0),
    ("train-n3", "train.parquet", 3),
    ("silver-n0", "train_silver.parquet", 0),
    ("silver-n3", "train_silver.parquet", 3),
]


def grid_model(recipe: str) -> str:
    return f"bioembed-bge-small-gd-{recipe}-s42"


def pick() -> None:
    dev = pd.read_csv(DEV_EVAL)
    dev = dev[dev.max_queries == 0]
    dev["name"] = dev.model.map(lambda m: Path(m).name)
    rows = []
    for recipe, data, negatives in GRID:
        r = dev[dev.name == grid_model(recipe)]
        if len(r):
            rows.append({"recipe": recipe, "data": data, "negatives": negatives} | r.iloc[-1].to_dict())
    missing = [g[0] for g in GRID if g[0] not in {r["recipe"] for r in rows}]
    if not rows:
        sys.exit(f"no grid rows in {DEV_EVAL}")
    best = max(rows, key=lambda r: r[METRIC])  # max keeps the first of ties
    out = {"metric": f"{METRIC} (practice haystack GeneDiseaseHaystackDev, curator-only gold)",
           "chosen": best["recipe"], "data": best["data"], "negatives": int(best["negatives"]),
           "model": grid_model(best["recipe"]), "missing_recipes": missing,
           "grid": [{k: r[k] for k in ("recipe", "data", "negatives", "raw_ndcg@10", "condensed_ndcg@10",
                                       "raw_recall@100", "condensed_recall@100", "small_hit@1")} for r in rows]}
    CHOSEN.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    dst = MODELS / "bioembed-bge-small-gd-s42"
    if not dst.exists():
        shutil.copytree(MODELS / grid_model(best["recipe"]), dst)
    print(f"-> {CHOSEN}, {dst}")


def chosen() -> dict:
    return json.loads(CHOSEN.read_text())


def check(model_dir: str, base: str) -> None:
    from sentence_transformers import SentenceTransformer

    def describe(m: SentenceTransformer) -> dict:
        tail = [(type(mod).__name__, mod.get_config_dict() if hasattr(mod, "get_config_dict") else None)
                for mod in list(m)[1:]]
        return {"modules": tail, "max_seq_length": m.max_seq_length, "similarity": m.similarity_fn_name}

    got, want = (describe(SentenceTransformer(x, device="cpu")) for x in (model_dir, base))
    print(f"{model_dir}: {got}\n{base}: {want}")
    if got != want:
        sys.exit("pooling/normalisation/length differ from the base model")
    print("ok: matches the base model")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pick", "args", "data", "check"])
    ap.add_argument("rest", nargs="*")
    a = ap.parse_args()
    if a.cmd == "pick":
        pick()
    elif a.cmd == "args":
        c = chosen()
        print(f"--data {FT / c['data']} --negatives {c['negatives']}")
    elif a.cmd == "data":
        print(FT / chosen()["data"])
    else:
        check(*a.rest)


if __name__ == "__main__":
    main()
