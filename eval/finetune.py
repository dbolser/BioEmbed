"""Fine-tune a bge model on data/finetune/pairs.parquet with CachedMNRL.

Usage: uv run python finetune.py BAAI/bge-small-en-v1.5 ../models/bioembed-bge-small-gaf
"""
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss

REPO = Path(__file__).resolve().parent.parent
base, out = sys.argv[1], Path(sys.argv[2])
df = pd.read_parquet(REPO / "data" / "finetune" / "pairs.parquet")
str_cols = [c for c in df.columns if df[c].dtype == object]
a, p = (["anchor", "positive"] if {"anchor", "positive"} <= set(df.columns) else
        ["query", "positive"] if {"query", "positive"} <= set(df.columns) else str_cols[:2])
print(f"columns {list(df.columns)} -> anchor={a}, positive={p}; n={len(df)}", flush=True)
ds = Dataset.from_pandas(df[[a, p]].rename(columns={a: "anchor", p: "positive"}).astype(str), preserve_index=False)

model = SentenceTransformer(base)
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=32)
args = SentenceTransformerTrainingArguments(
    output_dir=str(out.parent / f"{out.name}-ckpt"), num_train_epochs=1, per_device_train_batch_size=256,
    learning_rate=2e-5, warmup_ratio=0.1, seed=42, fp16=True, logging_steps=10, save_strategy="no",
    report_to=[], batch_sampler="no_duplicates",
)
SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
model.model_card_data.model_name = f"bioembed/{out.name.removeprefix('bioembed-')}"
model.save(str(out))
print("saved", out)
