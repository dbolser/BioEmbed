"""Fine-tune a cross-encoder reranker on curated gene-disease evidence
(docs/PREREG.md C8: our fine-tuned reranker vs the best off-the-shelf one).

Data: data/finetune_gd/train.parquet (anchor, positive, negative_1..3) from
scripts/build_finetune_gene_disease.py. Base: ncbi/MedCPT-Cross-Encoder
(default) or BAAI/bge-reranker-base, or any CrossEncoder-loadable model.

Losses (--loss):
  bce       pointwise BinaryCrossEntropyLoss on (anchor, doc, label) rows: the
            positive (1) and each of its first --negatives hard negatives (0),
            pos_weight = negatives per positive. Rows keep the negatives they
            have.
  listwise  CrossEncoder MultipleNegativesRankingLoss: softmax over the
            positive, the row's hard negatives and --in-batch random other
            docs of the batch (InfoNCE). Needs equal-width rows: rows with
            fewer than --negatives are dropped (counted). Batched by
            finetune.GuardedBatchSampler, so no in-batch doc is OT-linked to
            the anchor's pair.
Batch order follows --seed (bce: ST's seeded random batches; listwise:
finetune.Seeded). With --valid, eval loss is logged at step 0 and at every
checkpoint.

Usage (from eval/; on CPU pin torch as in finetune.py):
    uv run --with accelerate python finetune_ce.py --out ../models/bioembed-medcpt-ce-gd \\
        --valid ../data/finetune_gd/valid.parquet --loss bce --batch 32 --precision bf16 --save-steps 500
Output: --out (final model); <out>-ckpt/checkpoint-<step> every --save-steps.
On CPU, fp16/bf16 fall back to fp32. Score checkpoints with dev_eval.py.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss, MultipleNegativesRankingLoss

from finetune import GuardedBatchSampler, Seeded, load as load_listwise

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "finetune_gd" / "train.parquet"


def load(path: Path, negatives: int, loss: str) -> tuple[Dataset, float]:
    """Dataset for the loss, and negatives per positive."""
    if loss == "listwise":
        return load_listwise(path, negatives), float(negatives)
    df = pd.read_parquet(path)
    negs = [f"negative_{i + 1}" for i in range(negatives)]
    parts = [df[["anchor", "positive"]].rename(columns={"positive": "text"}).assign(label=1.0)]
    parts += [df[["anchor", c]].dropna().rename(columns={c: "text"}).assign(label=0.0) for c in negs]
    long = pd.concat(parts, ignore_index=True)
    ratio = (long.label == 0).sum() / max(1, (long.label == 1).sum())
    print(f"{path.name}: {len(df)} rows -> {len(long)} (anchor, text, label) pairs, "
          f"{ratio:.2f} negatives per positive", flush=True)
    return Dataset.from_pandas(long, preserve_index=False), float(ratio)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="ncbi/MedCPT-Cross-Encoder")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--valid", type=Path, help="eval-loss set, e.g. finetune_gd/valid.parquet")
    ap.add_argument("--loss", choices=["bce", "listwise"], default="bce")
    ap.add_argument("--negatives", type=int, default=3, help="hard-negative columns per row")
    ap.add_argument("--in-batch", type=int, default=4, help="listwise: random in-batch negatives")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--max-steps", type=int, default=-1, help="overrides --epochs (smoke tests)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    ap.add_argument("--save-steps", type=int, default=0, help="checkpoint (and eval) every N steps")
    args = ap.parse_args()
    precision = args.precision if torch.cuda.is_available() else "fp32"
    if precision != args.precision:
        print(f"no CUDA: {args.precision} -> fp32", flush=True)

    ds, ratio = load(args.data, args.negatives, args.loss)
    valid = load(args.valid, args.negatives, args.loss)[0] if args.valid else None
    model = CrossEncoder(args.base, num_labels=1, max_length=args.max_length)
    loss = (BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(ratio)) if args.loss == "bce"
            else MultipleNegativesRankingLoss(model, num_negatives=args.in_batch))
    save = "steps" if args.save_steps else "no"
    train_args = CrossEncoderTrainingArguments(
        output_dir=str(args.out.parent / f"{args.out.name}-ckpt"), num_train_epochs=args.epochs,
        max_steps=args.max_steps, per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch, learning_rate=args.lr, warmup_steps=0.1,
        seed=args.seed, fp16=precision == "fp16", bf16=precision == "bf16", logging_steps=10,
        save_strategy=save, save_steps=args.save_steps or None,
        eval_strategy=save if valid is not None else "no", eval_steps=args.save_steps or None,
        eval_on_start=valid is not None, report_to=[],
        # bce: ST's random batches (its RandomSampler generator is seeded with args.seed).
        batch_sampler=Seeded(GuardedBatchSampler, args.seed) if args.loss == "listwise" else "batch_sampler",
    )
    CrossEncoderTrainer(model=model, args=train_args, train_dataset=ds, eval_dataset=valid,
                        loss=loss).train()
    model.save(str(args.out))
    print("saved", args.out)


if __name__ == "__main__":
    main()
