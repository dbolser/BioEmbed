"""Fine-tune a sentence-transformers embedder with CachedMNRL.

Defaults reproduce the GO fine-tunes (bge-gaf): data/finetune/pairs.parquet
(anchor, positive), 1 epoch, batch 256, lr 2e-5, seed 42, fp16, no checkpoints.

Gene-disease (docs/PREREG.md C5, C6): data/finetune_gd/train.parquet from
scripts/build_finetune_gene_disease.py. --negatives N adds its first N
hard-negative columns (negative_1..N); rows with fewer are dropped (counted).
The builder caps BM25-mined negatives at 1 per row, so N = 1 keeps 98% of
rows (negative_1 is same-gene when the row has one) and N = 3 keeps 44%.
Loss: CachedMultipleNegativesRankingLoss, not CachedGISTEmbedLoss. The hard
negatives are already checked against every OT link of the gene, so they need
no guide filter, and a guide (an off-the-shelf embedder) would mask the
same-gene negatives it cannot tell from the positive: the very skill being
trained. It is also the loss of the GO fine-tunes, so C5 compares data, not
losses. In-batch false negatives are handled by the known links instead:
with guard_pmids in the data, GuardedBatchSampler never batches a row with
another whose docs are OT-linked to its pair. On train.parquet, plain
no_duplicates batches give 8% of anchors such a doc at batch 128 with 1
negative, 15% at 256 with 1, 22% at 128 with 3; guarded, 0%, with 0.01%
(128, 1), 0.9% (256, 1) and 3.5% (128, 3) of rows sitting out an epoch. Its
batch order follows --seed (ST's trainer alone always passes seed 0).

Same inputs as the base, so C5 compares like with like: --prompts mteb (the
default for data with guard_pmids, i.e. finetune_gd) trains with the base
model's mteb prompts (bge: query instruction) and saves them as the model's
'query'/'document' prompts; --max-seq-length is for training only, saved
models and checkpoints keep the base's. haystack.py and dev_eval.py then
encode exactly as for the off-the-shelf base. With --valid, eval loss is
logged at step 0 and at every checkpoint.

Usage (from eval/; on GPU, --with accelerate brings its own CUDA torch, on
CPU pin the env's: --with torch==2.14.0+cpu --index
https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --frozen):
    uv run --with accelerate python finetune.py BAAI/bge-small-en-v1.5 ../models/bioembed-bge-small-gaf
    uv run --with accelerate python finetune.py --base BAAI/bge-small-en-v1.5 --out ../models/bioembed-bge-small-gd \\
        --data ../data/finetune_gd/train.parquet --valid ../data/finetune_gd/valid.parquet \\
        --negatives 1 --batch 128 --precision bf16 --save-steps 200
Output: --out (final model); <out>-ckpt/checkpoint-<step> every --save-steps.
On CPU, fp16/bf16 fall back to fp32.
"""
import argparse
from pathlib import Path

import mteb
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.base.sampler import DefaultBatchSampler
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "finetune" / "pairs.parquet"


class Seeded:
    """Batch-sampler factory that passes the run's seed (ST's trainer passes 0;
    its samplers reseed with seed + epoch, so --seed never moved the batches)."""

    def __init__(self, sampler: type[DefaultBatchSampler], seed: int):
        self.sampler, self.seed = sampler, seed

    def __call__(self, dataset: Dataset, **kwargs) -> DefaultBatchSampler:
        return self.sampler(dataset, **kwargs | {"seed": self.seed})


class GuardedBatchSampler(DefaultBatchSampler):
    """no_duplicates by pair and PMID, plus no row whose docs are in another
    batch row's guard_pmids (or the reverse). Keys come from load(); a dataset
    without them is batched as no_duplicates on its texts. Rows, in a seeded
    order, go first-fit into ceil(n / batch_size) batches (floor with
    drop_last) starting round-robin, so a gene's conflicting rows spread over
    all batches. A row that conflicts with every batch sits out the epoch
    (printed): a heavy gene (FLNA, GJB2) can have more mutually conflicting
    rows than an epoch has batches; deferring them instead left ~75 tiny
    batches of ~170 at 256. Planned per epoch, so len() is exact."""

    def __init__(self, dataset: Dataset, batch_size: int, drop_last: bool, **kwargs):
        super().__init__(dataset, batch_size=batch_size, drop_last=drop_last, **kwargs)
        self.keys = getattr(dataset, "batch_keys", None) or (
            [frozenset(map(str, r.values())) for r in dataset], [frozenset()] * len(dataset))
        self.planned: tuple[int, list[list[int]]] | None = None

    def plan(self) -> list[list[int]]:
        if self.planned and self.planned[0] == self.epoch:
            return self.planned[1]
        docs, guards = self.keys
        generator = self.generator if self.generator is not None else torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        n = len(docs)
        n_bins = n // self.batch_size if self.drop_last else -(-n // self.batch_size)
        bins = [([], set(), set()) for _ in range(n_bins)]  # rows, their docs, their guards
        left_out = 0
        for k, i in enumerate(torch.randperm(n, generator=generator).tolist()):
            for j in range(n_bins):
                batch, used, guarded = bins[(k + j) % n_bins]
                if (len(batch) < self.batch_size and used.isdisjoint(docs[i])
                        and guarded.isdisjoint(docs[i]) and used.isdisjoint(guards[i])):
                    batch.append(i)
                    used |= docs[i]
                    guarded |= guards[i]
                    break
            else:
                left_out += 1
        batches = [b for b, _, _ in bins if len(b) == self.batch_size or (b and not self.drop_last)]
        print(f"GuardedBatchSampler epoch {self.epoch}: {len(batches)} batches, {left_out} of {n} rows "
              "left out (conflict with every batch)", flush=True)
        self.planned = (self.epoch, batches)
        return batches

    def __iter__(self):
        yield from self.plan()

    def __len__(self) -> int:
        return len(self.plan())


def load(path: Path, negatives: int) -> Dataset:
    """Dataset (anchor, positive, negative_1..N); with guard_pmids in the file,
    also GuardedBatchSampler keys as .batch_keys: per row {pair qid, doc PMIDs}
    and its guard PMIDs."""
    df = pd.read_parquet(path)
    str_cols = [c for c in df.columns if df[c].dtype == object]
    a, p = (["anchor", "positive"] if {"anchor", "positive"} <= set(df.columns) else
            ["query", "positive"] if {"query", "positive"} <= set(df.columns) else str_cols[:2])
    negs = [f"negative_{i + 1}" for i in range(negatives)]
    n = len(df)
    df = df[df[[a, p, *negs]].notna().all(axis=1)]
    print(f"{path.name}: columns {list(df.columns)} -> anchor={a}, positive={p}, negatives={negs}; "
          f"n={len(df)} ({n - len(df)} rows with < {negatives} negatives dropped)", flush=True)
    ds = Dataset.from_pandas(df[[a, p, *negs]].rename(columns={a: "anchor", p: "positive"}).astype(str),
                             preserve_index=False)
    if "guard_pmids" in df.columns:
        ds.batch_keys = ([frozenset([q, d, *ns[:negatives]]) for q, d, ns in zip(df.qid, df.pmid, df.neg_pmids)],
                         [frozenset(g) for g in df.guard_pmids])
    return ds


class BaseLengthTrainer(SentenceTransformerTrainer):
    """Saves checkpoints and the final model at the base model's
    max_seq_length, whatever length training used."""

    def __init__(self, *args, save_max_seq_length: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_max_seq_length = save_max_seq_length

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        train_length, self.model.max_seq_length = self.model.max_seq_length, self.save_max_seq_length
        try:
            super()._save(output_dir, state_dict)
        finally:
            self.model.max_seq_length = train_length


def mteb_prompts(base: str) -> dict[str, str]:
    """The base model's mteb 'query'/'document' prompts ({} if none)."""
    try:
        prompts = mteb.get_model_meta(base).loader_kwargs.get("model_prompts") or {}
    except KeyError:  # not in mteb's registry
        prompts = {}
    return {k: prompts[k] for k in ("query", "document") if prompts.get(k)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("legacy", nargs="*", metavar="BASE OUT", help="positional form of --base --out")
    ap.add_argument("--base")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--valid", type=Path, help="eval-loss set, e.g. finetune_gd/valid.parquet")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--max-steps", type=int, default=-1, help="overrides --epochs (smoke tests)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--mini-batch", type=int, default=32, help="CachedMNRL gradient-cache chunk")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--negatives", type=int, default=0, help="hard-negative columns per row")
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--prompts", choices=["none", "mteb"],
                    help="default: mteb for data with guard_pmids (finetune_gd), else none")
    ap.add_argument("--max-seq-length", type=int, help="training only; saved models keep the base's")
    ap.add_argument("--save-steps", type=int, default=0, help="checkpoint (and eval) every N steps")
    args = ap.parse_args()
    base = args.base or (args.legacy[0] if args.legacy else ap.error("--base required"))
    out = args.out or (Path(args.legacy[1]) if len(args.legacy) > 1 else ap.error("--out required"))
    precision = args.precision if torch.cuda.is_available() else "fp32"
    if precision != args.precision:
        print(f"no CUDA: {args.precision} -> fp32", flush=True)

    ds = load(args.data, args.negatives)
    valid = load(args.valid, args.negatives) if args.valid else None
    guarded = hasattr(ds, "batch_keys")
    prompts = mteb_prompts(base) if (args.prompts or ("mteb" if guarded else "none")) == "mteb" else {}
    columns = {c: prompts.get("query" if c == "anchor" else "document") for c in ds.column_names}
    print(f"training prompts: {prompts or 'none'}", flush=True)

    model = SentenceTransformer(base)
    base_length = model.max_seq_length
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length
    if prompts:  # saved with every checkpoint; encode-time only, training uses train_args.prompts
        model.prompts = {"query": prompts.get("query", ""), "document": prompts.get("document", "")}
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch)
    save = "steps" if args.save_steps else "no"
    train_args = SentenceTransformerTrainingArguments(
        output_dir=str(out.parent / f"{out.name}-ckpt"), num_train_epochs=args.epochs,
        max_steps=args.max_steps, per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch, learning_rate=args.lr, warmup_steps=0.1,
        seed=args.seed, fp16=precision == "fp16", bf16=precision == "bf16", logging_steps=10,
        save_strategy=save, save_steps=args.save_steps or None,
        eval_strategy=save if valid is not None else "no", eval_steps=args.save_steps or None,
        eval_on_start=valid is not None, report_to=[],
        batch_sampler=Seeded(GuardedBatchSampler, args.seed) if guarded else "no_duplicates",
        prompts={c: p for c, p in columns.items() if p} or None,
    )
    BaseLengthTrainer(model=model, args=train_args, train_dataset=ds, eval_dataset=valid, loss=loss,
                      save_max_seq_length=base_length).train()
    model.model_card_data.model_name = f"bioembed/{out.name.removeprefix('bioembed-')}"
    model.max_seq_length = base_length
    model.save(str(out))
    print("saved", out)


if __name__ == "__main__":
    main()
