"""Gene-disease fine-tuning plumbing: GuardedBatchSampler (eval/finetune.py),
one finetune_ce.py step per loss, and dev_eval's scoring (eval/dev_eval.py).

Run (the finetune_ce test needs accelerate and MedCPT-Cross-Encoder in the HF
cache, else it skips):
    cd eval && uv run --with pytest --with accelerate pytest ../tests/test_finetune_gd.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))

from finetune import GuardedBatchSampler, Seeded, load  # noqa: E402

TRAIN = REPO / "data" / "finetune_gd" / "train.parquet"
DEV_SCORES = REPO / "results" / "haystack" / "GeneDiseaseHaystackDev" / "scores.csv"


def batches(ds: Dataset, batch_size: int, seed: int = 0) -> list[list[int]]:
    return list(Seeded(GuardedBatchSampler, seed)(ds, batch_size=batch_size, drop_last=False,
                                                  valid_label_columns=None, generator=None))


def assert_guarded(ds: Dataset, plan: list[list[int]], batch_size: int) -> None:
    docs, guards = ds.batch_keys
    rows = [i for b in plan for i in b]
    assert len(rows) == len(set(rows)), "a row in two batches"
    for b in plan:
        assert len(b) <= batch_size
        for i in b:
            others = set().union(*(docs[j] for j in b if j != i))
            assert not docs[i] & others, f"row {i} shares a doc or pair with its batch"
            assert not guards[i] & others, f"row {i} has an OT-linked doc in its batch"


def toy() -> Dataset:
    """Gene g: rows of (g, d1) and (g, d2) whose docs are each other's known positives."""
    ds = Dataset.from_dict({"anchor": list("abcdef"), "positive": list("uvwxyz")})
    ds.batch_keys = (
        [frozenset(x) for x in (["g|d1", "1", "3"], ["g|d1", "2"], ["g|d2", "3", "1"],
                                ["g|d2", "4"], ["h|d1", "5"], ["k|d3", "6"])],
        [frozenset(x) for x in (["1", "2"], ["1", "2"], ["3", "4"], ["3", "4"], ["5"], ["6"])],
    )
    return ds


def test_guarded_toy():
    ds = toy()
    for seed in range(20):
        assert_guarded(ds, batches(ds, 3, seed), 3)


def test_seed_moves_batches():
    ds = Dataset.from_dict({"anchor": [str(i) for i in range(64)], "positive": [f"p{i}" for i in range(64)]})
    assert batches(ds, 8, seed=1) == batches(ds, 8, seed=1)
    assert batches(ds, 8, seed=1) != batches(ds, 8, seed=2)


@pytest.mark.skipif(not TRAIN.exists(), reason="run scripts/build_finetune_gene_disease.py")
def test_guarded_train_file():
    """The recommended setting (1 hard negative, batch 128) places ~all rows."""
    ds = load(TRAIN, 1)
    plan = batches(ds, 128)
    assert_guarded(ds, plan, 128)
    assert sum(map(len, plan)) >= 0.98 * len(ds)


@pytest.mark.parametrize("loss", ["bce", "listwise"])
def test_finetune_ce_one_step(tmp_path, monkeypatch, loss):
    """One training step plus step-0 eval per loss (bce once crashed on its first batch)."""
    pytest.importorskip("accelerate")
    from huggingface_hub import try_to_load_from_cache
    base = "ncbi/MedCPT-Cross-Encoder"
    if not isinstance(try_to_load_from_cache(base, "config.json"), str):
        pytest.skip(f"{base} not in the local HF cache")
    import finetune_ce
    rows = [{"qid": f"g{i}|d", "pmid": str(10 * i), "anchor": f"GENE{i} in disease", "positive": f"pos {i}",
             **{f"negative_{k}": f"neg {i}.{k}" for k in (1, 2, 3)},
             "neg_pmids": [str(10 * i + k) for k in (1, 2, 3)], "guard_pmids": []} for i in range(8)]
    data = tmp_path / "train.parquet"
    pd.DataFrame(rows).to_parquet(data)
    out = tmp_path / "ce"
    monkeypatch.setattr(sys, "argv", ["finetune_ce.py", "--base", base, "--out", str(out), "--data", str(data),
                                      "--valid", str(data), "--loss", loss, "--batch", "4", "--in-batch", "2",
                                      "--max-steps", "1", "--max-length", "32"])
    finetune_ce.main()
    assert (out / "config.json").exists()


@pytest.mark.skipif(not DEV_SCORES.exists(), reason="no practice scores.csv")
def test_dev_eval_reproduces_bm25():
    """A constant scorer keeps BM25 order: dev_eval must match score_haystack.py."""
    from dev_eval import FIRST_PASS, haystack

    class Constant:
        def score(self, queries, docs, pairs):
            return np.zeros(len(pairs), dtype=np.float32)

    got = haystack(Constant(), 1000, None)
    ref = pd.read_csv(DEV_SCORES).query("system == @FIRST_PASS.stem.split('__')[0] and track == 'gene'")
    for variant, r in ref.set_index("variant").iterrows():
        assert got[f"{variant}_ndcg@10"] == pytest.approx(r["ndcg@10"], abs=1e-4)
        assert got[f"{variant}_recall@100"] == pytest.approx(r["recall@100"], abs=1e-4)


def test_small_task_text_as_mteb():
    """dev_eval's small-task docs are what mteb encodes for results/v2_scores.csv."""
    from datasets import load_from_disk
    from dev_eval import SMALL, corpus_texts
    from mteb._create_dataloaders import _corpus_to_dict
    from tasks_local import GENE_DISEASE_V2

    task = next(t for t in GENE_DISEASE_V2 if t.metadata.name == SMALL.name)()
    task.load_data()
    mteb_texts = [_corpus_to_dict(r)["text"] for r in task.dataset["default"]["test"]["corpus"]]
    assert corpus_texts(load_from_disk(str(SMALL))["corpus"]) == mteb_texts
