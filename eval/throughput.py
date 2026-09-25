"""Sustained embedding throughput on this GPU, mirroring the paper's
data/embedding_throughput.csv method: seq len 512 (pad/truncate), largest
batch that fits (doubling until OOM), median/p5/p95 tok/s over timed
batches of bio-suite corpus texts.

Usage: uv run python throughput.py --rate 0.97 --out ../results/embedding_throughput_gpu.csv
"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "tasks"

HF_IDS = {  # results dir name -> HF id (for encoder weights)
    "ncbi__MedCPT": "ncbi/MedCPT-Article-Encoder",
}


def corpus_texts(n=4096):
    texts = []
    for name in ["GOPubMedRetrieval", "R2MEDBiologyPooled"]:
        texts += load_from_disk(str(DATA / name))["corpus"]["text"]
    texts += load_from_disk(str(DATA / "BioMeSHClassification"))["test"]["text"]
    rng = np.random.default_rng(42)
    rng.shuffle(texts)
    return texts[:n]


def model_id_for(dirname: str, local_models: Path):
    if dirname in HF_IDS:
        return HF_IDS[dirname]
    if dirname.startswith("bioembed__"):
        return str(local_models / dirname.split("__", 1)[1])
    return dirname.replace("__", "/")


@torch.no_grad()
def run_batches(model, batches):
    times = []
    for b in batches:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        model(**b)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
    return times


def measure(hf_id, texts, seq_len, dtype):
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id, dtype=dtype, device_map="cuda", trust_remote_code=True).eval()  # device_map: stream weights to GPU (8B OOMs host RAM otherwise)
    params = sum(p.numel() for p in model.parameters())
    enc = tok(texts, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
    keys = [k for k in ("input_ids", "attention_mask") if k in enc]

    def mk(bs, start=0):
        return {k: enc[k][start:start + bs].cuda() for k in keys}

    bs = 8
    while True:
        try:
            run_batches(model, [mk(bs)]); run_batches(model, [mk(bs)])
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); bs //= 2; break
        if bs * 2 > len(texts):
            break
        bs *= 2
        torch.cuda.empty_cache()
    # sustained: as many batches as the corpus supports (>= 8), warm-up 2
    n_batches = max(8, min(32, len(texts) // bs))
    batches = [mk(bs, (i * bs) % max(1, len(texts) - bs)) for i in range(n_batches + 2)]
    times = run_batches(model, batches)[2:]
    tps = np.array([bs * seq_len / t for t in times])
    del model; torch.cuda.empty_cache()
    return params, bs, float(np.median(tps)), float(np.percentile(tps, 5)), float(np.percentile(tps, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, required=True, help="GPU $/hr actually paid")
    ap.add_argument("--out", default=str(REPO / "results" / "embedding_throughput_gpu.csv"))
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--models", nargs="*")
    args = ap.parse_args()

    gpu = torch.cuda.get_device_name(0)
    texts = corpus_texts()
    dirs = args.models or sorted(p.name for p in (REPO / "results" / "embedding").iterdir() if p.is_dir())
    out = Path(args.out)
    done = {}
    if out.exists():
        done = {r["model_id"]: r for r in csv.DictReader(out.open())}
    cols = ["model_id", "params", "batch_size_used", "seq_len", "tokens_per_batch", "median_tok_per_sec",
            "p5_tok_per_sec", "p95_tok_per_sec", "cost_usd_per_mtok", "gpu", "gpu_rate_usd_per_hr", "status", "error"]
    for d in dirs:
        hf_id = model_id_for(d, REPO / "models")
        mid = d.replace("__", "/")
        if mid in done and done[mid]["status"] == "success":
            continue
        print(f"=== {mid} ({hf_id})", flush=True)
        row = dict.fromkeys(cols, ""); row.update(model_id=mid, seq_len=args.seq_len, gpu=gpu, gpu_rate_usd_per_hr=args.rate)
        try:
            dtype = torch.float16
            params, bs, med, p5, p95 = measure(hf_id, texts, args.seq_len, dtype)
            row.update(params=params, batch_size_used=bs, tokens_per_batch=bs * args.seq_len,
                       median_tok_per_sec=round(med, 1), p5_tok_per_sec=round(p5, 1), p95_tok_per_sec=round(p95, 1),
                       cost_usd_per_mtok=round(args.rate / (med * 3600 / 1e6), 6), status="success")
        except Exception as e:
            row.update(status="error", error=repr(e)[:300]); torch.cuda.empty_cache()
        print(json.dumps(row), flush=True)
        done[mid] = row
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(done.values())


if __name__ == "__main__":
    main()
