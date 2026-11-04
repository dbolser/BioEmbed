"""Embed texts with a large model on a GPU VM (the model check). Resumable: chunks of 2,000 are saved as they finish.
  python gpu_embed.py MODEL IN.parquet OUTDIR [--col text] [--batch 32] [--max-len 512]
Qwen3-Embedding: documents need no instruction (model card), so papers and concepts are encoded as plain text,
as MedEmbed saw them. Vectors are L2-normalised float16; OUTDIR/<stem>_<chunk>.npy, then OUTDIR/<stem>.npy when done.
"""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--col", default="text"); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--max-len", type=int, default=512)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); stem = Path(a.inp).stem
if (out / f"{stem}.npy").exists(): print("done already"); raise SystemExit
texts = pd.read_parquet(a.inp)[a.col].tolist()
m = SentenceTransformer(a.model, device="cuda", model_kwargs={"torch_dtype": torch.float16})
m.max_seq_length = a.max_len
C = 2000; t0 = time.time()
for s in range(0, len(texts), C):
    f = out / f"{stem}_{s // C:05d}.npy"
    if f.exists(): continue
    v = m.encode(texts[s:s + C], batch_size=a.batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    np.save(f, v.astype(np.float16))
    print(f"[{time.time() - t0:6.0f}s] {min(s + C, len(texts)):,}/{len(texts):,}", flush=True)
np.save(out / f"{stem}.npy", np.concatenate([np.load(out / f"{stem}_{s // C:05d}.npy") for s in range(0, len(texts), C)]))
print("done", flush=True)
