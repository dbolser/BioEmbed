"""Count embedding-side tokens for the bio suite with each model's own
tokenizer (the paper's accounting: a proxy tokenizer misestimates 22-54%).

Counts what the embedder encodes:
  classification: train + test texts | sts/pair: both sentences (test)
  clustering: documents (test)       | retrieval: corpus + queries

Output: results/bio_token_counts.csv (model, task, tokens) + per-model totals
joined with $/MTok (paper's measured H100 throughput; params-nearest proxy
for models the paper didn't measure, flagged proxy=True).
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
VENDOR = REPO / "vendor" / "embedders-dilemma"

TIER_A = {
    "LLMBIOSSES": ("sts", "mteb/llm-eval-biosses"),
    "LLMSTSBenchmark": ("sts", "mteb/llm-eval-stsbenchmark"),
    "LLMBiorxivClusteringP2PV2": ("clust", "mteb/llm-eval-biorxiv_clustering_p2p_v2"),
    "LLMMedrxivClusteringP2PV2": ("clust", "mteb/llm-eval-medrxiv_clustering_p2p_v2"),
    "LLMMedrxivClusteringS2SV2": ("clust", "mteb/llm-eval-medrxiv_clustering_s2s_v2"),
    "LLMToxicConversationsClassification": ("cls", "mteb/llm-eval-toxic_conversations"),
    "LLMPublicHealthQA": ("ret", "mteb/llm-eval-public-health-qa"),
}
LOCAL = {  # data/tasks/<name>, present once builders finish
    "BioMeSHClassification": "cls",
    "PubChemSynonymPC500": "pair",
    "R2MEDBiologyPooled": "ret",
    "GOPubMedRetrieval": "ret",
    "GOProteinPairCls": "pair",
}

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5",
    "intfloat/multilingual-e5-small",
    "google/embeddinggemma-300m",
    "Snowflake/snowflake-arctic-embed-l-v2.0",
    "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-8B",
    "NeuML/pubmedbert-base-embeddings",
    "ncbi/MedCPT",
    "abhinand/MedEmbed-small-v0.1", "abhinand/MedEmbed-base-v0.1", "abhinand/MedEmbed-large-v0.1",
    "FremyCompany/BioLORD-2023",
]


def texts_hf(kind: str, path: str) -> list[str]:
    out: list[str] = []
    if kind == "sts":
        d = load_dataset(path, split="test")
        out += [*d["sentence1"], *d["sentence2"]]
    elif kind == "clust":
        d = load_dataset(path, split="test")
        col = "sentences" if "sentences" in d.column_names else "text"
        for row in d[col]:
            out += row if isinstance(row, list) else [row]
    elif kind == "cls":
        for split in ("train", "test"):
            try:
                d = load_dataset(path, split=split)
                out += [str(t) for t in d["text"]]
            except Exception:
                pass
    elif kind == "ret":
        out += [r["text"] for r in load_dataset(path, "corpus", split="corpus")]
        out += [r["text"] for r in load_dataset(path, "queries", split="queries")]
    return out


def texts_local(kind: str, name: str) -> list[str] | None:
    p = REPO / "data" / "tasks" / name
    if not p.exists():
        return None
    dd = load_from_disk(str(p))
    out: list[str] = []
    if kind == "cls":
        for split in dd:
            out += [str(t) for t in dd[split]["text"]]
    elif kind == "pair":
        for split in dd:
            out += [*dd[split]["sentence1"], *dd[split]["sentence2"]]
    elif kind == "ret":
        out += [r["text"] for r in dd["corpus"]]
        out += [r["text"] for r in dd["queries"]]
    return out


def main() -> None:
    task_texts: dict[str, list[str]] = {}
    for name, (kind, path) in TIER_A.items():
        task_texts[name] = texts_hf(kind, path)
    for name, kind in LOCAL.items():
        t = texts_local(kind, name)
        if t is not None:
            task_texts[name] = t
        else:
            print(f"(skipping {name} — not built yet)")

    tokenizer_overrides = {"ncbi/MedCPT": "ncbi/MedCPT-Query-Encoder"}
    rows = []
    for model in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(
                tokenizer_overrides.get(model, model), trust_remote_code=True)
        except OSError as e:
            print(f"SKIP {model}: {str(e).splitlines()[0]}")
            continue
        for name, texts in task_texts.items():
            n = sum(len(ids) for ids in tok(texts, add_special_tokens=True).input_ids)
            rows.append({"model": model, "task": name, "tokens": n})
        total = sum(r["tokens"] for r in rows if r["model"] == model)
        print(f"{model:45s} {total/1e6:.2f}M tokens")

    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / "bio_token_counts.csv", index=False)

    # join $/MTok from the paper's measured throughput; proxy by nearest params
    thr = pd.read_csv(VENDOR / "data" / "embedding_throughput.csv")
    thr = thr[thr["status"] == "success"][["model_id", "params", "cost_usd_per_mtok"]]
    params = {  # millions
        "sentence-transformers/all-MiniLM-L6-v2": 22, "BAAI/bge-small-en-v1.5": 33,
        "BAAI/bge-base-en-v1.5": 109, "BAAI/bge-large-en-v1.5": 335,
        "intfloat/multilingual-e5-small": 118, "google/embeddinggemma-300m": 308,
        "Snowflake/snowflake-arctic-embed-l-v2.0": 568,
        "Qwen/Qwen3-Embedding-0.6B": 596, "Qwen/Qwen3-Embedding-4B": 4000,
        "Qwen/Qwen3-Embedding-8B": 7570, "NeuML/pubmedbert-base-embeddings": 109,
        "ncbi/MedCPT": 109, "abhinand/MedEmbed-small-v0.1": 33,
        "abhinand/MedEmbed-base-v0.1": 109, "abhinand/MedEmbed-large-v0.1": 335,
        "FremyCompany/BioLORD-2023": 109,
    }
    out = []
    for model, grp in df.groupby("model"):
        exact = thr[thr["model_id"] == model]
        if len(exact):
            rate, proxy = float(exact["cost_usd_per_mtok"].iloc[0]), ""
        else:
            near = thr.iloc[(thr["params"] - params[model] * 1e6).abs().argsort()[:1]]
            rate, proxy = float(near["cost_usd_per_mtok"].iloc[0]), near["model_id"].iloc[0]
        tokens = int(grp["tokens"].sum())
        out.append({"model": model, "tokens": tokens, "cost_usd_per_mtok": rate,
                    "total_cost_usd": tokens / 1e6 * rate, "throughput_proxy": proxy})
    pd.DataFrame(out).to_csv(REPO / "results" / "bio_embedding_costs.csv", index=False)
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
