"""Bio Pareto frontier: BioScore vs USD per bio-benchmark pass (log x).

Embedding costs: results/bio_embedding_costs.csv (count_tokens_bio.py), or
the paper's embedding_costs scaled by bio-suite token share for paper-only
models. LLM costs: recomputed per lifted task from their raw usage_stats at
the paper's prices; cached tokens allocated proportionally to input (the
per-task JSONs don't split caching out).

Until our own LLM runs cover Tier B/C, the plot uses the task subset both
paradigms cover (currently the 7 lifted Tier A tasks) — labelled as such.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
VENDOR = REPO / "vendor" / "embedders-dilemma"

# lifted task name -> file stem in llm_results
LLM_FILES = {
    "LLMBIOSSES": "BIOSSES",
    "LLMSTSBenchmark": "STSBenchmark",
    "LLMBiorxivClusteringP2PV2": "BiorxivClusteringP2P",
    "LLMMedrxivClusteringP2PV2": "MedrxivClusteringP2P",
    "LLMMedrxivClusteringS2SV2": "MedrxivClusteringS2S",
    "LLMToxicConversationsClassification": "ToxicConversationsClassification",
    "LLMPublicHealthQA": "LLMPublicHealthQA",
}
BIO_SPECIALISTS = {
    "NeuML__pubmedbert-base-embeddings", "ncbi__MedCPT-Query-Encoder",
    "abhinand__MedEmbed-small-v0.1", "abhinand__MedEmbed-base-v0.1",
    "abhinand__MedEmbed-large-v0.1", "FremyCompany__BioLORD-2023",
}


def llm_bio_costs() -> pd.DataFrame:
    usage = pd.read_csv(VENDOR / "data" / "llm_token_usage.csv").set_index("model")
    rows = []
    for model, u in usage.iterrows():
        base = VENDOR / "llm_results" / model / "results"
        files = list(base.rglob("*.json")) if base.exists() else []
        by_stem = {f.stem: f for f in files if not f.stem.endswith("_samples")}
        cache_frac = (u.cached_tokens / u.input_tokens) if u.input_tokens else 0.0
        cost = 0.0
        n = 0
        for task, stem in LLM_FILES.items():
            f = by_stem.get(stem)
            if f is None:
                continue
            s = json.loads(f.read_text())["scores"]["test"][0]
            us = s.get("usage_stats")
            if not us:
                continue
            inp, tot = us["input_tokens"], us["total_tokens"]
            cached = inp * cache_frac
            gen = tot - inp  # output + thinking, billed at output rate
            cost += ((inp - cached) * u.price_input_per_mtok
                     + cached * u.price_cached_per_mtok
                     + gen * u.price_output_per_mtok) / 1e6
            n += 1
        if n:
            rows.append({"model": model, "bio_cost_usd": cost, "n_cost_tasks": n})
    return pd.DataFrame(rows)


# current OpenRouter prices per MTok (in, cache_read, out) for models we run
OR_PRICES = {
    "deepseek__deepseek-v4-flash": (0.066, 0.013, 0.131),
    "qwen__qwen3.6-35b-a3b": (0.100, 0.050, 0.900),
    "qwen3.6-35b-a3b": (0.100, 0.050, 0.900),  # paper slug for the same model
    "google__gemini-3.1-flash-lite": (0.250, 0.025, 1.500),
    "google__gemini-3.1-flash-lite-preview": (0.250, 0.025, 1.500),
    "google__gemini-3.1-pro-preview": (2.000, 0.200, 12.000),
}


def our_llm_task_costs() -> pd.DataFrame:
    """Cost of OUR five tasks per LLM from measured usage at OpenRouter rates."""
    rows = []
    root = REPO / "results" / "llm"
    if not root.exists():
        return pd.DataFrame(columns=["model", "our_cost_usd", "n_our_tasks"])
    for mdir in root.iterdir():
        if mdir.name.endswith("__nothink"):
            continue
        prices = OR_PRICES.get(mdir.name)
        if prices is None:
            continue
        r_in, r_cache, r_out = prices
        cost, n = 0.0, 0
        for f in mdir.rglob("*.json"):
            if f.name == "model_meta.json" or f.stem.endswith("_samples"):
                continue
            u = json.loads(f.read_text())["scores"]["test"][0].get("usage_stats")
            if not u:
                continue
            fresh = u["input_tokens"] - u.get("cached_tokens", 0)
            gen = u["total_tokens"] - u["input_tokens"]
            cost += (fresh * r_in + u.get("cached_tokens", 0) * r_cache
                     + gen * r_out) / 1e6
            n += 1
        rows.append({"model": mdir.name, "our_cost_usd": cost, "n_our_tasks": n})
    return pd.DataFrame(rows)


def _pareto_panel(ax, bio, title, ylab):
    front = bio.sort_values("cost")
    best = -1
    fx, fy = [], []
    for _, r in front.iterrows():
        is_llm = r["model_type"] == "llm"
        is_bio = r["model"] in BIO_SPECIALISTS
        ax.scatter(r["cost"], r["bio_score"],
                   marker="^" if is_llm else ("s" if is_bio else "o"),
                   s=70, alpha=0.85,
                   color="#d62728" if is_llm else ("#2ca02c" if is_bio else "#1f77b4"))
        if r["bio_score"] > best:
            best = r["bio_score"]
            fx.append(r["cost"]); fy.append(r["bio_score"])
            ax.annotate(r["model"].split("__")[-1], (r["cost"], r["bio_score"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.step(fx, fy, where="post", color="gray", lw=1, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("USD per bio-benchmark pass (log)")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return [m.split("__")[-1] for m, sc in zip(front["model"], front["bio_score"])
            if sc in fy]


def main() -> None:
    scores = pd.read_csv(REPO / "results" / "bio_scores.csv")
    lifted = list(LLM_FILES)
    sub = scores[scores["task"].isin(lifted)]
    pivot = sub.pivot_table(index=["model", "model_type"], columns="task", values="score")
    pivot = pivot.dropna()  # complete coverage of the subset only
    bio = pivot.mean(axis=1).rename("bio_score").reset_index()

    emb_costs = pd.read_csv(REPO / "results" / "bio_embedding_costs.csv")
    emb_costs["model"] = emb_costs["model"].str.replace("/", "__")
    # paper-only embedding models: scale their 37-task cost by bio token share
    paper_costs = pd.read_csv(VENDOR / "data" / "embedding_costs_per_model.csv")
    paper_costs["model"] = paper_costs["model_id"].str.replace("/", "__")
    ours_total = emb_costs["tokens"].median()
    paper_costs["bio_cost_usd"] = paper_costs["cost_per_mtok"] * ours_total / 1e6

    cost_map = dict(zip(emb_costs["model"], emb_costs["total_cost_usd"]))
    for _, r in paper_costs.iterrows():
        cost_map.setdefault(r["model"], r["bio_cost_usd"])
    for _, r in llm_bio_costs().iterrows():
        cost_map[r["model"]] = r["bio_cost_usd"]

    bio["cost"] = bio["model"].map(cost_map)
    bio = bio.dropna(subset=["cost"])
    bio.to_csv(REPO / "results" / "bio_pareto.csv", index=False)

    # full 12-task suite panel: models with complete coverage
    all_tasks = scores["task"].unique()
    pivot_full = scores.pivot_table(index=["model", "model_type"],
                                    columns="task", values="score")
    pivot_full = pivot_full.dropna()
    full = pivot_full.mean(axis=1).rename("bio_score").reset_index()
    # full-suite LLM cost = lifted-task cost (paper usage, OpenRouter rates
    # where known, else paper prices) + our-task measured cost
    ours = our_llm_task_costs().set_index("model")
    lifted_llm = llm_bio_costs().set_index("model")
    full_cost = {}
    for _, r in full.iterrows():
        m = r["model"]
        if r["model_type"] == "embedding":
            if m in cost_map:
                full_cost[m] = cost_map[m]
        else:
            # scores use the paper's slug; our results/llm dirs use the
            # OpenRouter slug for the same model
            our_slug = {"qwen3.6-35b-a3b": "qwen__qwen3.6-35b-a3b",
                        "google__gemini-3.1-flash-lite-preview": "google__gemini-3.1-flash-lite",
                        }.get(m, m)
            lift = lifted_llm["bio_cost_usd"].get(m, None)
            our = ours["our_cost_usd"].get(our_slug, None)
            if lift is not None and our is not None:
                full_cost[m] = lift + our
    full["cost"] = full["model"].map(full_cost)
    full = full.dropna(subset=["cost"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    f1 = _pareto_panel(axes[0], bio,
                       f"Tier A subset ({len(lifted)} lifted tasks, paper LLM results)",
                       "BioScore (lifted subset)")
    f2 = _pareto_panel(axes[1], full,
                       f"Full BioMTEB suite ({pivot_full.shape[1]} tasks)",
                       "BioScore (full suite)")
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[
        Line2D([], [], marker="^", ls="", color="#d62728", label="LLM (API cost)"),
        Line2D([], [], marker="o", ls="", color="#1f77b4", label="Embedding, general"),
        Line2D([], [], marker="s", ls="", color="#2ca02c", label="Embedding, bio-specialised"),
    ], loc="lower right", fontsize=9)

    out = REPO / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "bio_pareto.png", dpi=160)
    fig.savefig(out / "bio_pareto.pdf")
    print(f"panel A: {len(bio)} models, frontier {' -> '.join(f1)}")
    print(f"panel B: {len(full)} models, frontier {' -> '.join(f2)}")


if __name__ == "__main__":
    main()
