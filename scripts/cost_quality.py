#!/usr/bin/env python
"""Quality vs cost on the 500-query test haystack (exploratory; see results/writeup/cost_notes.md).

Quality: graded nDCG@10 (pooled, both judges) and curator-only nDCG@10 (raw)
from results/haystack/graded/stats.csv; for LLM reranks also the score under
the other judge's grades (stats_haystack.py --judgments judgments_merged_no-<judge>.parquet,
output dirs given by --no-gemini / --no-deepseek).

Cost: whole-job L4 time (runner.log, start of the last try to exit) x one
on-demand rate, plus LLM API spend from the rerank sidecars. Test runs with
no log (made 2026-10-15 on Spot VMs) are estimated: from a same-architecture
model's logged test run, else from the model's practice-set encode time x the
median test-job / practice-encode ratio of the models that have both.

Run:  (cd scripts; for j in gemini deepseek; do uv run python stats_haystack.py --n-perm 200 \
          --judgments ../data/haystack/GeneDiseaseHaystack/judgments_merged_no-$j.parquet \
          --out /tmp/cq_no$j; done)   # means only; they do not depend on --n-perm
      uv run --with matplotlib python scripts/cost_quality.py --no-gemini /tmp/cq_nogemini \
          --no-deepseek /tmp/cq_nodeepseek
Output: results/writeup/cost_quality.csv, docs/writeup_frontier.png
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "data/gcp_logs"
USD_PER_L4_HOUR = 0.94  # g2-standard-8 + 1 L4, on-demand, europe-west1
FIRST = "rrf-bm25-symbol_disease-k0.9-b0.75+BAAI__bge-base-en-v1.5"
BM25 = "bm25-symbol_disease-k0.9-b0.75"
LLM_SIDE = ROOT / "results/llm_v2/haystack_rerank/GeneDiseaseHaystack"

# (name, group, run ids, first pass or None, how the GPU run is costed)
# group: base = BM25 baseline, single = off-the-shelf one-step, ours = our fine-tunes,
# rerank = first pass + off-the-shelf reranker, bm25llm = BM25 + LLM rerank (no GPU).
# Shape: no reranker vs reranked.
EMBEDDERS = [
    ("embeddinggemma-300m", "google__embeddinggemma-300m"),
    ("Qwen3-Embedding-0.6B", "Qwen__Qwen3-Embedding-0.6B"),
    ("BiCA-base", "bisectgroup__BiCA-base"),
    ("MedCPT", "ncbi__MedCPT"),
    ("BiCA-small", "bisectgroup__BiCA-small"),
    ("MedEmbed-small", "abhinand__MedEmbed-small-v0.1"),
    ("bge-base", "BAAI__bge-base-en-v1.5"),
    ("MedEmbed-base", "abhinand__MedEmbed-base-v0.1"),
    ("bge-small", "BAAI__bge-small-en-v1.5"),
    ("MiniLM-L6", "sentence-transformers__all-MiniLM-L6-v2"),
    ("BMRetriever-410M", "BMRetriever__BMRetriever-410M"),
    ("BioLORD-2023", "FremyCompany__BioLORD-2023"),
    ("e5-small (multilingual)", "intfloat__multilingual-e5-small"),
    ("PubMedBERT-embeddings", "NeuML__pubmedbert-base-embeddings"),
]
CROSS = [  # (name, group, model slug in the run id)
    ("Qwen3-Reranker-4B", "rerank", "Qwen__Qwen3-Reranker-4B"),
    ("Qwen3-Reranker-0.6B", "rerank", "Qwen__Qwen3-Reranker-0.6B"),
    ("MedCPT cross-encoder", "rerank", "ncbi__MedCPT-Cross-Encoder"),
    ("bge-reranker-v2-m3", "rerank", "BAAI__bge-reranker-v2-m3"),
    ("bge-reranker-base", "rerank", "BAAI__bge-reranker-base"),
    ("MedCPT cross-encoder, fine-tuned (ours)", "ours", "bioembed__medcpt-ce-gd-s42"),
    ("bge-reranker-base, fine-tuned (ours)", "ours", "bioembed__bge-reranker-base-gd-s42"),
]
LLMS = [  # (name, llm slug, judge family it is one of, first pass)
    ("Gemini 3.1 Flash-Lite rerank", "gemini-3.1-flash-lite", "gemini", FIRST),
    ("DeepSeek-V4-Flash rerank", "deepseek-v4-flash", "deepseek", FIRST),
    # not in the requested list: LLM rerank of BM25's top 20 (no GPU at all)
    ("BM25 + Gemini 3.1 Flash-Lite rerank", "gemini-3.1-flash-lite", "gemini", BM25),
    ("BM25 + DeepSeek-V4-Flash rerank", "deepseek-v4-flash", "deepseek", BM25),
]
SAME_ARCH = {  # phase-1 runs with no log -> logged test runs of the same architecture
    "BAAI__bge-base-en-v1.5": "bge-base-gd", "abhinand__MedEmbed-base-v0.1": "bge-base-gd",
    "bioembed__bge-base-gaf": "bge-base-gd",
    "BAAI__bge-small-en-v1.5": "bge-small-gd", "abhinand__MedEmbed-small-v0.1": "bge-small-gd",
    "bioembed__bge-small-gaf": "bge-small-gd",
}
SEEDS = (42, 43, 44)


def jobs(vm: str) -> list[dict]:
    """Each finished job's command and seconds from its last start to its exit."""
    start, out = {}, []
    for line in (LOGS / vm / "runner.log").read_text().splitlines():
        if m := re.match(r"(\S+) job (\d+) start \(try \d+\): (.*)", line):
            start[m[2]] = (datetime.fromisoformat(m[1]), m[3])
        elif m := re.match(r"(\S+) job (\d+) exit 0", line):
            t0, cmd = start[m[2]]
            out.append({"vm": vm, "job": int(m[2]), "cmd": cmd,
                        "seconds": (datetime.fromisoformat(m[1]) - t0).total_seconds()})
    return out


def encode_seconds(vm: str, job: int) -> float:
    """Sum of the per-chunk document-encoding times a haystack.py job printed."""
    text = (LOGS / vm / "logs" / f"{job}.log").read_text(errors="replace")
    return sum(int(s) for s in re.findall(r"docs [\d,]+/[\d,]+ \((\d+)s\)", text))


def slug(model: str) -> str:
    return model.replace("/", "__")


def gpu_times() -> tuple[dict, dict, dict, dict]:
    """Seconds per test embedding run, per test rerank, per training run; and the estimate notes."""
    all_jobs = [j for vm in ("hs-q1", "hs-q2", "hs-q3", "hs-ft", "hs-rr") for j in jobs(vm)]
    embed, practice, rerank, train = {}, {}, {}, {}
    for j in all_jobs:
        cmd = j["cmd"]
        if m := re.search(r"haystack\.py .*--models (\S+)", cmd):
            if "GeneDiseaseHaystackDev" in cmd:
                practice[slug(m[1])] = encode_seconds(j["vm"], j["job"])
            else:
                embed[slug(m[1])] = j["seconds"]
        elif "rerank_haystack.py" in cmd and "--haystack GeneDiseaseHaystack " in cmd:
            rerank[slug(re.search(r"--model (\S+)", cmd)[1])] = j["seconds"]
        elif m := re.search(r"finetune(?:_ce)?\.py .*--out \.\./models/bioembed-(\S+)", cmd):
            train[m[1]] = j["seconds"]
    chosen = json.loads((ROOT / "data/finetune_gd/runs/chosen.json").read_text())["model"]
    train["bge-small-gd-s42"] = train[chosen.removeprefix("bioembed-")]  # the grid run that became s42
    ratio = {k: embed[k] / practice[k] for k in embed if practice.get(k)}  # MedCPT-cosine reuses vectors
    r = float(np.median(list(ratio.values())))
    how = {k: "logged test run" for k in embed}
    for arch in ("bge-base-gd", "bge-small-gd"):
        embed[arch] = np.mean([embed[f"bioembed__{arch}-s{s}"] for s in SEEDS])
    for k, arch in SAME_ARCH.items():
        embed[k], how[k] = embed[arch], f"no log; = logged test runs of {arch} (same architecture)"
    for k, sec in practice.items():
        if k not in embed:
            embed[k], how[k] = sec * r, f"no log; practice encode time x {r:.2f}"
    how["_ratio"] = ratio
    return embed, rerank, train, how


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gemini", type=Path, required=True, help="stats_haystack.py --out dir, no-Gemini grades")
    ap.add_argument("--no-deepseek", type=Path, required=True, help="the same, no-DeepSeek grades")
    args = ap.parse_args()

    def scores(path: Path, variant: str, metric: str) -> pd.Series:
        s = pd.read_csv(path)
        s = s[(s.variant == variant) & (s.metric == metric)]
        return s.set_index("system")["mean"]

    stats = ROOT / "results/haystack/graded/stats.csv"
    graded, curator = scores(stats, "pooled", "graded_ndcg@10"), scores(stats, "raw", "ndcg@10")
    ci = pd.read_csv(stats).query("variant == 'pooled' and metric == 'graded_ndcg@10'").set_index("system")
    other = {"gemini": scores(args.no_gemini / "stats.csv", "pooled", "graded_ndcg@10"),
             "deepseek": scores(args.no_deepseek / "stats.csv", "pooled", "graded_ndcg@10")}
    embed, rerank, train, how = gpu_times()

    def usd(sec: float) -> float:
        return sec / 3600 * USD_PER_L4_HOUR

    rows = []

    def add(name, group, runs, first_usd, step_usd, step_hours, cost_how, train_s=None, **kw):
        g = [graded[r] for r in runs]
        row = {"system": name, "group": group, "reranked": kw.pop("reranked", False),
               "graded_ndcg10": round(np.mean(g), 4),
               "graded_ci_low": ci.loc[runs[0], "ci_low"] if len(runs) == 1 else None,
               "graded_ci_high": ci.loc[runs[0], "ci_high"] if len(runs) == 1 else None,
               "seed_min": round(min(g), 4) if len(runs) > 1 else None,
               "seed_max": round(max(g), 4) if len(runs) > 1 else None,
               "graded_ndcg10_other_judge": None, "other_judge_grades": None,
               "curator_ndcg10": round(np.mean([curator[r] for r in runs]), 4),
               "run_cost_usd": round(first_usd + step_usd, 3),
               "first_pass_cost_usd": round(first_usd, 3), "step_cost_usd": round(step_usd, 3),
               "l4_hours": round(first_usd / USD_PER_L4_HOUR + step_hours, 2),
               "train_cost_usd": None if train_s is None else round(usd(train_s), 3),
               "cost_basis": cost_how, "requested": kw.pop("requested", True), "run_ids": ";".join(runs)}
        row.update(kw)
        rows.append(row)

    first_usd = usd(embed["BAAI__bge-base-en-v1.5"])
    add("BM25, tuned (keyword search)", "base", [BM25], 0.0, 0.0, 0.0, "CPU, not priced ($0)")
    for name, run in EMBEDDERS:
        add(name, "single", [run], 0.0, usd(embed[run]), embed[run] / 3600, how[run])
    for arch, name in (("bge-base-gd", "bge-base, fine-tuned gene-disease (ours, 3 seeds)"),
                       ("bge-small-gd", "bge-small, fine-tuned gene-disease (ours, 3 seeds)")):
        runs = [f"bioembed__{arch}-s{s}" for s in SEEDS]
        add(name, "ours", runs, 0.0, usd(embed[arch]), embed[arch] / 3600,
            "logged test runs, mean of 3 seeds", np.mean([train[f"{arch}-s{s}"] for s in SEEDS]))
    for gaf, name in (("bge-base-gaf", "bge-base-GO (ours, GO fine-tune)"),
                      ("bge-small-gaf", "bge-small-GO (ours, GO fine-tune)")):
        run = f"bioembed__{gaf}"
        add(name, "ours", [run], 0.0, usd(embed[run]), embed[run] / 3600, how[run])
    add("First pass: BM25 + bge-base fusion", "rerank", [FIRST], first_usd, 0.0, 0.0,
        f"bge-base run ({how['BAAI__bge-base-en-v1.5']}); BM25 and fusion on CPU, $0")
    for name, group, model in CROSS:
        run = f"ce-{model}-top100-of-{FIRST}"
        add(name, group, [run], first_usd, usd(rerank[model]), rerank[model] / 3600,
            "first pass + logged rerank of top 100", train.get(model.removeprefix("bioembed__")),
            reranked=True)
    for name, llm, judge, first in LLMS:
        run = f"llm-{llm}-top20-of-{first}"
        spend = pd.read_parquet(LLM_SIDE / f"{run}.parquet", columns=["cost"]).cost.sum()
        fp = first_usd if first == FIRST else 0.0
        on_fp = first == FIRST
        add(name, "rerank" if on_fp else "bm25llm", [run], fp, spend, 0.0,
            f"{'first pass' if on_fp else 'BM25 (CPU, $0)'} + API spend (top 20, charged)",
            reranked=True, requested=on_fp,
            graded_ndcg10_other_judge=round(other[judge][run], 4),
            other_judge_grades=f"judgments_merged_no-{judge}")

    out = pd.DataFrame(rows)
    (ROOT / "results/writeup").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results/writeup/cost_quality.csv", index=False)
    print(out[["system", "graded_ndcg10", "graded_ndcg10_other_judge", "curator_ndcg10",
               "run_cost_usd", "train_cost_usd"]].to_string(index=False))
    print("test-job / practice-encode ratios:", {k: round(v, 2) for k, v in how["_ratio"].items()})
    print("hs-ft all training + dev_eval hours:",
          round(sum(j["seconds"] for j in jobs("hs-ft") if "haystack.py" not in j["cmd"]) / 3600, 2))
    plot(out)


INK, INK2, GRID = "#2b2b29", "#5c5c58", "#ebebe7"
COLOUR = {"base": "#8a8a85", "single": "#2a78d6", "ours": "#eb6834", "rerank": "#1baf7a",
          "bm25llm": "#4a3aa7"}  # dataviz categorical slots 1-3 + 7 (slot 4 fails the all-pairs floor)
LEGEND = {"base": "BM25 keyword search (baseline)", "single": "Off-the-shelf single retriever",
          "ours": "Our fine-tuned models", "rerank": "First pass, alone or + off-the-shelf reranker",
          "bm25llm": "BM25, then LLM rerank (no GPU)"}
FLOOR = 0.03  # where $0 (CPU-only) systems are drawn on the log axis
# label text, (dx, dy) offset in points; a leader line is drawn when the offset is long
LABELS = {
    "BM25, tuned (keyword search)": ("BM25, tuned", (14, -26)),
    "embeddinggemma-300m": ("embeddinggemma-300m\n(best off-the-shelf embedder)", (-20, 34)),
    "bge-base, fine-tuned gene-disease (ours, 3 seeds)": ("bge-base, fine-tuned (ours)", (-40, 14)),
    "BiCA-base": ("BiCA-base", (40, -14)),
    "First pass: BM25 + bge-base fusion": ("First pass: BM25 + bge-base fusion", (-30, -44)),
    "bge-reranker-base, fine-tuned (ours)": ("First pass + bge-reranker-base, fine-tuned (ours)", (-10, -60)),
    "Qwen3-Embedding-0.6B": ("Qwen3-Embedding-0.6B", (-4, -16)),
    "bge-base-GO (ours, GO fine-tune)": ("bge-base-GO (old GO fine-tune)", (-14, 12)),
    "bge-small-GO (ours, GO fine-tune)": ("bge-small-GO (old GO fine-tune)", (-14, 0)),
    "Gemini 3.1 Flash-Lite rerank": ("First pass +\nGemini rerank", (24, 10)),
    "MedCPT cross-encoder, fine-tuned (ours)": ("First pass + MedCPT cross-encoder, fine-tuned (ours)", (-30, 24)),
    "Qwen3-Reranker-4B": ("First pass +\nQwen3-Reranker-4B", (22, -36)),
    "BM25 + DeepSeek-V4-Flash rerank": ("BM25 + DeepSeek rerank\n(no GPU)", (-40, 34)),
    "BM25 + Gemini 3.1 Flash-Lite rerank": ("BM25 + Gemini rerank (no GPU)", (-24, -36)),
}


def plot(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    px = 72 / 100  # points per pixel at dpi 100
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(16, 10), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    x = df.run_cost_usd.clip(lower=FLOOR)
    for reranked, marker in ((True, "D"), (False, "o")):  # circles over diamonds (first pass vs MedCPT-CE)
        for g in ("base", "single", "bm25llm", "rerank", "ours"):  # ours on top
            m = (df.group == g) & (df.reranked == reranked)
            ax.scatter(x[m], df.graded_ndcg10[m], s=(13 * px) ** 2 if marker == "o" else (11 * px) ** 2,
                       marker=marker, color=COLOUR[g], edgecolor="white", linewidth=2 * px, zorder=3)

    # Pareto frontier: each point better than every cheaper one
    order = df.assign(x=x).sort_values(["x", "graded_ndcg10"], ascending=[True, False])
    front, best = [], -1.0
    for _, r in order.iterrows():
        if r.graded_ndcg10 > best:
            front.append((r.x, r.graded_ndcg10))
            best = r.graded_ndcg10
    fx, fy = zip(*front)
    ax.plot(fx, fy, color=INK2, linewidth=1.5 * px, linestyle=(0, (5, 4)), zorder=2)
    for _, r in df.iterrows():
        if r.system not in LABELS:
            continue
        text, (dx, dy) = LABELS[r.system]
        far = (dx * dx + dy * dy) ** 0.5 > 30
        ax.annotate(text, xy=(max(r.run_cost_usd, FLOOR), r.graded_ndcg10), xytext=(dx, dy),
                    textcoords="offset points", fontsize=14 * px, color=INK,
                    ha="left" if dx >= 0 else "right", va="center", linespacing=1.15,
                    arrowprops={"arrowstyle": "-", "color": INK2, "linewidth": 0.8, "shrinkA": 2, "shrinkB": 6}
                    if far else None, zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(FLOOR / 1.6, 18)
    ax.axvspan(FLOOR * 1.55, FLOOR * 1.75, color="#f6f6f3", linewidth=0, zorder=0)  # scale break after $0
    ax.text(FLOOR * 1.65, 0, "//\naxis break", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=12 * px, color=INK2, linespacing=1.3)
    ticks = [FLOOR, 0.1, 0.3, 1, 3, 10]
    ax.set_xticks(ticks, ["$0\n(CPU only)", "$0.10", "$0.30", "$1", "$3", "$10"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(0.52, 0.89)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.tick_params(labelsize=14 * px, length=0, pad=8)
    ax.grid(True, color=GRID, linewidth=1 * px, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Cost to run all 500 test queries, USD (log scale; one-off training cost not included)",
                  fontsize=15 * px, labelpad=10)
    ax.set_ylabel("Quality: graded nDCG@10 (higher is better)", fontsize=15 * px, labelpad=10)
    fig.suptitle("Quality vs cost: 500 gene–disease queries, each against ~1,100 candidate abstracts"
                 " (1.7M in all)",
                 x=0.06, ha="left", fontsize=18 * px, fontweight="bold", color=INK, y=0.965)
    fig.text(0.06, 0.895, "Exploratory, test set. Circles: no reranker. Diamonds: a reranker re-sorts"
             " the top results of a first search.\nPoints at the same cost can sit on top of each other;"
             " every value is in results/writeup/cost_quality.csv.",
             fontsize=14 * px, color=INK2)
    handles = [Line2D([], [], marker="s", linestyle="", markersize=12 * px, color=COLOUR[g],
                      label=LEGEND[g]) for g in COLOUR]
    handles += [Line2D([], [], marker="o", linestyle="", markersize=12 * px, color=INK2,
                       label="No reranker"),
                Line2D([], [], marker="D", linestyle="", markersize=10 * px, color=INK2,
                       label="Reranked: a first search, then a reranker"),
                Line2D([], [], color=INK2, linestyle=(0, (5, 4)), linewidth=1.5 * px,
                       label="Best value: nothing cheaper scores higher")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=14 * px,
              labelcolor=INK, handletextpad=0.6, borderaxespad=0.8)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.87, bottom=0.11)
    fig.savefig(ROOT / "docs/writeup_frontier.png", dpi=100, facecolor="white")


if __name__ == "__main__":
    main()
