"""Token/cost estimates for the ranked-list and rerank protocols.

Ranked-list: the input is identical to the existing corpus-in-context runs, so
input/cached tokens are taken from results/llm/<model>/**/<task>.json
usage_stats (actual, provider-reported); output is assumed ~10 IDs + the
"ID and TITLE" prelude (estimated from existing output plus ~120 tokens).

Rerank: 20 docs/query at len(chars)/4, calibrated by the ratio of actual
input tokens to the chars/4 proxy observed on the same corpus (per model),
plus prompt overhead and a JSON list of 20 ints (~70 output tokens).

Usage (from llm/):  uv run python estimate_cost.py
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_common import REPO_ROOT, RETRIEVAL_TASKS, load_first_stage, load_task_data

# $ per MTok: (input, cached input, output)
PRICES = {
    "deepseek/deepseek-v4-flash": (0.066, 0.013, 0.131),
    "google/gemini-3.1-flash-lite": (0.25, 0.025, 1.50),
}
TOP_K = 20
FIRST_STAGE = "Qwen/Qwen3-Embedding-0.6B"


def _existing_usage(model: str, task: str) -> dict | None:
    p = REPO_ROOT / "results" / "llm" / model.replace("/", "__") / "results"
    hits = list(p.glob(f"**/{task}.json"))
    if not hits:
        return None
    return json.loads(hits[0].read_text())["scores"]["test"][0]["usage_stats"]


def _cost(inp: float, cached: float, out: float, price: tuple) -> float:
    pi, pc, po = price
    return ((inp - cached) * pi + cached * pc + out * po) / 1e6


def main() -> None:
    rows = []
    for task in RETRIEVAL_TASKS:
        corpus, queries, qrels = load_task_data(task)
        nq = len(queries)
        corpus_chars = sum(len(d["text"]) for d in corpus.values())
        fs = load_first_stage(FIRST_STAGE, task, TOP_K)
        if fs:
            cand_chars = sum(sum(len(corpus[d]["text"]) for d in ids[:TOP_K])
                             for ids in fs["top_ranked"].values()) / nq
        else:  # no first-stage file yet: use the corpus mean
            cand_chars = TOP_K * corpus_chars / len(corpus)
        q_chars = sum(len(t) for t in queries.values()) / nq

        for model, price in PRICES.items():
            u = _existing_usage(model, task)
            # tokens-per-char calibration on this corpus for this model's tokenizer
            proxy_in = nq * (corpus_chars / 4)
            calib = (u["input_tokens"] / proxy_in) if u else 1.0

            # --- ranked list: same input as now, ~10 IDs (+ "ID and TITLE" prelude) out
            r_in = u["input_tokens"] if u else proxy_in
            r_cached = u["cached_tokens"] if u else 0.85 * r_in
            r_out = nq * ((u["output_tokens"] / nq if u else 100) + 120)
            r_think = u["thinking_tokens"] if u else 0
            rows.append(dict(task=task, model=model, protocol="ranked_list",
                             input_tokens=int(r_in), cached_tokens=int(r_cached),
                             output_tokens=int(r_out + r_think),
                             cost=_cost(r_in, r_cached, r_out + r_think, price),
                             calib=calib, per_query_in=int(r_in / nq)))

            # --- rerank: 20 candidate docs + query + instructions per query, JSON out
            overhead_tok = 120  # system prompt + framing
            per_q_in = calib * (cand_chars + q_chars) / 4 + overhead_tok + TOP_K * 4  # "[i] " tags
            k_in = nq * per_q_in
            k_out = nq * 70
            rows.append(dict(task=task, model=model, protocol="rerank_top20",
                             input_tokens=int(k_in), cached_tokens=0, output_tokens=int(k_out),
                             cost=_cost(k_in, 0, k_out, price), calib=calib,
                             per_query_in=int(per_q_in)))

    print(f"{'task':22} {'model':30} {'protocol':13} {'in_tok':>11} {'cached':>11} {'out_tok':>8} {'tok/q':>8} {'$':>8}")
    for r in rows:
        print(f"{r['task']:22} {r['model']:30} {r['protocol']:13} {r['input_tokens']:>11,} "
              f"{r['cached_tokens']:>11,} {r['output_tokens']:>8,} {r['per_query_in']:>8,} {r['cost']:>8.4f}")
    out = Path(__file__).with_name("cost_estimates.json")
    out.write_text(json.dumps(rows, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
