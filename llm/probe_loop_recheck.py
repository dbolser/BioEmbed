"""Re-probe the memory-probe replies that looped, at temperature 1 (C4 sensitivity check).

closed_book_probe.py runs every model at temperature 0. Google advises the
default 1.0 for Gemini 3 models, as lower values can loop; on the v2 small
task 12 of 375 Gemini probe replies did ("M. M. M. ..." in an author field up
to max_tokens, or a provider "error"). Those are scored on the papers listed
before the loop. This sends the same request again with temperature 1.0 (one
sample, everything else unchanged) for each v2 probe reply that did not stop,
scores it with the probe's own rule, and reports how the memorised count would
move. The pre-registered label in results/memorisation_probe.parquet is not
changed.

Run (from llm/):  uv run --with duckdb --with rapidfuzz python probe_loop_recheck.py [--budget 0.2]
Output: results/llm_v2/probe_loop_recheck.parquet  task, qid, model, memorised_t0, n_listed_t0,
            memorised_t1, n_listed_t1, finish_reason_t1, tokens_out_t1, cost_t1, raw_path_t1
        llm/llm_results/closed_book_t1/<model-slug>/<qid>.<request-hash>.json  full API responses
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

import closed_book_probe as cb

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CACHE_DIR = HERE / "llm_results" / "closed_book_t1"
OUT = REPO_ROOT / "results" / "llm_v2" / "probe_loop_recheck.parquet"
TASKS = {f"GeneDiseaseRetrieval_v2_{b}": REPO_ROOT / "data" / "tasks" / f"GeneDiseaseRetrieval_v2_{b}"
         for b in range(5)}
PRACTICE = {"GeneDiseaseRetrieval_v2_0"}
SHORT = {"deepseek/deepseek-v4-flash": "deepseek", "google/gemini-3.1-flash-lite": "gemini"}
TEMPERATURE = 1.0
MEMORISED_MIN = 15  # PREREG: C4 dropped if the tested LLM remembers fewer test pairs


def finish_reason(raw_path: str) -> str | None:
    return json.loads((REPO_ROOT / raw_path).read_text())["response"]["choices"][0].get("finish_reason")


async def send(todo: list[tuple[Path, str, dict, dict]], budget: float) -> float:
    spent = 0.0
    for path, qid, env, request in todo:  # a dozen requests: sequential is fine
        if path.exists():
            continue
        if spent >= budget:
            raise SystemExit(f"budget ${budget} reached; rerun to resume from cache")
        client = AsyncOpenAI(api_key=env["token"], base_url=env["base_url"], max_retries=8)
        resp = (await client.chat.completions.create(**request)).model_dump()
        await client.close()
        spent += cb.response_cost(request["model"], resp.get("usage") or {})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"qid": qid, "request": request, "created": datetime.now(UTC).isoformat(),
                                    "response": resp}, indent=1))
    return spent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, default=0.2, help="max new $")
    args = ap.parse_args()

    probe = pd.read_parquet(cb.PROBE_OUT)
    probe = probe[probe.task.isin(TASKS)]
    cut = probe[probe.raw_path.map(finish_reason) != "stop"].reset_index(drop=True)
    print(f"{len(cut)} v2 probe replies did not stop: {cut.model.value_counts().to_dict()}")

    texts = {(t, q): x for t, path in TASKS.items() for q, x in cb.load_queries(path).itertuples(index=False)}
    qrels = pd.concat([cb.load_qrels(path).assign(task=t) for t, path in TASKS.items()])
    qrels = qrels.merge(cut[["task", "qid"]], on=["task", "qid"])
    qrels = qrels.merge(cb.gold_meta(set(qrels.pmid)), on="pmid", how="left")
    gold = {k: (g.pmid.tolist(), g.title.fillna("").tolist()) for k, g in qrels.groupby(["task", "qid"])}

    todo = []
    for r in cut.itertuples():
        env_file, extra_body = cb.MODELS[SHORT[r.model]]
        env = cb.read_env(env_file)
        request = {**cb.build_request(r.model, cb.build_prompt(texts[(r.task, r.qid)]), extra_body),
                   "temperature": TEMPERATURE}
        h = hashlib.sha1(json.dumps(request, sort_keys=True).encode()).hexdigest()[:10]
        todo.append((CACHE_DIR / cb.model_slug(r.model) / f"{r.qid.replace('/', '_')}.{h}.json", r.qid, env, request))
    spent = asyncio.run(send(todo, args.budget))

    rows = []
    for r, (path, *_) in zip(cut.itertuples(), todo):
        resp = json.loads(path.read_text())["response"]
        papers, _ = cb.parse_papers(resp["choices"][0]["message"]["content"])
        s = cb.score(papers, *gold[(r.task, r.qid)])
        u = resp.get("usage") or {}
        rows.append({"task": r.task, "qid": r.qid, "model": r.model, "memorised_t0": r.memorised,
                     "n_listed_t0": r.n_listed, "memorised_t1": s["memorised"], "n_listed_t1": s["n_listed"],
                     "finish_reason_t1": resp["choices"][0].get("finish_reason"),
                     "tokens_out_t1": u.get("completion_tokens", 0), "cost_t1": cb.response_cost(r.model, u),
                     "raw_path_t1": str(path.relative_to(REPO_ROOT))})
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(out.drop(columns="raw_path_t1").to_string(index=False))

    test = probe[~probe.task.isin(PRACTICE)]
    for model, g in out[~out.task.isin(PRACTICE)].groupby("model"):
        base = int(test[test.model == model].memorised.sum())
        gained = int((g.memorised_t1 & ~g.memorised_t0).sum())
        print(f"{model}: test memorised {base} at T=0; +{gained} of {len(g)} looped replies at T=1 -> "
              f"{base + gained} (C4 needs {MEMORISED_MIN}); T=1 replies still not stopped "
              f"{int((g.finish_reason_t1 != 'stop').sum())}")
    print(f"cost of the T=1 replies ${out.cost_t1.sum():.4f} (new spend ${spent:.4f}) -> {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
