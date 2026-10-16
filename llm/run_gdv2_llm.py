"""Small task v2 (GeneDiseaseRetrieval_v2_0..4), corpus in context, for PREREG C3/C4.

Same protocol as run_bio_llm.py: the tasks_bio v2 task loads a batch and the
vendor's LLMRetrievalEvaluator sends each query with all 300 docs (LOFT format,
sorted ids, cached-prefix layout of send_request_multi) and parses its
"Final Answer: [...]". hit@1 = first parsed id is the gold (= the vendor's
recall@1, one gold per query). Nothing in vendor/ is changed; two wrappers:

- the vendor's module-level AsyncOpenAI client is proxied: each request gets
  the model's extra_body (reasoning off; DeepSeek pinned to Baidu as in
  closed_book_probe.py), and every full API response (text, usage, cost,
  provider) is cached per (qid, request hash), so reruns are free and an
  aborted batch resumes;
- _score_query keeps every query's raw text, parsed ranking and usage (the
  vendor writes only 5 samples).

Batch 0 is the practice batch, 1-4 are test (PREREG change log 22:45).
"memorised" joins results/memorisation_probe.parquet (each model's own label;
run closed_book_probe.py on the v2 tasks first). The vendor protocol sends no
temperature, so the provider default (1.0 on OpenRouter) applies and each
hit@1 is one sample.

The vendor prompt asks for every document that helps, not a ranking, so
hit@1 depends on the order of a multi-id answer, and models often list ids
ascending (= prompt order). Order-free scores sit next to hit@1: hit_any
(gold anywhere in the answer) and hit1_tiebreak (expected hit@1 if the
answer is an unordered tie broken at random: 1/n when the gold is among n
ids; equals hit@1 for a full ranking, so it is the like-for-like score
against embedders).

Run (from llm/):  uv run python run_gdv2_llm.py --model deepseek [--batches 0 1 2 3 4] [--limit N]
                  uv run python run_gdv2_llm.py --summary-only
Output: results/llm_v2/<model-slug>/<batch>.parquet  qid, gold, ranked_ids, hit1, tokens_in,
            cached_tokens, tokens_out, reasoning_tokens, cost, finish_reason, provider, raw, raw_path
        results/llm_v2/summary.csv  model x batch rows, plus 'test' (1-4) and 'all'; hit1,
            hit1_tiebreak, hit_any, n_multi (answers with >1 id), n_multi_ascending (of those,
            ids in ascending order); hit1 split by the model's own memorised label;
            n_probe_cut = probe replies cut off or errored (finish_reason != stop), scored on
            the papers listed before the cut
        results/llm_v2/pairs_test.csv  test batches, each LLM vs the other and vs the PREREG
            C3a/C3b embedders (results/v2_perquery.parquet): hit1 and hit_any by exact McNemar,
            hit1_tiebreak by paired sign-flip (10,000); uncorrected, context for C3, not its test
        llm/llm_results/llm_v2/<model-slug>/<qid>.<hash>.json  full API responses
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
OUT_DIR = REPO_ROOT / "results" / "llm_v2"
CACHE_DIR = HERE / "llm_results" / "llm_v2"
PROBE = REPO_ROOT / "results" / "memorisation_probe.parquet"
EMB_PERQUERY = REPO_ROOT / "results" / "v2_perquery.parquet"
EMBEDDERS = {"C3a": "google__embeddinggemma-300m", "C3b": "BMRetriever__BMRetriever-410M"}  # PREREG 2026-10-16 tie
N_PERM = 10_000
SEED = 0
BATCHES = range(5)
PRACTICE = {0}
# short name -> (env file, extra_body added to every request)
MODELS = {
    "deepseek": (".env.or-deepseek_deepseek-v4-flash",
                 {"reasoning": {"enabled": False},
                  "provider": {"order": ["baidu"], "allow_fallbacks": False}}),
    "gemini": (".env.or-google_gemini-3.1-flash-lite", {"reasoning": {"enabled": False}}),
}
CONCURRENCY = 8

CURRENT_QID: contextvars.ContextVar[str] = contextvars.ContextVar("qid")


def read_env(name: str) -> dict:
    env = {}
    for line in (HERE / name).read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#"):
            env[key.strip()] = value.strip()
    return env


def task_name(batch: int) -> str:
    return f"GeneDiseaseRetrieval_v2_{batch}"


class CachingClient:
    """Stands in for llm_judge.llm_client.client (chat.completions.create only)."""

    def __init__(self, inner, extra_body: dict, cache_dir: Path, budget: float):
        self.inner, self.extra_body, self.cache_dir, self.budget = inner, extra_body, cache_dir, budget
        self.api_key = inner.api_key  # the vendor reassigns this before each call
        self.spent = 0.0
        self.paths: dict[str, Path] = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        from openai.types.chat import ChatCompletion

        qid = CURRENT_QID.get()
        kw = {**kw, "extra_body": {**kw.get("extra_body", {}), **self.extra_body}}
        h = hashlib.sha1(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()[:10]
        path = self.cache_dir / f"{qid.replace('/', '_')}.{h}.json"
        if path.exists():
            raw = json.loads(path.read_text())["response"]
        else:
            if self.spent >= self.budget:
                raise RuntimeError(f"budget ${self.budget} spent; rerun to resume from cache")
            resp = await self.inner.chat.completions.create(**kw)
            raw = resp.model_dump()
            self.spent += (raw.get("usage") or {}).get("cost") or 0.0
            request = {k: v for k, v in kw.items() if k != "messages"}
            request["query_message"] = kw["messages"][-1]["content"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"qid": qid, "request": request,
                                        "created": datetime.now(UTC).isoformat(),
                                        "response": raw}, indent=1))
        self.paths[qid] = path
        return ChatCompletion.model_validate(raw)


def make_evaluator_class():
    from llm_judge.evaluators.llm_retrieval_evaluator import LLMRetrievalEvaluator

    class RecordingEvaluator(LLMRetrievalEvaluator):
        """Vendor evaluator; _score_query also tags the request with its qid and keeps the result."""

        def __init__(self, *a, task: str, **kw):
            super().__init__(*a, **kw)
            self.metadata = SimpleNamespace(name=task)  # for the vendor's sample dump
            self.records: dict[str, tuple[list[str], str]] = {}

        async def _score_query(self, query_id, query_text, candidate_ids):
            CURRENT_QID.set(query_id)
            retrieved, usage, raw = await super()._score_query(query_id, query_text, candidate_ids)
            self.records[query_id] = (retrieved, raw)
            return retrieved, usage, raw

    return RecordingEvaluator


def usage_row(path: Path) -> dict:
    resp = json.loads(path.read_text())["response"]
    u = resp.get("usage") or {}
    return {"tokens_in": u.get("prompt_tokens", 0),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
            "tokens_out": u.get("completion_tokens", 0),
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
            "cost": u.get("cost") or 0.0,
            "finish_reason": resp["choices"][0].get("finish_reason"),
            "provider": resp.get("provider")}


def run(short: str, batches: list[int], limit: int | None, budget: float) -> None:
    env_file, extra_body = MODELS[short]
    env = read_env(env_file)
    # Settings() is built when llm_judge is imported: environment beats llm/.env (the Vertex config).
    os.environ.update({k: env[k] for k in ("token", "base_url", "model", "use_strict_json")})
    os.environ.update({"use_vertex_ai": "false", "enable_thinking": "false",
                       "max_concurrency": str(CONCURRENCY)})
    for p in (str(REPO_ROOT / "vendor" / "embedders-dilemma"), str(HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(HERE)  # the vendor writes its 5 debug samples under ./llm_results/
    import llm_judge.llm_client as lc

    from tasks_bio import GENE_DISEASE_V2

    slug = env["model"].replace("/", "__")
    proxy = CachingClient(lc.client, extra_body, CACHE_DIR / slug, budget)
    lc.client = proxy
    Evaluator = make_evaluator_class()
    for b in batches:
        task = GENE_DISEASE_V2[b]()
        task.load_data()
        queries = dict(list(task._queries.items())[:limit]) if limit else task._queries
        # one query alone first, so the provider's prefix cache is warm before the concurrent burst
        warm = dict(list(queries.items())[:1])
        asyncio.run(Evaluator(corpus=task._corpus, queries=warm, qrels=task._qrels,
                              task=task_name(b)).evaluate_async())
        ev = Evaluator(corpus=task._corpus, queries=queries, qrels=task._qrels, task=task_name(b))
        scores = asyncio.run(ev.evaluate_async())
        rows = []
        for qid in queries:
            ranked, raw = ev.records[qid]
            gold = next(d for d, s in task._qrels[qid].items() if s > 0)
            path = proxy.paths[qid]
            rows.append({"task": task_name(b), "batch": b, "qid": qid, "gold": gold, "ranked_ids": ranked,
                         "hit1": bool(ranked) and ranked[0] == gold, **usage_row(path), "raw": raw,
                         "raw_path": str(path.relative_to(REPO_ROOT))})
        df = pd.DataFrame(rows)
        u = scores["usage_stats"]
        print(f"{env['model']} {task_name(b)}: hit@1={df.hit1.mean():.3f} (vendor recall@1 "
              f"{scores['recall@1']:.3f}) n={len(df)} empty={int((df.ranked_ids.map(len) == 0).sum())} "
              f"in={u['input_tokens']:,} cached={u['cached_tokens']:,} out={u['output_tokens']:,} "
              f"think={u['thinking_tokens']:,} cost=${df.cost.sum():.4f} (new spend ${proxy.spent:.4f})",
              flush=True)
        if limit:
            continue
        out = OUT_DIR / slug / f"{b}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)


def probe_cut(raw_path: str) -> bool:
    """True if the probe reply did not finish ("length" loop or "error"); scored on what it listed."""
    resp = json.loads((REPO_ROOT / raw_path).read_text())["response"]
    return resp["choices"][0].get("finish_reason") != "stop"


def summarise() -> pd.DataFrame:
    files = sorted(OUT_DIR.glob("*__*/[0-9].parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f).assign(model_slug=f.parent.name) for f in files], ignore_index=True)
    df["model"] = df.model_slug.str.replace("__", "/", n=1)
    df["n_listed"] = df.ranked_ids.map(len)
    df["hit_any"] = [g in set(r) for g, r in zip(df.gold, df.ranked_ids)]
    df["hit1_tiebreak"] = df.hit_any / df.n_listed.clip(lower=1)
    df["ascending"] = [len(r) > 1 and list(r) == sorted(r) for r in df.ranked_ids]  # vendor ids follow sorted ids
    if PROBE.exists():
        probe = pd.read_parquet(PROBE)[["task", "qid", "model", "memorised", "raw_path"]]
        probe = probe[probe.task.isin(set(df.task))]
        probe["probe_cut"] = probe.raw_path.map(probe_cut)
        df = df.merge(probe.drop(columns="raw_path"), on=["task", "qid", "model"], how="left")
    else:
        df["memorised"] = df["probe_cut"] = pd.NA
    groups = [(str(b), g) for b, g in df.groupby("batch")]
    groups += [("test", df[~df.batch.isin(PRACTICE)]), ("all", df)]
    rows = []
    for name, g in groups:
        split = "all" if name == "all" else "practice" if name in {str(b) for b in PRACTICE} else "test"
        for model, m in g.groupby("model"):
            mem = m[m.memorised.notna()]
            rows.append({
                "model": model, "batch": name, "split": split, "n": len(m), "hit1": m.hit1.mean(),
                "hit1_tiebreak": m.hit1_tiebreak.mean(), "hit_any": m.hit_any.mean(),
                "n_multi": int((m.n_listed > 1).sum()), "n_multi_ascending": int(m.ascending.sum()),
                "n_empty": int((m.n_listed == 0).sum()),
                "n_probed": len(mem), "n_memorised": int(mem.memorised.astype(bool).sum()),
                "n_probe_cut": int(mem.probe_cut.astype(bool).sum()),
                "hit1_memorised": mem[mem.memorised.astype(bool)].hit1.mean(),
                "hit1_not_memorised": mem[~mem.memorised.astype(bool)].hit1.mean(),
                "tokens_in": int(m.tokens_in.sum()), "cached_tokens": int(m.cached_tokens.sum()),
                "tokens_out": int(m.tokens_out.sum()), "reasoning_tokens": int(m.reasoning_tokens.sum()),
                "cost": m.cost.sum(), "cost_per_query": m.cost.mean(),
                "not_stopped": int((m.finish_reason != "stop").sum()),
            })
    summ = pd.DataFrame(rows)
    summ.round(5).to_csv(OUT_DIR / "summary.csv", index=False)
    compare(df[~df.batch.isin(PRACTICE)])
    return summ


def mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided McNemar p on paired booleans."""
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(n10, n01) + 1)) / 2**n) if n else 1.0


def sign_flip(d: np.ndarray) -> float:
    """Two-sided paired sign-flip permutation p, as scripts/stats_haystack.py."""
    signs = np.random.default_rng(SEED).choice([-1, 1], size=(N_PERM, len(d)))
    return (1 + int((np.abs(signs @ d) >= abs(d.sum()) - 1e-9).sum())) / (1 + N_PERM)


def compare(test: pd.DataFrame) -> pd.DataFrame:
    """Per-query paired tests on the test batches: LLM vs LLM and LLM vs the C3a/C3b embedders."""
    wide = {m: g.set_index("qid")[["hit1", "hit1_tiebreak", "hit_any"]] for m, g in test.groupby("model")}
    llms = list(wide)
    if EMB_PERQUERY.exists():
        emb = pd.read_parquet(EMB_PERQUERY)
        for claim, name in EMBEDDERS.items():
            e = emb[(emb.model == name) & ~emb.batch.isin(PRACTICE)].set_index("qid")
            wide[f"{claim} {name}"] = e[["hit1"]].assign(hit1_tiebreak=e.hit1.astype(float))
    pairs = [(a, b) for i, a in enumerate(llms) for b in llms[i + 1:]] + \
            [(a, b) for a in llms for b in wide if b not in llms]
    rows = []
    for a, b in pairs:
        m = wide[a].join(wide[b], how="inner", lsuffix="_a", rsuffix="_b")
        for metric in ["hit1", "hit1_tiebreak", "hit_any"]:
            if f"{metric}_b" not in m:
                continue
            x, y = m[f"{metric}_a"].to_numpy(), m[f"{metric}_b"].to_numpy()
            binary = metric != "hit1_tiebreak"
            rows.append({"system_a": a, "system_b": b, "metric": metric, "n": len(m),
                         "mean_a": x.mean(), "mean_b": y.mean(), "diff": x.mean() - y.mean(),
                         "a_only": int((x & ~y).sum()) if binary else pd.NA,
                         "b_only": int((~x & y).sum()) if binary else pd.NA,
                         "test": "exact McNemar" if binary else "sign-flip",
                         "p_value": mcnemar(x, y) if binary else sign_flip(x.astype(float) - y)})
    out = pd.DataFrame(rows).astype({"a_only": "Int64", "b_only": "Int64"})
    out.round(5).to_csv(OUT_DIR / "pairs_test.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--batches", nargs="+", type=int, default=list(BATCHES))
    ap.add_argument("--limit", type=int, help="first N queries per batch; paid calls cached, nothing written")
    ap.add_argument("--budget", type=float, default=4.0, help="max new $ this invocation")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    if not args.summary_only:
        if not args.model:
            ap.error("--model is required unless --summary-only")
        run(args.model, args.batches, args.limit, args.budget)
    if args.limit:
        return
    print(summarise().round(3).to_string(index=False))
    print(pd.read_csv(OUT_DIR / "pairs_test.csv").to_string(index=False))


if __name__ == "__main__":
    main()
