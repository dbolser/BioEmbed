"""Closed-book memorisation probe for the gene–disease tasks (PREREG claim C4).

No documents are given. Each model is asked from memory for up to 3 PubMed
papers that are the key evidence for a query's gene–disease link, as JSON
(title, first author, year, PMID). A gold paper is recalled if its PMID is
listed (pmid_hit) or a listed title matches its title with rapidfuzz
token_sort_ratio >= 90 (title_hit). A query is "memorised" if any of its gold
papers is recalled. Gold titles and publication years come from the IUK-IPF
records table.

The title rule is calibrated against query-only template titles ("SYMBOL
mutations in DISEASE", ...), which know nothing about the literature: the same
rule scored on them gives template_hit, the rule's false-hit floor, reported
next to the model rates. token_set_ratio >= 90 (the first rule tried) gave the
templates a hit on 23% of haystack queries, as often as the models; this rule
gives them 1%.

On multi-gold tasks (the haystack) "memorised" rises with gold count; stratify
by n_gold, or use n_gold_hit / n_gold (per gold paper).

Models (OpenRouter, ids from llm/.env.or-*), temperature 0: DeepSeek-V4-Flash
with reasoning off, pinned to one provider; Gemini 3.1 Flash-Lite at its
default reasoning. Raw responses are cached per request in
llm/llm_results/closed_book/<model-slug>/<qid>.<request-hash>.json, so reruns
are free, identical requests are shared across tasks, and any change to the
prompt or settings makes a new request.

Usage (from llm/):
    uv run --with duckdb --with rapidfuzz python closed_book_probe.py           # haystack test + v1
    uv run --with duckdb --with rapidfuzz python closed_book_probe.py --limit 3 # smoke: paid calls, cached, no results written
    uv run --with duckdb --with rapidfuzz python closed_book_probe.py \\
        --task GeneDiseaseRetrieval_v2_0 --queries ../data/tasks/GeneDiseaseRetrieval_v2_0

--queries/--qrels take a parquet file or a datasets save_to_disk dir (its
"queries"/"qrels" split); columns qid|_id|query-id, text, pmid|corpus-id
[, score > 0 kept]. --qrels defaults to --queries.

Output (upserted by task and model; not written unless every query has a response):
    results/memorisation_probe.parquet  task, qid, model, pmid_hit, title_hit, memorised, n_listed,
                                        n_gold, n_gold_hit, template_hit, raw_path
    results/gold_years.parquet          task, qid, pmid, publication_year
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

import duckdb
import pandas as pd
from openai import AsyncOpenAI
from rapidfuzz import fuzz, utils

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CACHE_DIR = HERE / "llm_results" / "closed_book"
GOLD_META = CACHE_DIR / "gold_meta.parquet"
PROBE_OUT = REPO_ROOT / "results" / "memorisation_probe.parquet"
YEARS_OUT = REPO_ROOT / "results" / "gold_years.parquet"
RECORDS = "/outsee5/home/danbolser/Work/IUK-IPF/data/embeddings-catalogue/records/*.parquet"

TASKS = {
    "GeneDiseaseHaystack": (REPO_ROOT / "data/haystack/GeneDiseaseHaystack/queries.parquet",
                            REPO_ROOT / "data/haystack/GeneDiseaseHaystack/qrels.parquet"),
    "GeneDiseaseRetrieval": (REPO_ROOT / "data/tasks/GeneDiseaseRetrieval",
                             REPO_ROOT / "data/tasks/GeneDiseaseRetrieval"),
}
# short name -> (env file, extra_body)
MODELS = {
    "deepseek": (".env.or-deepseek_deepseek-v4-flash",
                 {"reasoning": {"enabled": False},
                  "provider": {"order": ["baidu"], "allow_fallbacks": False}}),
    "gemini": (".env.or-google_gemini-3.1-flash-lite", {}),
}
# $ per MTok (input, output); fallback only when OpenRouter reports no usage.cost
PRICES = {
    "deepseek/deepseek-v4-flash": (0.066, 0.131),
    "google/gemini-3.1-flash-lite": (0.25, 1.50),
}
MAX_TOKENS = 2048
TEMPERATURE = 0.0
TITLE_THRESHOLD = 90  # rapidfuzz token_sort_ratio
TEMPLATES = [
    "{symbol} mutations in {disease}",
    "Mutations in {symbol} cause {disease}",
    "Mutations in the {symbol} gene cause {disease}",
    "{disease} is caused by mutations in {symbol}",
    "Mutations in {name} ({symbol}) cause {disease}",
    "A novel {symbol} mutation in a family with {disease}",
]
GOLD_BANDS = {"bins": [0, 1, 2, 4, 8, float("inf")], "labels": ["1", "2", "3-4", "5-8", ">8"]}
SYSTEM = "You are an expert in human genetics and the biomedical literature. Answer from memory. Reply with JSON only."
PROMPT = (
    "List up to 3 PubMed papers that are the key evidence that {gene} is involved in {disease}. "
    "For each give title, first author, year and PMID.\n\n"
    'Answer as JSON: {{"papers": [{{"title": "...", "first_author": "...", "year": 2000, "pmid": "..."}}]}}'
)
QUERY_RE = re.compile(r"^(?P<gene>(?P<symbol>\S+) \((?P<name>.*?)\)) in (?P<disease>.+)$")


def read_env(name: str) -> dict:
    env = {}
    for line in (HERE / name).read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#"):
            env[key.strip()] = value.strip()
    return env


def model_slug(model: str) -> str:
    return model.replace("/", "__")


def _split(path: Path, split: str) -> pd.DataFrame:
    if path.is_dir():
        from datasets import load_from_disk

        return load_from_disk(str(path))[split].to_pandas()
    return pd.read_parquet(path)


def load_queries(path: Path) -> pd.DataFrame:
    df = _split(path, "queries").rename(columns={"_id": "qid", "query-id": "qid"})
    return df[["qid", "text"]].astype(str)


def load_qrels(path: Path) -> pd.DataFrame:
    df = _split(path, "qrels").rename(columns={"query-id": "qid", "corpus-id": "pmid"})
    if "score" in df:
        df = df[df["score"] > 0]
    return df[["qid", "pmid"]].astype(str).drop_duplicates()


def gold_meta(pmids: set[str]) -> pd.DataFrame:
    """pmid, title, publication_year for gold PMIDs; records scanned only for PMIDs not yet cached."""
    cached = pd.read_parquet(GOLD_META) if GOLD_META.exists() else pd.DataFrame(
        {"pmid": pd.Series(dtype=str), "title": pd.Series(dtype=str),
         "publication_year": pd.Series(dtype="Int64")})
    missing = pd.DataFrame({"pmid": sorted(pmids - set(cached["pmid"]))})
    if len(missing):
        print(f"records lookup: {len(missing)} PMIDs")
        con = duckdb.connect()
        con.register("need", missing)
        new = con.sql(f"""
            SELECT pmid, title, publication_year FROM (
                SELECT replace(record_id, 'PMID:', '') AS pmid,
                       coalesce(trim(title), '') AS title,
                       TRY_CAST(json_extract_string(metadata, '$.publication_year') AS INTEGER)
                           AS publication_year,
                       row_number() OVER (PARTITION BY record_id ORDER BY record_hash) AS rn
                FROM read_parquet('{RECORDS}')
                WHERE record_type = 'pubmed_abstract'
                  AND record_id IN (SELECT 'PMID:' || pmid FROM need)
            ) WHERE rn = 1
        """).df()
        new["publication_year"] = new["publication_year"].astype("Int64")
        cached = pd.concat([cached, new], ignore_index=True)
        GOLD_META.parent.mkdir(parents=True, exist_ok=True)
        cached.sort_values("pmid").to_parquet(GOLD_META, index=False)
    return cached[cached["pmid"].isin(pmids)]


def build_prompt(text: str) -> str:
    m = QUERY_RE.match(text)
    gene, disease = (m["gene"], m["disease"]) if m else (text, "the disease named")
    return PROMPT.format(gene=gene, disease=disease)


def template_titles(text: str) -> list[str]:
    m = QUERY_RE.match(text)
    return [t.format(**m.groupdict()) for t in TEMPLATES] if m else []


def build_request(model: str, prompt: str, extra_body: dict) -> dict:
    return {"model": model, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "extra_body": extra_body,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}


def cache_path(qid: str, request: dict) -> Path:
    h = hashlib.sha1(json.dumps(request, sort_keys=True).encode()).hexdigest()[:10]
    return CACHE_DIR / model_slug(request["model"]) / f"{qid.replace('/', '_')}.{h}.json"


def response_cost(model: str, usage: dict) -> float:
    if usage.get("cost") is not None:
        return float(usage["cost"])
    pi, po = PRICES.get(model, (0.0, 0.0))
    return (usage.get("prompt_tokens", 0) * pi + usage.get("completion_tokens", 0) * po) / 1e6


async def run_model(short: str, prompts: dict[tuple, str], concurrency: int, budget: float) -> dict:
    """Send every uncached request; return {(task, qid): cache path}. Stops launching once new spend > budget."""
    env_file, extra_body = MODELS[short]
    env = read_env(env_file)
    model = env["model"]
    requests = {key: build_request(model, p, extra_body) for key, p in prompts.items()}
    paths = {key: cache_path(key[1], r) for key, r in requests.items()}
    todo = {paths[k]: (k[1], requests[k]) for k in requests if not paths[k].exists()}
    print(f"{model}: {len(set(paths.values())) - len(todo)} cached, {len(todo)} to query")
    client = AsyncOpenAI(api_key=env["token"], base_url=env["base_url"], max_retries=8)
    sem = asyncio.Semaphore(concurrency)
    spent, failed, skipped = 0.0, [], 0

    async def one(path: Path, qid: str, request: dict) -> None:
        nonlocal spent, skipped
        async with sem:
            if spent > budget:
                skipped += 1
                return
            try:
                resp = await client.chat.completions.create(**request)
            except Exception as e:  # noqa: BLE001 - left uncached, retried on rerun
                failed.append((qid, repr(e)[:200]))
                return
        raw = resp.model_dump()
        spent += response_cost(model, raw.get("usage") or {})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"qid": qid, "request": request,
                                    "created": datetime.now(timezone.utc).isoformat(),
                                    "response": raw}, indent=1))

    await asyncio.gather(*(one(p, *v) for p, v in todo.items()))
    await client.close()
    for qid, err in failed[:5]:
        print(f"  FAILED {qid}: {err}")
    if failed or skipped:
        print(f"  {len(failed)} failed, {skipped} skipped over budget; new spend ${spent:.4f} (budget left ${budget:.4f})")
    return {"model": model, "paths": paths, "new_spend": spent}


def parse_papers(content: str) -> tuple[list[dict], bool]:
    """(papers, valid JSON). Truncated or broken JSON (e.g. a looping author
    field hitting max_tokens) is salvaged field by field: every complete
    "title" and "pmid" string, paired in order."""
    s = re.sub(r"^```\w*\s*|\s*```$", "", (content or "").strip())
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        obj = obj.get("papers")
    if isinstance(obj, list):
        return [p for p in obj if isinstance(p, dict)], True
    titles = re.findall(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"', s)
    pmids = re.findall(r'"pmid"\s*:\s*"?(\d+)', s)
    return [{"title": t, "pmid": m} for t, m in zip_longest(titles, pmids)], False


def norm_pmid(x) -> str:
    return re.sub(r"\D", "", str(x or ""))


def title_match(listed: str, gold: str) -> bool:
    return fuzz.token_sort_ratio(listed, gold, processor=utils.default_process) >= TITLE_THRESHOLD


def score(papers: list[dict], gold_pmids: list[str], gold_titles: list[str]) -> dict:
    """Per gold paper: PMID listed, or a listed title matches its title."""
    pmids = {norm_pmid(p.get("pmid")) for p in papers} - {""}
    titles = [t for t in (str(p.get("title") or "").strip() for p in papers) if t]
    by_pmid = [g in pmids for g in gold_pmids]
    by_title = [bool(g) and any(title_match(t, g) for t in titles) for g in gold_titles]
    n_hit = sum(a or b for a, b in zip(by_pmid, by_title))
    return {"pmid_hit": any(by_pmid), "title_hit": any(by_title), "memorised": n_hit > 0,
            "n_listed": len(papers), "n_gold": len(gold_pmids), "n_gold_hit": n_hit}


def upsert(path: Path, new: pd.DataFrame, keys: list[str]) -> None:
    if path.exists():
        old = pd.read_parquet(path)
        old = old[~old.set_index(keys).index.isin(new[keys].drop_duplicates().set_index(keys).index)]
        if len(old):
            new = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values(list(new.columns[:3])).to_parquet(path, index=False)


def summarise(probe: pd.DataFrame, n_models: int) -> None:
    g = probe.groupby(["task", "model"])
    summ = g.agg(n=("qid", "size"), pmid_hit=("pmid_hit", "mean"), title_hit=("title_hit", "mean"),
                 memorised=("memorised", "mean"), template_hit=("template_hit", "mean"),
                 n_listed=("n_listed", "mean"), salvaged=("parsed", lambda s: int((~s).sum())))
    summ.insert(4, "per_gold", g["n_gold_hit"].sum() / g["n_gold"].sum())
    print("\nHit rates (template_hit = same rule on query-only template titles, the false-hit floor):")
    print(summ.round(3).to_string())

    multi = probe[probe.groupby("task")["n_gold"].transform("max") > 1]
    if len(multi):
        band = pd.cut(multi["n_gold"], **GOLD_BANDS).rename("n_gold")
        tab = multi.pivot_table(index=["task", band], columns="model", values="memorised",
                                aggfunc="mean", observed=True)
        tab["template"] = multi.groupby(["task", band], observed=True)["template_hit"].mean()
        tab["n"] = multi.groupby(["task", band], observed=True)["qid"].nunique()
        print("\nMemorised by gold-count band (multi-gold tasks):")
        print(tab.round(3).to_string())

    both = probe.groupby(["task", "qid"])["memorised"].agg(["sum", "size"])
    both = both[both["size"] == n_models]
    print("\nQueries memorised by all models run:")
    print(both.groupby("task")["sum"].agg(
        all_models=lambda s: int((s == n_models).sum()),
        any_model=lambda s: int((s > 0).sum()), n="size").to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", help="task name for --queries/--qrels (default: haystack test + v1)")
    ap.add_argument("--queries", type=Path)
    ap.add_argument("--qrels", type=Path)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--budget", type=float, default=4.0, help="max new $ per run, all models together")
    ap.add_argument("--limit", type=int, help="first N queries per task; paid calls are cached, no results written")
    args = ap.parse_args()

    if args.queries:
        if not args.task:
            ap.error("--queries needs --task")
        tasks = {args.task: (args.queries, args.qrels or args.queries)}
    else:
        tasks = TASKS

    queries, qrels = [], []
    for task, (qpath, rpath) in tasks.items():
        q = load_queries(qpath)
        if args.limit:
            q = q.head(args.limit)
        r = load_qrels(rpath)
        queries.append(q.assign(task=task))
        qrels.append(r[r["qid"].isin(q["qid"])].assign(task=task))
    queries = pd.concat(queries, ignore_index=True)
    qrels = pd.concat(qrels, ignore_index=True)
    meta = gold_meta(set(qrels["pmid"]))
    qrels = qrels.merge(meta, on="pmid", how="left")
    print(f"{len(queries)} queries, {len(qrels)} gold rows, {qrels['title'].isna().sum()} golds not in records")

    keys = list(zip(queries["task"], queries["qid"]))
    prompts = dict(zip(keys, queries["text"].map(build_prompt)))
    templates = dict(zip(keys, queries["text"].map(template_titles)))
    unparsed = queries.loc[~queries["text"].str.match(QUERY_RE), "qid"].tolist()
    if unparsed:
        print(f"{len(unparsed)} query texts not in 'SYMBOL (name) in disease' form, used whole: {unparsed[:3]}")
    gold = {k: (g["pmid"].tolist(), g["title"].fillna("").tolist()) for k, g in qrels.groupby(["task", "qid"])}
    template_hit = {k: any(title_match(t, g) for t in templates[k] for g in gold.get(k, ([], []))[1] if g)
                    for k in keys}

    rows, costs, missing = [], [], 0
    budget = args.budget
    for short in args.models:
        res = asyncio.run(run_model(short, prompts, args.concurrency, budget))
        budget -= res["new_spend"]
        model = res["model"]
        tok = {"in": 0, "out": 0, "reasoning": 0, "cost": 0.0, "n": 0, "truncated": 0}
        providers, seen = Counter(), set()
        for key in keys:
            p = res["paths"][key]
            if not p.exists():
                missing += 1
                continue
            resp = json.loads(p.read_text())["response"]
            if p not in seen:
                seen.add(p)
                usage = resp.get("usage") or {}
                tok["in"] += usage.get("prompt_tokens", 0)
                tok["out"] += usage.get("completion_tokens", 0)
                tok["reasoning"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                tok["cost"] += response_cost(model, usage)
                tok["n"] += 1
                tok["truncated"] += resp["choices"][0].get("finish_reason") == "length"
                providers[resp.get("provider")] += 1
            papers, parsed = parse_papers(resp["choices"][0]["message"]["content"])
            rows.append({"task": key[0], "qid": key[1], "model": model,
                         **score(papers, *gold.get(key, ([], []))), "template_hit": template_hit[key],
                         "parsed": parsed, "raw_path": str(p.relative_to(REPO_ROOT))})
        costs.append({"model": model, **tok, "new_spend": res["new_spend"], "providers": dict(providers)})

    probe = pd.DataFrame(rows)
    print("\nCost (unique responses used, incl. cached; truncated = hit max_tokens):")
    print(pd.DataFrame(costs).to_string(index=False))
    if len(probe):
        summarise(probe, len(args.models))

    if missing:
        print(f"\n{missing} query x model responses missing (failed or over budget); results not written. Rerun.")
        sys.exit(1)
    if args.limit:
        print("\n--limit: smoke run, results not written")
        return
    cols = ["task", "qid", "model", "pmid_hit", "title_hit", "memorised", "n_listed",
            "n_gold", "n_gold_hit", "template_hit", "raw_path"]
    upsert(PROBE_OUT, probe[cols], ["task", "model"])
    years = qrels[["task", "qid", "pmid", "publication_year"]].astype({"publication_year": "Int64"})
    upsert(YEARS_OUT, years, ["task"])
    print(f"\nwrote {PROBE_OUT.relative_to(REPO_ROOT)} ({len(probe)} rows), "
          f"{YEARS_OUT.relative_to(REPO_ROOT)} ({len(years)} rows)")


if __name__ == "__main__":
    main()
