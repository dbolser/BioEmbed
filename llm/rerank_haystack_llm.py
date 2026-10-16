"""LLM listwise rerank of a first-stage run's top 20 on a GeneDiseaseHaystack.

PREREG (change log 22:45): LLM rerankers re-sort the top 20 of the first pass;
not tuned. For C1 the first pass is the candidate (BM25, dense, RRF) with the
best curator-only recall@100 on the practice haystack; --run defaults to tuned
BM25, and reranks of it are observations until that pick is made. Per query
the model gets the gene (symbol, approved name, HGNC alias and previous
symbols), the disease (OT name and exact synonyms) and the 20 abstracts, each
cut to its first 300 words, shuffled by a qid-seeded RNG (the shown order is
recorded), and returns a full ranking as JSON. Parsing is lenient (code
fences, a bare list or any wrapping object, stray text, duplicates,
out-of-range ids are dropped); ids the model leaves out follow its ranking in
first-stage order. Output ranks 1-20 are that order, 21+ keep the first-stage
order; score = -rank. A reply that echoes the shown order (identity) is kept
as the model's ranking, i.e. a random top 20.

Models (OpenRouter, ids from llm/.env.or-*), temperature 0, reasoning off;
DeepSeek pinned to Baidu as in closed_book_probe.py. Google advises the
default 1.0 for Gemini 3 (lower can loop); kept at 0 as run, and no rerank
reply looped (all finish "stop"). Full API responses are
cached per request in llm/llm_results/haystack_rerank/<model-slug>/<haystack>/
<qid>.<request-hash>.json, so reruns are free and any change to the prompt or
settings makes a new request. Nothing is written unless every query has a
response.

Run (from llm/):
    uv run --with duckdb python rerank_haystack_llm.py --haystack GeneDiseaseHaystackDev \\
        --run bm25-symbol_disease-k0.9-b0.75__gene --model deepseek [--limit N] [--budget 2]
Output: data/haystack/<haystack>/runs/llm-<model>-top20-of-<first stage>__gene.parquet
            qid, pmid, score, rank
        results/llm_v2/haystack_rerank/<haystack>/llm-<model>-top20-of-<first stage>.parquet
            per query: shown (pmids in prompt order), ranked (pmids as the model ranked them),
            n_valid, parsed_json, identity (returned 1..20 as shown), tokens_in, cached_tokens,
            tokens_out, reasoning_tokens, cost, finish_reason, provider, raw_path
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
from openai import AsyncOpenAI

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
HAYSTACKS = REPO_ROOT / "data" / "haystack"
CACHE_DIR = HERE / "llm_results" / "haystack_rerank"
SIDE_DIR = REPO_ROOT / "results" / "llm_v2" / "haystack_rerank"
IUK = "/outsee5/home/danbolser/Work/IUK-IPF/data"
OT = f"{IUK}/sources/opentargets/26.06"
HGNC = f"{IUK}/sources/hgnc/2026-10-14/hgnc_complete_set.txt"
# short name -> (env file, extra_body)
MODELS = {
    "deepseek": (".env.or-deepseek_deepseek-v4-flash",
                 {"reasoning": {"enabled": False},
                  "provider": {"order": ["baidu"], "allow_fallbacks": False}}),
    "gemini": (".env.or-google_gemini-3.1-flash-lite", {"reasoning": {"enabled": False}}),
}
TOP_K = 20
MAX_WORDS = 300
MAX_TOKENS = 1024
TEMPERATURE = 0.0
SYSTEM = ("You are an expert biomedical curator. You rank PubMed abstracts by how strongly they "
          "report evidence that a gene is involved in a disease. Reply with JSON only.")
PROMPT = (
    "Gene: {gene}\n"
    "Disease: {disease}\n\n"
    "Rank all {n} abstracts below by how well each reports evidence that this gene is involved in "
    "this disease (or a synonym or subtype of it, or the same phenotype in a model organism), "
    "most relevant first.\n\n"
    "{docs}\n\n"
    'Answer as JSON with all {n} ids, best first: {{"ranking": [3, 17, ...]}}'
)


def read_env(name: str) -> dict:
    env = {}
    for line in (HERE / name).read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#"):
            env[key.strip()] = value.strip()
    return env


def query_info(haystack: Path) -> dict[str, tuple[str, str]]:
    """qid -> (gene line, disease line)."""
    q = duckdb.sql(f"""
        SELECT q.qid, t.approvedSymbol AS symbol, t.approvedName AS name, d.name AS disease,
               coalesce(d.exactSynonyms, []) AS disease_syn, g.gene_syn
        FROM '{haystack / "queries.parquet"}' q
        JOIN '{OT}/target/*.parquet' t ON t.id = q.targetId
        JOIN '{OT}/disease/*.parquet' d ON d.id = q.diseaseId
        LEFT JOIN (
            SELECT o.id AS targetId,
                   string_agg(concat_ws('|', h.alias_symbol, h.prev_symbol), '|') AS gene_syn
            FROM '{OT}/target/*.parquet' o, unnest(o.dbXrefs) x,
                 read_csv('{HGNC}', delim='\t', all_varchar=true) h
            WHERE x.unnest.source = 'HGNC' AND h.hgnc_id = 'HGNC:' || x.unnest.id
            GROUP BY 1
        ) g USING (targetId)
        ORDER BY q.qid
    """).df()
    info = {}
    for r in q.itertuples():
        aliases = [a for a in dict.fromkeys((r.gene_syn or "").split("|")) if a and a != r.symbol]
        syns = [s for s in dict.fromkeys(r.disease_syn) if s and s.lower() != r.disease.lower()]
        gene = f"{r.symbol} ({r.name})" + (f"; also known as {', '.join(aliases)}" if aliases else "")
        disease = r.disease + (f"; synonyms: {'; '.join(syns)}" if syns else "")
        info[r.qid] = (gene, disease)
    return info


def truncate(text: str, max_words: int = MAX_WORDS) -> str:
    words = text.split()
    return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")


def shuffled(qid: str, pmids: list[str]) -> list[str]:
    seed = int(hashlib.sha1(qid.encode()).hexdigest()[:8], 16)
    order = list(pmids)
    random.Random(seed).shuffle(order)
    return order


def build_request(model: str, extra_body: dict, gene: str, disease: str, docs: list[str]) -> dict:
    body = "\n\n".join(f"[{i}] {d}" for i, d in enumerate(docs, 1))
    prompt = PROMPT.format(gene=gene, disease=disease, n=len(docs), docs=body)
    return {"model": model, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "extra_body": extra_body,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}


def cache_path(model: str, haystack: str, qid: str, request: dict) -> Path:
    h = hashlib.sha1(json.dumps(request, sort_keys=True).encode()).hexdigest()[:10]
    return CACHE_DIR / model.replace("/", "__") / haystack / f"{qid.replace('/', '_')}.{h}.json"


def parse_ranking(content: str, n: int) -> tuple[list[int], bool]:
    """(ids in 1..n in the model's order, deduplicated; whether the answer was valid JSON)."""
    s = re.sub(r"^```\w*\s*|\s*```$", "", (content or "").strip())
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        obj = obj.get("ranking", next((v for v in obj.values() if isinstance(v, list)), None))
    parsed = isinstance(obj, list)
    if not parsed:
        m = re.search(r"\[([^\[\]]*)\]", s.split("ranking", 1)[-1])
        obj = re.findall(r"\d+", m.group(1) if m else s)
    ids = []
    for x in obj:
        d = re.search(r"\d+", str(x))
        i = int(d.group()) if d else 0
        if 1 <= i <= n and i not in ids:
            ids.append(i)
    return ids, parsed


async def send_all(requests: dict[str, tuple[Path, dict]], env: dict, concurrency: int, budget: float) -> float:
    """Send every uncached request; returns new spend. Stops launching once spend reaches budget."""
    todo = {qid: v for qid, v in requests.items() if not v[0].exists()}
    print(f"{env['model']}: {len(requests) - len(todo)} cached, {len(todo)} to query", flush=True)
    client = AsyncOpenAI(api_key=env["token"], base_url=env["base_url"], max_retries=8)
    sem = asyncio.Semaphore(concurrency)
    spent, failed, skipped = 0.0, [], 0

    async def one(qid: str, path: Path, request: dict) -> None:
        nonlocal spent, skipped
        async with sem:
            if spent >= budget:
                skipped += 1
                return
            try:
                resp = await client.chat.completions.create(**request)
            except Exception as e:  # noqa: BLE001 - left uncached, retried on rerun
                failed.append((qid, repr(e)[:200]))
                return
        raw = resp.model_dump()
        spent += (raw.get("usage") or {}).get("cost") or 0.0
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"qid": qid, "request": request,
                                    "created": datetime.now(UTC).isoformat(),
                                    "response": raw}, indent=1))

    await asyncio.gather(*(one(q, *v) for q, v in todo.items()))
    await client.close()
    for qid, err in failed[:5]:
        print(f"  FAILED {qid}: {err}")
    if failed or skipped:
        print(f"  {len(failed)} failed, {skipped} skipped over budget")
    print(f"  new spend ${spent:.4f}", flush=True)
    return spent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--haystack", default="GeneDiseaseHaystack")
    ap.add_argument("--run", default="bm25-symbol_disease-k0.9-b0.75__gene", help="first-stage run file stem")
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--budget", type=float, default=2.0, help="max new $ this invocation")
    ap.add_argument("--limit", type=int, help="first N queries; paid calls cached, nothing written")
    args = ap.parse_args()

    haystack = HAYSTACKS / args.haystack
    env_file, extra_body = MODELS[args.model]
    env = read_env(env_file)
    model = env["model"]
    system, track = args.run.rsplit("__", 1)
    name = f"llm-{model.split('/')[-1]}-top{TOP_K}-of-{system}"

    run = pd.read_parquet(haystack / "runs" / f"{args.run}.parquet").sort_values(["qid", "rank"])
    qids = sorted(run.qid.unique())[: args.limit]
    run = run[run.qid.isin(qids)]
    top = run[run["rank"] <= TOP_K]
    docs = duckdb.sql(f"""
        SELECT pmid, text FROM '{haystack / "docs_gene.parquet"}'
        WHERE pmid IN (SELECT DISTINCT pmid FROM top)
    """).df().set_index("pmid").text.map(truncate).to_dict()
    info = query_info(haystack)

    first = {q: g.pmid.tolist() for q, g in top.groupby("qid")}
    rest = {q: g.pmid.tolist() for q, g in run[run["rank"] > TOP_K].groupby("qid")}
    shown = {q: shuffled(q, first[q]) for q in qids}
    requests = {}
    for q in qids:
        req = build_request(model, extra_body, *info[q], [docs[p] for p in shown[q]])
        requests[q] = (cache_path(model, args.haystack, q, req), req)
    spent = asyncio.run(send_all(requests, env, args.concurrency, args.budget))

    rows, meta, missing = [], [], []
    for q in qids:
        path = requests[q][0]
        if not path.exists():
            missing.append(q)
            continue
        resp = json.loads(path.read_text())["response"]
        ids, parsed = parse_ranking(resp["choices"][0]["message"]["content"], len(shown[q]))
        ranked = [shown[q][i - 1] for i in ids]
        order = ranked + [p for p in first[q] if p not in ranked]
        order += rest.get(q, [])
        rows += [(q, p, -float(r), r) for r, p in enumerate(order, 1)]
        u = resp.get("usage") or {}
        meta.append({"qid": q, "shown": shown[q], "ranked": ranked, "n_valid": len(ids), "parsed_json": parsed,
                     "identity": ids == list(range(1, len(shown[q]) + 1)),
                     "tokens_in": u.get("prompt_tokens", 0),
                     "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                     "tokens_out": u.get("completion_tokens", 0),
                     "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
                     "cost": u.get("cost") or 0.0, "finish_reason": resp["choices"][0].get("finish_reason"),
                     "provider": resp.get("provider"), "raw_path": str(path.relative_to(REPO_ROOT))})
    meta = pd.DataFrame(meta)
    if len(meta):
        print(f"{len(meta)} queries: full rankings {int((meta.n_valid == TOP_K).sum())}, "
              f"mean ids returned {meta.n_valid.mean():.1f}/{TOP_K}, invalid JSON {int((~meta.parsed_json).sum())}, "
              f"not stopped {int((meta.finish_reason != 'stop').sum())}, identity (shown order kept) "
              f"{int(meta.identity.sum())}, top-1 changed "
              f"{sum(m.ranked[:1] != first[m.qid][:1] for m in meta.itertuples())}")
        print(f"tokens in {meta.tokens_in.sum():,} (cached {meta.cached_tokens.sum():,}), "
              f"out {meta.tokens_out.sum():,} (reasoning {meta.reasoning_tokens.sum():,}), "
              f"cost ${meta.cost.sum():.4f} (new ${spent:.4f}), providers {meta.provider.value_counts().to_dict()}")
    if missing:
        print(f"{len(missing)} queries without a response; nothing written. Rerun.")
        sys.exit(1)
    if args.limit:
        print("--limit: smoke run, nothing written")
        return
    out = haystack / "runs" / f"{name}__{track}.parquet"
    pd.DataFrame(rows, columns=["qid", "pmid", "score", "rank"]).to_parquet(out, index=False)
    side = SIDE_DIR / args.haystack / f"{name}.parquet"
    side.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(side, index=False)
    print(f"-> {out.relative_to(REPO_ROOT)} ({len(rows):,} rows), {side.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
