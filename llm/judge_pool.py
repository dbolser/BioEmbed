"""Judging pool and LLM relevance grades for a gene-disease haystack (PREREG "The answer key").

Pool, per query: the top 10 of every run in runs/ (minus known-bad run files, matched by content
hash, and any --skip), all curated gold, and up to 5 PubTator lookup hits per lookup.parquet tier
(exact, descendants, parents, ancestors): a seeded md5 sample of the tier's non-gold hits, so it
does not move when runs are added. best_rank is the best rank over the pooled runs at any depth
(null if none ranks it); systems lists the runs with it in their top 10; lookup_tier is its
lookup.parquet tier (null if not a hit); lookup_weight, set only on pairs drawn by the lookup
sample, is the tier's non-gold hits for the query / hits drawn from that tier, the inverse chance
of being drawn, to weight the sample back to all hits.

Judging: each (qid, pmid) is graded 0/1/2 with the PREREG rubric (verbatim; the script stops if
docs/PREREG.md no longer contains it) by two judges, DeepSeek-V4-Flash (pinned provider) and
Gemini 3.1 Flash-Lite, at temperature 0 with reasoning off. One request = one query and up to 10
abstracts (a query's pending pairs split into near-equal chunks), in a shuffled order seeded per
judge and query (seed in each cache file; practice requests made before 2026-10-16 have seed 42,
one order and grouping per query shared by both judges), labelled A, B, ... so no PMID is
shown. A query that already has grades from that judge (a run added later, pairs a response
left out) has each request filled up to 10 with a seeded sample of its graded pairs, so new
pairs are never judged in batches of their own; a pair's first grade is the one kept. The
prompt holds the gene symbol and name, HGNC aliases cleaned as labeller/make_demo_items.py
does, the disease name and Open Targets exact synonyms, and the abstracts; never rank, system,
gold or silver. JSON out: {id: {reason, grade}}, reason first, mapped back to PMIDs.

One grade per pair (judgments_merged*.parquet, for scripts/stats_haystack.py --judgments): the
mean of the judges' grades rounded down (1 and 2 give 1, 0 and 2 give 1), for pairs every judge
of the set graded. Self-grading: a run whose file name holds a judge's model family (e.g.
'deepseek' in llm-deepseek-v4-flash-top20-of-...) conflicts with that judge
(judgments_conflicts.parquet); a comparison involving it uses, for both runs,
judgments_merged_no-<judge>.parquet, and curator-only scores if no judge is left. Test-haystack
judging stops unless docs/PREREG.md registers this rule (MERGE_RULE).

Budget: nothing is sent unless --budget > 0 (the default 0 only builds the pool and prints the
cost of what is pending). Before each attempt, spend so far plus a reserve for requests in flight
(list-price upper bound or mean cost so far, whichever is higher) must stay within --budget, all
judges together.

Raw responses are cached per request in llm/llm_results/judge/<haystack>/<judge>/<qid>.<hash>.json
and judgments.parquet is rebuilt from that cache on every run (only files made with the current
prompt and settings count), so reruns pay only for pairs not yet graded. Pairs
a response leaves out go into a new batch next run; a response with no usable grade is retried
once, and every such attempt is kept as *.<time>.failed.json; a request that raises is kept as
*.<time>.error.json with its list-price ceiling as an estimated cost. Both count in the cost table.

Judge the test pool once, after all its runs are final.

Run (from llm/):
    uv run --with duckdb python judge_pool.py --haystack GeneDiseaseHaystackDev --max-queries 15 --budget 0.2
    uv run --with duckdb python judge_pool.py --haystack GeneDiseaseHaystack            # pool + cost, no calls
    uv run --with duckdb python judge_pool.py --haystack GeneDiseaseHaystack --budget 6 # judge the test pool

Output (data/haystack/<NAME>/):
    pool.parquet                 qid, pmid, in_gold, in_silver, best_rank, systems, lookup_tier, lookup_weight
    judgments.parquet            qid, pmid, judge, grade, reason, request_hash (one row per judge)
    judgments_merged.parquet     qid, pmid, grade (one per pair, all judges)
    judgments_merged_no-<judge>.parquet  the same without that judge, for each judge with a conflict
    judgments_conflicts.parquet  system, judge (runs that judge may not grade)
    pool_context.parquet         qid, gene_symbol, gene_name, gene_aliases, disease_name, disease_synonyms
                                 (the names the judges, and the labeller, are shown)
    pool_extra_docs.parquet      pmid, text for pool PMIDs not in docs_gene (from the records table)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import string
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from openai import APIStatusError, AsyncOpenAI

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
HAYSTACKS = REPO_ROOT / "data" / "haystack"
CACHE_DIR = HERE / "llm_results" / "judge"
PREREG = REPO_ROOT / "docs" / "PREREG.md"
IUK = "/outsee5/home/danbolser/Work/IUK-IPF/data"
OT = f"{IUK}/sources/opentargets/26.06"
HGNC = f"{IUK}/sources/hgnc/2026-10-14/hgnc_complete_set.txt"
RECORDS = f"{IUK}/embeddings-catalogue/records/*.parquet"
ABSTRACT_MARK = "\nAbstract: "  # as scripts/build_tier_c_haystack.py

TOP_K = 10
LOOKUP_PER_TIER = 5
DOCS_PER_REQUEST = 10
SEED = 42
ORDER_SEED = "{seed}|{judge}|{qid}"  # doc order and filler sample per request
PRACTICE = {"GeneDiseaseHaystackDev"}  # haystacks judged without MERGE_RULE in PREREG
MERGE_RULE = "mean of the judges' grades, rounded down"  # must be in docs/PREREG.md to judge a test haystack
# run files never pooled, by sha1 of the file (a fixed rerun under the same name is pooled)
BAD_RUNS = {"44e09ab115c10d26a54f75594866ca1a7e57e4cb": "ncbi__MedCPT__gene before the CLS-pooling fix"}

OR_ENV = ".env.or-deepseek_deepseek-v4-flash"  # OpenRouter key and base_url, the same for every judge
REASONING_OFF = {"reasoning": {"enabled": False}}
# short name -> request settings; family (first word of the model slug) marks self-grading conflicts
JUDGES = {
    "deepseek": {"model": "deepseek/deepseek-v4-flash", "temperature": 0.0,
                 "extra_body": REASONING_OFF | {"provider": {"order": ["baidu"], "allow_fallbacks": False}}},
    "gemini": {"model": "google/gemini-3.1-flash-lite", "temperature": 0.0, "extra_body": REASONING_OFF},
}
# $ per MTok (input, output): budget reserve, and cost when OpenRouter reports no usage.cost
PRICES = {
    "deepseek/deepseek-v4-flash": (0.091, 0.182),
    "google/gemini-3.1-flash-lite": (0.25, 1.50),
}
MAX_TOKENS = 2048

RUBRIC = {  # verbatim from docs/PREREG.md
    2: "the abstract reports evidence that this gene is involved in this disease (or a synonym, "
       "subtype, or the same phenotype in a model organism of the gene).",
    1: "the abstract discusses both, but the link is not its evidence (a review mention, a "
       "candidate in a screen with no result, or a related disease).",
    0: "otherwise.",
}
SYSTEM = ("You are an expert biocurator. You grade PubMed abstracts as evidence for a gene-disease link. "
          "Reply with JSON only.")
PROMPT = """Gene: {symbol} ({name})
Other names for the gene: {aliases}
Disease: {disease}
Other names for the disease: {synonyms}

Grade each abstract below for this gene and this disease:
{rubric}

{docs}

Answer as JSON with one entry per abstract id, the reason first (at most 25 words):
{{"A": {{"reason": "...", "grade": 0}}, ...}}"""
DOC = '<abstract id="{id}">\n{text}\n</abstract>'
PROMPT_SHA = hashlib.sha1("\x1f".join([SYSTEM, PROMPT, DOC, json.dumps(RUBRIC)]).encode()).hexdigest()[:8]


def check_prereg(haystack: str, sending: bool) -> None:
    text = " ".join(PREREG.read_text().split())
    if bad := [g for g, t in RUBRIC.items() if f"**{g}**: {t}" not in text]:
        raise SystemExit(f"rubric for grade {bad} no longer matches {PREREG}; update RUBRIC (re-judges everything)")
    if sending and haystack not in PRACTICE and MERGE_RULE not in text:
        raise SystemExit(f"register the judges and merge rule ('{MERGE_RULE}') in {PREREG} before judging {haystack}")


def build_pool(con: duckdb.DuckDBPyConnection, hay: Path, skip: list[str]) -> pd.DataFrame:
    """Writes pool.parquet and returns it."""
    runs = []
    for p in sorted((hay / "runs").glob("*.parquet")):
        sha = hashlib.sha1(p.read_bytes()).hexdigest()
        why = BAD_RUNS.get(sha) or ("--skip" if p.stem in skip else "")
        print(f"  {'skip' if why else 'pool'} {sha[:8]} {p.stem}{f'  ({why})' if why else ''}")
        if not why:
            runs.append(p)
    if not runs:
        raise SystemExit(f"no runs in {hay / 'runs'}")
    con.sql("CREATE OR REPLACE TEMP TABLE runs AS " + " UNION ALL ".join(
        f"SELECT '{p.stem}' AS system, qid, pmid, rank FROM '{p}'" for p in runs))
    print(f"{len(runs)} runs pooled")
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE pool AS
        WITH gold AS (SELECT DISTINCT qid, pmid, true AS in_gold FROM '{hay}/qrels.parquet'),
        silver AS (SELECT DISTINCT qid, pmid, true AS in_silver FROM '{hay}/silver.parquet'),
        top AS (SELECT qid, pmid, list(system ORDER BY system) AS systems
                FROM runs WHERE rank <= {TOP_K} GROUP BY qid, pmid),
        hits AS (SELECT qid, pmid, tier AS lookup_tier FROM '{hay}/lookup.parquet'),
        lookup AS (SELECT qid, pmid, count(*) OVER w / least(count(*) OVER w, {LOOKUP_PER_TIER}) AS lookup_weight
                   FROM hits ANTI JOIN gold USING (qid, pmid)
                   WINDOW w AS (PARTITION BY qid, lookup_tier)
                   QUALIFY row_number() OVER (PARTITION BY qid, lookup_tier
                                              ORDER BY md5('{SEED}|' || qid || '|' || pmid)) <= {LOOKUP_PER_TIER}),
        pairs AS (SELECT qid, pmid FROM top UNION SELECT qid, pmid FROM gold UNION SELECT qid, pmid FROM lookup),
        best AS (SELECT qid, pmid, min(rank)::INTEGER AS best_rank
                 FROM runs SEMI JOIN pairs USING (qid, pmid) GROUP BY qid, pmid)
        SELECT qid, pmid, coalesce(in_gold, false) AS in_gold, coalesce(in_silver, false) AS in_silver,
               best_rank, coalesce(systems, []::VARCHAR[]) AS systems, lookup_tier, lookup_weight
        FROM pairs LEFT JOIN gold USING (qid, pmid) LEFT JOIN silver USING (qid, pmid)
             LEFT JOIN best USING (qid, pmid) LEFT JOIN top USING (qid, pmid)
             LEFT JOIN hits USING (qid, pmid) LEFT JOIN lookup USING (qid, pmid)
        ORDER BY qid, pmid::BIGINT""")
    con.sql(f"COPY pool TO '{hay / 'pool.parquet'}' (FORMAT parquet)")
    return con.sql("SELECT * FROM pool").df()


def write_conflicts(hay: Path) -> pd.DataFrame:
    """judgments_conflicts.parquet: (system, judge) for every run in runs/ whose name holds the
    judge's model family, e.g. llm-gemini-... and google/gemini-3.1-flash-lite."""
    rows = [(p.stem, s["model"]) for p in sorted((hay / "runs").glob("*.parquet")) for s in JUDGES.values()
            if s["model"].split("/")[1].split("-")[0] in p.stem.lower()]
    conflicts = pd.DataFrame(rows, columns=["system", "judge"])
    conflicts.to_parquet(hay / "judgments_conflicts.parquet", index=False)
    for system, judge in rows:
        print(f"  conflict: {judge} may not grade {system}")
    return conflicts


def build_context(con: duckdb.DuckDBPyConnection, hay: Path) -> pd.DataFrame:
    """Gene and disease names per query. Aliases (HGNC alias + previous symbols) drop those of
    2 characters or fewer, any approved symbol or symbol shared by 2+ genes, and any Open Targets
    disease name or synonym, as labeller/make_demo_items.py."""
    ctx = con.sql(f"""
        WITH h AS (SELECT * FROM read_csv('{HGNC}', delim='\t', header=true, quote='"', all_varchar=true)),
        syn AS (SELECT DISTINCT hgnc_id, unnest(string_split(
                    coalesce(alias_symbol, '') || '|' || coalesce(prev_symbol, ''), '|')) AS s FROM h),
        bad AS (SELECT symbol AS s FROM h UNION SELECT s FROM syn WHERE s <> '' GROUP BY s HAVING count(*) > 1),
        dterm AS (SELECT lower(name) AS t FROM '{OT}/disease/disease.parquet'
                  UNION SELECT lower(unnest(exactSynonyms)) FROM '{OT}/disease/disease.parquet'),
        gene AS (SELECT id AS targetId, approvedSymbol AS gene_symbol, approvedName AS gene_name,
                        'HGNC:' || list_filter(dbXrefs, x -> x.source = 'HGNC')[1].id AS hgnc_id
                 FROM '{OT}/target/*.parquet'),
        alias AS (SELECT g.targetId, list(y.s ORDER BY y.s) AS gene_aliases
                  FROM gene g JOIN syn y USING (hgnc_id)
                  ANTI JOIN bad b ON b.s = y.s
                  ANTI JOIN dterm d ON d.t = lower(y.s)
                  WHERE length(y.s) > 2 AND y.s <> g.gene_symbol
                  GROUP BY g.targetId)
        SELECT q.qid, g.gene_symbol, g.gene_name, coalesce(a.gene_aliases, []::VARCHAR[]) AS gene_aliases,
               d.name AS disease_name, coalesce(d.exactSynonyms, []::VARCHAR[]) AS disease_synonyms
        FROM '{hay}/queries.parquet' q
        JOIN gene g USING (targetId)
        LEFT JOIN alias a USING (targetId)
        JOIN '{OT}/disease/disease.parquet' d ON d.id = q.diseaseId
        ORDER BY q.qid""").df()
    n = con.sql(f"SELECT count(*) FROM '{hay}/queries.parquet'").fetchone()[0]
    assert len(ctx) == n, f"lost queries joining gene/disease names: {len(ctx)} of {n}"
    return ctx


def fetch_extra_docs(con: duckdb.DuckDBPyConnection, hay: Path, pool: pd.DataFrame) -> None:
    """Add records-table texts for pool PMIDs that are neither in docs_gene nor already fetched."""
    extra = hay / "pool_extra_docs.parquet"
    con.register("pool_pmids", pool[["pmid"]].drop_duplicates())
    have = f"SELECT pmid FROM '{hay}/docs_gene.parquet'" + (f" UNION ALL SELECT pmid FROM '{extra}'"
                                                             if extra.exists() else "")
    need = con.sql(f"SELECT pmid FROM pool_pmids ANTI JOIN ({have}) USING (pmid)").df()
    if not len(need):
        return
    print(f"records lookup: {len(need)} pool PMIDs outside docs_gene")
    con.register("need", need)
    new = con.sql(f"""
        SELECT pmid, CASE WHEN title <> '' THEN title || chr(10) || chr(10) || abstract ELSE abstract END AS text
        FROM (SELECT replace(record_id, 'PMID:', '') AS pmid,
                     coalesce(trim(title), '') AS title,
                     trim(substr(text, position('{ABSTRACT_MARK}' IN text) + {len(ABSTRACT_MARK)})) AS abstract,
                     row_number() OVER (PARTITION BY record_id ORDER BY record_hash) AS rn
              FROM read_parquet('{RECORDS}')
              WHERE record_type = 'pubmed_abstract'
                AND record_id IN (SELECT 'PMID:' || pmid FROM need))
        WHERE rn = 1""").df()
    print(f"  found {len(new)}; {len(need) - len(new)} have no local abstract and cannot be judged")
    if extra.exists():
        new = pd.concat([pd.read_parquet(extra), new], ignore_index=True)
    new.sort_values("pmid", key=lambda s: s.astype(int)).to_parquet(extra, index=False)


def doc_texts(con: duckdb.DuckDBPyConnection, hay: Path, pmids) -> dict[str, str]:
    con.register("want", pd.DataFrame({"pmid": sorted(set(pmids))}))
    extra = hay / "pool_extra_docs.parquet"
    src = f"SELECT pmid, text FROM '{hay}/docs_gene.parquet'" + (
        f" UNION ALL SELECT pmid, text FROM '{extra}'" if extra.exists() else "")
    return dict(con.sql(f"SELECT pmid, text FROM ({src}) SEMI JOIN want USING (pmid)").fetchall())


def read_env(name: str) -> dict:
    env = {}
    for line in (HERE / name).read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#"):
            env[key.strip()] = value.strip()
    return env


def judge_settings(short: str) -> tuple[dict, dict, str]:
    """(env, request settings, config hash). The hash covers model, settings and prompt template,
    not the doc order (each request's seed is in its cache file)."""
    settings = {"max_tokens": MAX_TOKENS, **JUDGES[short]}
    config = hashlib.sha1(json.dumps({**settings, "prompt": PROMPT_SHA}, sort_keys=True).encode()).hexdigest()[:8]
    return read_env(OR_ENV), settings, config


def build_prompt(ctx: dict, texts: list[str]) -> str:
    return PROMPT.format(
        symbol=ctx["gene_symbol"], name=ctx["gene_name"],
        aliases=", ".join(ctx["gene_aliases"]) or "none",
        disease=ctx["disease_name"], synonyms="; ".join(ctx["disease_synonyms"]) or "none",
        rubric="\n".join(f"{g}: {t}" for g, t in RUBRIC.items()),
        docs="\n\n".join(DOC.format(id=i, text=t.strip()) for i, t in zip(string.ascii_uppercase, texts)))


def request_hash(request: dict) -> str:
    return hashlib.sha1(json.dumps(request, sort_keys=True).encode()).hexdigest()[:12]


def choice(raw: dict) -> dict:
    """First choice of a raw response; {} when there is none (an OpenRouter error payload)."""
    return (raw.get("choices") or [None])[0] or {}


def as_grade(g) -> int | None:
    """0, 1 or 2 from an int, an integral float or a numeric string; None otherwise."""
    if isinstance(g, bool):
        return None
    try:
        f = float(str(g).strip())
    except ValueError:
        return None
    return int(f) if f in (0, 1, 2) else None


def parse_grades(content: str | None, pmids: list[str]) -> dict[str, tuple[int, str]]:
    """{pmid: (grade, reason)} for shown abstracts (ids A, B, ... in pmids order; 'Abstract_B'
    and the like accepted) with a grade of 0, 1 or 2; the rest are left out."""
    s = re.sub(r"^```\w*\s*|\s*```$", "", (content or "").strip())
    obj = None
    for cand in (s, (m := re.search(r"\{.*\}", s, re.DOTALL)) and m[0]):
        try:
            obj = json.loads(cand)
            break
        except (TypeError, json.JSONDecodeError):
            pass
    ids = dict(zip(string.ascii_uppercase, pmids))
    out = {}
    for k, v in (obj.items() if isinstance(obj, dict) else []):
        k = re.sub(r"(?i)^[\W_]*(abstract[\W_]*)?|[\W_]+$", "", str(k)).upper()
        g, r = (v.get("grade"), v.get("reason")) if isinstance(v, dict) else (v, "")
        if k in ids and (g := as_grade(g)) is not None:
            out[ids[k]] = (g, str(r or "").strip())
    return out


def response_cost(model: str, usage: dict) -> float:
    if usage.get("cost") is not None:
        return float(usage["cost"])
    pi, po = PRICES.get(model, (0.0, 0.0))
    return (usage.get("prompt_tokens", 0) * pi + usage.get("completion_tokens", 0) * po) / 1e6


def price_ceiling(request: dict) -> float:
    """List-price upper bound for one request: ~4 characters per prompt token, max_tokens out."""
    pi, po = PRICES.get(request["model"], (0.0, 0.0))
    return (sum(len(m["content"]) for m in request["messages"]) / 4 * pi + request["max_tokens"] * po) / 1e6


def load_cache(hay_name: str, short: str) -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    """(judgments, per-request usage incl. failed attempts, cost of each stale file) from the
    cache files; only those made with the current config give judgments and usage."""
    _, settings, config = judge_settings(short)
    rows, reqs, stale = [], [], []
    for p in sorted((CACHE_DIR / hay_name / short).glob("*.json")):
        d = json.loads(p.read_text())
        resp = d["response"]
        if d["config"] != config:
            stale.append(d.get("est_cost") or response_cost(settings["model"], resp.get("usage") or {}))
            continue
        failed = p.name.endswith((".failed.json", ".error.json"))
        grades = {} if failed else parse_grades((choice(resp).get("message") or {}).get("content"), d["pmids"])
        rows += [(d["qid"], pmid, settings["model"], g, r, d["request_hash"], d["created"])
                 for pmid, (g, r) in grades.items()]
        usage = resp.get("usage") or {}
        reqs.append({"judge": settings["model"], "request_hash": d["request_hash"], "failed": failed,
                     "n_docs": 0 if failed else len(d["pmids"]), "n_graded": len(grades),
                     "prompt_tokens": usage.get("prompt_tokens", 0),
                     "completion_tokens": usage.get("completion_tokens", 0),
                     "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
                     "cost": d.get("est_cost") or response_cost(settings["model"], usage)})
    cols = ["qid", "pmid", "judge", "grade", "reason", "request_hash", "created"]
    judg = (pd.DataFrame(rows, columns=cols).sort_values("created")
            .drop_duplicates(["qid", "pmid", "judge"]).drop(columns="created"))
    return judg, pd.DataFrame(reqs), stale


def make_batches(pending: pd.DataFrame, graded: pd.DataFrame, ctx: pd.DataFrame, texts: dict[str, str],
                 short: str, out_dir: Path) -> list[dict]:
    """Requests for pending pairs; each judge gets its own seeded doc order and grouping. Requests
    of a query with graded pairs are filled up to DOCS_PER_REQUEST with a sample of them."""
    _, settings, config = judge_settings(short)
    cmap = ctx.set_index("qid").to_dict("index")
    old = graded.groupby("qid").pmid.apply(lambda s: sorted(s, key=int)).to_dict()
    batches = []
    for qid, g in pending.groupby("qid", sort=True):
        pmids = sorted(g.pmid, key=int)
        seed = ORDER_SEED.format(seed=SEED, judge=short, qid=qid)
        rng = random.Random(seed)
        rng.shuffle(pmids)
        for chunk in np.array_split(np.array(pmids), -(-len(pmids) // DOCS_PER_REQUEST)):
            chunk = chunk.tolist()
            if pad := old.get(qid, []):
                chunk += rng.sample(pad, min(len(pad), DOCS_PER_REQUEST - len(chunk)))
                rng.shuffle(chunk)
            prompt = build_prompt(cmap[qid], [texts[p] for p in chunk])
            request = {**settings, "messages": [{"role": "system", "content": SYSTEM},
                                                {"role": "user", "content": prompt}]}
            h = request_hash(request)
            batches.append({"qid": qid, "pmids": chunk, "request": request, "request_hash": h, "config": config,
                            "seed": seed, "path": out_dir / f"{qid.replace('/', '_')}.{h}.json"})
    return batches


async def send_all(env: dict, batches: list[dict], budget: float, concurrency: int, client=None) -> float:
    """Send every batch; return new spend. An attempt starts only if spend so far plus a reserve for
    it and every attempt in flight stays within budget; the reserve per attempt is its list-price
    ceiling or the mean cost so far, whichever is higher. Spend can pass the budget only by what
    in-flight attempts cost over their reserve (at most `concurrency` attempts with no list price).
    An attempt that raises (it may still be billed) is logged as *.error.json and counted at its
    list-price ceiling."""
    client = client or AsyncOpenAI(api_key=env["token"], base_url=env["base_url"], max_retries=8)
    sem = asyncio.Semaphore(concurrency)
    spent, n_cost, in_flight, failed, skipped = 0.0, 0, {}, [], 0

    def save(b: dict, suffix: str, **extra) -> None:
        now = datetime.now(UTC)
        path = b["path"].with_suffix(f".{now:%Y%m%dT%H%M%S%f}{suffix}") if suffix else b["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {k: b[k] for k in ["qid", "pmids", "request_hash", "config", "seed", "request"]}
        path.write_text(json.dumps(rec | {"created": now.isoformat()} | extra, indent=1))

    async def one(b: dict) -> None:
        nonlocal spent, n_cost, skipped
        err = ""
        async with sem:
            for _ in range(2):  # one retry when nothing usable comes back
                mean = spent / n_cost if n_cost else 0.0
                ceiling = price_ceiling(b["request"])
                if spent + sum(max(c, mean) for c in [*in_flight.values(), ceiling]) > budget:
                    skipped += 1
                    return
                in_flight[id(b)] = ceiling
                try:
                    raw = (await client.chat.completions.create(**b["request"])).model_dump()
                except Exception as e:  # noqa: BLE001 - logged with an estimated cost, retried on rerun
                    err = repr(e)[:200]
                    if isinstance(e, APIStatusError) and 400 <= e.status_code < 500 and e.status_code != 408:
                        break  # rejected, so not billed (402 no credit, 429 rate limit, ...); not retried
                    spent += ceiling
                    save(b, ".error.json", response={}, error=err, est_cost=ceiling)
                    continue
                finally:
                    del in_flight[id(b)]
                spent += response_cost(b["request"]["model"], raw.get("usage") or {})
                n_cost += 1
                c = choice(raw)
                ok = bool(parse_grades((c.get("message") or {}).get("content"), b["pmids"]))
                save(b, "" if ok else ".failed.json", response=raw)
                if ok:
                    return
                err = f"no usable grade (finish_reason={c.get('finish_reason')}, error={raw.get('error')})"
        failed.append((b["qid"], err))

    errors = [e for e in await asyncio.gather(*(one(b) for b in batches), return_exceptions=True) if e]
    await client.close()
    for qid, err in failed[:5]:
        print(f"  FAILED {qid}: {err}")
    for e in errors[:5]:
        print(f"  ERROR {e!r}")
    if failed or skipped or errors:
        print(f"  {len(failed)} failed, {skipped} skipped over budget, {len(errors)} errors")
    return spent


AGREEMENT = ["exact", "kappa_lin", "kappa_quad", "kappa_ge1", "kappa_ge2"]


def merge_judgments(hay: Path, judg: pd.DataFrame, conflicts: pd.DataFrame) -> None:
    """judgments_merged.parquet over all judges, and judgments_merged_no-<judge>.parquet without
    each judge that has a conflict run: one grade per pair, the judges' mean grade rounded down,
    for pairs every judge of the set graded."""
    short = {s["model"]: k for k, s in JUDGES.items()}
    sets = {"": list(short)} | {f"_no-{short[m]}": [x for x in short if x != m] for m in sorted(set(conflicts.judge))}
    for suffix, models in sets.items():
        g = judg[judg.judge.isin(models)].groupby(["qid", "pmid"], sort=False).grade.agg(["sum", "count"])
        full = g["count"] == len(models)
        out = (g[full]["sum"] // len(models)).astype("int8").rename("grade").reset_index()
        out.to_parquet(hay / f"judgments_merged{suffix}.parquet", index=False)
        print(f"wrote judgments_merged{suffix}.parquet: {len(out)} pairs ({', '.join(short[m] for m in models)}), "
              f"grades {out.grade.value_counts().sort_index().to_dict()}; {int((~full).sum())} pairs left out "
              f"(not graded by every judge)")


def agreement_stats(obs: np.ndarray) -> np.ndarray:
    """AGREEMENT from a 3x3 count matrix: exact agreement, Cohen's weighted kappa (weights |i - j|,
    then (i - j)^2), and Cohen's kappa with grades cut to relevant or not at >= 1 and at >= 2."""
    def kappa(o: np.ndarray, w: np.ndarray) -> float:
        return 1 - (w * o).sum() / (w * np.outer(o.sum(1), o.sum(0))).sum()

    obs = obs / obs.sum()
    d = np.subtract.outer(np.arange(3), np.arange(3))
    cuts = [np.add.reduceat(np.add.reduceat(obs, [0, t], 0), [0, t], 1) for t in (1, 2)]
    return np.array([np.trace(obs), kappa(obs, np.abs(d)), kappa(obs, d**2)] + [kappa(c, 1 - np.eye(2)) for c in cuts])


def agreement(qids: pd.Series, a: pd.Series, b: pd.Series, reps: int = 2000) -> str:
    """Agreement stats, each with a 95% percentile CI from resampling queries (pairs of one query
    move together)."""
    q = pd.factorize(qids)[0]
    per_q = np.zeros((q.max() + 1, 3, 3))
    np.add.at(per_q, (q, a.to_numpy(), b.to_numpy()), 1)
    boot = np.array([agreement_stats(per_q[i].sum(0)) for i in
                     np.random.default_rng(SEED).integers(len(per_q), size=(reps, len(per_q)))])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return " ".join(f"{name}={x:.3f} [{lo_:.3f}, {hi_:.3f}]" for name, x, lo_, hi_ in
                    zip(AGREEMENT, agreement_stats(per_q.sum(0)), lo, hi))


def report(pool: pd.DataFrame, judg: pd.DataFrame, reqs: pd.DataFrame) -> None:
    if not len(judg):
        return
    j = judg.merge(pool[["qid", "pmid", "in_gold"]], on=["qid", "pmid"])
    print("\nGrades by judge (share of pairs; gold = curated pairs only):")
    tab = pd.crosstab(j.judge, j.grade, normalize="index").round(3)
    tab["pairs"] = j.groupby("judge").size()
    tab["gold_2"] = j[j.in_gold].groupby("judge").grade.apply(lambda s: (s == 2).mean()).round(3)
    tab["gold_n"] = j[j.in_gold].groupby("judge").size()
    print(tab.to_string())

    sizes = {p.parent.name: len(pd.read_parquet(p, columns=["qid"])) for p in sorted(HAYSTACKS.glob("*/pool.parquet"))}
    cost = reqs.groupby("judge").agg(requests=("request_hash", "size"), failed=("failed", "sum"),
                                     pairs=("n_graded", "sum"), missed=("n_docs", "sum"),
                                     prompt_tok=("prompt_tokens", "sum"), completion_tok=("completion_tokens", "sum"),
                                     reasoning_tok=("reasoning_tokens", "sum"), usd=("cost", "sum"))
    cost["missed"] -= cost["pairs"]
    cost["usd_per_1k_pairs"] = 1000 * cost.usd / cost.pairs
    for name, n in sizes.items():
        cost[f"usd_{name}({n})"] = cost.usd_per_1k_pairs * n / 1000
    print("\nCost (all cached requests for this haystack, failed attempts included; "
          "projection = whole pool at this rate):")
    print(cost.round(4).to_string())

    judges = sorted(j.judge.unique())
    if len(judges) < 2:
        return
    w = (j.pivot_table(index=["qid", "pmid"], columns="judge", values="grade").dropna().astype(int)
         .join(pool.set_index(["qid", "pmid"]).systems.map(len).gt(0).rename("top")).reset_index())
    print("\nAgreement (95% CI over queries); top-10 = pairs in some run's top 10:")
    for x, y in [(x, y) for i, x in enumerate(judges) for y in judges[i + 1:]]:
        for name, s in [("all", w), ("top-10", w[w.top])]:
            print(f"  {x} vs {y} {name}: pairs={len(s)} queries={s.qid.nunique()} "
                  f"{agreement(s.qid, s[x], s[y])}")
        print(pd.crosstab(w[x].rename(x), w[y].rename(y)).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--haystack", required=True, help="e.g. GeneDiseaseHaystackDev (practice), GeneDiseaseHaystack")
    ap.add_argument("--skip", nargs="*", default=[], help="run file stems left out of the pool (BAD_RUNS always are)")
    ap.add_argument("--judges", nargs="+", default=list(JUDGES), choices=list(JUDGES))
    ap.add_argument("--max-queries", type=int, help="judge only the first N queries (by qid)")
    ap.add_argument("--pool-only", action="store_true", help="write pool.parquet and stop (no calls)")
    ap.add_argument("--budget", type=float, default=0.0,
                    help="max new $ per run, all judges together; 0 = no calls, print the cost of what is pending")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    check_prereg(args.haystack, args.budget > 0)
    hay = HAYSTACKS / args.haystack
    con = duckdb.connect()
    pool = build_pool(con, hay, args.skip)
    conflicts = write_conflicts(hay)
    per_q = pool.groupby("qid").size()
    print(f"pool: {len(pool)} pairs over {len(per_q)} queries (median {per_q.median():.0f}/query); "
          f"top-{TOP_K} {int(pool.systems.map(len).gt(0).sum())}, gold {int(pool.in_gold.sum())}, "
          f"lookup-only {int((pool.systems.map(len).eq(0) & ~pool.in_gold).sum())}, silver {int(pool.in_silver.sum())}")
    if args.pool_only:
        return

    ctx = build_context(con, hay)
    ctx.to_parquet(hay / "pool_context.parquet", index=False)
    fetch_extra_docs(con, hay, pool)
    qids = sorted(pool.qid.unique())[: args.max_queries]
    sel = pool[pool.qid.isin(qids)]
    texts = doc_texts(con, hay, sel.pmid)
    if n_none := int((~sel.pmid.isin(texts)).sum()):
        print(f"{n_none} selected pairs have no text and are skipped")
    sel = sel[sel.pmid.isin(texts)]

    budget = args.budget
    for short in args.judges:
        env, settings, _ = judge_settings(short)
        done, _, stale = load_cache(args.haystack, short)
        both = sel.merge(done[["qid", "pmid"]], on=["qid", "pmid"], how="left", indicator=True)
        pending, graded = both[both._merge == "left_only"], both[both._merge == "both"]
        batches = make_batches(pending, graded, ctx, texts, short, CACHE_DIR / args.haystack / short)
        n_pad = sum(len(b["pmids"]) for b in batches) - len(pending)
        print(f"{settings['model']}: {len(graded)} of {len(sel)} pairs graded, {len(batches)} requests to send"
              f"{f' ({n_pad} graded pairs added as filler)' if n_pad else ''}; "
              f"{len(stale)} stale cache files ignored (${sum(stale):.4f} spent on them)")
        if batches and budget <= 0:
            cost = n = 0
            for p in CACHE_DIR.glob(f"*/{short}"):  # $ per abstract shown over every haystack cached
                reqs = load_cache(p.parent.name, short)[1]
                cost, n = cost + reqs.get("cost", pd.Series()).sum(), n + reqs.get("n_docs", pd.Series()).sum()
            print(f"  not sent (--budget 0): list-price ceiling ${sum(price_ceiling(b['request']) for b in batches):.2f}"
                  + (f", ${cost / n * (len(pending) + n_pad):.2f} at the cached rate" if n else ""))
        elif batches:
            spent = asyncio.run(send_all(env, batches, budget, args.concurrency))
            budget -= spent
            print(f"  new spend ${spent:.4f}")

    loaded = [load_cache(args.haystack, s) for s in JUDGES]
    judg = pd.concat([x[0] for x in loaded], ignore_index=True).merge(pool[["qid", "pmid"]], on=["qid", "pmid"])
    judg = judg.astype({"grade": "int8"}).sort_values(["qid", "pmid", "judge"])
    judg.to_parquet(hay / "judgments.parquet", index=False)
    print(f"\nwrote {hay.relative_to(REPO_ROOT)}/judgments.parquet: {len(judg)} rows, "
          f"{judg.groupby('judge').size().to_dict()}")
    merge_judgments(hay, judg, conflicts)
    report(pool, judg, pd.concat([x[1] for x in loaded], ignore_index=True))


if __name__ == "__main__":
    main()
