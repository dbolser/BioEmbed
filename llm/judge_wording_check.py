"""Do the judges grade lower than the humans because of the rubric wording? (PREREG 2026-11-03,
exploratory.) Both judges re-grade the calibration abstracts in their original test-pool requests
(same batch, doc order and settings, from the judge cache), two arms:
  same   the request resent unchanged (run-to-run drift)
  human  the rubric lines swapped for the human guide's final wording (labeller/index.html)
Grades for the calibration pairs are compared with Dan's pass-0 labels; the answer key is untouched.

Run (from llm/):
    uv run --with duckdb python judge_wording_check.py               # cost of what is pending, no calls
    uv run --with duckdb python judge_wording_check.py --budget 3    # send
Output: llm/llm_results/judge/wording_check/<arm>/<judge>/*.json, results/judge_wording_check.csv
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3

import numpy as np
import pandas as pd

import judge_pool as jp

HAY = "GeneDiseaseHaystack"
DB = jp.REPO_ROOT / "labeller/labels_calibration.sqlite"
OUT = jp.CACHE_DIR / "wording_check"
CSV = jp.REPO_ROOT / "results/judge_wording_check.csv"
OLD = "\n".join(f"{g}: {t}" for g, t in jp.RUBRIC.items())
HUMAN = (  # labeller/index.html, the guide's grade list and the same-or-related note
    "2: Evidence. The abstract reports evidence that this gene (or a homolog) is involved in this disease, "
    "a synonym or a subtype, including in a model of the disease.\n"
    "1: Related or mention. The abstract discusses both the gene and the disease, but either indirectly or "
    "in a related disease (see below).\n"
    "0: Unrelated. Either an unrelated disease, the gene alone (e.g. basic research), or not this gene at all.\n"
    "Same disease, or related? This disease by another name, a subtype of a broader disease, or the disease "
    "described only by its symptoms: all count as the same disease. Rate a broader disease as 1 or 0 unless "
    "the study includes this disease. Use your own scientific judgement to decide whether two different "
    "diseases are related, e.g. epilepsy and schizophrenia.")
ARMS = {"same": lambda s: s, "human": lambda s: s.replace(OLD, HUMAN)}


def calibration() -> pd.DataFrame:
    q = "select round_id qid, pmid, grade dan from labels where labeller='dan' and pass=0"
    return pd.read_sql(q, sqlite3.connect(DB))


def batches(cal: pd.DataFrame) -> list[dict]:
    """The original requests holding a calibration pair, once per arm and judge."""
    judg = pd.read_parquet(jp.HAYSTACKS / HAY / "judgments.parquet")
    hashes = set(judg.merge(cal[["qid", "pmid"]]).request_hash)
    out = []
    for short in jp.JUDGES:
        for p in sorted((jp.CACHE_DIR / HAY / short).glob("*.json")):
            if p.name.endswith((".failed.json", ".error.json")):
                continue
            d = json.loads(p.read_text())
            if d["request_hash"] not in hashes:
                continue
            for arm, edit in ARMS.items():
                req = json.loads(json.dumps(d["request"]))
                user = req["messages"][1]
                assert OLD in user["content"]
                user["content"] = edit(user["content"])
                h = jp.request_hash(req)
                out.append({"qid": d["qid"], "pmids": d["pmids"], "request": req, "request_hash": h,
                            "config": f"{arm}|{d['config']}", "seed": d["seed"], "arm": arm, "judge": short,
                            "path": OUT / arm / short / f"{d['qid'].replace('/', '_')}.{h}.json"})
    return out


def grades() -> pd.DataFrame:
    rows = []
    for p in OUT.glob("*/*/*.json"):
        if p.name.endswith((".failed.json", ".error.json")):
            continue
        d = json.loads(p.read_text())
        content = (jp.choice(d["response"]).get("message") or {}).get("content")
        arm, short = d["config"].split("|")[0], p.parent.name
        rows += [(arm, short, d["qid"], pmid, g) for pmid, (g, _) in jp.parse_grades(content, d["pmids"]).items()]
    return pd.DataFrame(rows, columns=["arm", "judge", "qid", "pmid", "grade"]).drop_duplicates(
        ["arm", "judge", "qid", "pmid"])


def compare(cal: pd.DataFrame) -> pd.DataFrame:
    orig = pd.read_parquet(jp.HAYSTACKS / HAY / "judgments.parquet")
    orig["judge"] = orig.judge.map({s["model"]: k for k, s in jp.JUDGES.items()})
    orig["arm"] = "original"
    g = pd.concat([orig[["arm", "judge", "qid", "pmid", "grade"]], grades()])
    w = g.pivot_table(index=["arm", "qid", "pmid"], columns="judge", values="grade").dropna().reset_index()
    w["judges"] = np.floor(w[list(jp.JUDGES)].mean(axis=1))
    d = cal.dropna(subset=["dan"]).merge(w, on=["qid", "pmid"])
    rows = []
    for arm, x in d.groupby("arm"):
        for who in [*jp.JUDGES, "judges"]:
            rows.append({"arm": arm, "judge": who, "n": len(x), "exact": (x[who] == x.dan).mean(),
                         "relevant": ((x[who] >= 1) == (x.dan >= 1)).mean(),
                         "lower": int((x[who] < x.dan).sum()), "higher": int((x[who] > x.dan).sum()),
                         "mean_grade": x[who].mean(), "dan_mean": x.dan.mean()})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    cal = calibration()
    todo = [b for b in batches(cal) if not b["path"].exists()]
    print(f"{len(todo)} requests pending, list-price ceiling ${sum(jp.price_ceiling(b['request']) for b in todo):.2f}")
    if todo and a.budget > 0:
        spent = asyncio.run(jp.send_all(jp.read_env(jp.OR_ENV), todo, a.budget, a.concurrency))
        print(f"spent ${spent:.2f}")
    t = compare(cal)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(CSV, index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
