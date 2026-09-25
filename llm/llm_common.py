"""Shared plumbing for the two extra LLM retrieval protocols
(retrieval_ranked.py, rerank_pipeline.py): env loading, local data loading,
result-JSON writing, and an offline stub for the vendor's AsyncOpenAI client.

The stub replaces `llm_judge.llm_client.client` only; send_request /
send_request_multi (prompt assembly, thinking-strip, usage parsing, semaphore)
run unchanged, so a --dry-run exercises the real usage_stats path.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
VENDOR_ROOT = REPO_ROOT / "vendor" / "embedders-dilemma"
DATA_DIR = REPO_ROOT / "data" / "tasks"
for p in (str(VENDOR_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

RETRIEVAL_TASKS = ["GOPubMedRetrieval", "R2MEDBiologyPooled"]


def load_env(dry_run: bool = False) -> None:
    """Load llm/.env into os.environ (existing vars win), as run_bio_llm does.

    With dry_run, fill any missing required Settings fields with dummies so the
    vendor client can be imported without a real key.
    """
    env_path = HERE / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
    if dry_run:
        os.environ.setdefault("token", "dummy")
        os.environ.setdefault("base_url", "https://openrouter.ai/api/v1")
        os.environ.setdefault("model", "stub/dry-run")


def load_task_data(task: str):
    """corpus {id: {title, text}}, queries {id: text}, qrels {qid: {docid: score}}.

    Title is blanked as in tasks_bio.AbsTaskLocalLLMRetrieval (text already
    contains the title where available).
    """
    from datasets import load_from_disk

    dd = load_from_disk(str(DATA_DIR / task))
    corpus = {r["_id"]: {"title": "", "text": r["text"]} for r in dd["corpus"]}
    queries = {r["_id"]: r["text"] for r in dd["queries"]}
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for r in dd["qrels"]:
        qrels[r["query-id"]][r["corpus-id"]] = int(r["score"])
    return corpus, queries, dict(qrels)


def model_slug(settings, suffix: str = "") -> str:
    return settings.model.replace("/", "__") + suffix


def write_result_json(path: Path, task: str, scores: dict, t0: float, extra: dict | None = None) -> None:
    """Same shape as the vendor/mteb result files: {scores: {test: [{...}]}}."""
    import mteb

    row = {**scores, "hf_subset": "default", "languages": ["eng-Latn"]}
    doc = {
        "dataset_revision": "local",
        "task_name": task,
        "mteb_version": mteb.__version__,
        "scores": {"test": [row]},
        "evaluation_time": round(time.time() - t0, 2),
        **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))


# ---------------------------------------------------------------------------
# Offline stub client
# ---------------------------------------------------------------------------

class StubClient:
    """Minimal AsyncOpenAI look-alike: `client.chat.completions.create(**kw)`.

    `respond(messages) -> str` produces the assistant text. Token usage is a
    len/4 proxy; `cached_tokens` mimics prefix caching by reporting the size
    of every message but the last as cached from the second call onwards.
    Records every call in `self.calls` for assertions.
    """

    def __init__(self, respond):
        self._respond = respond
        self.calls: list[dict] = []
        self.api_key = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, *, messages, model, max_tokens, **kwargs):
        content = self._respond(messages)
        n_in = sum(len(m["content"]) for m in messages) // 4
        n_prefix = sum(len(m["content"]) for m in messages[:-1]) // 4
        n_out = max(1, len(content) // 4)
        cached = n_prefix if self.calls else 0
        self.calls.append({"model": model, "n_messages": len(messages), "kwargs": kwargs})
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason="stop")],
            usage=SimpleNamespace(
                prompt_tokens=n_in, completion_tokens=n_out, total_tokens=n_in + n_out,
                prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
        )


def install_stub(respond) -> StubClient:
    """Swap the vendor's module-level AsyncOpenAI client for a StubClient."""
    import llm_judge.llm_client as lc

    stub = StubClient(respond)
    lc.client = stub
    return stub


def load_first_stage(emb_model: str, task: str, top_k: int = 20) -> dict | None:
    p = REPO_ROOT / "results" / "first_stage" / emb_model.replace("/", "__") / f"{task}_top{top_k}.json"
    return json.loads(p.read_text()) if p.exists() else None
