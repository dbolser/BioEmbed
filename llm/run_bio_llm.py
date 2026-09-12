"""Run the five BioEmbed bio tasks against an LLM judge.

Adapted from vendor/embedders-dilemma/llm_judge/main.py (pure_llm mode only).
Results go to results/llm/<model-slug>/ under the repo root.

Usage (from llm/):
    uv run --env-file .env python run_bio_llm.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
VENDOR_ROOT = REPO_ROOT / "vendor" / "embedders-dilemma"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load_env_file() -> None:
    """Load llm/.env into os.environ (existing env vars win).

    llm_judge instantiates Settings() at import time with env_file=".env"
    relative to the CWD; loading here makes the run CWD-independent.
    """
    env_path = HERE / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file()

from typing import Any

import numpy as np
import mteb
from mteb.models.model_meta import ModelMeta

from llm_judge.settings import Settings

from tasks_bio import BIO_TASKS


class _DummyEncoder:
    """Encoder stub for our LLM tasks, which override _evaluate_subset and never
    encode. mteb>=2.6 calls meta.load_model() before running, so the loader
    points back at this instance to avoid a HuggingFace lookup for the anonymous
    name. mteb_model_meta is a settable attribute because load_model() reassigns it.
    """

    def __init__(self) -> None:
        meta = ModelMeta._from_hub(None)
        meta.loader = lambda *args, **kwargs: self
        self.mteb_model_meta = meta

    def encode(self, inputs: Any, **kwargs: Any) -> np.ndarray:
        n = len(inputs) if hasattr(inputs, "__len__") else 1
        return np.zeros((n, 1), dtype=np.float32)

    def similarity(self, embeddings1: Any, embeddings2: Any) -> np.ndarray:
        return np.zeros((len(embeddings1), len(embeddings2)), dtype=np.float32)

    def similarity_pairwise(self, embeddings1: Any, embeddings2: Any) -> np.ndarray:
        return np.zeros(len(embeddings1), dtype=np.float32)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="task class names to run (default: all five)")
    args = ap.parse_args()

    settings = Settings()

    if settings.token in ("", "dummy") or settings.token.endswith("..."):
        sys.exit(
            "run_bio_llm.py: settings.token is a placeholder "
            f"({settings.token!r}). Put a real API key in llm/.env "
            "(see llm/.env.example) before running."
        )

    model_slug = settings.model.replace("/", "__")
    model_folder = REPO_ROOT / "results" / "llm" / model_slug
    model_folder.mkdir(parents=True, exist_ok=True)

    tasks = [cls() for cls in BIO_TASKS
             if args.tasks is None or cls.__name__ in args.tasks]
    print(f"Running {len(tasks)} bio tasks against {settings.model} "
          f"(base_url={settings.base_url}, max_concurrency={settings.max_concurrency})")
    print(f"Results -> {model_folder}")

    cache = mteb.cache.ResultCache(model_folder)
    mteb.evaluate(
        model=_DummyEncoder(),
        tasks=tasks,
        cache=cache,
    )


if __name__ == "__main__":
    main()
