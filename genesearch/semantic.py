"""Meaning (AI embedding) search for genesearch: MedEmbed-small vectors for papers and queries.

Paper vectors come from one of two places. Both use the same model and text ('title\\n\\nabstract'):
  FULL   index/emb_medembed_small/: whole-PubMed float16 .npy shards + pmids.npy (built on a GPU),
         memory-mapped. Used as soon as it is complete; nothing is recomputed.
  CACHE  index/cache/emb/<HGNC id>.npz: vectors computed here on CPU for one gene's papers when first
         needed, by a pool of worker processes, then kept on disk.

Slow work (CPU embedding, maps) runs as a Job the page polls for progress; see Pending.
"""

import hashlib
import multiprocessing as mp
import os
import re
import sys
import threading
import time
import traceback
import types
from collections import deque
from functools import cache, lru_cache
from pathlib import Path

import numpy as np

MODEL = "abhinand/MedEmbed-small-v0.1"
REVISION = "40a5850d046cfdb56154e332b4d7099b63e8d50e"  # the revision mteb's ModelMeta pins
DIM = 384
IDX = Path(__file__).resolve().parent / "index"
FULL_DIR = IDX / "emb_medembed_small"
CACHE_DIR = IDX / "cache"
# CPU embedding: several small processes beat one big one (torch scales poorly past ~8 threads here).
WORKERS = int(os.environ.get("GENESEARCH_EMBED_WORKERS", "12"))
THREADS = int(os.environ.get("GENESEARCH_EMBED_THREADS", "4"))
CHUNK = 64  # papers per worker task


def load_model(threads: int):
    """MedEmbed-small as eval/run_embedders.py get_model loads it: mteb's loader for this model is a
    plain SentenceTransformer at mteb's revision (modules.json: CLS pooling, then Normalize), and
    get_model casts it to fp32 on CPU. No prompt on either side: mteb's ModelMeta, the HF card and
    config_sentence_transformers.json all give none, so queries and papers are encoded as plain text."""
    import torch
    torch.set_num_threads(threads)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL, revision=REVISION, device="cpu").float()


def doc_text(title, abstract) -> str:
    return f"{title or ''}\n\n{abstract or ''}"  # as the GPU-built index


# ---------------------------------------------------------------- query side (in the server process)
_qmodel, _qlock = None, threading.Lock()


@lru_cache(maxsize=512)
def query_vector(text: str) -> np.ndarray:
    global _qmodel
    with _qlock:
        if _qmodel is None:
            _qmodel = load_model(4)
        return _qmodel.encode([text], convert_to_numpy=True)[0].astype(np.float32)


# ---------------------------------------------------------------- jobs the page polls
class Pending(Exception):
    """Raised when an answer needs a background job; the API returns the job's progress instead."""

    def __init__(self, job: "Job"):
        self.job = job


class Stopped(Exception):
    """A job gave up early because nobody was waiting for it (the part done is kept)."""


class Job:
    """A background task: a phase description, done/total counts, then a result or an error.
    If nobody has polled for STALE seconds the task may stop early (see .abandoned). While it
    waits on another job (.sub), its status is that job's."""
    STALE = 60

    def __init__(self, fn):
        self.phase, self.done, self.total, self.note = "Starting", 0, 0, ""
        self.result, self.error, self.finished, self.stopped = None, None, False, False
        self.t0 = self.seen = self.ended = time.time()
        self.rate_t0, self.rate_done, self.sub = None, 0, None
        threading.Thread(target=self._run, args=(fn,), daemon=True).start()

    def _run(self, fn):
        try:
            self.result = fn(self)
        except Stopped as e:
            self.error, self.stopped = str(e), True
        except Exception as e:  # shown on the page
            self.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            self.finished, self.ended = True, time.time()

    def step(self, phase: str, total: int = 0):
        self.phase, self.done, self.total = phase, 0, total
        self.rate_t0, self.rate_done = None, 0

    def advance(self, n: int):
        if self.rate_t0 is None:  # time from the first finished chunk (skips worker start-up)
            self.rate_t0, self.rate_done = time.time(), self.done
        self.done += n

    @property
    def abandoned(self) -> bool:
        return time.time() - self.seen > self.STALE

    def status(self) -> dict:
        self.seen = time.time()
        if self.sub is not None and not self.sub.finished:
            return self.sub.status()
        eta = None
        if self.total and self.rate_t0 and self.done > self.rate_done:
            rate = (self.done - self.rate_done) / max(1e-3, time.time() - self.rate_t0)
            eta = round((self.total - self.done) / rate)
        return {"phase": self.phase, "done": self.done, "total": self.total, "note": self.note,
                "elapsed_s": round(time.time() - self.t0), "eta_s": eta,
                "error": None if self.stopped else self.error}

    def wait_for(self, fn):
        """fn(), waiting out (and showing the progress of) any job it is Pending on."""
        try:
            while True:
                try:
                    return fn()
                except Pending as p:
                    if p.job.error and not p.job.stopped:
                        raise RuntimeError(p.job.error)
                    self.sub = p.job
                    p.job.seen = self.seen  # alive while someone polls this job
                    if self.abandoned:
                        raise Stopped("stopped: nobody was waiting")
                    time.sleep(0.5)
        finally:
            self.sub = None


JOBS: dict = {}
_jobs_lock = threading.Lock()


def job(key, fn) -> Job:
    """The job for this key: the running or finished one, or a new one (also after a failure)."""
    with _jobs_lock:
        for k in [k for k, j in JOBS.items() if j.finished and time.time() - j.seen > 600]:
            del JOBS[k]
        j = JOBS.get(key)  # retry a stopped job at once, a failed one after a pause
        if j is None or (j.finished and j.error and (j.stopped or time.time() - j.ended > 30)):
            j = JOBS[key] = Job(fn)
        return j


def digest(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.tobytes() if isinstance(p, np.ndarray) else repr(p).encode())
    return h.hexdigest()[:20]


# ---------------------------------------------------------------- FULL: the whole-PubMed index
def _natural(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


class FullIndex:
    """index/emb_medembed_small/: pmids.npy (N,) plus float16 (rows, 384) .npy shards whose rows,
    in file-name order, line up with pmids.npy. Memory-mapped: only the rows asked for are read."""

    def __init__(self, path: Path):
        pmids = np.asarray(np.load(path / "pmids.npy", mmap_mode="r"), dtype=np.int64)
        self.shards = []
        for f in sorted(path.glob("*.npy"), key=_natural):
            if f.name != "pmids.npy":
                a = np.load(f, mmap_mode="r")
                if a.ndim == 2 and a.shape[1] == DIM:
                    self.shards.append(a)
        self.starts = np.cumsum([0] + [len(s) for s in self.shards])
        if not self.shards or self.starts[-1] != len(pmids):
            raise ValueError(f"{self.starts[-1]:,} vectors for {len(pmids):,} PMIDs: not finished")
        self.order = None if np.all(pmids[1:] >= pmids[:-1]) else np.argsort(pmids, kind="stable")
        self.pmids = pmids if self.order is None else pmids[self.order]

    def get(self, pmids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pmids = np.asarray(pmids, dtype=np.int64)
        i = np.searchsorted(self.pmids, pmids).clip(0, len(self.pmids) - 1)
        ok = self.pmids[i] == pmids
        rows = (i if self.order is None else self.order[i])[ok]
        out = np.zeros((len(pmids), DIM), np.float32)
        at = np.flatnonzero(ok)
        shard = np.searchsorted(self.starts, rows, side="right") - 1
        for s in np.unique(shard):
            m = shard == s
            local = rows[m] - self.starts[s]
            o = np.argsort(local)  # read each shard front to back
            out[at[m][o]] = self.shards[s][local[o]]
        return out, ok


_full, _full_checked = None, 0.0


def full_index() -> FullIndex | None:
    """The whole-PubMed index once it is complete (looked for at most once a minute)."""
    global _full, _full_checked
    if _full is None and time.time() - _full_checked > 60:
        _full_checked = time.time()
        if (FULL_DIR / "pmids.npy").exists():
            try:
                _full = FullIndex(FULL_DIR)
                print(f"semantic: using {FULL_DIR} ({len(_full.pmids):,} papers)", flush=True)
            except Exception as e:
                print(f"semantic: {FULL_DIR} not usable yet: {e}", flush=True)
    return _full


def source() -> str:
    return "full" if full_index() is not None else "cpu"


# ---------------------------------------------------------------- CACHE: computed here, per gene
class GeneCache:
    """Vectors computed on this machine for one gene's papers (sorted by pmid, float16)."""

    def __init__(self, hgnc_id: str):
        self.path = CACHE_DIR / "emb" / f"{hgnc_id.replace(':', '_')}.npz"
        self.lock, self.save_lock = threading.Lock(), threading.Lock()
        self.pmids, self.vecs, self.mtime = np.zeros(0, np.int64), np.zeros((0, DIM), np.float16), None
        if self.path.exists():
            self.mtime = self.path.stat().st_mtime
            z = np.load(self.path)
            self.pmids, self.vecs = z["pmids"], z["vecs"]

    def get(self, pmids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self.lock:
            p, v = self.pmids, self.vecs
        if not len(p):
            return np.zeros((len(pmids), DIM), np.float32), np.zeros(len(pmids), bool)
        i = np.searchsorted(p, pmids).clip(0, len(p) - 1)
        ok = p[i] == pmids
        return np.where(ok[:, None], v[i], 0).astype(np.float32), ok

    def add(self, pmids: np.ndarray, vecs: np.ndarray):
        with self.lock:
            p, first = np.unique(np.concatenate([self.pmids, pmids]), return_index=True)
            self.pmids, self.vecs = p, np.concatenate([self.vecs, vecs.astype(np.float16)])[first]

    def save(self):
        """Write to disk, first merging in anything another process saved meanwhile. One save at a
        time per gene here; a temp file of its own, so saves from other processes can't collide."""
        with self.save_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_mtime != self.mtime:
                z = np.load(self.path)
                self.add(z["pmids"], z["vecs"])
            with self.lock:
                p, v = self.pmids, self.vecs
            tmp = self.path.with_name(f"{self.path.stem}.{os.getpid()}.{threading.get_ident()}.tmp.npz")
            np.savez(tmp, pmids=p, vecs=v)
            os.replace(tmp, self.path)
            self.mtime = self.path.stat().st_mtime


@cache
def gene_cache(hgnc_id: str) -> GeneCache:
    return GeneCache(hgnc_id)


_pool, _pool_lock = None, threading.Lock()


def _worker_init(threads: int):
    global _wmodel
    _wmodel = load_model(threads)


def _worker_encode(texts: list[str]) -> np.ndarray:
    return _wmodel.encode(texts, batch_size=32, convert_to_numpy=True).astype(np.float16)


def pool():
    global _pool
    with _pool_lock:
        if _pool is None:
            # Spawned workers re-run the parent's __main__ script first (app.py, which loads the
            # whole index); they need only this module, so hide __main__ while they start.
            main, sys.modules["__main__"] = sys.modules["__main__"], types.ModuleType("__main__")
            try:
                _pool = mp.get_context("spawn").Pool(WORKERS, _worker_init, (THREADS,))
            finally:
                sys.modules["__main__"] = main
    return _pool


def _embed(cache: GeneCache, pmids: np.ndarray, texts_fn, job: Job):
    """Compute and cache vectors for the papers not cached yet. Keeps at most WORKERS chunks in
    flight, so a small job started later is not stuck behind a big one; skips papers another job
    has done meanwhile; stops early if nobody is waiting."""
    todo = pmids[~cache.get(pmids)[1]]
    job.note = (f"First time only for these papers: the AI model reads each title and abstract on "
                f"this computer ({len(todo):,} papers). They are saved for next time.")
    job.step("Starting the AI model", len(todo))
    texts = texts_fn(todo)
    order = np.argsort([len(t) for t in texts], kind="stable")  # similar lengths batch with less padding
    chunks = [order[i:i + CHUNK] for i in range(0, len(order), CHUNK)]
    p = pool()
    flight, nxt, since_save = deque(), 0, 0
    while nxt < len(chunks) or flight:
        while nxt < len(chunks) and len(flight) < WORKERS and not job.abandoned:
            idx = chunks[nxt][~cache.get(todo[chunks[nxt]])[1]]
            job.advance(len(chunks[nxt]) - len(idx))
            if len(idx):
                flight.append((idx, p.apply_async(_worker_encode, ([texts[i] for i in idx],))))
            nxt += 1
        if not flight:
            break
        idx, res = flight.popleft()
        cache.add(todo[idx], res.get())
        job.phase = "Reading the papers with the AI model"
        job.advance(len(idx))
        since_save += len(idx)
        if since_save >= 2000:
            cache.save()
            since_save = 0
    cache.save()
    if nxt < len(chunks):
        raise Stopped("stopped: nobody was waiting (the part done is saved)")


def vectors(hgnc_id: str, pmids: np.ndarray, texts_fn) -> tuple[np.ndarray, np.ndarray]:
    """Paper vectors (float32, n x 384, unit length) and a found mask. From the whole-PubMed index
    if it is ready; else from this gene's cache, raising Pending while missing ones are computed.
    texts_fn(pmids) -> list of 'title\\n\\nabstract' strings (only called for missing papers)."""
    pmids = np.asarray(pmids, dtype=np.int64)
    if (full := full_index()) is not None:
        return full.get(pmids)
    cache = gene_cache(hgnc_id)
    vecs, ok = cache.get(pmids)
    if ok.all():
        return vecs, ok
    j = job(("embed", hgnc_id, digest(np.sort(pmids))), lambda jb: _embed(cache, pmids, texts_fn, jb))
    if j.finished and not j.error:  # computed since we looked
        return cache.get(pmids)
    raise Pending(j)
