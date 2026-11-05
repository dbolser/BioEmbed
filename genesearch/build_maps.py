"""Precompute the topic map (the Map and Map Explorer tabs) of every gene and disease into index/cache/maps, so
opening one reads a file instead of making it (up to a minute on a small server; topics named by the LLM).

Run:  uv run python build_maps.py [--genes | --diseases] [--workers N] [--threads T] [--limit N] [--no-llm]

The map is the one a plain search shows (no words or year limits). The same map serves any search without year
limits or keyword-ranked words: items picked in the boxes and 'Named in the abstract' only grey papers out on it,
and semantic or hybrid words keep every paper. Each map comes from app.api_map, the function the page calls, so it
lands under the key the app looks for. Maps already made are kept (a rebuild would move every dot and renumber the
topics in shared links); one named by topic terms (the LLM failed) is named again. Ctrl-C stops after the maps in
progress, and a rerun carries on. About 36,000 maps (~16 GB, ~$12 of LLM names) take a few hours with 14 workers
of 8 threads on the 112-core box. A map's key is its exact list of papers: after an index rebuild that changes a
subject's papers, or a MAP_VERSION bump, its map is made again when opened (or by rerunning this). Map files don't
say whose they are: index/cache/maps_manifest.jsonl lists each subject's key.
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import signal
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

TIMEOUT = 900  # s for one map; a longer one is reported as an error


def threads_env(threads: int):
    """Before numpy loads (here and in the workers, which inherit it): numba (UMAP) gets `threads`, BLAS and
    OpenMP one each. Their default pools (64-112 threads per process) only add CPU when several processes run."""
    os.environ.update(GENESEARCH_MAP_JOBS=str(threads), NUMBA_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="1",
                      OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def init_worker(names_dir: str, no_llm: bool):
    global app, semantic, per_subject
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C: the parent lets the maps in progress finish
    import app
    import litmap
    import semantic
    litmap.NAMES = Path(names_dir) / f"names.{os.getpid()}.json"  # (a shared file is rewritten whole: names get lost)
    if no_llm:
        litmap.LLM_ENV = Path(names_dir) / "no-llm"  # (no key: topics named by their terms)
    per_subject = [f for f in vars(app).values() if callable(f) and getattr(f, "cache_parameters", dict)().get("maxsize")]
    litmap.warm_up()  # compile UMAP once here, not in the first map


def one(sid: str) -> dict:
    """Make (or find) one subject's map the way the page asks for it."""
    t0, built = time.time(), False
    try:
        while True:
            t = time.time()
            if "pending" not in (r := app.api_map(sid)):
                break
            if err := r["pending"].get("error"):
                return {"sid": sid, "status": "error", "error": err}
            if time.time() - t0 > TIMEOUT:
                return {"sid": sid, "status": "error", "error": f"no map after {TIMEOUT} s"}
            built = True
            time.sleep(max(0.3, 3 * (time.time() - t)))  # (a call works out the list again: 1.6 s for 4.4M papers)
        out = {"sid": sid, "n_list": r["n_list"], "n_shown": r["n_shown"]}
        if r.get("too_few"):
            return {**out, "status": "too_few"}
        return {**out, "status": "built" if built else "cached", "key": r["key"], "topics": len(r["clusters"]),
                "named_by": r["named_by"], "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001 (any error: reported; the page makes this map when it is opened)
        return {"sid": sid, "status": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        end = time.time() + TIMEOUT  # a build given up on still runs (and saves): not alongside the next one
        while any(not j.finished for j in semantic.JOBS.values()) and time.time() < end:
            time.sleep(0.5)
        semantic.JOBS.clear()
        for f in per_subject:  # app's bounded caches are per subject (paper sets, tags, BM25...): a worker would
            f.cache_clear()  # otherwise keep the last 128 subjects' (GBs for big diseases)


def default_workers(threads: int) -> int:
    """cores / threads (UMAP keeps a worker's threads about half busy, which leaves room for the live servers),
    and at most one per 6 GB of free memory (each holds ~3 GB of its own; the index pages are shared)."""
    avail = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                 if line.startswith("MemAvailable:")) / 2**20
    return max(1, min((os.cpu_count() or 1) // threads, int(avail // 6)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    kind = ap.add_mutually_exclusive_group()
    kind.add_argument("--genes", action="store_true", help="genes only")
    kind.add_argument("--diseases", action="store_true", help="diseases only")
    ap.add_argument("--threads", type=int, default=8, help="numba threads per worker (default 8)")
    ap.add_argument("--workers", type=int, help="processes (default: cores / threads, fewer if memory is short)")
    ap.add_argument("--limit", type=int, help="only the N biggest subjects (a trial run)")
    ap.add_argument("--no-llm", action="store_true", help="name topics by their terms (no LLM calls)")
    a = ap.parse_args()
    threads_env(a.threads)
    import app
    import litmap
    import semantic

    if not litmap._llm_config() and not a.no_llm:
        sys.exit(f"No LLM key at {litmap.LLM_ENV}: topics would be named by their terms. Add it, or --no-llm.")
    if semantic.source() != "full":
        sys.exit(f"The whole-PubMed vectors ({semantic.FULL_DIR}) aren't loaded: map keys would say 'cpu'.")
    app.MAPS.mkdir(parents=True, exist_ok=True)
    if (free := shutil.disk_usage(app.MAPS).free / 1e9) < 25:
        sys.exit(f"{free:.0f} GB free at {app.MAPS}: all the maps take ~16 GB.")
    subjects = ([] if a.diseases else list(zip(app.GENES.n_papers, app.GENES.hgnc_id))) + \
               ([] if a.genes else list(zip(app.DISEASES.n_papers, app.DISEASES.disease_id)))
    subjects = [s for _, s in sorted(subjects, reverse=True)][:a.limit]  # biggest first: no straggler at the end
    workers = a.workers or default_workers(a.threads)
    manifest = app.MAPS.parent / "maps_manifest.jsonl"
    listed = {(r["sid"], r["key"]) for r in map(json.loads, manifest.open())} if manifest.exists() else set()
    print(f"{len(subjects):,} subjects, {workers} workers x {a.threads} threads, into {app.MAPS}", flush=True)

    t0, last, count, terms = time.time(), time.time(), Counter(), 0

    def tally() -> str:
        return ", ".join(f"{v:,} {k}" for k, v in sorted(count.items()))

    with tempfile.TemporaryDirectory(prefix="build_maps.") as tmp, manifest.open("a") as log:
        try:  # (a worker killed, e.g. out of memory, stops the run here instead of hanging it)
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=init_worker,
                                     initargs=(tmp, a.no_llm)) as pool:
                try:
                    for i, f in enumerate(as_completed([pool.submit(one, s) for s in subjects]), 1):
                        r = f.result()
                        count[r["status"]] += 1
                        if r["status"] == "error":
                            print(f"  error: {r['sid']}: {r['error']}", flush=True)
                        if "key" in r:
                            terms += r["named_by"] == "terms" and r["topics"] > 0
                            if (r["sid"], r["key"]) not in listed:
                                log.write(json.dumps({**r, "t": round(time.time())}) + "\n")
                                log.flush()
                        if time.time() - last > 60 or i == len(subjects):
                            last = time.time()
                            print(f"{i:,}/{len(subjects):,} subjects, {(time.time() - t0) / 60:.0f} min: {tally()}; "
                                  f"{terms:,} named by terms", flush=True)
                except KeyboardInterrupt:  # (else the pool would still make every queued map before stopping)
                    print("Stopping: finishing the maps in progress...", flush=True)
                    pool.shutdown(wait=True, cancel_futures=True)
                    sys.exit("Stopped: rerun to carry on.")
                except BaseException:
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise
        except BrokenProcessPool as e:
            sys.exit(f"A worker died ({e}). The maps made so far are kept: rerun (fewer --workers?).")
    print(f"Done in {(time.time() - t0) / 3600:.1f} h: {tally()}. Manifest: {manifest}")
    if terms and not a.no_llm:
        print(f"{terms:,} maps have topics named by their terms (the LLM failed): rerun to name them.")


if __name__ == "__main__":
    main()
