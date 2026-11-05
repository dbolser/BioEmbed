"""Precompute 'Named in the abstract' (app.named_live) for every gene, and with --diseases every disease, into
index/named_pmids.parquet: one row per subject, its sorted named pmids. The app looks them up instead of reading
every abstract of the subject at search time (minutes for the biggest genes on a cloud disk, and it evicts the
other genes from the page cache).

Run:  uv run python build_named.py [--diseases] [--workers N]

Build it where the index is in memory (~4 min for the genes on the 754 GB box, 16 workers of up to ~5 GB each),
then copy the file to a server along with any index files that changed, keeping their times (rsync -a). Rebuild
after a change to the naming code (app.naming_code) or to an index file it reads (app.NAMED_INPUTS, by size and
time): until then the app counts live, and its startup line and /api/about say why.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

APP_SOURCE = (Path(__file__).parent / "app.py").read_text(encoding="utf-8")  # (before importing it: an edit since shows below)

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import app


def named(subject: str) -> tuple[str, np.ndarray | None, str]:
    """(subject, its named pmids, error). Uncached, and the subject's lookups are dropped after: a worker would
    otherwise keep the biggest genes' paper sets."""
    try:
        return subject, app.named_live.__wrapped__(subject).astype(np.int32), ""
    except Exception as e:  # noqa: BLE001 (any error: left out of the table, so the app computes it live)
        return subject, None, f"{type(e).__name__}: {e}"
    finally:
        for f in (app.subject_tags, app.paper_set, app.epmc_links):
            f.cache_clear()


def default_workers() -> int:
    """16, or fewer where memory is short: a worker can take ~5 GB on the biggest subjects."""
    avail = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                 if line.startswith("MemAvailable:")) / 2**20
    return max(1, min(16, os.cpu_count() or 1, int(avail // 6)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--diseases", action="store_true", help="every disease too (genes only by default)")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help="processes, forked after the index is loaded (default: 16, fewer if memory is short)")
    a = ap.parse_args()
    if app.SOURCE != APP_SOURCE:
        sys.exit("app.py changed while it was being imported: run again.")
    t0, stamp = time.time(), app.named_stamp(a.diseases)
    subjects = list(zip(app.GENES.n_papers, app.GENES.hgnc_id))
    if a.diseases:
        subjects += list(zip(app.DISEASES.n_papers, app.DISEASES.disease_id))
    subjects = [s for _, s in sorted(subjects, reverse=True)]  # biggest first: no straggler at the end
    assert len(set(subjects)) == len(subjects), "duplicate subject ids"
    assert app.PMIDS.max() < 2**31, "pmids no longer fit int32"
    print(f"{len(subjects):,} subjects, {a.workers} workers", flush=True)
    out, errors, last = {}, {}, time.time()
    try:  # (a worker killed, e.g. out of memory, stops the build here instead of hanging it)
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork")) as pool:
            for i, f in enumerate(as_completed([pool.submit(named, s) for s in subjects]), 1):
                s, pm, err = f.result()
                if err:
                    errors[s] = err
                else:
                    out[s] = pm
                if time.time() - last > 30 or i == len(subjects):
                    last = time.time()
                    print(f"{i:,}/{len(subjects):,} subjects, {time.time() - t0:.0f} s", flush=True)
    except BrokenProcessPool as e:
        sys.exit(f"A worker died ({e}); nothing written. Fewer --workers?")
    if Path(app.__file__).read_text(encoding="utf-8") != app.SOURCE or app.named_stamp(a.diseases) != stamp:
        sys.exit("The naming code or an index file changed during the build: nothing written, run it again.")
    order = sorted(out)
    table = pa.table({"subject": order, "pmids": pa.array([out[s] for s in order], pa.list_(pa.int32()))})
    table = table.replace_schema_metadata(stamp)
    tmp = app.NAMED_FILE.with_name(f".{app.NAMED_FILE.name}.{os.getpid()}.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, app.NAMED_FILE)  # (a running app keeps what it loaded; a restart reads this)
    n = sum(len(v) for v in out.values())
    print(f"{app.NAMED_FILE}: {len(out):,} subjects, {n:,} named papers, "
          f"{app.NAMED_FILE.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f} s")
    for s, err in list(errors.items())[:20]:
        print(f"  not precomputed: {s}: {err}")
    if errors:
        print(f"{len(errors):,} subjects not precomputed (the app computes them live)")


if __name__ == "__main__":
    main()
