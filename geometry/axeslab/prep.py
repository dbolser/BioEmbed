"""Extract what the axis experiments need, once: for each subject, its genesearch map (the cached default view,
via app.api_map) and its papers' MedEmbed vectors; plus a background sample of human-gene papers.

Run (genesearch env, from anywhere):
  genesearch/.venv/bin/python geometry/axeslab/prep.py  (from the repo root)
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/geometry/axeslab"
GS = ROOT / "genesearch"
sys.path.insert(0, str(GS))

import numpy as np  # noqa: E402

import app  # noqa: E402
import semantic  # noqa: E402

SUBJECTS = {"ZCCHC8": "gene", "SFTPC": "gene", "CFTR": "gene", "AQP4": "gene",
            "EFO_0000768": "disease"}  # the last: idiopathic pulmonary fibrosis


def sid_of(name, kind):
    return app.GENES.hgnc_id[app.GENES.symbol == name].iat[0] if kind == "gene" else name


def main():
    out = DATA / "subjects"
    out.mkdir(parents=True, exist_ok=True)
    for name, kind in SUBJECTS.items():
        sid = sid_of(name, kind)
        t0 = time.time()
        while "pending" in (m := app.api_map(sid)):
            time.sleep(1)
        pm = np.asarray(m["pmid"], np.int64)
        vecs, ok = semantic.vectors(sid, pm, app.texts_for)
        about = app.map_subject(sid)[0]
        keep = np.flatnonzero(ok)
        np.savez(out / f"{name}.npz", X=vecs[keep].astype(np.float32), pmid=pm[keep],
                 c=np.asarray(m["c"])[keep], year=np.asarray(m["year"])[keep],
                 curated=np.asarray(m["curated"])[keep])
        (out / f"{name}.json").write_text(json.dumps({
            "name": name, "sid": sid, "kind": kind, "about": about, "key": m["key"],
            "titles": [m["title"][i] for i in keep], "n_list": m["n_list"], "n_shown": m["n_shown"],
            "clusters": [{k: c[k] for k in ("id", "n", "name", "terms")} for c in m["clusters"]]}))
        print(f"{name} ({sid}): {len(keep):,} papers with vectors, {len(m['clusters'])} topics, "
              f"{time.time() - t0:.0f} s", flush=True)
    X = np.load(ROOT / "data/geometry/medembed_small/X.npy", mmap_mode="r")
    idx = np.sort(np.random.default_rng(0).choice(len(X), 50_000, replace=False))
    np.save(DATA / "background.npy", np.asarray(X[idx], np.float32))
    print("background: 50,000 random human-gene papers", flush=True)


if __name__ == "__main__":
    main()
