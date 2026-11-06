"""Shared harness for trying automatic Map Explorer projections. Every method only has to propose directions
(384-d vectors); this module scores, names, plots and records them the same way, so methods can be compared.

Use (root env):  cd geometry/axeslab && ../../.venv/bin/python my_method.py  (data and results: data/geometry/axeslab)
  import axislab as al
  S = al.load("CFTR")                     # X (n x 384, unit vectors), c (topic), year, curated, titles, clusters
  al.record("pca", S, [v1, v2], why=["PC1 of the gene's papers", "PC2"])   # names, metrics, plot, JSON

The explorer (after PR #18) puts a paper at (x, y) = (p . a, p . b_perp), b_perp = b with its part along a taken
out; record() does the same. Subjects: ZCCHC8, SFTPC, CFTR, AQP4 (genes), EFO_0000768 (idiopathic pulmonary
fibrosis). Banks: named_axes() (67 named global axes from the manifold work), concepts() (71k MeSH/GO concept
vectors, centred), sae() (4,096-feature sparse autoencoder), background() (50k random human-gene papers), MU (the
human-gene corpus mean: the space is a cone, centre before PCA-like steps).
"""

import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
from functools import cache
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/geometry/axeslab"  # (made by prep.py; gitignored)
MAN = ROOT / "data/geometry/medembed_small/manifold"
RESULTS = DATA / "results"
VERTEX = ("https://aiplatform.googleapis.com/v1/projects/agentic-ai-iuk-ipf/locations/global/endpoints/openapi/"
          "chat/completions")
LLM_MODEL = "google/gemini-3.1-flash-lite"
SUBJECTS = ["ZCCHC8", "SFTPC", "CFTR", "AQP4", "EFO_0000768"]
MU = np.load(MAN / "mean.npy").astype(np.float32)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]  # the page's
NOISE = "#d6d5cf"


def unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def perp(b, a) -> np.ndarray:
    """b with its part along a taken out, unit length (the explorer's up axis)."""
    a, b = unit(a), unit(b)
    return unit(b - (a @ b) * a)


@cache
def load(name: str) -> dict:
    z = np.load(DATA / "subjects" / f"{name}.npz")
    meta = json.loads((DATA / "subjects" / f"{name}.json").read_text())
    return {**meta, **{k: z[k] for k in z.files}}


@cache
def background() -> np.ndarray:
    return np.load(DATA / "background.npy")


@cache
def named_axes() -> tuple[np.ndarray, list[dict]]:
    """(67 x 384 unit vectors, their metadata: id, kind, name, minus_pole, plus_pole, description, confidence)."""
    return np.load(MAN / "atlas/axes.npy"), json.loads((MAN / "atlas/axes.json").read_text())


@cache
def concepts() -> tuple[np.ndarray, "pd.DataFrame"]:  # noqa: F821
    """(71,324 x 384 'name: definition' vectors, centred by the concept mean and unit length, as the manifold work
    found places them best; the table: source mesh|go, id, name, definition, npapers)."""
    import pandas as pd
    C = np.load(MAN / "concept_def.npy").astype(np.float32)
    C -= C.mean(0)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C, pd.read_parquet(MAN / "concepts.parquet")


@cache
def sae() -> dict:
    """The manifold work's TopK sparse autoencoder: x = (paper - MU) * scale; z = TopK_32(relu((x - b_dec) W_enc^T +
    b_enc)); x_hat = W_dec z + b_dec. Returns W_enc (4096 x 384), b_enc, W_dec (384 x 4096, unit columns), b_dec,
    scale, k, labels (feature -> 'mesh' and 'concept' strings) and encode(X) -> (n x 4096) codes."""
    import pandas as pd
    import torch
    s = torch.load(MAN / "atlas/sae.pt", map_location="cpu", weights_only=False)
    W = {k: v.numpy().astype(np.float32) for k, v in s["state"].items()}
    lab = pd.read_csv(MAN / "atlas/sae_features.csv").set_index("feature")

    def encode(X: np.ndarray) -> np.ndarray:
        pre = np.maximum(((X - MU) * s["scale"] - W["b_dec"]) @ W["W_enc"].T + W["b_enc"], 0)
        thresh = -np.partition(-pre, s["k"] - 1, axis=1)[:, s["k"] - 1:s["k"]]
        return np.where(pre >= thresh, pre, 0)

    return {**W, "scale": s["scale"], "k": s["k"], "labels": lab, "encode": encode}


def scores(S: dict, v) -> np.ndarray:
    return S["X"] @ unit(v)


def total_var(S: dict) -> float:
    return float(S["X"].var(0).sum())


def _spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def pole_terms(S: dict, s: np.ndarray, top: int = 8) -> dict:
    """The most distinctive title words of each end (bottom and top 10% of papers), c-TF-IDF style."""
    from sklearn.feature_extraction.text import CountVectorizer
    lo, hi = s <= np.quantile(s, 0.1), s >= np.quantile(s, 0.9)
    mid = ~(lo | hi)
    skip = {w.lower() for w in re.findall(r"[A-Za-z0-9-]+", S["about"])} | {S["name"].lower()}
    docs = [" ".join(np.asarray(S["titles"], object)[m]) for m in (lo, hi, mid)]
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9-]+\b")
    M = cv.fit_transform(docs).toarray().astype(float)
    vocab = np.array(cv.get_feature_names_out())
    tf = M / M.sum(1, keepdims=True)
    idf = np.log(1 + M.sum(1).mean() / (M.sum(0) + 1))
    out = {}
    for i, k in ((0, "minus"), (1, "plus")):
        order = np.argsort(-(tf[i] * idf))
        out[k] = [w for w in vocab[order] if not set(w.split()) & skip][:top]
    return out


def describe(S: dict, v) -> dict:
    """What one direction means for this subject's papers."""
    s = scores(S, v)
    t = np.asarray(S["titles"], object)
    o = np.argsort(s)
    yr = S["year"] > 0
    z = (s - s.mean()) / s.std()
    topics = {int(k): round(float(z[S["c"] == k].mean()), 2) for k in np.unique(S["c"]) if k >= 0}
    cur = S["curated"].astype(bool)
    return {"titles_minus": [str(x) for x in t[o[:8]]], "titles_plus": [str(x) for x in t[o[::-1][:8]]],
            "terms": pole_terms(S, s), "year_rho": round(_spearman(s[yr], S["year"][yr]), 3),
            "curated_shift_sd": round(float(z[cur].mean() - z[~cur].mean()), 2) if 0 < cur.sum() < len(cur) else None,
            "topic_mean_sd": topics, "var_frac": round(float(s.var() / total_var(S)), 4)}


_token = {"value": "", "until": 0.0}


def _access_token() -> str:
    """A Google Cloud access token (Application Default Credentials, via gcloud), reused for 45 minutes."""
    if time.time() > _token["until"]:
        _token["value"] = subprocess.run(["gcloud", "auth", "application-default", "print-access-token"],
                                         capture_output=True, text=True, check=True).stdout.strip()
        _token["until"] = time.time() + 45 * 60
    return _token["value"]


def _llm(prompt: str) -> dict:
    """The LLM's JSON answer (Gemini Flash-Lite on Vertex AI), cached one file per prompt (several processes may run
    at once)."""
    f = DATA / "llm_cache" / f"{hashlib.sha1(prompt.encode()).hexdigest()}.json"
    if f.exists():
        return json.loads(f.read_text())
    body = {"model": LLM_MODEL, "temperature": 0, "max_tokens": 400,
            "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}}
    req = urllib.request.Request(VERTEX, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {_access_token()}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        text = json.loads(r.read())["choices"][0]["message"]["content"]
    out = json.loads(text.strip().removeprefix("```json").removeprefix("```").removesuffix("```"))
    f.parent.mkdir(exist_ok=True)
    tmp = f.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out))
    os.replace(tmp, f)
    return out


def name_axis(S: dict, d: dict) -> dict:
    """{'minus', 'plus', 'axis'}: short names for the two ends, from their papers (the LLM; cached)."""
    prompt = (f"These are PubMed papers about {S['about']}, ordered along one direction of a text-embedding space. "
              "Name what the papers at each end are about, so a biologist can read the direction as an axis. "
              f"Low end, most extreme titles:\n" + "\n".join(f"- {x[:160]}" for x in d["titles_minus"]) +
              f"\nLow end, distinctive words: {', '.join(d['terms']['minus'])}\n"
              f"High end, most extreme titles:\n" + "\n".join(f"- {x[:160]}" for x in d["titles_plus"]) +
              f"\nHigh end, distinctive words: {', '.join(d['terms']['plus'])}\n"
              'Answer with only JSON: {"minus": "<2-5 words>", "plus": "<2-5 words>", "axis": "<minus> ↔ <plus>, '
              'or a better 3-8 word name for the contrast>"}. Do not put the gene or disease name in them.')
    return _llm(prompt)


def metrics(S: dict, x: np.ndarray, y: np.ndarray) -> dict:
    """How the 2-D projection looks: eta2 = share of the spread that is between the map's topics (higher = topics
    separate); r = correlation of x and y; var = share of the papers' total variance each axis shows."""
    P = np.c_[(x - x.mean()) / x.std(), (y - y.mean()) / y.std()]
    k = S["c"] >= 0
    Pk, ck = P[k], S["c"][k]
    between = sum((ck == t).sum() * ((Pk[ck == t].mean(0) - Pk.mean(0)) ** 2).sum() for t in np.unique(ck))
    tv = total_var(S)
    return {"eta2": round(float(between / ((Pk - Pk.mean(0)) ** 2).sum()), 3), "r_xy": round(float(np.corrcoef(x, y)[0, 1]), 3),
            "var_x": round(float(x.var() / tv), 4), "var_y": round(float(y.var() / tv), 4)}


def plot(S: dict, x, y, xlab: str, ylab: str, title: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7), dpi=100)
    names = {c["id"]: c["name"] for c in S["clusters"]}
    for k in [-1, *sorted(t for t in np.unique(S["c"]) if t >= 0)]:
        m = S["c"] == k
        ax.scatter(x[m], y[m], s=6 if len(x) > 3000 else 14, alpha=0.55, lw=0,
                   c=NOISE if k < 0 else SLOTS[(k - 1) % len(SLOTS)], label=None if k < 0 else f"{k}. {names.get(k, '')}"[:45])
    cur = S["curated"].astype(bool)
    ax.scatter(x[cur], y[cur], s=40, facecolors="none", edgecolors="#222", lw=0.8, label="curated")
    ax.set_xlabel(xlab, fontsize=11)
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="best", markerscale=2, framealpha=0.8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def record(method: str, S: dict, vecs: list, why: list[str], note: str = "", names: list[dict] | None = None) -> dict:
    """Score, name and plot a pair of directions for one subject, as the explorer would draw them (up made
    perpendicular to across), and save results/<method>/<subject>.json + .png. names: optional ready-made
    {'minus','plus','axis'} per axis (e.g. from a named bank); otherwise the LLM names them from their papers."""
    a, b = unit(vecs[0]), perp(vecs[1], vecs[0])
    x, y = scores(S, a), scores(S, b)
    axes = []
    for i, (v, w) in enumerate(zip((a, b), why)):
        d = describe(S, v)
        nm = (names[i] if names and names[i] else None) or name_axis(S, d)
        axes.append({"why": w, "name": nm, **d, "vec": [round(float(t), 6) for t in v]})
    angle = float(np.degrees(np.arccos(np.clip(unit(vecs[0]) @ unit(vecs[1]), -1, 1))))
    res = {"method": method, "subject": S["name"], "about": S["about"], "n": len(x), "note": note,
           "angle_between_raw_axes": round(angle, 1), "metrics": metrics(S, x, y), "axes": axes}
    out = RESULTS / method
    lab = lambda nm: f"{nm['minus']}  ←  →  {nm['plus']}"  # noqa: E731
    plot(S, x, y, lab(axes[0]["name"]), lab(axes[1]["name"]) + " (beyond across)",
         f"{S['about']}: {method}  (topics η²={res['metrics']['eta2']})", out / f"{S['name']}.png")
    res["plot"] = str(out / f"{S['name']}.png")
    (out / f"{S['name']}.json").write_text(json.dumps(res, indent=1))
    return res
