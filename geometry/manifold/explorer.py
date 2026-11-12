"""Manifold explorer: navigate the space of biological ideas in the human-gene literature.

Everything happens in the centred 384-D MedEmbed space (paper vectors minus the paper mean; text typed in minus the
concept mean, the centring that put 60% of MeSH concepts on their own papers' centroid, concepts.py).
  locate   text, a topic or a PMID -> nearest topics, concepts (MeSH, GO), papers, and its coordinates on the named axes
  walk     step a point along a named axis (within-topic themes W*, between-topic B*, time, analogy themes A*)
  analogy  c + (b - a), each of a, b, c text or a topic
  path     shortest route between two topics over the topic neighbour graph
  pairs    diverse pairs of neighbouring topics whose step points along an axis (one per map region)
Results always name real topics, concepts and papers near the point, and say how far the point is from the
nearest paper (cosine), so a walk that leaves the papers behind shows it. Data: atlas.py.
Run (root env): uv run --with fastapi --with uvicorn python geometry/manifold/explorer.py --port 8768
"""
import argparse, json, sys, threading
from functools import lru_cache
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from sklearn.cluster import KMeans

REPO = Path(__file__).resolve().parents[2]
A = REPO / "data/geometry/medembed_small/manifold/atlas"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "genesearch"))
torch.set_num_threads(8)

print("loading atlas ...", flush=True)
P = np.load(A / "papers.npz")
pmean = P["mean"].astype(np.float32)
Xc = torch.from_numpy(P["vec"].astype(np.float32) - pmean)
Xc = torch.nn.functional.normalize(Xc, dim=1)
p_pmid, p_year, p_topic, p_x, p_y = P["pmid"], P["year"], P["topic"], P["x"], P["y"]
titles = pd.read_parquet(A / "titles.parquet").title.values
TOP = pd.read_parquet(A / "topics.parquet")
Tc = np.load(A / "topics.npy"); Tn = Tc / np.linalg.norm(Tc, axis=1, keepdims=True)
CON = pd.read_parquet(A / "concepts.parquet")
Craw = np.load(A / "concepts.npy"); cmean = Craw.mean(0)
Cn = Craw - cmean; Cn /= np.linalg.norm(Cn, axis=1, keepdims=True)
AX = json.load(open(A / "axes.json")); AV = np.load(A / "axes.npy")
AXI = {a["id"]: i for i, a in enumerate(AX)}
MAP = np.load(A / "map.npz")
# coordinates are z-scores against the population a start point belongs to: typed text and concepts against
# the concepts, topics and papers against the papers (a short text and an abstract do not share an origin)
_cp = (Craw - cmean) @ AV.T
CSTAT = (_cp.mean(0), _cp.std(0))
_pp = (P["vec"][:100_000].astype(np.float32) - pmean) @ AV.T
PSTAT = (_pp.mean(0), _pp.std(0))
pmid_row = {int(p): i for i, p in enumerate(p_pmid)}
# the learned dictionary (sae.py): which of its 4,096 "ideas" a point is made of
SAE = None
if (A / "sae.pt").exists():
    _ck = torch.load(A / "sae.pt", weights_only=False)  # our own checkpoint (sae.py)
    SAE = {k: v.float() for k, v in _ck["state"].items()}
    SAE_K = int(_ck["k"])
    _sf = pd.read_csv(A / "sae_features.csv").set_index("feature")
    SAE_NAME = {int(f): " · ".join(h.rsplit(" (", 1)[0] for h in str(m).split(" | ")[:3]) for f, m in _sf.mesh.items()}


def ideas(v: np.ndarray, n=6) -> list:
    """Top active SAE features of a point (direction only: the SAE was trained on unit-average-length inputs)."""
    if SAE is None: return []
    x = torch.from_numpy((v / max(np.linalg.norm(v), 1e-9)).astype(np.float32))[None]
    pre = torch.relu((x - SAE["b_dec"]) @ SAE["W_enc"].T + SAE["b_enc"])[0]
    val, idx = pre.topk(n)
    return [dict(feature=int(i), act=round(float(a), 3), name=SAE_NAME.get(int(i), "")) for a, i in zip(val, idx) if a > 0]
# topic graph: each topic to its 10 nearest; edge weight 1 - cos
sim = Tn @ Tn.T; np.fill_diagonal(sim, -1)
nb = np.argsort(-sim, 1)[:, :10]
r, c = np.repeat(np.arange(len(Tn)), 10), nb.ravel()
G = csr_matrix((np.maximum(1 - sim[r, c], 1e-6), (r, c)), shape=sim.shape); G = G.maximum(G.T)
# the steps along that graph's edges (the between-topic vectors), and 20 coarse map regions (k-means of the topic
# directions, as in relations.py) so that a list of pairs can be kept to one per region
STEP_E = np.unique(np.sort(np.stack([r, c], 1), 1), axis=0)
STEP_U = Tc[STEP_E[:, 1]] - Tc[STEP_E[:, 0]]; STEP_U /= np.linalg.norm(STEP_U, axis=1, keepdims=True)
REGION = KMeans(20, n_init=4, random_state=0).fit_predict(Tn)
print(f"ready: {len(p_pmid):,} papers, {len(TOP)} topics, {len(CON):,} concepts, {len(AX)} axes", flush=True)

_model, _lock = None, threading.Lock()


@lru_cache(maxsize=2048)
def embed(text: str) -> tuple:
    global _model
    from semantic import load_model
    with _lock:
        if _model is None: _model = load_model(4)
        v = _model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
    return tuple(v)


def point(spec: dict) -> tuple[np.ndarray, str, str]:
    """A start point in centred space from {"text"}, {"topic"}, {"pmid"} or {"concept"} (index into CON),
    with the population it is scored against ("paper" or "concept")."""
    if spec.get("topic") is not None:
        t = int(spec["topic"]); return Tc[t].copy(), f"topic: {TOP.label[t]}", "paper"
    if spec.get("pmid"):
        i = pmid_row.get(int(spec["pmid"]))
        if i is None: raise HTTPException(404, "PMID not in the 1.5M-paper sample")
        return (P["vec"][i].astype(np.float32) - pmean), f"PMID {spec['pmid']}", "paper"
    if spec.get("concept") is not None:
        j = int(spec["concept"]); return Craw[j] - cmean, f"{CON.source[j].upper()}: {CON.name[j]}", "concept"
    text = (spec.get("text") or "").strip()
    if not text: raise HTTPException(400, "empty")
    # typed text is concept-like: centre it like the concepts
    return np.array(embed(text), np.float32) - cmean, f"“{text}”", "concept"


def describe(v: np.ndarray, n_papers=8, pop="paper") -> dict:
    vn = v / max(np.linalg.norm(v), 1e-9)
    ts = Tn @ vn; ti = np.argsort(-ts)[:8]
    cs = Cn @ vn; ci = np.argsort(-cs)[:12]
    ps = (Xc @ torch.from_numpy(vn)).numpy(); pi = np.argpartition(-ps, 40)[:40]; pi = pi[np.argsort(-ps[pi])]
    m_, s_ = CSTAT if pop == "concept" else PSTAT
    z = (v @ AV.T - m_) / s_
    coords = {a["id"]: round(float(z[i]), 2) for i, a in enumerate(AX) if "analogy" not in a["kind"]}
    return dict(
        topics=[dict(topic=int(t), label=str(TOP.label[t]), blurb=str(TOP.blurb[t]), sim=round(float(ts[t]), 3),
                     size=int(TOP["size"][t])) for t in ti],
        concepts=[dict(idx=int(j), name=str(CON.name[j]), source=str(CON.source[j]), sim=round(float(cs[j]), 3)) for j in ci],
        papers=[dict(pmid=int(p_pmid[i]), year=int(p_year[i]), title=str(titles[i]), sim=round(float(ps[i]), 3),
                     topic=str(TOP.label[p_topic[i]])) for i in pi[:n_papers]],
        xy=[float(np.median(p_x[pi[:15]])), float(np.median(p_y[pi[:15]]))],
        nearest_paper=round(float(ps[pi[0]]), 3), coords=coords, ideas=ideas(v))


app = FastAPI(); app.add_middleware(GZipMiddleware, minimum_size=2000)


@app.get("/")
def index(): return FileResponse(HERE / "explorer.html")


@app.get("/favicon.ico")
def favicon(): return Response(status_code=204)


@app.get("/api/meta")
def meta():
    return dict(axes=AX, topics=TOP[["topic", "label", "blurb", "kind", "size", "x", "y", "median_year"]]
                .fillna("").to_dict("records"))


@app.get("/api/map")
def mapdata():
    b = np.column_stack([MAP["x"], MAP["y"]]).astype(np.float32).tobytes() + MAP["topic"].astype(np.int16).tobytes()
    return Response(b, media_type="application/octet-stream")


@lru_cache(maxsize=64)
def _mapaxis(axis: str) -> bytes:
    """Each background point's coordinate on an axis in units (sd of papers), as int8 = round(z * 40), clipped."""
    i = AXI[axis]; xs = Xc[torch.from_numpy(MAP["idx"].astype(np.int64))].numpy()
    # Xc rows are unit length; scale back to the papers' own centred length is not needed for a z-score
    s = xs @ AV[i]; z = (s - s.mean()) / s.std()
    return np.clip(np.round(z * 40), -127, 127).astype(np.int8).tobytes()


@app.get("/api/mapaxis")
def mapaxis(axis: str):
    if axis not in AXI: raise HTTPException(400, "unknown axis")
    return Response(_mapaxis(axis), media_type="application/octet-stream")


@app.get("/api/concepts")
def concepts(q: str, n: int = 12):
    m = CON[CON.name.str.contains(q, case=False, regex=False)].copy()
    m["len"] = m.name.str.len()
    m = m.sort_values(["len", "npapers"], ascending=[True, False]).head(n)
    return [dict(idx=int(i), name=r.name, source=r.source) for i, r in m.iterrows()]


class Q(BaseModel):
    start: dict = {}
    axis: str | None = None
    steps: list[float] = [-3, -2, -1, 0, 1, 2, 3]
    a: dict = {}
    b: dict = {}
    c: dict = {}
    to: int | None = None
    n: int = 8


def tdict(i: int) -> dict:
    return dict(topic=int(i), label=str(TOP.label[i]), blurb=str(TOP.blurb[i]), x=float(TOP.x[i]), y=float(TOP.y[i]))


@app.post("/api/locate")
def locate(q: Q):
    v, lab, pop = point(q.start); return dict(label=lab, population=pop, **describe(v, pop=pop))


@app.post("/api/walk")
def walk(q: Q):
    if q.axis not in AXI: raise HTTPException(400, "unknown axis")
    v, lab, pop = point(q.start); i = AXI[q.axis]; ax = AV[i]
    # one step = one sd of the start's own population on this axis (analogy themes: one typical hop)
    u = AX[i]["unit"] if "analogy" in AX[i]["kind"] else float((CSTAT if pop == "concept" else PSTAT)[1][i])
    out = []
    for s in q.steps:
        d = describe(v + s * u * ax, n_papers=4, pop=pop); d["step"] = s; out.append(d)
    return dict(label=lab, axis=AX[i], steps=out)


@app.post("/api/analogy")
def analogy(q: Q):
    (va, la, pa), (vb, lb, pb), (vc, lc, pc) = point(q.a), point(q.b), point(q.c)
    res = describe(vc + (vb - va), pop=pc)
    inputs = {int(s["topic"]) for s in (q.a, q.b, q.c) if s.get("topic") is not None}
    res["topics"] = [t for t in res["topics"] if t["topic"] not in inputs]  # as usual for analogies: inputs excluded
    base = describe(vc, n_papers=4, pop=pc)  # what c alone looks like: the honest baseline
    return dict(label=f"{lc} + ({lb} − {la})", result=res, baseline=base)


@app.post("/api/path")
def path(q: Q):
    s, t = int(q.start.get("topic")), int(q.to)
    dist, pred = dijkstra(G, indices=s, return_predecessors=True)
    if not np.isfinite(dist[t]): raise HTTPException(404, "no path")
    route = [t]
    while route[-1] != s: route.append(int(pred[route[-1]]))
    return dict(route=[tdict(i) for i in route[::-1]])


@app.post("/api/pairs")
def pairs(q: Q):
    """Diverse pairs of neighbouring topics whose step points along an axis, from its - end to its + end: the most
    parallel steps first, keeping at most one pair per map region and each topic once."""
    if q.axis not in AXI: raise HTTPException(400, "unknown axis")
    cs = STEP_U @ AV[AXI[q.axis]]; ab = np.abs(cs)
    regions, topics, out = set(), set(), []
    for e in np.argsort(-ab):
        i, j = (int(STEP_E[e, 0]), int(STEP_E[e, 1]))[:: 1 if cs[e] > 0 else -1]
        rg = {int(REGION[i]), int(REGION[j])}
        if rg & regions or {i, j} & topics: continue
        regions |= rg; topics |= {i, j}
        out.append(dict(frm=tdict(i), to=tdict(j), cos=round(float(ab[e]), 3), top_share=float(np.mean(ab >= ab[e]))))
        if len(out) >= max(1, min(q.n, 20)): break
    return dict(axis=AX[AXI[q.axis]], steps=len(STEP_E), random_cos=round(float(np.sqrt(2 / np.pi / AV.shape[1])), 3),
                share_over_03=float(np.mean(ab > 0.3)), pairs=out)


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8768); ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args(); uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
