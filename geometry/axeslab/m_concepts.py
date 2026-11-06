"""Concepts family: draw each subject's map on MeSH/GO concept directions (al.concepts(): 71,324 centred, unit
'name: definition' vectors), named by the concepts themselves.

Every paper is tagged with its 10 nearest concepts (cosine, paper centred by the corpus mean, as the manifold work
placed concepts). A subject's vocabulary = concepts among the 10 of at least max(3, 0.5% of) its papers (~200); only
these may build an axis, so no axis is named after a concept the subject's papers never touch.

Variants, recorded as 'concepts_<variant>' for every subject:
  spread     single concepts. x = the vocabulary concept along which the subject's papers spread most relative to
             50k random human-gene papers (variance ratio); y = the same ratio for the displayed (perpendicular)
             direction, among concepts with |cos| <= 0.5 to x and |r| <= 0.3 with x on the subject's papers.
  pcends     concept A minus concept B, for PC1 (x) and PC2 (y) of the subject's centred papers: A = the concept
             tagged most more often in the top 10% than the bottom 10% of papers, B the reverse.
  topicpair  concept contrasts between topics: every pair of the map's topics gives one (concept most over-tagged
             in one topic minus the one most over-tagged in the other). x = the pair whose centroids lie farthest
             apart; y = the contrast that, made perpendicular to x, best separates the topics (the harness eta2 in
             closed form), same |cos| and |r| limits.
Tried and dropped: the single-concept pair that best separates the topics (eta2 a little higher, but it picks
concepts such as 'Cytochrome-c Oxidase Deficiency' or 'Forensic Genetics' whose names do not match their ends), and
the best-eta2 pair of topic contrasts (stability 0.69/0.54 against 1.00/1.00).

Names: a contrast's poles are B (minus) and A (plus). A single concept names its plus end; its minus end gets the
concept most over-tagged in the bottom 10% of the displayed axis if that tags >= 8% of those papers, else 'not <A>'.
Checks: each pole concept's tag rate among the 10% of papers at its own end and at the other end, and an LLM judge
(cached, ~$0.0004 a call) on the 8 most extreme and 8 random titles of each end. Stability: HALVES random halves of
each subject's papers, |cos| of the half-fit vs full-fit displayed axes (x, and y made perpendicular to x).

Run: cd geometry/axeslab && nice -n 10 ../../.venv/bin/python m_concepts.py [--dry]
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "8"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from functools import cache  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import axislab as al  # noqa: E402

torch.set_num_threads(8)
C32, TABLE = al.concepts()
CT = torch.from_numpy(C32)
NAMES = TABLE.name.values
K_TAG, MIN_SHARE, MAX_COS, MAX_R, MIN_NAME_HIT, HALVES = 10, 0.005, 0.5, 0.3, 0.08, 10


def tags(X: np.ndarray) -> np.ndarray:
    """(n x K_TAG) indices of each paper's nearest concepts (cosine after centring the paper by the corpus mean)."""
    Xc = X - al.MU
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
    Xt = torch.from_numpy(Xc.astype(np.float32))
    return torch.cat([torch.topk(Xt[i:i + 4000] @ CT.T, K_TAG, dim=1).indices
                      for i in range(0, len(Xt), 4000)]).numpy()


def vocab(tg: np.ndarray) -> np.ndarray:
    cnt = np.bincount(tg.ravel(), minlength=len(C32))
    return np.flatnonzero(cnt >= max(3, MIN_SHARE * len(tg)))


def rate(tg: np.ndarray, m: np.ndarray, voc: np.ndarray) -> np.ndarray:
    """Share of the papers in mask m that have each concept of voc among their tags."""
    pos = np.full(len(C32), -1)
    pos[voc] = np.arange(len(voc))
    p = pos[tg[m]]  # a paper's tags are distinct, so each counts once
    return np.bincount(p[p >= 0], minlength=len(voc)) / max(m.sum(), 1)


def ends(s, q=0.1):
    return s <= np.quantile(s, q), s >= np.quantile(s, 1 - q)


@cache
def bg_cov() -> np.ndarray:
    return np.cov(al.background().astype(np.float64), rowvar=False)


def scatter(X: np.ndarray, c: np.ndarray):
    """Covariance of all papers, and (as the harness's eta2 counts them: topic papers only) total and between-topic
    scatter."""
    X = X.astype(np.float64)
    k = c >= 0
    Xk = X[k] - X[k].mean(0)
    ids, inv = np.unique(c[k], return_inverse=True)
    H = np.eye(len(ids))[inv]
    n = H.sum(0)
    M = H.T @ Xk / n[:, None]
    return np.cov(X, rowvar=False, bias=True), Xk.T @ Xk, (M * n[:, None]).T @ M


def pair_q(U, G, M):
    """(u_j - g u_i)' M (u_j - g u_i) for every x = u_i, y = u_j (g = cos(u_i, u_j)): the displayed y direction
    (unnormalised) as a quadratic form; also U M U'."""
    Q = U @ M @ U.T
    d = np.diag(Q)
    return d[None, :] - 2 * G * Q + G ** 2 * d[:, None], Q


def pairs(U, Sc, St, Sb):
    """eta2 (the harness formula) and r_xy of every ordered pair of unit directions in U: x = row, y = column made
    perpendicular to x."""
    G = U @ U.T
    cy, Qc = pair_q(U, G, Sc)
    ty, Qt = pair_q(U, G, St)
    by, Qb = pair_q(U, G, Sb)
    cx = np.diag(Qc)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):  # an axis against itself is 0/0; never allowed
        eta = (np.diag(Qb)[:, None] / cx + by / cy) / (np.diag(Qt)[:, None] / cx + ty / cy)
        r = (Qc - G * np.diag(Qc)[:, None]) / np.sqrt(cx * cy)
    return eta, r, G, cy


def allowed(G, r):
    return (np.abs(G) <= MAX_COS) & (np.abs(r) <= MAX_R) & ~np.eye(len(G), dtype=bool)


def fit_spread(X, c, tg, tname=str):
    voc = vocab(tg)
    U = C32[voc].astype(np.float64)
    eta, r, G, cy = pairs(U, *scatter(X, c))
    by_bg, Qbg = pair_q(U, G, bg_cov())
    ratio1 = np.diag(U @ np.cov(X.astype(np.float64), rowvar=False) @ U.T) / np.diag(Qbg)
    i = int(np.argmax(ratio1))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio2 = np.where(allowed(G, r)[i], cy[i] / by_bg[i], -np.inf)
    j = int(np.argmax(ratio2))
    return [U[i], U[j]], [voc[i], voc[j]], [
        f"concept '{NAMES[voc[i]]}': the subject's papers spread {ratio1[i]:.2f}x as widely as random human-gene papers "
        f"along it, the most of {len(voc)} vocabulary concepts",
        f"concept '{NAMES[voc[j]]}': widest spread relative to random papers ({ratio2[j]:.2f}x) of the displayed direction, "
        f"among concepts with |cos| <= {MAX_COS} to x and |r| <= {MAX_R}"], {"eta2_pair": float(eta[i, j])}


def contrast(tg, m_plus, m_minus, voc):
    """A = concept most over-tagged in papers m_plus against m_minus, B the reverse, and the two rate differences."""
    d = rate(tg, m_plus, voc) - rate(tg, m_minus, voc)
    return voc[int(np.argmax(d))], voc[int(np.argmin(d))], float(d.max()), float(-d.min())


def fit_pcends(X, c, tg, tname=str):
    voc = vocab(tg)
    Xc = X.astype(np.float64) - X.mean(0)
    pcs = np.linalg.eigh(Xc.T @ Xc)[1][:, ::-1][:, :2].T
    vecs, ids, why = [], [], []
    for k, pc in enumerate(pcs):
        lo, hi = ends(X @ pc)
        a, b, da, db = contrast(tg, hi, lo, voc)
        v = C32[a].astype(np.float64) - C32[b]
        vecs.append(v)
        ids.append((a, b))
        why.append(f"PC{k + 1} of the subject's papers as a concept contrast: '{NAMES[a]}' (tags {da:.0%} more of the top "
                   f"10% than of the bottom 10%) minus '{NAMES[b]}' ({db:.0%} the other way); cos(A, B) "
                   f"{C32[a] @ C32[b]:.2f}, |cos| of A - B with PC{k + 1} {abs(al.unit(v) @ pc):.2f}")
    return vecs, ids, why, {}


def fit_topicpair(X, c, tg, tname=str):
    voc = vocab(tg)
    ids_t = np.unique(c[c >= 0])
    M = np.array([X[c == t].mean(0) for t in ids_t]).astype(np.float64)
    cand = []
    for i in range(len(ids_t)):
        for j in range(i + 1, len(ids_t)):
            a, b, da, db = contrast(tg, c == ids_t[i], c == ids_t[j], voc)
            if a != b and (a, b) not in [x[:2] for x in cand]:
                cand.append((a, b, ids_t[i], ids_t[j], da, db, np.linalg.norm(M[i] - M[j])))
    V = np.array([al.unit(C32[a].astype(np.float64) - C32[b]) for a, b, *_ in cand])
    eta, r, G, _ = pairs(V, *scatter(X, c))
    i = int(np.argmax([x[-1] for x in cand]))
    j = int(np.argmax(np.where(allowed(G, r)[i], eta[i], -np.inf)))
    why = [f"topics '{tname(t1)}' vs '{tname(t2)}' as a concept contrast: '{NAMES[a]}' (tags {da:.0%} more of the first "
           f"topic's papers) minus '{NAMES[b]}' ({db:.0%} more of the second's); " + lab
           for (a, b, t1, t2, da, db, _), lab in zip((cand[i], cand[j]), (
               f"the two topics whose centroids lie farthest apart (of {len(cand)} topic-pair contrasts)",
               f"of the {len(cand)} topic-pair contrasts, the best topic separation with x (eta2 {eta[i, j]:.3f})"))]
    return [V[i], V[j]], [cand[i][:2], cand[j][:2]], why, {"eta2_pair": float(eta[i, j])}


FIT = {"spread": fit_spread, "pcends": fit_pcends, "topicpair": fit_topicpair}


def shown(vecs):
    return al.unit(vecs[0]), al.perp(vecs[1], vecs[0])


def name_and_check(S, tg, voc, a, b, ids):
    """Names for the two displayed axes from their concepts, and how often each pole concept tags the papers at its
    own end and at the other end (10% each)."""
    names, checks = [], []
    for v, cid in zip((a, b), ids):
        lo, hi = ends(S["X"] @ v)
        plus, minus = cid if isinstance(cid, tuple) else (cid, None)
        if minus is None:
            cand = voc[int(np.argmax(rate(tg, lo, voc) - rate(tg, hi, voc)))]
            minus = cand if rate(tg, lo, np.array([cand]))[0] >= MIN_NAME_HIT else None
        pm = np.array([plus] + ([minus] if minus is not None else []))
        rh, rl = rate(tg, hi, pm), rate(tg, lo, pm)
        ck = {"plus": NAMES[plus], "plus_at_plus": round(float(rh[0]), 3), "plus_at_minus": round(float(rl[0]), 3)}
        if minus is not None:
            ck |= {"minus": NAMES[minus], "minus_at_minus": round(float(rl[1]), 3), "minus_at_plus": round(float(rh[1]), 3)}
        checks.append(ck)
        mname = NAMES[minus] if minus is not None else f"not {NAMES[plus]}"
        names.append({"minus": mname, "plus": NAMES[plus], "axis": f"{mname} ↔ {NAMES[plus]}"})
    return names, checks


def judge(S, v, nm):
    """LLM check: do the names fit the papers at each end (8 most extreme + 8 random from each 10%)?"""
    s = S["X"] @ v
    t = np.asarray(S["titles"], object)
    o = np.argsort(s)
    lo, hi = ends(s)
    rng = np.random.default_rng(0)
    pick = lambda m, ext: list(t[ext[:8]]) + list(rng.choice(t[m], 8, replace=False))  # noqa: E731
    tl, th = pick(lo, o), pick(hi, o[::-1])
    prompt = (f"PubMed papers about {S['about']} are ordered along one axis. Proposed names for the two ends: "
              f"low end '{nm['minus']}', high end '{nm['plus']}'.\nLow-end titles:\n" + "\n".join(f"- {x[:150]}" for x in tl) +
              "\nHigh-end titles:\n" + "\n".join(f"- {x[:150]}" for x in th) +
              '\nDoes each name describe what its end\'s papers are about, as opposed to the other end? '
              'Answer with only JSON: {"minus": "fits|partly|no", "plus": "fits|partly|no", '
              '"better_minus": "<2-5 words>", "better_plus": "<2-5 words>"}')
    return al._llm(prompt)


def stability(S, tg, variant, full):
    n = len(S["X"])
    a0, b0 = full
    out = []
    for seed in range(HALVES):
        h = np.sort(np.random.default_rng(seed).choice(n, n // 2, replace=False))
        a, b = shown(FIT[variant](S["X"][h], S["c"][h], tg[h])[0])
        out.append((abs(a @ a0), abs(b @ b0)))
    o = np.array(out)
    return {"cos_x": float(np.median(o[:, 0])), "cos_y": float(np.median(o[:, 1])),
            "min_cos_x": float(o[:, 0].min()), "min_cos_y": float(o[:, 1].min())}


def main(dry=False):
    summary = {v: [] for v in FIT}
    for name in al.SUBJECTS:
        S = al.load(name)
        tnames = {cl["id"]: cl["name"] for cl in S["clusters"]}
        t0 = time.time()
        tg = tags(S["X"])
        t_tag = time.time() - t0
        voc = vocab(tg)
        for v, fit in FIT.items():
            t1 = time.time()
            vecs, ids, why, info = fit(S["X"], S["c"], tg, tname=lambda t: tnames.get(int(t), str(t)))
            t_fit = time.time() - t1
            a, b = shown(vecs)
            names, checks = name_and_check(S, tg, voc, a, b, ids)
            stab = stability(S, tg, v, (a, b))
            verdict = [judge(S, w, nm) for w, nm in zip((a, b), names)]
            note = (f"vocabulary {len(voc)} concepts; tagging {len(tg):,} papers {t_tag:.1f} s, fit {t_fit * 1000:.0f} ms; "
                    f"stability over {HALVES} random halves, |cos| half vs full median (min): x {stab['cos_x']:.2f} "
                    f"({stab['min_cos_x']:.2f}), y {stab['cos_y']:.2f} ({stab['min_cos_y']:.2f}); tag rate of each pole "
                    f"concept at its own end and the other end: {json.dumps(checks)}; LLM judge of the names: "
                    f"{json.dumps(verdict)}")
            if dry:
                m = al.metrics(S, S["X"] @ a, S["X"] @ b)
            else:
                m = al.record(f"concepts_{v}", S, vecs, why=why, names=names, note=note)["metrics"]
            summary[v].append({"subject": name, "names": names, "metrics": m, "stab": stab, "checks": checks,
                               "judge": verdict, "t_tag": round(t_tag, 2), "t_fit": round(t_fit, 3), "why": why})
            print(f"{v:9s} {name:12s} x: {names[0]['axis'][:75]}\n{'':23s}y: {names[1]['axis'][:75]}\n{'':23s}{m}  "
                  f"stab x {stab['cos_x']:.2f} y {stab['cos_y']:.2f}  fit {t_fit * 1000:.0f} ms  tag {t_tag:.1f} s  "
                  f"judge {[(j.get('minus'), j.get('plus')) for j in verdict]}", flush=True)
    for v, rows in summary.items():
        fits = [j[k] for r in rows for j in r["judge"] for k in ("minus", "plus")]
        print(f"\n{v}: median eta2 {np.median([r['metrics']['eta2'] for r in rows]):.3f}; median |r_xy| "
              f"{np.median([abs(r['metrics']['r_xy']) for r in rows]):.2f}; stability median over subjects: x "
              f"{np.median([r['stab']['cos_x'] for r in rows]):.2f}, y {np.median([r['stab']['cos_y'] for r in rows]):.2f}; "
              f"judge on {len(fits)} pole names: {', '.join(f'{k} {fits.count(k)}' for k in ('fits', 'partly', 'no'))}")
    if not dry:
        (al.RESULTS / "concepts_summary.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main(dry="--dry" in sys.argv)
