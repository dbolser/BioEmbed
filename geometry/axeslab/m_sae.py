"""SAE family: automatic Map Explorer axes from the manifold work's sparse autoencoder (al.sae(), 4,096 features).

Encode the subject's papers; keep features that fire on 3-70% of them (not near-universal); score each by how much
more it varies here than in 50k random human-gene papers: gain = var_s * log(var_s / var_bg) (variance times log
variance-lift, 0 if it varies less than in the background). Variants (recorded for all subjects as sae_<variant>):
  single    x, y = decoder directions (W_dec columns) of the best feature and the best one whose code correlates
            < 0.3 with it (the commoner of the two goes across)
  contrast  x, y = d_f - d_g for the two best feature pairs (codes correlate < 0.1; pair score = geometric mean of
            the two gains; the second pair shares no feature with the first), so both ends of an axis are ideas
  cpca      x, y = top-2 eigenvectors of the codes' covariance here minus in the background (contrastive PCA in
            feature space), cut to their 6 largest features and mapped back through W_dec
Checks: stability (refit on random halves of the papers, |cos| with the full fit), monosemanticity (share of the
axis' extreme 10% of papers with its feature on; Spearman of the axis with the feature's code; an LLM judge on
whether the extreme titles share one idea and whether it matches the feature's MeSH label).
Run: cd geometry/axeslab && nice -n 10 ../../.venv/bin/python m_sae.py [--dry] [variant ...]
"""

import json
import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "8"

from functools import cache  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

import axislab as al  # noqa: E402

torch.set_num_threads(8)
SAE = al.sae()
D = SAE["W_dec"].astype(np.float64)  # 384 x 4096, unit columns: where each feature writes in paper space
LAB = SAE["labels"]
P_MIN, P_MAX = 0.03, 0.70  # a feature must fire on 3-70% of the subject's papers
MAX_R = 0.3  # single: two features "fire on different papers" if their codes correlate below 0.3
_T = {k: torch.from_numpy(np.asarray(SAE[k], np.float32)) for k in ("W_enc", "b_enc", "b_dec")}
_MU = torch.from_numpy(al.MU)


def encode(X: np.ndarray, chunk: int = 10_000) -> np.ndarray:
    """The same codes as al.sae()['encode'] (checked identical on CFTR), ~10x faster: torch topk, not np.partition."""
    out = []
    for i in range(0, len(X), chunk):
        x = torch.from_numpy(np.asarray(X[i:i + chunk], np.float32))
        pre = torch.relu(((x - _MU) * float(SAE["scale"]) - _T["b_dec"]) @ _T["W_enc"].T + _T["b_enc"])
        v, j = pre.topk(SAE["k"], dim=1)
        out.append(torch.zeros_like(pre).scatter_(1, j, v).numpy())
    return np.concatenate(out)


@cache
def bg() -> tuple:
    """Background codes (sparse, 50k x 4096) and per-feature variance. In the app this is a one-off precompute."""
    B = al.background()
    Zb = sp.vstack([sp.csr_matrix(encode(B[i:i + 10_000])) for i in range(0, len(B), 10_000)]).tocsc()
    m = np.asarray(Zb.mean(0)).ravel()
    return Zb, np.asarray(Zb.multiply(Zb).mean(0)).ravel() - m ** 2


def gain(vs, vb):
    """Variance times log variance-lift over the background; 0 where the subject varies no more than background."""
    vs, vb = np.asarray(vs, float), np.maximum(np.asarray(vb, float), 1e-6)
    return np.where(vs > vb, vs * np.log(np.maximum(vs, 1e-12) / vb), 0.0)


def candidates(Z: np.ndarray, n: int) -> np.ndarray:
    p = (Z > 0).mean(0)
    g = np.where((p >= P_MIN) & (p <= P_MAX), gain(Z.var(0), bg()[1]), 0.0)
    top = np.argsort(-g)[:n]
    return top[g[top] > 0]


def fit_single(Z: np.ndarray) -> dict:
    """The best feature and the best one whose code correlates < MAX_R with it; the commoner one goes across."""
    top = candidates(Z, 60)
    f1 = int(top[0])
    f2 = next(int(f) for f in top[1:] if abs(np.corrcoef(Z[:, f1], Z[:, f])[0, 1]) < MAX_R)
    f1, f2 = sorted((f1, f2), key=lambda f: -(Z[:, f] > 0).mean())
    return {"feats": [[f1], [f2]], "vecs": [D[:, f1], D[:, f2]]}


def fit_contrast(Z: np.ndarray, n: int = 30) -> dict:
    """Pairs (f, g) among the n best features whose codes do not go together (r < 0.1), scored by the geometric
    mean of their gains (both ends must be strong ideas here); the second pair shares no feature with the first."""
    top = candidates(Z, n)
    Zt, g = Z[:, top], gain(Z[:, top].var(0), bg()[1][top])
    R = np.corrcoef(Zt.T)
    pairs = sorted(((float(np.sqrt(g[i] * g[j])), i, j) for i in range(len(top)) for j in range(i + 1, len(top))
                    if R[i, j] < 0.1), reverse=True)
    (_, i, j), out = pairs[0], None
    for _, k, m in pairs:
        if not {k, m} & {i, j} and abs(al.unit(D[:, top[i]] - D[:, top[j]]) @ al.unit(D[:, top[k]] - D[:, top[m]])) < 0.5:
            out = (k, m)
            break
    fe = [[int(top[i]), int(top[j])], [int(top[out[0]]), int(top[out[1]])]]
    fe = [sorted(p, key=lambda f: -(Z[:, f] > 0).mean()) for p in fe]  # the commoner feature at the + end
    return {"feats": fe, "vecs": [D[:, p] - D[:, q] for p, q in fe]}


def fit_cpca(Z: np.ndarray, n: int = 300, m: int = 6) -> dict:
    """Contrastive PCA of the codes (subject covariance minus background covariance), sparsified to m features."""
    top = candidates(Z, n)
    C = np.cov(Z[:, top].T) - np.cov(bg()[0][:, top].toarray().T)
    w, V = np.linalg.eigh(C)
    feats, vecs = [], []
    for k in (-1, -2):
        l = V[:, k]
        keep = np.argsort(-np.abs(l))[:m]
        sgn = np.sign(l[keep[0]])
        feats.append([(int(top[i]), round(float(sgn * l[i]), 3)) for i in keep])
        vecs.append(D[:, top[keep]] @ (sgn * l[keep]))
    return {"feats": feats, "vecs": vecs}


FITS = {"single": fit_single, "contrast": fit_contrast, "cpca": fit_cpca}


def plotted(vecs) -> tuple:
    """The two directions as the explorer draws them (up made perpendicular to across)."""
    return al.unit(vecs[0]), al.perp(vecs[1], vecs[0])


def stability(fit, Z: np.ndarray, full: dict, n_halves: int = 5) -> tuple:
    """|cos| between half-fit and full-fit plotted axes for n random halves (n_halves x 2), and the halves' features."""
    a, b = plotted(full["vecs"])
    rng, cos, feats = np.random.default_rng(0), [], []
    for _ in range(n_halves):
        h = fit(Z[rng.choice(len(Z), len(Z) // 2, replace=False)])
        ha, hb = plotted(h["vecs"])
        cos.append([abs(a @ ha), abs(b @ hb)])
        feats.append(h["feats"])
    return np.array(cos), feats


def mesh(f: int, n: int = 3) -> str:
    return " | ".join(str(LAB.loc[f, "mesh"]).split(" | ")[:n])


def describe_feats(variant: str, feats, Z) -> list[str]:
    p = lambda f: f"{100 * (Z[:, f] > 0).mean():.0f}% here vs {100 * LAB.loc[f, 'fires']:.1f}% of all"  # noqa: E731
    if variant == "single":
        return [f"SAE feature {f[0]} ({p(f[0])}): {mesh(f[0])}" for f in feats]
    if variant == "contrast":
        return [f"SAE feature {g} ({p(g)}: {mesh(g, 2)}) → feature {f} ({p(f)}: {mesh(f, 2)})" for f, g in feats]
    return ["SAE contrastive PC: " + "; ".join(f"{w:+.2f} f{f} {mesh(f, 2)}" for f, w in fe) for fe in feats]


def poles(variant: str, fe) -> tuple:
    """(feature expected at the + end, feature expected at the - end or None) of an axis."""
    if variant == "single":
        return fe[0], None
    if variant == "contrast":
        return fe[0], fe[1]
    pos = [f for f, w in fe if w > 0]
    neg = [f for f, w in fe if w < 0]
    return pos[0], (neg[0] if neg else None)


def mono(S: dict, Z: np.ndarray, v, plus: int, minus) -> dict:
    """Do the axis' extreme papers carry the feature(s) it was built from?"""
    s = al.scores(S, v)
    hi, lo = s >= np.quantile(s, 0.9), s <= np.quantile(s, 0.1)
    code = Z[:, plus] - (Z[:, minus] if minus is not None else 0)
    out = {"plus_feature": plus, "plus_on_in_top10pct": round(float((Z[hi, plus] > 0).mean()), 2),
           "plus_on_overall": round(float((Z[:, plus] > 0).mean()), 2),
           "spearman_axis_vs_code": round(al._spearman(s, code), 2)}
    if minus is not None:
        out |= {"minus_feature": minus, "minus_on_in_bottom10pct": round(float((Z[lo, minus] > 0).mean()), 2),
                "minus_on_overall": round(float((Z[:, minus] > 0).mean()), 2)}
    return out


def judge(S: dict, ax: dict, plus: int, minus) -> dict:
    """LLM check of monosemanticity: do the extreme titles at each end share one idea, and is it the feature's?"""
    def block(end, f):
        lab = mesh(f, 4) if f is not None else "(none: the low end of a single feature)"
        return (f"{end} end; feature label (MeSH headings over-represented where the feature fires): {lab}\n"
                + "\n".join(f"- {t[:160]}" for t in ax[f"titles_{end}"]))
    prompt = (f"PubMed papers about {S['about']} were ordered along one direction built from sparse-autoencoder "
              "features of a text-embedding space. For each end, judge from the 8 most extreme titles: do they "
              "share one idea (a topic, method or kind of paper)? Does that idea match the feature label? Treat "
              "sharing only a word or an abbreviation as not one idea.\n\n" + block("plus", plus) + "\n\n"
              + block("minus", minus) + '\n\nAnswer with only JSON: {"plus": {"idea": "<3-8 words>", "one_idea": '
              'true|false, "matches_label": "yes|partly|no"}, "minus": {"idea": "<3-8 words>", "one_idea": true|false, '
              '"matches_label": "yes|partly|no|n/a"}}')
    return al._llm(prompt)


def run(variant: str, dry: bool) -> dict:
    fit = FITS[variant]
    bg()  # the background codes are a one-off precompute, not part of the live cost
    summary = {}
    for name in al.SUBJECTS:
        S = al.load(name)
        t0 = time.time()
        Z = encode(S["X"])
        t_enc = time.time() - t0
        full = fit(Z)
        t_fit = time.time() - t0
        st, half_feats = stability(fit, Z, full)
        med = np.median(st, 0)
        why = describe_feats(variant, full["feats"], Z)
        note = (f"live: encode {t_enc:.2f} s + pick {t_fit - t_enc:.2f} s for {len(Z):,} papers (background stats "
                f"precomputed); stability over 5 random halves: median |cos| x {med[0]:.2f}, y {med[1]:.2f} "
                f"(min {st[:, 0].min():.2f}, {st[:, 1].min():.2f})")
        a, b = plotted(full["vecs"])
        if dry:
            m = al.metrics(S, al.scores(S, a), al.scores(S, b))
            print(f"[{variant}] {name}: {m}  stab x {med[0]:.2f} y {med[1]:.2f}  ({t_enc:.2f}+{t_fit - t_enc:.2f} s)")
            for w, v in zip(why, (a, b)):
                pl, mi = poles(variant, full["feats"][why.index(w)])
                print("    ", w[:150], "|", mono(S, Z, v, pl, mi))
            summary[name] = {"metrics": m, "stab": med.tolist()}
            continue
        res = al.record(f"sae_{variant}", S, full["vecs"], why=why, note=note)
        checks = []
        for i, (ax, fe) in enumerate(zip(res["axes"], full["feats"])):
            pl, mi = poles(variant, fe)
            checks.append({**mono(S, Z, (a, b)[i], pl, mi), "llm_judge": judge(S, ax, pl, mi)})
        res["sae_checks"] = checks
        res["sae_features"] = full["feats"]
        res["stability_halves"] = {"abs_cos_xy": st.round(3).tolist(), "features": half_feats}
        res["live_seconds"] = {"encode": round(t_enc, 2), "pick": round(t_fit - t_enc, 2)}
        (al.RESULTS / f"sae_{variant}" / f"{name}.json").write_text(json.dumps(res, indent=1))
        summary[name] = {"metrics": res["metrics"], "stab": med.tolist(), "names": [ax["name"]["axis"] for ax in res["axes"]],
                         "checks": checks}
        print(f"[{variant}] {name}: {res['metrics']} stab {med.round(2)} | "
              + " || ".join(ax["name"]["axis"] for ax in res["axes"]), flush=True)
        for w, c in zip(why, checks):
            print("    ", w[:140], "|", c)
    st = np.array([v["stab"] for v in summary.values()])
    print(f"[{variant}] median over subjects of median |cos| half vs full: x {np.median(st[:, 0]):.2f}, "
          f"y {np.median(st[:, 1]):.2f}", flush=True)
    return summary


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for variant in args or list(FITS):
        run(variant, dry="--dry" in sys.argv)
