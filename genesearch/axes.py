"""Map Explorer's suggested axes: two directions to lay a map's papers out on, found from the papers themselves, with
each end named from the papers near it.

- discover: the two directions in which the map's papers differ most, compared with how much human-gene papers in
  general differ along them. These are generalised eigenvectors of the map's covariance against the background's,
  shrunk 20% towards equal spread in every direction. The map's own direction is taken out first, or it would come
  first for most maps. That direction is the map's centre minus the background's: 'about the subject <-> mentions
  it in passing'.
- spread: the two directions in which the map's papers differ most (PCA), whatever gene papers do in general.
- ontopic: across is the map's own direction (+ = like the map's centre, - = like gene papers in general); up is
  discover's first direction.

discover and ontopic need the background: the mean and covariance of 50,000 random human-gene papers' vectors. They
are computed once from the whole-PubMed vectors and kept in index/cache/background.npz. Each axis except ontopic's
across is signed so that its + end has the newer papers; up is made perpendicular to across. These three were picked
from 18 kinds of axis tried on 5 maps (data/geometry/axeslab); three blind judges ranked discover first.
"""

import os

import numpy as np
import scipy.linalg

import litmap
import semantic

KINDS = ("discover", "spread", "ontopic")
N_BACKGROUND, SHRINK = 50_000, 0.2
BACKGROUND = semantic.CACHE_DIR / "background.npz"
_bg = None


def background(gene_pmids) -> tuple[np.ndarray, np.ndarray]:
    """(mean, covariance) of 50,000 random human-gene papers' vectors; raises Pending while they are computed (once,
    in about 5 s on a big server: then they are read from disk). gene_pmids(): every human-gene paper's PMID."""
    global _bg
    if _bg is None and BACKGROUND.exists():
        with np.load(BACKGROUND) as f:
            _bg = f["mean"], f["cov"]
    if _bg is None:
        j = semantic.job(("background",), lambda jb: _make_background(jb, gene_pmids))
        if not j.finished or j.error:
            raise semantic.Pending(j)
        _bg = j.result
    return _bg


def _make_background(job: semantic.Job, gene_pmids) -> tuple[np.ndarray, np.ndarray]:
    job.step("Measuring 50,000 gene papers in general (needed once)")
    full = semantic.full_index()
    if full is None:
        raise RuntimeError("needs the whole-PubMed vectors")
    pm = np.unique(gene_pmids())  # (sorted: the seeded draw then takes the same papers every time)
    X, ok = full.get(np.sort(np.random.default_rng(0).choice(pm, min(N_BACKGROUND, len(pm)), replace=False)))
    X = X[ok].astype(np.float64)
    mean, cov = X.mean(0), _cov(X)
    BACKGROUND.parent.mkdir(parents=True, exist_ok=True)
    tmp = BACKGROUND.with_name(f"background.{os.getpid()}.tmp.npz")
    np.savez(tmp, mean=mean, cov=cov, n=len(X))
    os.replace(tmp, BACKGROUND)
    return mean, cov


def _cov(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0)
    return Xc.T @ Xc / max(len(X) - 1, 1)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def directions(kind: str, X: np.ndarray, years: np.ndarray, bg=None) -> tuple[np.ndarray, np.ndarray]:
    """(across, up): unit vectors, up perpendicular to across. X: the map's papers' vectors; years: 0 = unknown;
    bg: background() (for discover and ontopic)."""
    X = np.asarray(X, np.float64)
    if kind == "spread":
        _, V = np.linalg.eigh(_cov(X))
        a, b = V[:, -1], V[:, -2]
    else:
        mean, Cb = bg
        own = _unit(X.mean(0) - mean)
        P = np.eye(len(own)) - np.outer(own, own)  # (adding own's outer product back keeps the right side invertible)
        R = (1 - SHRINK) * Cb + SHRINK * np.trace(Cb) / len(own) * np.eye(len(own))
        _, V = scipy.linalg.eigh(P @ _cov(X) @ P, P @ R @ P + np.outer(own, own))
        a, b = (own, V[:, -1]) if kind == "ontopic" else (V[:, -1], V[:, -2])
    newer = years > 0

    def signed(v):  # + end = the newer papers
        if newer.sum() > 2 and np.corrcoef(X[newer] @ v, years[newer])[0, 1] < 0:
            return -v
        return v

    a = _unit(a) if kind == "ontopic" else signed(_unit(a))
    return a, signed(_unit(b - (b @ a) * a))


def name_axes(scores: dict[str, np.ndarray], titles: list[str], named: np.ndarray, about: str, label: str,
              skip: set[str]) -> dict:
    """{axis: {'minus', 'plus', 'by'}} for scores = {'x': across, 'y': up} (each paper's place on them): what the
    papers at each end are about, named by the LLM in one go (so the axes get names that tell them apart); without an
    answer, each end's most distinctive words (by = 'terms').

    An end is its 10% of the papers, less any that are also at an end of the other axis (often a third or more):
    its words are the most distinctive against the other three ends and the middle (c-TF-IDF), and its titles are
    up to 8 spread over 2-10% in from it (the most extreme papers are often odd ones out). named: whether each paper
    names the subject (label) in its title or abstract; each end's share of them goes to the LLM, as about the
    subject or not can be what sets an end apart."""
    n = len(titles)
    tenth = max(1, n // 10)
    ends = {}  # (axis, 'low' | 'high'): papers, nearest the end first
    for k, s in scores.items():
        order = np.argsort(s, kind="stable")
        ends[k, "low"], ends[k, "high"] = order, order[::-1]
    seen = np.zeros(n, int)
    for ranked in ends.values():
        seen[ranked[:tenth]] += 1
    labels = np.full(n, -1)
    for c, ranked in enumerate(ends.values()):
        own = ranked[:tenth][seen[ranked[:tenth]] == 1]
        labels[own] = c
    terms = litmap.ctfidf(titles, labels, skip, top=8)
    blocks, words = [], {}
    for c, ((k, end), ranked) in enumerate(ends.items()):
        band = ranked[int(0.02 * n):max(tenth, int(0.02 * n) + 1)]
        band = band[seen[band] == 1] if (seen[band] == 1).sum() >= 3 else band
        pick = band[np.unique(np.linspace(0, len(band) - 1, min(8, len(band))).astype(int))]
        words[k, end] = terms.get(c, [])
        blocks.append(f"{'Across' if k == 'x' else 'Up'}, {end} end ({round(100 * named[ranked[:tenth]].mean())}% of its "
                      f"papers name {label} in the title or abstract)\nTypical titles:\n" +
                      "\n".join(f"- {titles[i][:160]}" for i in pick) + f"\nDistinctive words: {', '.join(words[k, end])}")
    got = litmap.ask(
        f"These are PubMed papers linked to {about} by text mining: most are about it, some only mention it, and a few "
        "are linked by mistake. They are laid out on two independent directions of a text-embedding space, across and "
        "up. Name what the papers at each end of each direction are about, so that a biologist can read each direction "
        "as an axis, and make the two names of a direction contrast. Each end's titles and words leave out papers that "
        "are at an end of both directions, so name each end from its own. Leave out the name " + label + " unless "
        "naming it, or not, is what sets that end apart (for example, one end is about it and the other only mentions "
        "it). If an end's papers are not about it, name what they are about.\n\n" + "\n\n".join(blocks) +
        "\n\nAnswer with only a JSON object: "
        '{"across_low": "<2-5 words>", "across_high": "<2-5 words>", "up_low": "<2-5 words>", "up_high": "<2-5 words>"}.')
    out = {}
    for k in scores:
        word = "across" if k == "x" else "up"
        if got.get(f"{word}_low") and got.get(f"{word}_high"):
            out[k] = {"minus": got[f"{word}_low"], "plus": got[f"{word}_high"], "by": "llm"}
        else:
            out[k] = {"minus": ", ".join(words[k, "low"][:3]) or "lower", "plus": ", ".join(words[k, "high"][:3]) or "higher",
                      "by": "terms"}
    return out
