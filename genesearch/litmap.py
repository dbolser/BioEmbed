"""Literature map: topic clusters of a list of papers, from their MedEmbed vectors.

UMAP (cosine) to 12-D -> HDBSCAN clusters -> c-TF-IDF top terms per cluster -> short names from an
LLM (Gemini Flash-Lite via OpenRouter, cached; the top terms if there is no key or no answer) ->
UMAP to 2-D for display. Both UMAP runs share one nearest-neighbour graph.
"""

import json
import os
import re
import threading
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import Stemmer

from semantic import CACHE_DIR, digest

LLM_ENV = Path(__file__).resolve().parent.parent / "llm" / ".env.or-google_gemini-3.1-flash-lite"
NAMES = CACHE_DIR / "cluster_names.json"
N_JOBS = int(os.environ.get("GENESEARCH_MAP_JOBS", "32"))
K = 15  # neighbours (UMAP's default)
MAX_TOPICS = 8  # the page has 8 distinct topic colours
STEMMER = Stemmer.Stemmer("english")

# Words every biomedical abstract uses; c-TF-IDF down-weights them anyway, but big clusters can
# still surface them.
BOILERPLATE = """study studies result results method methods conclusion conclusions background
objective objectives aim aims purpose using used use based associated association analysis analyses
showed show shows shown found observed significant significantly also however suggest suggests
suggested data patients patient case cases including included increased decreased level levels
effect effects role potential novel new report reported present review clinical important total""".split()


def warm_up():
    """Compile the numba code (neighbour search, both UMAPs) once at start-up; the first map would
    otherwise wait ~30 s for it."""
    build(np.random.default_rng(0).standard_normal((300, 384)).astype(np.float32), None, None, "", set(), None)


def build(vecs: np.ndarray, texts: list[str] | None, titles: list[str] | None, symbol: str, skip_words: set[str],
          job, within: str = "") -> dict:
    """Coordinates, cluster per paper (-1 = no cluster) and the clusters, biggest first.
    Without texts (warm-up) it stops after the layout. within = the topic all these papers come
    from (a drill-in), for naming."""
    from umap import UMAP
    from umap.umap_ import nearest_neighbors

    n, k = len(vecs), min(K, len(vecs) - 1)
    took, t = {}, time.perf_counter()

    def phase(name, text=None):
        nonlocal t
        took[name], t = round(time.perf_counter() - t, 2), time.perf_counter()
        if job and text:
            job.step(text)

    phase("start", "Finding similar papers")
    knn = nearest_neighbors(vecs, k, "cosine", {}, False, None, n_jobs=N_JOBS)
    phase("neighbours", "Grouping papers by topic (UMAP to 12 dimensions, then HDBSCAN)")
    u12 = UMAP(n_components=12, n_neighbors=k, min_dist=0.0, metric="cosine", precomputed_knn=knn,
               n_jobs=N_JOBS).fit_transform(vecs)
    phase("umap_12d")
    labels = topics(u12)
    phase("hdbscan", "Drawing the map (UMAP to 2 dimensions)")
    xy = UMAP(n_components=2, n_neighbors=k, min_dist=0.1, metric="cosine", precomputed_knn=knn,
              n_jobs=N_JOBS).fit_transform(vecs)
    phase("umap_2d", "Naming the topics")
    if texts is None:
        return {}

    ids = [c for c in np.unique(labels) if c >= 0]
    ids.sort(key=lambda c: -(labels == c).sum())  # cluster 1 = biggest
    renum = {c: i + 1 for i, c in enumerate(ids)}
    labels = np.array([renum.get(c, -1) for c in labels])
    terms = ctfidf(texts, labels, skip_words)
    clusters = []
    for c in range(1, len(ids) + 1):
        m = np.flatnonzero(labels == c)
        near = m[np.argsort(((u12[m] - u12[m].mean(0)) ** 2).sum(1))[:6]]  # most typical papers
        clusters.append({"id": c, "n": len(m), "terms": terms.get(c, []),
                         "x": float(np.median(xy[m, 0])), "y": float(np.median(xy[m, 1])),
                         "examples": [titles[i] for i in near if titles[i]]})
    phase("terms")
    named = name_clusters(symbol, clusters, within)
    phase("names")
    return {"x": np.round(xy[:, 0], 3).tolist(), "y": np.round(xy[:, 1], 3).tolist(), "c": labels.tolist(),
            "clusters": clusters, "named_by": "llm" if named else "terms", "timings_s": took}


def topics(u12: np.ndarray, most: int = MAX_TOPICS, outliers: float = 0.1) -> np.ndarray:
    """HDBSCAN topics (-1 = none), at most `most` of them (one colour each on the page).

    On a continuous literature HDBSCAN either returns one topic holding half the papers ('eom'
    selection) or small dense cores with half the papers left over ('leaf'). So: try both selection
    methods over a range of minimum topic sizes; in each, give every left-over paper the topic of
    its nearest topic paper (12-D), except the `outliers` share that HDBSCAN's own outlier score
    (GLOSH) rates least typical, which stay grey; keep the result with the most even topic sizes
    (effective number of topics: exp of the entropy of their sizes) times the share of papers in a
    topic. The fits share one minimum spanning tree: ~6 s for 20,000 papers."""
    import tempfile

    import hdbscan
    import joblib
    from sklearn.neighbors import NearestNeighbors

    best, best_score, seen = np.full(len(u12), -1), 0.0, set()
    with tempfile.TemporaryDirectory() as tmp:
        memory = joblib.Memory(tmp, verbose=0)
        for method in ("eom", "leaf"):
            for div in (5, 8, 12, 20, 30, 45, 60, 80):
                h = hdbscan.HDBSCAN(min_cluster_size=max(8, round(len(u12) / div)), min_samples=5,
                                    cluster_selection_method=method, memory=memory,
                                    core_dist_n_jobs=1).fit(u12)  # parallel spawns 32 processes, 2x slower
                labels = h.labels_.copy()
                if not 1 <= labels.max() + 1 <= most or (key := labels.tobytes()) in seen:
                    continue
                seen.add(key)
                glosh = np.nan_to_num(h.outlier_scores_)  # NaN for exact duplicates: not outliers
                lost = np.flatnonzero((labels < 0) & (glosh < np.quantile(glosh, 1 - outliers)))
                if len(lost):
                    core = np.flatnonzero(labels >= 0)
                    near = NearestNeighbors(n_neighbors=1).fit(u12[core]).kneighbors(u12[lost], return_distance=False)
                    labels[lost] = labels[core[near[:, 0]]]
                share = np.bincount(labels[labels >= 0]) / len(labels)
                p = share / share.sum()
                score = np.exp(-(p * np.log(p)).sum()) * share.sum()
                if score > best_score:
                    best, best_score = labels, score
    return best


def name_clusters(symbol: str, clusters: list[dict], within: str = "") -> bool:
    """Set each cluster's name: the LLM's if it answers (True), else its top 3 terms, taking first
    the terms no other cluster has ('cystic fibrosis' is a top term of most CFTR topics, so it
    doesn't tell them apart)."""
    names = llm_names(symbol, clusters, within)
    shared = Counter(t for cl in clusters for t in set(cl["terms"]))
    for cl in clusters:
        terms = sorted(cl["terms"], key=lambda t: shared[t] > 1)  # stable: own terms first, in rank order
        cl["name"] = names.get(str(cl["id"])) or ", ".join(terms[:3]) or f"Topic {cl['id']}"
    return bool(names)


PHRASE_END = re.compile(r"[^\w\s'-]+")  # brackets, commas, full stops: no term spans them
TOKEN = re.compile(r"\b[a-z][a-z0-9-]+\b")


def terms_of(stop: set[str]):
    """Words and word pairs of a text, lower-cased, without stop words. A pair never spans
    punctuation or a stop word: 'Alzheimer disease (AD)' gives 'alzheimer disease', not 'disease ad'."""
    def terms(text: str) -> list[str]:
        out = []
        for phrase in PHRASE_END.split(text.lower()):
            words = TOKEN.findall(phrase)
            keep = [w not in stop for w in words]
            out += [w for w, k in zip(words, keep) if k]
            out += [f"{a} {b}" for a, b, ka, kb in zip(words, words[1:], keep, keep[1:]) if ka and kb]
        return out
    return terms


def ctfidf(texts: list[str], labels: np.ndarray, skip_words: set[str], top: int = 6) -> dict[int, list[str]]:
    """Top terms per cluster by class-based TF-IDF (BERTopic's c-TF-IDF): a term's share of the
    cluster's words, times log(1 + average words per cluster / the term's count over all clusters).
    Unclustered papers form a class too, so they count towards 'common everywhere'."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

    stop = set(ENGLISH_STOP_WORDS) | set(BOILERPLATE) | {w.lower() for w in skip_words}
    cv = CountVectorizer(analyzer=terms_of(stop), min_df=max(3, len(texts) // 2000), dtype=np.float32)
    try:
        x = cv.fit_transform(texts)
    except ValueError:  # too few papers for min_df
        return {}
    classes = np.unique(labels)
    member = sp.csr_matrix((np.ones(len(labels), np.float32), (np.searchsorted(classes, labels), np.arange(len(labels)))),
                           shape=(len(classes), len(labels)))
    tf = np.asarray((member @ x).todense())
    w = tf / np.maximum(tf.sum(1, keepdims=True), 1) * np.log(1 + tf.sum() / len(classes) / np.maximum(tf.sum(0), 1))
    vocab = cv.get_feature_names_out()
    out = {}
    for i, c in enumerate(classes):
        if c < 0:
            continue
        chosen, stems = [], []
        for j in np.argsort(-w[i])[:50]:
            st = set(STEMMER.stemWords(vocab[j].split()))
            inside = [k for k, o in enumerate(stems) if o < st]  # chosen single words this pair holds
            if any(st & o for k, o in enumerate(stems) if k not in inside):
                continue  # 'proteins' after 'protein'; 'fibrosis' after 'cystic fibrosis'; 'cell lung' after 'lung cancer'
            if inside:  # 'cystic fibrosis' takes the place of 'cystic' (and drops 'fibrosis')
                chosen[inside[0]], stems[inside[0]] = vocab[j], st
                for k in reversed(inside[1:]):
                    del chosen[k], stems[k]
            elif len(chosen) < top:
                chosen.append(vocab[j])
                stems.append(st)
        out[int(c)] = chosen
    return out


# ---------------------------------------------------------------- LLM names (optional, cached)
_names_lock = threading.Lock()
_llm_down_until = 0.0  # after a failure, don't ask again for a while


def _llm_config() -> dict | None:
    if not LLM_ENV.exists():
        return None
    cfg = dict(line.split("=", 1) for line in LLM_ENV.read_text().splitlines() if "=" in line)
    return cfg if cfg.get("token") else None


def llm_names(symbol: str, clusters: list[dict], within: str = "") -> dict[str, str]:
    """{cluster id: short name} from the LLM, or {} (no key, no answer, or nothing to name)."""
    global _llm_down_until
    cfg = _llm_config()
    if not cfg or not clusters or time.time() < _llm_down_until:
        return {}
    blocks = [f"Cluster {c['id']} ({c['n']} papers)\nTerms: {', '.join(c['terms'])}\nExample titles:\n" +
              "\n".join(f"- {t[:160]}" for t in c["examples"][:5]) for c in clusters]
    about = f"the human gene {symbol}" + (f", all from its topic \"{within}\", so name what sets each "
                                           "cluster apart within it" if within else "")
    prompt = (f"These are topic clusters from a map of PubMed papers about {about}. For each "
              "cluster you get its most distinctive terms and some typical titles. Give each cluster a short, "
              "plain-English name of 2 to 5 words that a biologist would recognise, naming its topic (for example "
              "a disease, a process, a method, a tissue or an organism). Do not put the gene symbol in the names, "
              "and make the names different from each other. Answer with only a JSON object mapping each cluster "
              'number to its name, like {"1": "Alzheimer disease risk", "2": "Lipoprotein metabolism"}.\n\n'
              + "\n\n".join(blocks))
    key = digest(cfg.get("model"), prompt)
    with _names_lock:
        cache = json.loads(NAMES.read_text()) if NAMES.exists() else {}
    if key in cache:
        return cache[key]
    body = {"model": cfg.get("model", "google/gemini-3.1-flash-lite"), "temperature": 0, "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}}
    req = urllib.request.Request(cfg.get("base_url", "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {cfg['token']}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            text = json.loads(r.read())["choices"][0]["message"]["content"]
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        names = {str(k): str(v).strip() for k, v in json.loads(text).items() if str(v).strip()}
    except Exception as e:  # fall back to the top terms
        print(f"litmap: no LLM names ({type(e).__name__}: {e})", flush=True)
        _llm_down_until = time.time() + 600
        return {}
    with _names_lock:
        cache = json.loads(NAMES.read_text()) if NAMES.exists() else {}
        cache[key] = names
        NAMES.parent.mkdir(parents=True, exist_ok=True)
        tmp = NAMES.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(cache))
        os.replace(tmp, NAMES)
    return names
