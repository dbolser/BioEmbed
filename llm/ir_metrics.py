"""pytrec_eval-based retrieval metrics shared by the first-stage (eval/ env,
mteb 2.11.6) and LLM-stage (llm/ env, mteb 2.6.5) scripts.

Both mteb versions expose `mteb._evaluators.retrieval_metrics.calculate_retrieval_scores`
(pytrec_eval under the hood), so the numbers here are the same ones written to
results/embedding/<model>/**/<task>.json — directly comparable across sides.
"""

from __future__ import annotations

from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores

K_VALUES = [1, 3, 5, 10, 20]


def ranked_list_to_scores(ranked_ids: list[str]) -> dict[str, float]:
    """Rank position -> descending score, as the vendor reranker does (k - pos)."""
    k = len(ranked_ids)
    return {doc_id: float(k - pos) for pos, doc_id in enumerate(ranked_ids)}


def retrieval_metrics(
    results: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    k_values: list[int] = K_VALUES,
) -> dict[str, float]:
    """Flat {ndcg_at_k, map_at_k, recall_at_k, precision_at_k, mrr_at_k} dict.

    Queries missing from `results` are scored as empty (pytrec_eval gives 0).
    """
    full = {qid: results.get(qid, {}) for qid in qrels}
    r = calculate_retrieval_scores(full, qrels, k_values)
    out: dict[str, float] = {}
    for name, d in (("ndcg", r.ndcg), ("map", r.map), ("recall", r.recall),
                    ("precision", r.precision), ("mrr", r.mrr)):
        for key, v in d.items():
            # keys look like "NDCG@10" / "MRR@10" -> "ndcg_at_10"
            key = key.lower().replace("@", "_at_")
            if key.startswith("p_at_"):
                key = "precision" + key[1:]
            out[key] = round(float(v), 5)
    return out
