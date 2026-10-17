"""Task-level paired bootstrap CIs for BioMTEB(LLM) results.

Method (the paper's, vendor/embedders-dilemma/scripts/verify_numbers.py):
resample the *task set* with replacement 10,000 times (numpy default_rng,
seed 42), recompute each model's macro-mean BioScore and each pair's mean
difference on every resample, take the 2.5/97.5 percentiles. The two-sided
p-value is 2 * min(P(diff <= 0), P(diff >= 0)) over resamples. Every pair
shares one resample index matrix, so the test is paired.

Exact sign-flip test: for each pair, the per-task differences d_t are
sign-flipped in all 2^n patterns (4,096 for 12 tasks); p is the share of
patterns whose |mean| is at least the observed |mean d|. wins = tasks where
a scores above b. This is the pre-registered test for suite claims
(docs/PREREG.md, C9).

Level: TASK ONLY. Item-level bootstrap is not possible from what is on
disk: the mteb result JSONs under results/embedding/**/*.json and
results/llm/**/*.json store aggregate metrics per split (main_score,
cosine_ap, recall_at_1, ...) with no per-item predictions or scores, and
the paper's lifted scores are one number per (model, task). Item-level
CIs would need the models re-run with prediction dumping.

Suites:
  full   — the 12-task bio suite; every model in results/bio_scores.csv with
           all 12 tasks.
  lifted — the 7 Tier A tasks lifted from the paper; models with all 7.

Outputs:
  results/bootstrap_scores.csv  model, suite, n_tasks, bio_score, ci_low, ci_high
  results/bootstrap_pairs.csv   model_a, model_b, suite, n_tasks, mean_diff,
                                ci_low, ci_high, p_value (bootstrap), wins, ties,
                                signflip_p, headline  (a > b by point score)
"""

import itertools

import numpy as np
import pandas as pd

from aggregate_bio import REPO, SUITE

N_BOOT = 10_000
SEED = 42

LIFTED = [t for t in SUITE if t.startswith("LLM")]  # 7 Tier A tasks
FULL = list(SUITE)  # 12

HEADLINE = {
    "full": [
        ("google__gemini-3.1-pro-preview", "Qwen__Qwen3-Embedding-8B"),
        ("google__gemini-3.1-pro-preview", "codefuse-ai__F2LLM-v2-1.7B"),
        ("bioembed__bge-small-gaf", "BAAI__bge-small-en-v1.5"),
        ("bioembed__bge-base-gaf", "BAAI__bge-base-en-v1.5"),
        ("bioembed__bge-small-gaf", "abhinand__MedEmbed-small-v0.1"),
        ("bioembed__bge-base-gaf", "abhinand__MedEmbed-base-v0.1"),
        ("Qwen__Qwen3-Embedding-4B", "Qwen__Qwen3-Embedding-0.6B"),
        ("Qwen__Qwen3-Embedding-0.6B", "abhinand__MedEmbed-base-v0.1"),
        ("abhinand__MedEmbed-base-v0.1", "deepseek__deepseek-v4-flash"),
        ("Qwen__Qwen3-Embedding-4B", "deepseek__deepseek-v4-flash"),
        ("abhinand__MedEmbed-small-v0.1", "BAAI__bge-small-en-v1.5"),
        ("abhinand__MedEmbed-base-v0.1", "BAAI__bge-base-en-v1.5"),
        ("abhinand__MedEmbed-large-v0.1", "BAAI__bge-large-en-v1.5"),
    ],
    "lifted": [
        ("google__gemini-3.1-pro-preview", "Qwen__Qwen3-Embedding-8B"),
    ],
}


def suite_pivot(scores: pd.DataFrame, tasks: list[str]) -> pd.DataFrame:
    """tasks x models matrix, keeping only models with full coverage."""
    piv = scores[scores.task.isin(tasks)].pivot_table(index="task", columns="model", values="score")
    piv = piv.reindex(tasks)
    return piv.dropna(axis=1)


def sign_flip(d: np.ndarray) -> float:
    """Exact two-sided sign-flip p for mean(d) over all 2^n sign patterns."""
    n = len(d)
    signs = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
    return float((np.abs(signs @ d) >= abs(d.sum()) - 1e-12).mean())


def bootstrap(piv: pd.DataFrame, suite: str, rng: np.random.Generator):
    n = len(piv)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    X = piv.values  # tasks x models
    boots = X[idx].mean(axis=1)  # N_BOOT x models
    models = list(piv.columns)

    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    score_rows = [
        dict(model=m, suite=suite, n_tasks=n, bio_score=X[:, j].mean(), ci_low=lo[j], ci_high=hi[j])
        for j, m in enumerate(models)
    ]

    pair_rows = []
    for i, j in itertools.combinations(range(len(models)), 2):
        if X[:, i].mean() < X[:, j].mean():
            i, j = j, i
        d = boots[:, i] - boots[:, j]
        dt = X[:, i] - X[:, j]
        pair_rows.append(dict(
            model_a=models[i], model_b=models[j], suite=suite, n_tasks=n,
            mean_diff=X[:, i].mean() - X[:, j].mean(),
            ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5),
            p_value=2 * min((d <= 0).mean(), (d >= 0).mean()),
            wins=int((dt > 0).sum()), ties=int((dt == 0).sum()), signflip_p=sign_flip(dt),
        ))
    return pd.DataFrame(score_rows), pd.DataFrame(pair_rows)


def main():
    scores = pd.read_csv(REPO / "results" / "bio_scores.csv")
    rng = np.random.default_rng(SEED)
    all_scores, all_pairs = [], []
    for suite, tasks in (("full", FULL), ("lifted", LIFTED)):
        piv = suite_pivot(scores, tasks)
        s, p = bootstrap(piv, suite, rng)
        all_scores.append(s)
        all_pairs.append(p)
    scores_df = pd.concat(all_scores).sort_values(["suite", "bio_score"], ascending=[True, False])
    pairs_df = pd.concat(all_pairs)

    full = scores_df[scores_df.suite == "full"].merge(
        scores[["model", "model_type"]].drop_duplicates(), on="model")
    best = full.sort_values("bio_score").groupby("model_type").model.last()
    headline = {k: list(v) for k, v in HEADLINE.items()}
    if (best.llm, best.embedding) not in headline["full"]:
        headline["full"].append((best.llm, best.embedding))
    key = {(suite, frozenset(p)) for suite, pairs in headline.items() for p in pairs}
    pairs_df["headline"] = [(r.suite, frozenset((r.model_a, r.model_b))) in key for r in pairs_df.itertuples()]
    scores_df.round(4).to_csv(REPO / "results" / "bootstrap_scores.csv", index=False)
    pairs_df.round(4).to_csv(REPO / "results" / "bootstrap_pairs.csv", index=False)

    for suite in ("full", "lifted"):
        s = scores_df[scores_df.suite == suite]
        w = s.ci_high - s.ci_low
        print(f"\n== {suite}: {len(s)} models, {s.n_tasks.iloc[0]} tasks, "
              f"BioScore 95% CI width median {w.median():.3f} (range {w.min():.3f}-{w.max():.3f})")
        for _, r in s.iterrows():
            print(f"  {r.model:45s} {r.bio_score:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]")

    print(f"\n== Best LLM vs best embedder (full): {best.llm} vs {best.embedding}")
    print("== Headline comparisons (task-level paired bootstrap 95% CI and p; exact sign-flip p; wins)")
    for suite, pairs in headline.items():
        for a, b in pairs:
            r = pairs_df[(pairs_df.suite == suite) & (pairs_df.model_a.isin([a, b])) & (pairs_df.model_b.isin([a, b]))]
            if r.empty:
                print(f"  [{suite}] {a} vs {b}: missing coverage")
                continue
            r = r.iloc[0]
            sign = 1 if r.model_a == a else -1
            lo, hi = sorted([sign * r.ci_low, sign * r.ci_high])
            sig = "SIG" if r.p_value < 0.05 else "ns"
            wins = r.wins if sign == 1 else r.n_tasks - r.wins - r.ties
            print(f"  [{suite}] {a} - {b}: {sign * r.mean_diff:+.3f} [{lo:+.3f}, {hi:+.3f}] "
                  f"boot p={r.p_value:.3f} {sig}; sign-flip p={r.signflip_p:.4f}; wins {wins}/{r.n_tasks}")


if __name__ == "__main__":
    main()
