"""Task-level paired bootstrap CIs for BioMTEB(LLM) results.

Method (the paper's, vendor/embedders-dilemma/scripts/verify_numbers.py):
resample the *task set* with replacement 10,000 times (numpy default_rng,
seed 42), recompute each model's macro-mean BioScore and each pair's mean
difference on every resample, take the 2.5/97.5 percentiles. The two-sided
p-value is 2 * min(P(diff <= 0), P(diff >= 0)) over resamples. Every pair
shares one resample index matrix, so the test is paired.

Level: TASK ONLY. Item-level bootstrap is not possible from what is on
disk: the mteb result JSONs under results/embedding/**/*.json and
results/llm/**/*.json store aggregate metrics per split (main_score,
cosine_ap, recall_at_1, ...) with no per-item predictions or scores, and
the paper's lifted scores are one number per (model, task). Item-level
CIs would need the models re-run with prediction dumping.

Suites:
  full   — the 12-task bio suite; models with all 12 tasks.
  lifted — the 7 Tier A tasks lifted from the paper; models with all 7.

Outputs:
  results/bootstrap_scores.csv  model, suite, n_tasks, bio_score, ci_low, ci_high
  results/bootstrap_pairs.csv   model_a, model_b, suite, n_tasks, mean_diff,
                                ci_low, ci_high, p_value  (a > b by point score)
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
        pair_rows.append(dict(
            model_a=models[i], model_b=models[j], suite=suite, n_tasks=n,
            mean_diff=X[:, i].mean() - X[:, j].mean(),
            ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5),
            p_value=2 * min((d <= 0).mean(), (d >= 0).mean()),
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
    scores_df.round(4).to_csv(REPO / "results" / "bootstrap_scores.csv", index=False)
    pairs_df.round(4).to_csv(REPO / "results" / "bootstrap_pairs.csv", index=False)

    for suite in ("full", "lifted"):
        s = scores_df[scores_df.suite == suite]
        w = s.ci_high - s.ci_low
        print(f"\n== {suite}: {len(s)} models, {s.n_tasks.iloc[0]} tasks, "
              f"BioScore 95% CI width median {w.median():.3f} (range {w.min():.3f}-{w.max():.3f})")
        for _, r in s.iterrows():
            print(f"  {r.model:45s} {r.bio_score:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]")

    print("\n== Headline comparisons (task-level paired bootstrap, 95%)")
    for suite, pairs in HEADLINE.items():
        for a, b in pairs:
            r = pairs_df[(pairs_df.suite == suite) & (pairs_df.model_a.isin([a, b])) & (pairs_df.model_b.isin([a, b]))]
            if r.empty:
                print(f"  [{suite}] {a} vs {b}: missing coverage")
                continue
            r = r.iloc[0]
            sign = 1 if r.model_a == a else -1
            lo, hi = sorted([sign * r.ci_low, sign * r.ci_high])
            sig = "SIG" if r.p_value < 0.05 else "ns"
            print(f"  [{suite}] {a} - {b}: {sign * r.mean_diff:+.3f} [{lo:+.3f}, {hi:+.3f}] p={r.p_value:.3f} {sig}")


if __name__ == "__main__":
    main()
