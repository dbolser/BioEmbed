"""Judge calibration: how often the LLM judges agree with the human labellers, next to how often the
humans agree with each other (labeller/labels_calibration.sqlite x data/haystack/<NAME>/judgments.parquet).

Exact = same 0/1/2 grade. Relevant = same answer to "grade >= 1?" (PREREG); only2 = same answer to
"grade == 2?" (the check PREREG reports alongside). Pass-0 labels only (the first look);
"dan, repeat" is Dan's pass 1 vs pass 0. Item ENSG00000008853|MONDO_0800491|29276004 is left out
(PREREG 2026-10-19 12:31). Unsure labels are left out. Dan's labels are also split at the change to
the human guide's wording (commit fde21fc, 2026-10-19 14:32 BST; PREREG 2026-10-19 14:32): "dan, old
wording" is his first 16 labels, "dan, new wording" the rest.

Run: uv run python scripts/judge_agreement.py   # -> results/figures/judge_agreement.{png,csv}
"""
import sqlite3
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
HAYSTACK = ROOT / "data/haystack/GeneDiseaseHaystack"
DB = ROOT / "labeller/labels_calibration.sqlite"
OUT = ROOT / "results/figures/judge_agreement"
EXCLUDED = {"ENSG00000008853|MONDO_0800491|29276004"}
REWORDED = "2026-10-19T13:32:14"  # UTC; labels first seen before this used the judges' rubric wording
JUDGES = {"deepseek/deepseek-v4-flash": "DeepSeek", "google/gemini-3.1-flash-lite": "Gemini"}
BLUE, ORANGE = "#2a78d6", "#eb6834"


def agreement(a, b):
    return len(a), float((a == b).mean()), float(((a >= 1) == (b >= 1)).mean()), float(((a == 2) == (b == 2)).mean())


def main():
    with sqlite3.connect(DB) as con:
        lab = pd.read_sql("select labeller, item_id, pass, round_id qid, pmid, grade, first_at from labels", con)
    lab = lab[~lab.item_id.isin(EXCLUDED) & lab.grade.notna()]
    first = lab[lab["pass"] == 0]
    j = duckdb.sql(f"select qid, pmid, judge, grade from '{HAYSTACK}/judgments.parquet'").df()
    j["judge"] = j.judge.map(JUDGES)
    jw = j.pivot_table(index=["qid", "pmid"], columns="judge", values="grade").reset_index()
    jw["Judges"] = np.floor((jw.DeepSeek + jw.Gemini) / 2)  # PREREG: mean, rounded down
    d = first.merge(jw, on=["qid", "pmid"], how="left")
    assert d.Judges.notna().all(), f"no judge grade for {d[d.Judges.isna()][['qid', 'pmid']].values.tolist()}"

    rows = []
    for name, g in d.groupby("labeller"):
        rows.append(("human vs judges", f"{name} vs judges", *agreement(g.grade, g.Judges)))
    dan = d[d.labeller == "dan"]
    old = dan.first_at < REWORDED
    for tag, g in (("old", dan[old]), ("new", dan[~old])):
        rows.append(("human vs judges", f"dan, {tag} wording, vs judges", *agreement(g.grade, g.Judges)))
    p = first.pivot_table(index="item_id", columns="labeller", values="grade")
    names = list(p.columns)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m = p[[a, b]].dropna()
            if len(m) >= 5:
                rows.append(("human vs human", f"{a} vs {b}", *agreement(m[a], m[b])))
    rep = lab.pivot_table(index=["labeller", "item_id"], columns="pass", values="grade").dropna()
    for name, g in rep.groupby(level=0):
        rows.append(("human vs human", f"{name} vs {name}, repeat", *agreement(g[0], g[1])))
    one = d.drop_duplicates(["qid", "pmid"])
    rows.append(("judge vs judge", "DeepSeek vs Gemini", *agreement(one.DeepSeek, one.Gemini)))
    t = pd.DataFrame(rows, columns=["kind", "pair", "n", "exact", "relevant", "only2"])
    t.to_csv(OUT.with_suffix(".csv"), index=False)
    print(t.round(2).to_string(index=False))

    order = ["human vs judges", "human vs human", "judge vs judge"]
    t = t.set_index("kind").loc[order].reset_index()
    t = pd.concat([t[t.kind == k].sort_values("exact") for k in order])
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(t) + 2.4))
    y, head = [], {}
    pos = 0.0
    for k in order:
        head[k] = pos
        pos -= 0.9
        for _ in range(int((t.kind == k).sum())):
            y.append(pos)
            pos -= 1.0
        pos -= 0.5
    y = np.array(y)
    ax.barh(y + 0.19, t.exact, 0.34, color=BLUE, label="same grade (0/1/2)")
    ax.barh(y - 0.19, t.relevant, 0.34, color=ORANGE, label="same answer to “relevant?”")
    for yi, e, r, n in zip(y, t.exact, t.relevant, t.n):
        ax.text(e + 0.012, yi + 0.19, f"{e:.0%}", va="center", fontsize=8.5, color="#52514e")
        ax.text(r + 0.012, yi - 0.19, f"{r:.0%}  (n={n})", va="center", fontsize=8.5, color="#52514e")
    ax.set_yticks(y, t.pair, fontsize=9.5)
    for k, hy in head.items():
        ax.text(-0.02, hy, k, ha="right", va="center", fontsize=9, fontweight="bold",
                color="#0b0b0b", transform=ax.get_yaxis_transform())
    ax.set_ylim(y.min() - 0.8, 0.6)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.xaxis.grid(True, color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("agreement on the calibration abstracts")
    ax.set_title("The LLM judges agree with the human labellers\nabout as often as the humans agree with each other",
                 fontsize=11, loc="left", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".png"), dpi=170)
    print("->", OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
