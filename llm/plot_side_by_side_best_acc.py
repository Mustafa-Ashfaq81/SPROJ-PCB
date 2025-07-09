#!/usr/bin/env python3
"""
plot_side_by_side_best_acc.py
─────────────────────────────
• For each player, plot the *best* Top-1 and Top-2 accuracy reached by
  four GPT variants.
• Bars inside every player-group are ordered by model size
  (nano → mini → 4.1 → o4-mini).
• An integer id under each bar shows which feature-block won; the full
  id → block mapping is printed beneath the plot.
"""

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# --------------------------------------------------------------
# 1.  Configuration
# --------------------------------------------------------------
PLAYERS = [
    "Babar Azam",
    "Shai Hope",
    "Ibrahim Zadran",
    "Daryl Mitchell",
    "Shubman Gill",
    "Harry Tector",
    "Charith Asalanka",
]

# name shown on chart   → folder/file key
MODELS = {
    "gpt-4.1 mini": "gpt-4.1-mini",
    "gpt-4.1": "gpt-4.1",
    "gpt-4.1 nano": "gpt-4.1-nano",
    "gpt-o4 mini": "o4-mini",
}

# required size-based plotting order
MODEL_ORDER = [
    "gpt-4.1 nano",  # smallest
    "gpt-4.1 mini",
    "gpt-4.1",
    "gpt-o4 mini",  # largest
]

MODEL_COLOURS = [
    "#440154",  # nano → dark purple
    "#414487",  # mini → indigo
    "#2A788E",  # 4.1 → teal
    "#22A884",  # o4-mini → bright green
]


MODEL2COLOR = dict(zip(MODEL_ORDER, MODEL_COLOURS))

METRICS = ["top1_acc", "top2_acc"]

# -------- fixed category-id mapping (0-8) --------
BLOCK_ID_MAP = {
    "Baseline": 0,
    "Baseline\n(all off)": 0,
    "General Stats": 1,
    "General Stats + Venue Stats": 2,
    "General Stats + Venue Stats + Vs-Team Stats": 3,
    "General Stats + Venue Stats + Vs-Team Stats + Commentary": 4,
    "Venue Stats": 5,
    "Vs-Team Stats": 6,
    "Venue Stats + Vs-Team Stats": 7,
    "Venue Stats + Vs-Team Stats + Commentary": 8,
}

# ensure folders exist
os.makedirs("plots", exist_ok=True)
os.makedirs("plots/csv", exist_ok=True)
os.makedirs("plots/figures", exist_ok=True)

# Save mapping as CSV
pd.Series(BLOCK_ID_MAP, name="category_id").sort_values().to_csv(
    "plots/csv/category_id_mapping.csv", header=True
)

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)


# --------------------------------------------------------------
# 2.  Helper – load / rebuild ablation summary
# --------------------------------------------------------------
def load_or_make_summary(player: str, model_key: str) -> pd.DataFrame:
    fp = f"plots/{player}/{model_key}/ablation_accuracy_summary.csv"
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        if {"top1_acc", "top2_acc", "block"}.issubset(df.columns):
            return df

    raw = f"{player}_prediction_results_{model_key}.csv"
    if not os.path.exists(raw):
        raise FileNotFoundError(f"Missing data for {player}/{model_key}")

    df = pd.read_csv(raw, quotechar='"')

    # numeric flags
    for col in (
        "general_stats_flag",
        "venuewise_stats_flag",
        "against_teams_stats_flag",
        "commentary",
    ):
        df[col] = df.get(col, 0).replace({"on": 1, "off": 0}).astype(int)

    # readable block label
    def label(r):
        parts = []
        if r["general_stats_flag"]:
            parts.append("General Stats")
        if r["venuewise_stats_flag"]:
            parts.append("Venue Stats")
        if r["against_teams_stats_flag"]:
            parts.append("Vs-Team Stats")
        if r["commentary"]:
            parts.append("Commentary")
        return " + ".join(parts) if parts else "Baseline"

    df["block"] = df.apply(label, axis=1)

    # correctness flags
    df["top1_correct"] = (
        df["most_likely_mode"].str.strip()
        == df["mode_of_dismissal_ground_truth"].str.strip()
    )
    df["prob_vec"] = (
        df["mode_of_dismissal_predicted_probability"]
        .str.split(",")
        .apply(lambda lst: [float(x) for x in lst])
    )
    df["label_vec"] = (
        df["mode_of_dismissal_predicted"]
        .str.split(",")
        .apply(lambda lst: [x.strip() for x in lst])
    )

    def hit_topk(row, k=2):
        if not row["label_vec"]:
            return False
        gt = row["mode_of_dismissal_ground_truth"].strip()
        if gt not in row["label_vec"]:
            return False
        top_ix = np.argsort(row["prob_vec"])[-k:]
        return gt in {row["label_vec"][i] for i in top_ix}

    df["top2_correct"] = df.apply(hit_topk, axis=1)

    summ = (
        df.groupby("block", observed=False)
        .agg(top1_acc=("top1_correct", "mean"), top2_acc=("top2_correct", "mean"))
        .reset_index()
    )
    summ[["top1_acc", "top2_acc"]] *= 100
    return summ


# --------------------------------------------------------------
# 3.  Build CSVs & charts
# --------------------------------------------------------------
for metric in METRICS:
    rows = []
    for player in PLAYERS:
        for m_name, m_key in MODELS.items():
            try:
                summ = load_or_make_summary(player, m_key)
            except FileNotFoundError as e:
                print("⚠️", e)
                continue
            best = summ.loc[summ[metric].idxmax()]
            rows.append(
                dict(
                    player=player,
                    model=m_name,
                    accuracy=best[metric],
                    block=best["block"],
                    cat_id=BLOCK_ID_MAP.get(best["block"], -1),
                )
            )
    df = pd.DataFrame(rows)
    if df.empty:
        continue

    csv_out = f"plots/figures/best_{metric}.csv"
    df.to_csv(csv_out, index=False)
    print(f"✓ Saved {csv_out}")

    # pivot for heights
    piv = df.pivot(index="player", columns="model", values="accuracy").reindex(PLAYERS)

    # ------------ plotting ------------
    n_p, n_m = len(PLAYERS), len(MODEL_ORDER)
    bar_w, gap = 0.24, 0.35
    centres = np.arange(n_p) * (n_m * bar_w + gap)

    fig_h = 8
    fig, ax = plt.subplots(figsize=(1.1 * n_p + 4, fig_h), dpi=150)
    ax.set_facecolor("white")

    for j, m_name in enumerate(MODEL_ORDER):
        xs = centres + (j - (n_m - 1) / 2) * bar_w
        ax.bar(
            xs,
            piv[m_name],
            width=bar_w,
            color=MODEL2COLOR[m_name],
            edgecolor="black",
            linewidth=0.5,
            label=m_name if metric == METRICS[0] else None,
        )

        for x, player in zip(xs, PLAYERS):
            cid = df.loc[(df.player == player) & (df.model == m_name), "cat_id"].iat[0]
            ax.text(x, -2, str(cid), ha="center", va="top", fontsize=9)

    # axes & legend
    ax.set_xticks(centres)
    ax.set_xticklabels(PLAYERS, rotation=18, ha="right")
    ylab = (
        "Best Top-1 Accuracy (%)" if metric == "top1_acc" else "Best Top-2 Accuracy (%)"
    )
    ax.set_ylabel(ylab)
    ax.set_ylim(-8, 100)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.65)
    ax.spines[["right", "top"]].set_visible(False)

    ax.legend(
        handles=[plt.Line2D([0], [0], color=MODEL2COLOR[m], lw=8) for m in MODEL_ORDER],
        labels=MODEL_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        frameon=False,
        ncol=4,
        columnspacing=0.8,
    )
    ax.set_title(f"{ylab} by Player and Model", pad=18, weight="semibold")

    # -------- mapping legend below --------
    id_to_block = {v: k.replace("\n", " ").strip() for k, v in BLOCK_ID_MAP.items()}
    legend_txt = "\n".join(f"{i} – {id_to_block[i]}" for i in range(9))
    fig.subplots_adjust(bottom=0.25)
    sub_ax = fig.add_axes([0, 0, 1, 0.2])
    sub_ax.axis("off")
    sub_ax.text(
        0.01,
        0.05,
        "Category ID Mapping:\n" + legend_txt,
        ha="left",
        va="bottom",
        fontsize=9,
        family="monospace",
    )

    plt.tight_layout(rect=[0, 0.25, 1, 0.93])
    out_png = f"plots/figures/best_{metric}_grouped_clean.png"
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {out_png}")
