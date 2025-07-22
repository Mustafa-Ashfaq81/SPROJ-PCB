import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm

# ========== Setup ==========
players = [
    "Babar Azam", "Charith Asalanka", "Daryl Mitchell",
    "Harry Tector", "Ibrahim Zadran", "Shai Hope", "Shubman Gill"
]

teams = {
    "Babar Azam": "Pakistan",
    "Charith Asalanka": "Sri Lanka",
    "Daryl Mitchell": "New Zealand",
    "Harry Tector": "Ireland",
    "Ibrahim Zadran": "Afghanistan",
    "Shai Hope": "West Indies",
    "Shubman Gill": "India",
}

models = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1", "o4-mini"]
out_dir = "plots/aggregated_by_player"
os.makedirs(out_dir, exist_ok=True)

# ========== Wilson Score ==========
def wilson_score_interval(successes, total, confidence=0.95):
    if total == 0:
        return 0.0, 0.0
    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / total
    denom = 1 + (z ** 2) / total
    center = (p_hat + (z ** 2) / (2 * total)) / denom
    margin = (z / denom) * np.sqrt((p_hat * (1 - p_hat)) / total + (z ** 2) / (4 * total ** 2))
    return max(0.0, center - margin), min(1.0, center + margin)

# ========== Accuracy Calculation ==========
def parse_float_list(txt): return [float(t.strip()) for t in txt.split(",")]
def parse_label_list(txt): return [t.strip() for t in txt.split(",")]

def topk_hit(row, k=2):
    probs, labels = row["prob_vec"], row["label_vec"]
    gt = row["mode_of_dismissal_ground_truth"].strip()
    if gt not in labels:
        return False
    topk_ix = np.argsort(probs)[-k:]
    return gt in {labels[i] for i in topk_ix}

# ========== Main Aggregation ==========
for player in players:
    all_dfs = []
    for model in models:
        file = f"{player}_prediction_results_{model}.csv"
        if not os.path.exists(file):
            print(f"[WARNING] File not found: {file}")
            continue
        df = pd.read_csv(file)
        df["prob_vec"] = df["mode_of_dismissal_predicted_probability"].apply(parse_float_list)
        df["label_vec"] = df["mode_of_dismissal_predicted"].apply(parse_label_list)
        df["top1_correct"] = df["most_likely_mode"].str.strip() == df["mode_of_dismissal_ground_truth"].str.strip()
        df["top2_correct"] = df.apply(topk_hit, axis=1)

        # Normalize flags to 0/1
        flag_cols = ["general_stats_flag", "venuewise_stats_flag", "against_teams_stats_flag", "commentary"]
        df[flag_cols] = df[flag_cols].replace({"on": 1, "off": 0})

        # Construct feature block
        def make_block(row):
            parts = []
            if row["general_stats_flag"]: parts.append("General Stats")
            if row["venuewise_stats_flag"]: parts.append("Venue Stats")
            if row["against_teams_stats_flag"]: parts.append("Vs-Team Stats")
            if row["commentary"]: parts.append("Commentary")
            return " + ".join(parts) if parts else "Baseline\n(all off)"
        
        df["block"] = df.apply(make_block, axis=1)
        all_dfs.append(df)

    if not all_dfs:
        print(f"[SKIP] No data found for {player}")
        continue

    combined = pd.concat(all_dfs, ignore_index=True)

    # Compute true top modes from ground truth distribution
    dismissal_counts = combined["mode_of_dismissal_ground_truth"].str.strip().value_counts()
    top1_mode = dismissal_counts.index[0]
    top2_modes = dismissal_counts.index[:2]

    # Approximate per-match baseline by first 1/N rows per block
    blocks = combined["block"].unique()
    n_blocks = len(blocks)
    n_matches = len(combined) // n_blocks
    df_true = combined.iloc[:n_matches].copy()

    df_true["top1_true_mode_correct"] = df_true["mode_of_dismissal_ground_truth"].str.strip() == top1_mode
    df_true["top2_true_mode_correct"] = df_true["mode_of_dismissal_ground_truth"].str.strip().isin(top2_modes)

    true_n = len(df_true)
    true_top1_acc = df_true["top1_true_mode_correct"].mean() * 100
    true_top2_acc = df_true["top2_true_mode_correct"].mean() * 100

    # Create Random Classifier and Top Career Dismissal rows
    row_random = {
        "block": "Random Classifier",
        "n": true_n,
        "top1_successes": round(true_n * 0.125),
        "top2_successes": round(true_n * 0.25),
    }
    row_random["top1_acc"] = row_random["top1_successes"] / row_random["n"] * 100
    row_random["top2_acc"] = row_random["top2_successes"] / row_random["n"] * 100

    row_true = {
        "block": "Top Career Dismissal",
        "n": true_n,
        "top1_successes": df_true["top1_true_mode_correct"].sum(),
        "top2_successes": df_true["top2_true_mode_correct"].sum(),
    }
    row_true["top1_acc"] = row_true["top1_successes"] / row_true["n"] * 100
    row_true["top2_acc"] = row_true["top2_successes"] / row_true["n"] * 100

    # Group real predictions by block
    summary = (
        combined.groupby("block", observed=False)
        .agg(
            n=("top1_correct", "size"),
            top1_successes=("top1_correct", "sum"),
            top2_successes=("top2_correct", "sum"),
        )
        .reset_index()
    )
    summary["top1_acc"] = summary["top1_successes"] / summary["n"] * 100
    summary["top2_acc"] = summary["top2_successes"] / summary["n"] * 100

    # Combine all rows
    summary = pd.concat([pd.DataFrame([row_random, row_true]), summary], ignore_index=True)

    # Sort with Random and True at top
    block_order = ["Random Classifier", "Top Career Dismissal"] + sorted(summary["block"].unique()[2:], key=lambda x: x)
    summary["block"] = pd.Categorical(summary["block"], categories=block_order, ordered=True)
    summary = summary.sort_values("block")


    # Wilson CI
    ci_lows_1, ci_highs_1, err_below_1, err_above_1 = [], [], [], []
    ci_lows_2, ci_highs_2, err_below_2, err_above_2 = [], [], [], []

    for _, row in summary.iterrows():
        n = row["n"]
        top1_lo, top1_hi = wilson_score_interval(row["top1_successes"], n)
        top2_lo, top2_hi = wilson_score_interval(row["top2_successes"], n)
        acc1 = row["top1_acc"]
        acc2 = row["top2_acc"]
        ci_lows_1.append(top1_lo * 100)
        ci_highs_1.append(top1_hi * 100)
        ci_lows_2.append(top2_lo * 100)
        ci_highs_2.append(top2_hi * 100)
        err_below_1.append(acc1 - top1_lo * 100)
        err_above_1.append(top1_hi * 100 - acc1)
        err_below_2.append(acc2 - top2_lo * 100)
        err_above_2.append(top2_hi * 100 - acc2)

    # Plot
    def plot_bar(summary, col, err_low, err_high, title, fname):
        fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
        bars = ax.bar(summary["block"], summary[col], yerr=[err_low, err_high],
                      capsize=6, ecolor="black", color="#2A788E", edgecolor="black", linewidth=0.5)

        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    height / 2,  # halfway up the bar
                    f"{height:.1f}%",
                    ha="center", va="center",
                    color="white" if height > 10 else "black",
                    fontsize=9, weight="bold")

        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
        ax.set_ylim(0, min(105, max(summary[col] + err_high) + 5))
        ax.set_xticklabels(summary["block"], rotation=30, ha="right")
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.65)
        ax.spines[["right", "top"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"[SAVED] {fname}")

    plot_bar(
        summary, "top1_acc", err_below_1, err_above_1,
        f"{player} – Aggregated Top‑1 Accuracy (95% CI)",
        os.path.join(out_dir, f"{player}_top1_accuracy.png")
    )

    plot_bar(
        summary, "top2_acc", err_below_2, err_above_2,
        f"{player} – Aggregated Top‑2 Accuracy (95% CI)",
        os.path.join(out_dir, f"{player}_top2_accuracy.png")
    )

