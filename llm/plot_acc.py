import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
import subprocess
import json

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
})

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
PLAYER = "Charith Asalanka"
MODEL = "gpt-4.1"
TEAM = "Sri Lanka"
CSV_PATH = f"{PLAYER}_prediction_results_{MODEL}.csv"
PLOTS_DIR = f"plots/{PLAYER}/{MODEL}"
os.makedirs(PLOTS_DIR, exist_ok=True)
plot_path = lambda fname: os.path.join(PLOTS_DIR, fname)
TOP1_PLOT_PATH   = plot_path("top1_accuracy.png")
TOP2_PLOT_PATH   = plot_path("top2_accuracy.png")
WATER_TOP1_PNG   = plot_path("waterfall_top1.png")
WATER_TOP2_PNG   = plot_path("waterfall_top2.png")
ERR_TOP1_PNG     = plot_path("top1_accuracy_ci.png")
ERR_TOP2_PNG     = plot_path("top2_accuracy_ci.png")
HEAT_PNG         = plot_path("recall_heatmap.png")

# ---------------------------------------------------------------
# NEW: Utilities for loading true dismissal stats
# ---------------------------------------------------------------

def run_subprocess(command):
    """Run a shell command and raise error if it fails."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Subprocess failed: {command}")
    return result.stdout

def load_json_file(path):
    """Load a JSON file from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
try:
    df = pd.read_csv(CSV_PATH, quotechar='"')
except FileNotFoundError as e:
    raise FileNotFoundError(
        f"CSV file not found at '{CSV_PATH}'. "
        "Place the CSV in the working directory or update CSV_PATH."
    ) from e

df = df.reset_index(drop=True)

# ------------------------------------------------------------------
# NEW: Compute True Dismissal statistics
# ------------------------------------------------------------------

# 1) Run external script to generate player stats
print("\n[INFO] Running get_player_stats.py ...")
safe_player_name = PLAYER.lower().replace(' ', '_')
cmd_general = (
    f"python ../cricinfo_data_scrapping/Scripts/get_player_stats.py "
    f"\"{PLAYER}\" \"{TEAM}\" all all"
)
try:
    run_subprocess(cmd_general)
except RuntimeError:
    print("[WARNING] get_player_stats script may have failed.")

# 2) Load JSON output
general_stats_filename = f"{safe_player_name}_all_span_stats.json"
general_stats_path = os.path.join(
    "..", "cricinfo_data_scrapping", "aggregated_stats", general_stats_filename
)
general_stats = load_json_file(general_stats_path)

# 3) Sum dismissal frequencies
dismissal_map = {
    "bowled": "Bowled",
    "caught-fielder": "Caught-Fielder",
    "caught wk": "Caught-Wicketkeeper",
    "lbw": "LBW",
    "stump": "Stumped",
    "hit wicket": "Hit-Wicket",
    "caught and bowled": "Caught-and-Bowled",
}

agg_counts = {label: 0 for label in dismissal_map.values()}
for entry in general_stats["dismissals"]:
    for key, label in dismissal_map.items():
        agg_counts[label] += entry.get(key, 0)

# Add Not-Outs
agg_counts["Not-Out"] = general_stats["batting"]["Not Outs"]
print(agg_counts)

# Determine top-1 mode
top1_mode = max(agg_counts.items(), key=lambda x: x[1])[0]

# Determine top-2 modes
sorted_modes = sorted(
    agg_counts.items(), key=lambda x: x[1], reverse=True
)
top2_modes = [sorted_modes[0][0]]
if len(sorted_modes) > 1:
    top2_modes.append(sorted_modes[1][0])

print(f"[INFO] True top-1 mode: {top1_mode}")
print(f"[INFO] True top-2 modes: {top2_modes}")

# ------------------------------------------------------------------
# 2. Normalize feature flag formats ("on"/"off" → 1/0)
# ------------------------------------------------------------------
flag_cols = [
    "general_stats_flag",
    "venuewise_stats_flag",
    "against_teams_stats_flag",
    "commentary"
]

# Only convert if these columns are present
existing_flag_cols = [col for col in flag_cols if col in df.columns]
df[existing_flag_cols] = df[existing_flag_cols].replace({"on": 1, "off": 0})

# ------------------------------------------------------------------
# 3. Dynamically Create Feature Block Labels
# ------------------------------------------------------------------
def make_block_label(row):
    parts = []
    if row.get("general_stats_flag", 0):
        parts.append("General Stats")
    if row.get("venuewise_stats_flag", 0):
        parts.append("Venue Stats")
    if row.get("against_teams_stats_flag", 0):
        parts.append("Vs-Team Stats")
    if row.get("commentary", 0):
        parts.append("Commentary")
    return " + ".join(parts) if parts else "Baseline\n(all off)"

df["block"] = df.apply(make_block_label, axis=1)
unique_order = df["block"].drop_duplicates().tolist()
df["block"] = pd.Categorical(df["block"], categories=unique_order, ordered=True)

# ------------------------------------------------------------------
# 4. Parse probability & label lists
# ------------------------------------------------------------------
def parse_float_list(txt: str):
    return [float(t.strip()) for t in txt.split(",")]

def parse_label_list(txt: str):
    return [t.strip() for t in txt.split(",")]

df["prob_vec"] = df["mode_of_dismissal_predicted_probability"].apply(parse_float_list)
df["label_vec"] = df["mode_of_dismissal_predicted"].apply(parse_label_list)

# ------------------------------------------------------------------
# 5. Top-1 & Top-2 correctness flags
# ------------------------------------------------------------------
df["top1_correct"] = (
    df["most_likely_mode"].str.strip()
    == df["mode_of_dismissal_ground_truth"].str.strip()
)

def topk_hit(row, k=2):
    probs, labels = row["prob_vec"], row["label_vec"]
    gt = row["mode_of_dismissal_ground_truth"].strip()
    if gt not in labels:
        return False
    topk_ix = np.argsort(probs)[-k:]
    topk_labels = {labels[i] for i in topk_ix}
    return gt in topk_labels

df["top2_correct"] = df.apply(topk_hit, axis=1)

# ------------------------------------------------------------------
# NEW: Compute true-mode correctness
# ------------------------------------------------------------------

# Add constant columns for true top-1 and top-2 modes
df["most_frequent_mode"] = top1_mode
df["top1_true_mode_correct"] = (
    df["most_frequent_mode"].str.strip() ==
    df["mode_of_dismissal_ground_truth"].str.strip()
)

df["top2_true_mode_correct"] = df["mode_of_dismissal_ground_truth"].str.strip().isin(top2_modes)

# Compute overall true top-1 and top-2 accuracy
true_top1_accuracy = df["top1_true_mode_correct"].mean() * 100
true_top2_accuracy = df["top2_true_mode_correct"].mean() * 100

# Store n
true_n = len(df)

print(f"[INFO] True Top-1 Accuracy: {true_top1_accuracy:.2f}%")
print(f"[INFO] True Top-2 Accuracy: {true_top2_accuracy:.2f}%")

# ------------------------------------------------------------------
# 6. Aggregate accuracy by block
# ------------------------------------------------------------------
summary = (
    df.groupby("block", observed=False)
      .agg(top1_acc=("top1_correct", "mean"),
           top2_acc=("top2_correct", "mean"),
           n=("top1_correct", "size"))
      .reset_index()
)

summary["block"] = pd.Categorical(summary["block"], categories=unique_order, ordered=True)
summary = summary.sort_values("block")

summary["top1_acc"] = (summary["top1_acc"] * 100).round(2)
summary["top2_acc"] = (summary["top2_acc"] * 100).round(2)

# ------------------------------------------------------------------
# NEW: Add True Dismissal row to summary
# ------------------------------------------------------------------

true_row = pd.DataFrame([{
    "block": "Top Career Dismissal",
    "top1_acc": round(true_top1_accuracy, 2),
    "top2_acc": round(true_top2_accuracy, 2),
    "n": true_n
}])

# Prepend True dismissal row at the start
summary = pd.concat([true_row, summary], ignore_index=True)

# Update categorical ordering
new_order = ["Top Career Dismissal"] + unique_order
summary["block"] = pd.Categorical(summary["block"], categories=new_order, ordered=True)
summary = summary.sort_values("block")


print("\n=== Accuracy per Feature Block ===")
print(summary.to_string(index=False))

# ------------------------------------------------------------------
# 7. Bar Plots
# ------------------------------------------------------------------
def make_bar(summary_col: str, ylabel: str, title: str, save_path: str):
    values = summary[summary_col].values
    colors = []
    for i in range(len(values)):
        if i == 0:
            colors.append("#999999")   # baseline grey
        else:
            colors.append("#2A788E")   # teal

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    bars = ax.bar(
        summary["block"],
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.set_xticklabels(summary["block"], rotation=25, ha="right")

    # Add grid and style
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.65)
    ax.spines[["right", "top"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {os.path.abspath(save_path)}")

make_bar("top1_acc", "Top-1 Accuracy (%)",
         "Ablation Study - Top-1 Accuracy per Feature Set", TOP1_PLOT_PATH)
make_bar("top2_acc", "Top-2 Accuracy (%)",
         "Ablation Study - Top-2 Accuracy per Feature Set", TOP2_PLOT_PATH)

# ------------------------------------------------------------------
# 8. Waterfall Plots
# ------------------------------------------------------------------
summary["d_top1"] = summary["top1_acc"].diff().fillna(summary["top1_acc"])
summary["d_top2"] = summary["top2_acc"].diff().fillna(summary["top2_acc"])

def waterfall(col_delta: str, col_cum: str, out_path: str, title: str):
    dx = summary["block"]
    delta = summary[col_delta]
    cum = summary[col_cum]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    base = 0
    for i, (lbl, gain) in enumerate(zip(dx, delta)):
        color = "#2A788E" if gain >= 0 else "#A52A2A"
        ax.bar(
            i,
            gain,
            bottom=base,
            width=0.6,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.text(
            i,
            base + gain / 2,
            f"{gain:+.2f} pp",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            weight="bold",
        )
        base += gain

    ax.set_xticks(range(len(dx)))
    ax.set_xticklabels(dx, rotation=25, ha="right")
    ax.set_ylabel("Top Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)

    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.65)
    ax.spines[["right", "top"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {os.path.abspath(out_path)}")

waterfall("d_top1", "top1_acc", WATER_TOP1_PNG, "Incremental Lift – Top‑1 Accuracy")
waterfall("d_top2", "top2_acc", WATER_TOP2_PNG, "Incremental Lift – Top‑2 Accuracy")

# ------------------------------------------------------------------
# 9. Error Bar Plots
# ------------------------------------------------------------------
def save_errorbar(col_acc: str, out_path: str, title: str):
    z = norm.ppf(0.975)
    p_hat = summary[col_acc] / 100.0
    n = summary["n"]
    center = (p_hat + (z**2) / (2 * n)) / (1 + (z**2) / n)
    margin = (z * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))) / (1 + (z**2) / n)

    summary["ci_low"] = (center - margin) * 100
    summary["ci_high"] = (center + margin) * 100
    summary["error_below"] = summary[col_acc] - summary["ci_low"]
    summary["error_above"] = summary["ci_high"] - summary[col_acc]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    bars = ax.bar(
        summary["block"],
        summary[col_acc],
        yerr=[summary["error_below"], summary["error_above"]],
        capsize=6,
        color="#2A788E",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title + " (95% Wilson CI with bounds)")
    ax.set_ylim(0, 100)

    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.65)
    ax.spines[["right", "top"]].set_visible(False)

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {os.path.abspath(out_path)}")

save_errorbar("top1_acc", ERR_TOP1_PNG, "Top‑1 Accuracy per Feature Set")
save_errorbar("top2_acc", ERR_TOP2_PNG, "Top‑2 Accuracy per Feature Set")

# ------------------------------------------------------------------
# 10. Recall Heatmap (first vs last block)
# ------------------------------------------------------------------
labels = df["label_vec"].iloc[0]

def recall_for_block(block_name):
    sub = df[df["block"] == block_name]
    gt = sub["mode_of_dismissal_ground_truth"].str.strip()
    pred = sub["most_likely_mode"].str.strip()
    recalls = []
    for lab in labels:
        total = (gt == lab).sum()
        correct = ((gt == lab) & (pred == lab)).sum()
        recalls.append(correct / total if total else np.nan)
    return np.array(recalls)

blocks_cmp = ["Top Career Dismissal", unique_order[-1]]
rec_baseline = recall_for_block("Top Career Dismissal")
rec_full = recall_for_block(unique_order[-1])
rec_matrix = np.vstack([rec_baseline, rec_full]) * 100

fig, ax = plt.subplots(figsize=(11, 3))
im = ax.imshow(rec_matrix, cmap="YlOrRd", vmin=0, vmax=100)
for i in range(rec_matrix.shape[0]):
    for j in range(rec_matrix.shape[1]):
        val = rec_matrix[i, j]
        txt = f"{val:.0f}" if not np.isnan(val) else "–"
        ax.text(j, i, txt, ha="center", va="center", color="black" if val < 60 else "white", fontsize=8)
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=40, ha="right")
ax.set_yticks([0, 1])
ax.set_yticklabels(blocks_cmp)
ax.set_title("Per‑Class Recall (%) – Baseline vs All Features")
cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
cbar.set_label("Recall (%)")
plt.tight_layout()
plt.savefig(HEAT_PNG, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved plot → {os.path.abspath(HEAT_PNG)}")

# ------------------------------------------------------------------
# 11. McNemar Significance Test
# ------------------------------------------------------------------
try:
    from statsmodels.stats.contingency_tables import mcnemar
    print("\n=== McNemar p-values (Top-1) ===")
    for i in range(1, len(unique_order)):
        prev = df[df["block"] == unique_order[i - 1]]["top1_correct"].values
        curr = df[df["block"] == unique_order[i]]["top1_correct"].values
        contingency = [[np.sum(prev & curr), np.sum(prev & ~curr)],
                       [np.sum(~prev & curr), np.sum(~prev & ~curr)]]
        res = mcnemar(contingency, exact=False, correction=True)
        print(f"{unique_order[i-1]} → {unique_order[i]}: p = {res.pvalue:.4f}")
except ImportError:
    print("\n(statsmodels not installed – McNemar test skipped)")

# ------------------------------------------------------------------
# 12. Save Accuracy Summary CSV
# ------------------------------------------------------------------
summary.to_csv(f"{PLOTS_DIR}/ablation_accuracy_summary.csv", index=False)
print("Saved extended summary →", os.path.abspath(f"{PLOTS_DIR}/ablation_accuracy_summary.csv"))
