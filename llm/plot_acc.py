import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
CSV_PATH = "llm_prediction_results.csv"   # ← change if your file name differs
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
plot_path = lambda fname: os.path.join(PLOTS_DIR, fname)
TOP1_PLOT_PATH   = plot_path("top1_accuracy.png")
TOP2_PLOT_PATH   = plot_path("top2_accuracy.png")
WATER_TOP1_PNG   = plot_path("waterfall_top1.png")
WATER_TOP2_PNG   = plot_path("waterfall_top2.png")
ERR_TOP1_PNG     = plot_path("top1_accuracy_ci.png")
ERR_TOP2_PNG     = plot_path("top2_accuracy_ci.png")
HEAT_PNG         = plot_path("recall_heatmap.png")

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

# Keep deterministic row order
df = df.reset_index(drop=True)
# print(df["prediction_match"][98:102])

# ------------------------------------------------------------------
# 2. Map each row to its feature-block (100-row chunks)
# ------------------------------------------------------------------
block_labels = {
    0: "Baseline\n(all off)",
    1: "+General Stats",
    2: "+Venue Stats",
    3: "+Vs-Team Stats",
    4: "+Commentary",
}
df["_block_id"] = np.clip(df.index // 100, 0, 4) 
df["block"] = df["_block_id"].map(block_labels)

# ------------------------------------------------------------------
# 3. Parse probability & label lists
# ------------------------------------------------------------------
def parse_float_list(txt: str):
    """Convert comma-separated numerics to list[float]."""
    return [float(t.strip()) for t in txt.split(",")]

def parse_label_list(txt: str):
    """Convert comma-separated strings to list[str]."""
    return [t.strip() for t in txt.split(",")]

df["prob_vec"] = df["mode_of_dismissal_predicted_probability"].apply(parse_float_list)
df["label_vec"] = df["mode_of_dismissal_predicted"].apply(parse_label_list)

# ------------------------------------------------------------------
# 4. Top-1 & Top-2 correctness flags
# ------------------------------------------------------------------
# Top-1: simply compare most_likely_mode with ground truth
df["top1_correct"] = (
    df["most_likely_mode"].str.strip()
    == df["mode_of_dismissal_ground_truth"].str.strip()
)

# Top-2: derive 2 highest-prob labels
def topk_hit(row, k=2):
    probs, labels = row["prob_vec"], row["label_vec"]
    gt = row["mode_of_dismissal_ground_truth"].strip()
    if gt not in labels:                # ground truth outside 8-class set
        return False
    # indices of k largest probs
    topk_ix = np.argsort(probs)[-k:]
    topk_labels = {labels[i] for i in topk_ix}
    return gt in topk_labels

df["top2_correct"] = df.apply(topk_hit, axis=1)

# ------------------------------------------------------------------
# 5. Aggregate accuracy by block
# ------------------------------------------------------------------
summary = (
    df.groupby("block")
      .agg(top1_acc=("top1_correct", "mean"),
           top2_acc=("top2_correct", "mean"),
           n=("top1_correct", "size"))
      .reset_index()
)

order = ["Baseline\n(all off)", "+General Stats", "+Venue Stats", "+Vs-Team Stats", "+Commentary"]
summary["block"] = pd.Categorical(summary["block"], categories=order, ordered=True)
summary = summary.sort_values("block")

summary["top1_acc"] = (summary["top1_acc"] * 100).round(2)
summary["top2_acc"] = (summary["top2_acc"] * 100).round(2)

print("\n=== Accuracy per Feature Block ===")
print(summary.to_string(index=False))

# ------------------------------------------------------------------
# 6. Plotting helper
# ------------------------------------------------------------------
def make_bar(summary_col: str, ylabel: str, title: str, save_path: str):
    plt.figure(figsize=(8, 5))
    plt.bar(summary["block"], summary[summary_col])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 100)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {os.path.abspath(save_path)}")

# ------------------------------------------------------------------
# 7. Generate & save plots
# ------------------------------------------------------------------
make_bar("top1_acc",
         "Top-1 Accuracy (%)",
         "Ablation Study - Top-1 Accuracy per Feature Set",
         TOP1_PLOT_PATH)

make_bar("top2_acc",
         "Top-2 Accuracy (%)",
         "Ablation Study - Top-2 Accuracy per Feature Set",
         TOP2_PLOT_PATH)

# ------------------------------------------------------------------
# 8. Incremental‑gain “waterfall” plots  ----------------------------
# ------------------------------------------------------------------

# compute deltas (already sorted)
summary["d_top1"] = summary["top1_acc"].diff().fillna(summary["top1_acc"])
summary["d_top2"] = summary["top2_acc"].diff().fillna(summary["top2_acc"])

def waterfall(col_delta: str, col_cum: str, out_path: str, title: str):
    dx   = summary["block"]
    delta = summary[col_delta]
    cum   = summary[col_cum]
    fig, ax = plt.subplots(figsize=(9, 5))
    base = 0
    for i, (lbl, gain) in enumerate(zip(dx, delta)):
        color = "tab:blue" if gain >= 0 else "tab:red"
        ax.bar(i, gain, bottom=base, width=0.6, color=color)
        ax.text(i, base + gain/2, f"{gain:+.2f} pp",
                ha="center", va="center", color="white", fontsize=9, weight="bold")
        base += gain
    ax.set_xticks(range(len(dx)))
    ax.set_xticklabels(dx, rotation=25, ha="right")
    ax.set_ylabel("Top Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {os.path.abspath(out_path)}")

waterfall("d_top1", "top1_acc", WATER_TOP1_PNG,
          "Incremental Lift – Top‑1 Accuracy")

waterfall("d_top2", "top2_acc", WATER_TOP2_PNG,
          "Incremental Lift – Top‑2 Accuracy")

# ------------------------------------------------------------------
# 9. Bar charts with 95 % CI error bars  ---------------------------
# ------------------------------------------------------------------

def save_errorbar(col_acc: str, out_path: str, title: str):
    p  = summary[col_acc] / 100.0     # back to proportion
    n  = summary["n"]
    se = np.sqrt(p * (1 - p) / n)
    ci = 1.96 * se * 100              # ± in percentage points
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(summary["block"], summary[col_acc], yerr=ci,
           capsize=4, color="tab:purple")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title + " (95% CI)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {os.path.abspath(out_path)}")

save_errorbar("top1_acc", ERR_TOP1_PNG, "Top‑1 Accuracy per Feature Set")
save_errorbar("top2_acc", ERR_TOP2_PNG, "Top‑2 Accuracy per Feature Set")

# ------------------------------------------------------------------
# 10. Per‑class recall heat‑map (Baseline vs All‑features) ----------
# ------------------------------------------------------------------
labels = df["label_vec"].iloc[0]  # the 8 dismissal classes

def recall_for_block(bid):
    sub = df[df["_block_id"] == bid]
    gt  = sub["mode_of_dismissal_ground_truth"].str.strip()
    pred = sub["most_likely_mode"].str.strip()
    recalls = []
    for lab in labels:
        total = (gt == lab).sum()
        correct = ((gt == lab) & (pred == lab)).sum()
        recalls.append( correct / total if total else np.nan )
    return np.array(recalls)

rec_baseline = recall_for_block(0)
rec_full     = recall_for_block(4)
rec_matrix   = np.vstack([rec_baseline, rec_full]) * 100   # %
blocks_cmp   = ["Baseline", "All Features"]

# simple heatmap
fig, ax = plt.subplots(figsize=(11, 3))
im = ax.imshow(rec_matrix, cmap="YlOrRd", vmin=0, vmax=100)
for i in range(rec_matrix.shape[0]):
    for j in range(rec_matrix.shape[1]):
        val = rec_matrix[i, j]
        txt = f"{val:.0f}" if not np.isnan(val) else "–"
        ax.text(j, i, txt, ha="center", va="center",
                color="black" if val<60 else "white", fontsize=8)
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=40, ha="right")
ax.set_yticks([0,1])
ax.set_yticklabels(blocks_cmp)
ax.set_title("Per‑Class Recall (%) – Baseline vs All Features")
cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
cbar.set_label("Recall (%)")
plt.tight_layout()
plt.savefig(HEAT_PNG, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved plot → {os.path.abspath(HEAT_PNG)}")

# ------------------------------------------------------------------
# 11. McNemar significance test between adjacent blocks ------------
# ------------------------------------------------------------------
try:
    from statsmodels.stats.contingency_tables import mcnemar
    print("\n=== McNemar p-values (Top-1) ===")

    for b in range(1, 5):
        prev = df[df["_block_id"] == (b - 1)]["top1_correct"].values
        curr = df[df["_block_id"] == b]["top1_correct"].values
        # contingency: [[both correct, only prev], [only curr, both wrong]]
        contingency = [[np.sum(prev & curr),  np.sum(prev & ~curr)],
                    [np.sum(~prev & curr), np.sum(~prev & ~curr)]]
        res = mcnemar(contingency, exact=False, correction=True)
        print(f"Block {b-1} → {b}: p = {res.pvalue:.4f}")
except ImportError:
    print("\n(statsmodels not installed – McNemar test skipped)")

# ------------------------------------------------------------------
# 12. Save extended numeric summary to CSV -------------------------
# ------------------------------------------------------------------
summary.to_csv("ablation_accuracy_summary.csv", index=False)
print("Saved extended summary →", os.path.abspath("ablation_accuracy_summary.csv"))
