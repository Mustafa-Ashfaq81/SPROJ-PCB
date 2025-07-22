import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# --- CONFIG ---
MODELS = {
    "gpt-4.1 nano": "gpt-4.1-nano",
    "gpt-4.1 mini": "gpt-4.1-mini",
    "gpt-4.1": "gpt-4.1",
    "gpt-o4 mini": "o4-mini",
}

MODEL_ORDER = [
    "gpt-4.1 nano",
    "gpt-4.1 mini",
    "gpt-4.1",
    "gpt-o4 mini",
]

MODEL_COLOURS = [
    "#440154",  # nano → dark purple
    "#414487",  # mini → indigo
    "#2A788E",  # 4.1 → teal
    "#22A884",  # o4-mini → bright green
]

MODEL2COLOR = dict(zip(MODEL_ORDER, MODEL_COLOURS))

EXTRA_BAR_COLOR = {
    -2: "#999999",  # Random
    # TCD will be handled inline below
}

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

SUMMARY_DIR = "plots/aggregated_by_model"
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(os.path.join(SUMMARY_DIR, "best_category_by_model"), exist_ok=True)

METRICS = ["top1_acc", "top2_acc"]

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "font.family": "monospace",
})

from scipy.stats import norm

def wilson_ci(successes, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return (centre - margin) * 100, (centre + margin) * 100


# === PLOTTING ===
for metric in METRICS:
    rows = []
    for model_label, model_key in MODELS.items():
        summary_path = os.path.join(SUMMARY_DIR, f"{model_key}_summary.csv")
        if not os.path.exists(summary_path):
            print(f"⚠️ Missing summary: {summary_path}")
            continue
        df = pd.read_csv(summary_path)
        df["block"] = df["block"].str.strip()  # Clean before any filtering

        # Keep regular blocks and TCD
        valid_blocks = list(BLOCK_ID_MAP.keys()) + ["Top Career Dismissal"]
        df = df[df["block"].isin(valid_blocks)]

        top2 = df.nlargest(2, metric)
        # Add Random Classifier baseline (1/8 for top1, 2/8 for top2)
        rand_n = int(df["n"].iloc[0])  # use same sample size as other blocks
        rand_successes = int(rand_n * (1 / 8 if metric == "top1_acc" else 2 / 8))  # floor to integer
        rand_acc = 12.5 if metric == "top1_acc" else 25.0
        low, high = wilson_ci(rand_successes, rand_n)

        rows.append(dict(
            model=model_label,
            block="Random Classifier",
            accuracy=rand_acc,
            cat_id=-2,
            ci_low=low,
            ci_high=high
        ))

        # Add Top Career Dismissal from summary row
        df["block"] = df["block"].str.strip()  # add this line to clean all
        top_row = df[df["block"] == "Top Career Dismissal"]

        if not top_row.empty:
            # print(f"[✔] TCD added for {model_label}, {top_row}")
            row = top_row.iloc[0]
            successes = row["top1_successes"] if metric == "top1_acc" else row["top2_successes"]
            n = row["n"]
            acc = row[metric]
            low, high = wilson_ci(successes, n)
            rows.append(dict(
                model=model_label,
                block="Top Career Dismissal",
                accuracy=acc,
                cat_id=-1,
                ci_low=low,
                ci_high=high
            ))

        for _, row in top2.iterrows():
            if metric == "top1_acc":
                successes = row["top1_successes"]
            elif metric == "top2_acc":
                successes = row["top2_successes"]
            n = row["n"]
            low, high = wilson_ci(successes, n)

            rows.append(dict(
                model=model_label,
                block=row["block"],
                accuracy=row[metric],
                cat_id=BLOCK_ID_MAP.get(row["block"], -1),
                ci_low=low,
                ci_high=high
            ))

    final_df = pd.DataFrame(rows)
    if final_df.empty:
        print(f"No data found for {metric}")
        continue

    final_df.to_csv(os.path.join(SUMMARY_DIR, f"best_category_by_model/best_{metric}_top2_per_model.csv"), index=False)

    # === Pivot table ===
    pivot = final_df.pivot(index="model", columns="cat_id", values="accuracy").reindex(MODEL_ORDER)

    n_models, n_blocks = pivot.shape
    bar_w, gap = 0.6, 0.7
    centres = np.arange(n_models) * (n_blocks * bar_w + gap)

    fig, ax = plt.subplots(figsize=(1.3 * n_models + 4, 7), dpi=150)
    ax.set_facecolor("white")

    for i, model in enumerate(MODEL_ORDER):
        # 1. Filter all rows for current model
        model_df = final_df[final_df["model"] == model]

        # 2. Extract the two extra bars
        extras_df = model_df[model_df["cat_id"].isin([-1, -2])]

        # 3. Select top 2 performing feature blocks (cat_id ≥ 0)
        top2_df = model_df[model_df["cat_id"] >= 0].nlargest(2, "accuracy")

        # 4. Combine and sort all 4 bars by accuracy
        plot_df = pd.concat([extras_df, top2_df], ignore_index=True)
        plot_df = plot_df.sort_values("accuracy").reset_index(drop=True)

        # 5. Prepare data for plotting
        blocks = plot_df["cat_id"].tolist()
        heights = plot_df["accuracy"].tolist()
        xs = centres[i] + np.arange(len(blocks)) * bar_w

        colors = []
        for cid in blocks:
            if cid in [-2, -1]:  # both random classifier and TCD → grey
                colors.append(EXTRA_BAR_COLOR[-2])
            else:
                colors.append(MODEL2COLOR[model])

        bars = ax.bar(xs, heights, width=bar_w, color=colors, 
                                    edgecolor="black", linewidth=0.5, label=model if metric == METRICS[0] else None)
        
        for x, block_id in zip(xs, blocks):
            r = plot_df[plot_df["cat_id"] == block_id].iloc[0]
            err_low = r["accuracy"] - r["ci_low"]
            err_high = r["ci_high"] - r["accuracy"]
            ax.errorbar(x, r["accuracy"], yerr=[[err_low], [err_high]], fmt="none", ecolor="black", capsize=4, linewidth=1)

        for x, h, cid in zip(xs, heights, blocks):
            print(f"[{model}] Block {cid}: {r['block']} → n = {int(n)}, accuracy = {r['accuracy']:.2f}%")
            ax.text(x, -2, str(cid), ha="center", va="top", fontsize=9)
            ax.text(x, h / 2 - 1.5, f"{h:.1f}%", ha="center", va="center", color="white" if h > 10 else "black", fontsize=7, weight="bold")

    ax.set_xticks(centres + 2 * bar_w)
    ax.set_xticklabels(MODEL_ORDER, rotation=18, ha="right")

    ylab = "Top‑1 Accuracy (%)" if metric == "top1_acc" else "Top‑2 Accuracy (%)"
    ax.set_ylabel(ylab)
    ax.set_ylim(-8, 100)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.65)
    ax.spines[["right", "top"]].set_visible(False)

    if metric == "top1_acc":
        ax.legend(
            handles=[plt.Line2D([0], [0], color=MODEL2COLOR[m], lw=8) for m in MODEL_ORDER],
            labels=MODEL_ORDER,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.07),
            frameon=False,
            ncol=4,
            columnspacing=0.8,
        )

    ax.set_title(f"Top 2 {ylab} Blocks per Model", pad=18, weight="semibold")

    # ID → category mapping below
    id_to_block = {v: k.replace("\n", " ").strip() for k, v in BLOCK_ID_MAP.items()}
    id_to_block[-2] = "Random Classifier"
    id_to_block[-1] = "Top Career Dismissal"
    legend_txt = "\n".join(f"{i:2} – {id_to_block[i]}" for i in sorted(id_to_block))

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
    out_path = os.path.join(SUMMARY_DIR, f"best_category_by_model/best_{metric}_top2_by_model_grouped_clean.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {out_path}")
