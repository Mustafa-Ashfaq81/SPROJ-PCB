#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beautiful accuracy comparisons for LLM vs. traditional models
• Generates four publication‑ready bar charts
• Saves them at 300 DPI in the current folder
"""

import json, pathlib, textwrap
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ───────────────────────────── SETTINGS ──────────────────────────────
LLM_JSON = "llm_comparison_results.json"
TRAD_JSON = "traditional_model_results.json"
OUT_DIR = pathlib.Path("plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)  # ← creates the 'plots/' folder
PALETTE = ["#386cb0", "#7fc97f", "#f0027f", "#fdc086", "#beaed4"]

sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update(
    {
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    }
)

# ──────────────────────────── LOAD & TIDY ────────────────────────────
with open(LLM_JSON) as f:
    llm = json.load(f)
with open(TRAD_JSON) as f:
    trad = json.load(f)

overall_df = pd.DataFrame(
    {
        "Group": ["Traditional", "LLM"],
        "Top‑1": [
            trad["best_performance"]["top1_accuracy"],
            llm["overall_llm_performance"]["top1_accuracy"],
        ],
        "Top‑2": [
            trad["best_performance"]["top2_accuracy"],
            llm["overall_llm_performance"]["top2_accuracy"],
        ],
    }
)

llm_df = pd.DataFrame(
    [
        {
            "Model": m["model_name"],
            "Top‑1": m["top1_accuracy"],
            "Top‑2": m["top2_accuracy"],
        }
        for m in llm["all_models_individual_performance"].values()
    ]
).sort_values("Top‑1", ascending=False)

trad_df = pd.DataFrame(
    [
        {"Model": k, "Top‑1": v["top1_accuracy"], "Top‑2": v["top2_accuracy"]}
        for k, v in trad["model_results"].items()
    ]
).sort_values("Top‑1", ascending=False)

# Combined for full model comparison plot
df_all_models = pd.concat(
    [llm_df.assign(Type="LLM"), trad_df.assign(Type="Traditional")]
)
df_all_models["Model"] = df_all_models["Model"].astype(str)
df_all_models = df_all_models.sort_values("Top‑1", ascending=False)


# ──────────────────────────── UTILITIES ──────────────────────────────
def barplot(df, index_col, title, fname, colors):
    ax = df.set_index(index_col).plot(
        kind="bar",
        figsize=(8, 5),
        color=colors[: len(df.columns)],
        width=0.75,
        edgecolor="0.2",
    )
    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda y, pos: f"{y:.0%}")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{v:.1%}" for v in container.datavalues],
            label_type="edge",
            padding=3,
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=300)
    plt.close()


def barplot_grouped(df, index_col, title, fname, colors):
    """Grouped bar chart for multiple models and metrics (Top‑1 / Top‑2)."""
    df = df.copy()
    df = df[[index_col, "Top‑1", "Top‑2"]].set_index(index_col)
    df = df.sort_values("Top‑1", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    x = range(len(df))
    bars1 = ax.bar(
        [i - width / 2 for i in x],
        df["Top‑1"],
        width=width,
        label="Top‑1",
        color=colors[0],
        edgecolor="0.2",
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        df["Top‑2"],
        width=width,
        label="Top‑2",
        color=colors[1],
        edgecolor="0.2",
    )

    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda y, pos: f"{y:.0%}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False)

    # Add labels on top of bars
    for bars in [bars1, bars2]:
        ax.bar_label(
            bars, labels=[f"{v:.1%}" for v in bars.datavalues], padding=3, fontsize=9
        )

    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=300)
    plt.close()


# NEW FUNCTION — Combined grouped barplot for all models
def barplot_all_models(df, title, fname):
    melted = df.melt(
        id_vars=["Model", "Type"],
        value_vars=["Top‑1", "Top‑2"],
        var_name="Metric",
        value_name="Accuracy",
    )
    melted["Model"] = pd.Categorical(
        melted["Model"], categories=df["Model"], ordered=True
    )

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=melted,
        x="Model",
        y="Accuracy",
        hue="Metric",
        palette={"Top‑1": "#386cb0", "Top‑2": "#7fc97f"},
    )
    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    plt.xticks(rotation=30, ha="right")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)

    plt.tight_layout()
    plt.legend(title="")
    plt.savefig(OUT_DIR / fname, dpi=300)
    plt.close()


# ───────────────────────────── PLOTS ─────────────────────────────────
barplot(
    overall_df,
    "Group",
    "Overall Accuracy: LLM vs. Traditional",
    "plot_overall_llm_vs_traditional.png",
    PALETTE,
)
barplot(
    llm_df, "Model", "Accuracy by LLM Model", "plot_llm_model_accuracy.png", PALETTE
)
barplot(
    trad_df,
    "Model",
    "Accuracy by Traditional Model",
    "plot_traditional_model_accuracy.png",
    PALETTE,
)

# NEW PLOT: All Models Grouped Top‑1/Top‑2
barplot_grouped(
    df_all_models[["Model", "Top‑1", "Top‑2"]],
    "Model",
    "Top‑1 and Top‑2 Accuracy: All Models (LLM + Traditional)",
    "plot_all_models_grouped_accuracy.png",
    PALETTE,
)
print(
    textwrap.dedent(
        f"""
✅ Plots saved to: {OUT_DIR.resolve()}
 • plot_overall_llm_vs_traditional.png
 • plot_llm_model_accuracy.png
 • plot_traditional_model_accuracy.png
 • plot_all_models_grouped_accuracy.png  ← new!
"""
    )
)
