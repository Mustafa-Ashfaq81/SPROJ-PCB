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

# Baseline methods to exclude from LLM calculations
BASELINE_METHODS = {"Random Classifier", "Top Career Dismissal"}

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


# Calculate corrected overall LLM performance (excluding baselines)
def calculate_corrected_llm_performance(llm_data):
    """Calculate LLM performance excluding baseline methods"""
    total_top1_successes = 0
    total_top2_successes = 0
    total_scenarios = 0

    for model_data in llm_data["all_models_individual_performance"].values():
        # Filter out baseline methods from raw_summary_data
        actual_llm_configs = [
            config
            for config in model_data["raw_summary_data"]
            if config["block"] not in BASELINE_METHODS
        ]

        # Use the best performing configuration for each model
        best_config = max(actual_llm_configs, key=lambda x: x["top1_acc"])
        total_top1_successes += best_config["top1_successes"]
        total_top2_successes += best_config["top2_successes"]
        total_scenarios += best_config["n"]

    return {
        "total_scenarios": total_scenarios,
        "total_top1_successes": total_top1_successes,
        "total_top2_successes": total_top2_successes,
        "top1_accuracy": total_top1_successes / total_scenarios,
        "top2_accuracy": total_top2_successes / total_scenarios,
    }


# Get corrected LLM performance
corrected_llm_perf = calculate_corrected_llm_performance(llm)

overall_df = pd.DataFrame(
    {
        "Group": ["Traditional", "LLM"],
        "Top‑1": [
            trad["best_performance"]["top1_accuracy"],
            corrected_llm_perf["top1_accuracy"],
        ],
        "Top‑2": [
            trad["best_performance"]["top2_accuracy"],
            corrected_llm_perf["top2_accuracy"],
        ],
    }
)

# Get individual LLM model performance (using best config, excluding baselines)
llm_models = []
for model_name, model_data in llm["all_models_individual_performance"].items():
    # Filter out baseline methods
    actual_llm_configs = [
        config
        for config in model_data["raw_summary_data"]
        if config["block"] not in BASELINE_METHODS
    ]

    # Find best configuration
    best_config = max(actual_llm_configs, key=lambda x: x["top1_acc"])

    llm_models.append(
        {
            "Model": model_name,
            "Top‑1": best_config["top1_acc"] / 100,  # Convert percentage to decimal
            "Top‑2": best_config["top2_acc"] / 100,  # Convert percentage to decimal
            "Best_Config": best_config["block"],
        }
    )

llm_df = pd.DataFrame(llm_models).sort_values("Top‑1", ascending=False)

trad_df = pd.DataFrame(
    [
        {"Model": k, "Top‑1": v["top1_accuracy"], "Top‑2": v["top2_accuracy"]}
        for k, v in trad["model_results"].items()
    ]
).sort_values("Top‑1", ascending=False)

# Combined for full model comparison plot
df_all_models = pd.concat(
    [
        llm_df[["Model", "Top‑1", "Top‑2"]].assign(Type="LLM"),
        trad_df.assign(Type="Traditional"),
    ]
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


# ───────────────────────────── PLOTS ─────────────────────────────────
barplot(
    overall_df,
    "Group",
    "Overall Accuracy: LLM vs. Traditional (Corrected)",
    "plot_overall_llm_vs_traditional_corrected.png",
    PALETTE,
)

barplot(
    llm_df,
    "Model",
    "Accuracy by LLM Model (Best Configuration)",
    "plot_llm_model_accuracy_corrected.png",
    PALETTE,
)

barplot(
    trad_df,
    "Model",
    "Accuracy by Traditional Model",
    "plot_traditional_model_accuracy.png",
    PALETTE,
)

# Combined plot with all models
barplot_grouped(
    df_all_models[["Model", "Top‑1", "Top‑2"]],
    "Model",
    "Top‑1 and Top‑2 Accuracy: All Models (LLM + Traditional, Corrected)",
    "plot_all_models_grouped_accuracy_corrected.png",
    PALETTE,
)


# NEW PLOT: Include baseline methods for context
def create_baseline_comparison_plot():
    """Create a plot that includes baseline methods alongside actual models"""
    # Extract baseline methods from first LLM model (they're the same across all models)
    first_model_data = list(llm["all_models_individual_performance"].values())[0]
    baseline_configs = [
        config
        for config in first_model_data["raw_summary_data"]
        if config["block"] in BASELINE_METHODS
    ]

    # Create baseline dataframe
    baseline_df = pd.DataFrame(
        [
            {
                "Model": config["block"],
                "Top‑1": config["top1_acc"] / 100,  # Convert percentage to decimal
                "Top‑2": config["top2_acc"] / 100,  # Convert percentage to decimal
                "Type": "Baseline",
            }
            for config in baseline_configs
        ]
    )

    # Combine all models with baselines
    df_with_baselines = pd.concat(
        [
            llm_df[["Model", "Top‑1", "Top‑2"]].assign(Type="LLM"),
            trad_df.assign(Type="Traditional"),
            baseline_df,
        ]
    )

    # Sort by Top-1 accuracy for better visualization
    df_with_baselines = df_with_baselines.sort_values("Top‑1", ascending=False)

    # Create the plot
    melted = df_with_baselines.melt(
        id_vars=["Model", "Type"],
        value_vars=["Top‑1", "Top‑2"],
        var_name="Metric",
        value_name="Accuracy",
    )
    melted["Model"] = pd.Categorical(
        melted["Model"], categories=df_with_baselines["Model"], ordered=True
    )

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=melted,
        x="Model",
        y="Accuracy",
        hue="Metric",
        palette={"Top‑1": "#386cb0", "Top‑2": "#7fc97f"},
    )

    # Color-code the x-axis labels by type
    colors_by_type = {"LLM": "#2E8B57", "Traditional": "#B8860B", "Baseline": "#CD5C5C"}
    for i, (_, row) in enumerate(df_with_baselines.iterrows()):
        ax.get_xticklabels()[i].set_color(colors_by_type[row["Type"]])

    ax.set_title("Model Performance Comparison (Including Baselines)")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    plt.xticks(rotation=45, ha="right")

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{v:.1%}" for v in container.datavalues],
            padding=3,
            fontsize=8,
        )

    plt.tight_layout()
    plt.legend(title="")
    plt.savefig(OUT_DIR / "plot_all_models_with_baselines.png", dpi=300)
    plt.close()


create_baseline_comparison_plot()

# Print summary of corrections
print(
    textwrap.dedent(
        f"""
✅ CORRECTED PLOTS saved to: {OUT_DIR.resolve()}
 • plot_overall_llm_vs_traditional_corrected.png
 • plot_llm_model_accuracy_corrected.png
 • plot_traditional_model_accuracy.png
 • plot_all_models_grouped_accuracy_corrected.png
 • plot_all_models_with_baselines.png  ← NEW! (includes baselines)

📊 PERFORMANCE SUMMARY (Excluding Baselines):
   Traditional Best: Top-1: {trad['best_performance']['top1_accuracy']:.1%}, Top-2: {trad['best_performance']['top2_accuracy']:.1%}
   LLM Average:      Top-1: {corrected_llm_perf['top1_accuracy']:.1%}, Top-2: {corrected_llm_perf['top2_accuracy']:.1%}
   
🔧 EXCLUDED from LLM calculations: {', '.join(BASELINE_METHODS)}
🎯 INCLUDED in baseline plot: {', '.join(BASELINE_METHODS)}
"""
    )
)

# Show best configuration for each model
print("\n🏆 BEST CONFIGURATIONS PER MODEL:")
for _, row in llm_df.iterrows():
    model_data = llm["all_models_individual_performance"][row["Model"]]
    actual_configs = [
        c for c in model_data["raw_summary_data"] if c["block"] not in BASELINE_METHODS
    ]
    best_config = max(actual_configs, key=lambda x: x["top1_acc"])
    print(f"   {row['Model']}: {best_config['block']} (Top-1: {row['Top‑1']:.1%})")
