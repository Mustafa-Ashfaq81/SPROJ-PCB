import json, pathlib, textwrap
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import numpy as np

# ───────────────────────────── SETTINGS ──────────────────────────────
LLM_JSON = "llm_comparison_results.json"
TRAD_JSON = "traditional_model_results.json"
OUT_DIR = pathlib.Path("plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)  # ← creates the 'plots/' folder

# Professional color palette with excellent contrast and visual appeal
PALETTE = [
    "#1f77b4",  # Professional blue
    "#ff7f0e",  # Vibrant orange
    "#2ca02c",  # Forest green
    "#d62728",  # Deep red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]

# Specialized color schemes for different plot types
COMPARISON_COLORS = ["#3498db", "#e74c3c"]  # Blue and red for comparisons
LLM_COLORS = ["#2ecc71", "#27ae60"]  # Green shades for LLM models
TRADITIONAL_COLORS = ["#f39c12", "#e67e22"]  # Orange shades for traditional models
BASELINE_COLORS = ["#95a5a6", "#7f8c8d"]  # Gray shades for baselines

# Baseline methods to exclude from LLM calculations
BASELINE_METHODS = {"Random Classifier", "Top Career Dismissal"}


# ──────────────────────────── WILSON CI CALCULATION ────────────────────────────
def calculate_wilson_ci(p_hat, n, confidence=0.95):
    """
    Calculate Wilson score confidence interval for binomial proportion.

    Args:
        p_hat: observed proportion (accuracy as decimal)
        n: number of trials
        confidence: confidence level (default 0.95 for 95% CI)

    Returns:
        tuple: (ci_low, ci_high, error_below, error_above)
    """
    if n == 0 or p_hat < 0 or p_hat > 1:
        return 0, 0, 0, 0

    z = norm.ppf((1 + confidence) / 2)  # 1.96 for 95% CI

    # Wilson score interval
    center = (p_hat + (z**2) / (2 * n)) / (1 + (z**2) / n)
    margin = (z * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))) / (
        1 + (z**2) / n
    )

    ci_low = max(0, center - margin)  # Ensure not below 0
    ci_high = min(1, center + margin)  # Ensure not above 1

    # Error bars relative to point estimate (ensure non-negative)
    error_below = max(0, p_hat - ci_low)
    error_above = max(0, ci_high - p_hat)

    return ci_low, ci_high, error_below, error_above


# Improved matplotlib and seaborn styling
plt.style.use("default")  # Start with clean default
sns.set_palette(PALETTE)
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 1.2,
        "axes.edgecolor": "#333333",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
    }
)

# ──────────────────────────── LOAD & TIDY ────────────────────────────
with open(LLM_JSON) as f:
    llm = json.load(f)
with open(TRAD_JSON) as f:
    trad = json.load(f)

# Use the already-corrected overall LLM performance from JSON file
corrected_llm_perf = llm["overall_llm_performance"]

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
        "n": [
            trad["best_performance"].get("total_scenarios", 500),
            corrected_llm_perf["total_scenarios"],
        ],
        "top1_successes": [
            int(
                trad["best_performance"]["top1_accuracy"]
                * trad["best_performance"].get("total_scenarios", 500)
            ),
            corrected_llm_perf["total_top1_successes"],
        ],
        "top2_successes": [
            int(
                trad["best_performance"]["top2_accuracy"]
                * trad["best_performance"].get("total_scenarios", 500)
            ),
            corrected_llm_perf["total_top2_successes"],
        ],
    }
)

overall_df = overall_df.sort_values("Top‑1", ascending=True)

# Get individual LLM model performance
llm_models = []
for model_name, model_data in llm["all_models_individual_performance"].items():
    llm_models.append(
        {
            "Model": model_name,
            "Top‑1": model_data["best_top1_accuracy"],
            "Top‑2": model_data["best_top2_accuracy"],
            "Best_Config": model_data["best_configuration"],
            "n": model_data["best_total_scenarios"],
            "top1_successes": model_data["best_top1_successes"],
            "top2_successes": model_data["best_top2_successes"],
        }
    )

llm_df = pd.DataFrame(llm_models).sort_values("Top‑1", ascending=True)

trad_df = pd.DataFrame(
    [
        {
            "Model": k,
            "Top‑1": v["top1_accuracy"],
            "Top‑2": v["top2_accuracy"],
            "n": v.get("total_scenarios", 500),
            "top1_successes": int(v["top1_accuracy"] * v.get("total_scenarios", 500)),
            "top2_successes": int(v["top2_accuracy"] * v.get("total_scenarios", 500)),
        }
        for k, v in trad["model_results"].items()
    ]
).sort_values("Top‑1", ascending=True)

# Combined for full model comparison plot
df_all_models = pd.concat(
    [
        llm_df[
            ["Model", "Top‑1", "Top‑2", "n", "top1_successes", "top2_successes"]
        ].assign(Type="LLM"),
        trad_df[
            ["Model", "Top‑1", "Top‑2", "n", "top1_successes", "top2_successes"]
        ].assign(Type="Traditional"),
    ]
)
df_all_models["Model"] = df_all_models["Model"].astype(str)
df_all_models = df_all_models.sort_values("Top‑1", ascending=True)


# ──────────────────────────── IMPROVED UTILITIES ──────────────────────────────
def barplot(df, index_col, title, fname, colors):
    """Create bar plot with Wilson CI error bars and improved formatting"""
    df_plot = df.set_index(index_col)

    # Calculate Wilson CIs for Top-1 and Top-2
    top1_errors_below, top1_errors_above = [], []
    top2_errors_below, top2_errors_above = [], []

    for idx in df_plot.index:
        row = df.loc[df[index_col] == idx].iloc[0]

        # Top-1 Wilson CI
        _, _, err_below1, err_above1 = calculate_wilson_ci(row["Top‑1"], row["n"])
        top1_errors_below.append(err_below1)
        top1_errors_above.append(err_above1)

        # Top-2 Wilson CI
        _, _, err_below2, err_above2 = calculate_wilson_ci(row["Top‑2"], row["n"])
        top2_errors_below.append(err_below2)
        top2_errors_above.append(err_above2)

    # Create plot with improved sizing
    fig, ax = plt.subplots(figsize=(10, 7))

    # Use appropriate color scheme based on plot type
    plot_colors = colors if colors else COMPARISON_COLORS

    # Create bars with better styling and gradients
    bars = df_plot[["Top‑1", "Top‑2"]].plot(
        kind="bar",
        ax=ax,
        color=plot_colors[:2],
        width=0.7,
        edgecolor="white",
        linewidth=2,
        yerr=[
            [top1_errors_below, top2_errors_below],
            [top1_errors_above, top2_errors_above],
        ],
        capsize=6,
        error_kw={"linewidth": 2, "capthick": 2, "ecolor": "#2c3e50"},
    )

    ax.set_title(title, fontweight="bold", pad=20, fontsize=16)
    ax.set_ylabel("Accuracy", fontweight="bold", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)

    # Improved y-axis formatting
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))

    # Better grid with subtle styling
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#bdc3c7", linewidth=0.8)
    ax.set_axisbelow(True)

    # Improved legend with better styling
    ax.legend(
        frameon=True,
        fancybox=True,
        shadow=True,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=12,
        facecolor="white",
        edgecolor="#34495e",
        framealpha=0.95,
    )

    # Add value labels with better positioning and styling
    # Get the bar containers (skip error bar containers)
    bar_containers = [
        container for container in ax.containers if hasattr(container, "datavalues")
    ]

    for i, container in enumerate(bar_containers):
        # Use contrasting colors for text based on bar color
        text_color = "white" if i == 0 else "white"  # White text for both bars
        ax.bar_label(
            container,
            labels=[f"{v:.1%}" for v in container.datavalues],
            label_type="center",
            fontsize=11,
            fontweight="bold",
            color=text_color,
        )

    # Improve x-axis labels
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0,
        ha="center",
        fontsize=12,
        fontweight="semibold",
    )

    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def barplot_single_metric(df, index_col, metric, title, fname, colors):
    """Create bar plot for a single metric (Top-1 or Top-2) with Wilson CI error bars"""
    df_plot = df.set_index(index_col)

    # Calculate Wilson CIs for the specific metric
    errors_below, errors_above = [], []

    for idx in df_plot.index:
        row = df.loc[df[index_col] == idx].iloc[0]

        # Wilson CI for the metric
        _, _, err_below, err_above = calculate_wilson_ci(row[metric], row["n"])
        errors_below.append(err_below)
        errors_above.append(err_above)

    # Create plot with improved sizing
    fig, ax = plt.subplots(figsize=(10, 7))

    # Use appropriate color scheme
    plot_color = colors[0] if colors else PALETTE[0]

    # Create bars with better styling
    bars = ax.bar(
        range(len(df_plot)),
        df_plot[metric],
        color=plot_color,
        width=0.7,
        edgecolor="white",
        linewidth=2,
        yerr=[errors_below, errors_above],
        capsize=6,
        error_kw={"linewidth": 2, "capthick": 2, "ecolor": "#2c3e50"},
    )

    ax.set_title(title, fontweight="bold", pad=20, fontsize=16)
    ax.set_ylabel("Accuracy", fontweight="bold", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)

    # Improved y-axis formatting
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))

    # Better grid with subtle styling
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#bdc3c7", linewidth=0.8)
    ax.set_axisbelow(True)

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, df_plot[metric])):
        if value > 0.05:  # Only show label if bar is tall enough
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value / 2,
                f"{value:.1%}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white",
            )

    # Set x-axis labels
    ax.set_xticks(range(len(df_plot)))
    ax.set_xticklabels(
        df_plot.index, rotation=0, ha="center", fontsize=12, fontweight="semibold"
    )

    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def barplot_single_metric_grouped(df, index_col, metric, title, fname):
    """Single metric bar chart for multiple models with type-based coloring"""
    df = df.copy()
    df = df.set_index(index_col)
    df = df.sort_values(metric, ascending=True)

    # Calculate Wilson CIs
    errors_below, errors_above = [], []

    for idx in df.index:
        row = df.loc[idx]

        # Wilson CI for the metric
        _, _, err_below, err_above = calculate_wilson_ci(row[metric], row["n"])
        errors_below.append(err_below)
        errors_above.append(err_above)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(df))

    # Define colors by type - UPDATED COLORS
    type_colors = {
        "LLM": "#414487",  # Indigo
        "Traditional": "#2A788E",  # Teal
        "Baseline": "#95a5a6",  # Professional gray
    }

    # Get colors for each bar based on type
    if "Type" in df.columns:
        bar_colors = [type_colors.get(t, PALETTE[0]) for t in df["Type"]]
    else:
        bar_colors = [PALETTE[0]] * len(df)

    # Create bars
    bars = ax.bar(
        x,
        df[metric],
        color=bar_colors,
        edgecolor="white",
        linewidth=2,
        yerr=[errors_below, errors_above],
        capsize=6,
        error_kw={"linewidth": 2, "capthick": 2, "ecolor": "#2c3e50"},
    )

    ax.set_title(title, fontweight="bold", pad=20, fontsize=16)
    ax.set_ylabel("Accuracy", fontweight="bold", fontsize=14)
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(
        df.index, rotation=45, ha="right", fontsize=11, fontweight="semibold"
    )
    ax.set_ylim(0, 1.05)

    # Improved y-axis formatting
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))

    # Better grid
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#bdc3c7", linewidth=0.8)
    ax.set_axisbelow(True)

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, df[metric])):
        if value > 0.05:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value / 2,
                f"{value:.1%}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
            )

    # Color-code x-axis labels by type if Type column exists
    if "Type" in df.columns:
        colors_by_type = {
            "LLM": "#414487",  # Indigo
            "Traditional": "#2A788E",  # Teal
            "Baseline": "#95a5a6",  # Professional gray
        }
        for i, (_, row) in enumerate(df.iterrows()):
            ax.get_xticklabels()[i].set_color(
                colors_by_type.get(row["Type"], "#333333")
            )
            ax.get_xticklabels()[i].set_fontweight("bold")

        # Add legend for model types
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=colors_by_type["LLM"], label="LLM Models"),
            Patch(facecolor=colors_by_type["Traditional"], label="Traditional Models"),
            Patch(facecolor=colors_by_type["Baseline"], label="Baseline Methods"),
        ]
        ax.legend(
            handles=legend_elements,
            frameon=True,
            fancybox=True,
            shadow=True,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            fontsize=12,
            facecolor="white",
            edgecolor="#34495e",
            framealpha=0.95,
        )

    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def create_baseline_comparison_plots():
    """Create separate plots for Top-1 and Top-2 that include baseline methods"""
    # Extract baseline methods from first LLM model
    first_model_data = list(llm["all_models_individual_performance"].values())[0]
    baseline_configs = [
        config
        for config in first_model_data["raw_summary_data"]
        if config["block"] in BASELINE_METHODS
    ]

    baseline_data = []
    for config in baseline_configs:
        if config["block"] == "Random Classifier":
            baseline_data.append(
                {
                    "Model": config["block"],
                    "Top‑1": 0.125,  # 12.5%
                    "Top‑2": 0.250,  # 25%
                    "Type": "Baseline",
                    "n": config["n"],
                    "top1_successes": int(0.125 * config["n"]),
                    "top2_successes": int(0.125 * config["n"]),
                }
            )
        else:
            # Use original data for other baseline methods
            baseline_data.append(
                {
                    "Model": config["block"],
                    "Top‑1": config["top1_acc"] / 100,
                    "Top‑2": config["top2_acc"] / 100,
                    "Type": "Baseline",
                    "n": config["n"],
                    "top1_successes": config["top1_successes"],
                    "top2_successes": config["top2_successes"],
                }
            )

    baseline_df = pd.DataFrame(baseline_data)

    # Combine all models with baselines
    df_with_baselines = pd.concat(
        [
            llm_df[
                ["Model", "Top‑1", "Top‑2", "n", "top1_successes", "top2_successes"]
            ].assign(Type="LLM"),
            trad_df[
                ["Model", "Top‑1", "Top‑2", "n", "top1_successes", "top2_successes"]
            ].assign(Type="Traditional"),
            baseline_df,
        ]
    )

    # Create separate plots for Top-1 and Top-2
    barplot_single_metric_grouped(
        df_with_baselines,
        "Model",
        "Top‑1",
        "Top-1 Accuracy Comparison: All Models (Including Baselines)",
        "plot_top1_all_models_with_baselines.png",
    )

    barplot_single_metric_grouped(
        df_with_baselines,
        "Model",
        "Top‑2",
        "Top-2 Accuracy Comparison: All Models (Including Baselines)",
        "plot_top2_all_models_with_baselines.png",
    )


# ───────────────────────────── GENERATE PLOTS ─────────────────────────────────
print("🎨 Generating improved plots...")

# ───────────────────────────── GENERATE PLOTS ─────────────────────────────────
print("🎨 Generating improved plots...")

# Overall comparison plots (Top-1 and Top-2 separate)
barplot_single_metric(
    overall_df,
    "Group",
    "Top‑1",
    "Top-1 Accuracy: LLM vs. Traditional Models",
    "plot_overall_top1_llm_vs_traditional.png",
    COMPARISON_COLORS,
)

barplot_single_metric(
    overall_df,
    "Group",
    "Top‑2",
    "Top-2 Accuracy: LLM vs. Traditional Models",
    "plot_overall_top2_llm_vs_traditional.png",
    COMPARISON_COLORS,
)

# LLM model comparison plots (Top-1 and Top-2 separate)
barplot_single_metric(
    llm_df,
    "Model",
    "Top‑1",
    "Top-1 Accuracy: LLM Models (Best Configuration)",
    "plot_llm_top1_accuracy.png",
    LLM_COLORS,
)

barplot_single_metric(
    llm_df,
    "Model",
    "Top‑2",
    "Top-2 Accuracy: LLM Models (Best Configuration)",
    "plot_llm_top2_accuracy.png",
    LLM_COLORS,
)

# Traditional model comparison plots (Top-1 and Top-2 separate)
barplot_single_metric(
    trad_df,
    "Model",
    "Top‑1",
    "Top-1 Accuracy: Traditional Models",
    "plot_traditional_top1_accuracy.png",
    TRADITIONAL_COLORS,
)

barplot_single_metric(
    trad_df,
    "Model",
    "Top‑2",
    "Top-2 Accuracy: Traditional Models",
    "plot_traditional_top2_accuracy.png",
    TRADITIONAL_COLORS,
)

# All models comparison plots (Top-1 and Top-2 separate)
barplot_single_metric_grouped(
    df_all_models,
    "Model",
    "Top‑1",
    "Top-1 Accuracy: Comprehensive Model Comparison",
    "plot_all_models_top1_accuracy.png",
)

barplot_single_metric_grouped(
    df_all_models,
    "Model",
    "Top‑2",
    "Top-2 Accuracy: Comprehensive Model Comparison",
    "plot_all_models_top2_accuracy.png",
)

# Baseline comparison plots (Top-1 and Top-2 separate)
create_baseline_comparison_plots()

# Print summary
print(
    textwrap.dedent(
        f"""
✨ SEPARATE TOP-1 & TOP-2 PLOTS saved to: {OUT_DIR.resolve()}

📊 OVERALL COMPARISONS:
 • plot_overall_top1_llm_vs_traditional.png
 • plot_overall_top2_llm_vs_traditional.png

🤖 LLM MODEL COMPARISONS:
 • plot_llm_top1_accuracy.png
 • plot_llm_top2_accuracy.png

🔧 TRADITIONAL MODEL COMPARISONS:
 • plot_traditional_top1_accuracy.png
 • plot_traditional_top2_accuracy.png

🔍 COMPREHENSIVE COMPARISONS:
 • plot_all_models_top1_accuracy.png
 • plot_all_models_top2_accuracy.png

📈 WITH BASELINES:
 • plot_top1_all_models_with_baselines.png
 • plot_top2_all_models_with_baselines.png

🎯 FORMATTING IMPROVEMENTS:
 • Separate graphs for Top-1 and Top-2 for clearer comparison
 • Type-based color coding (Indigo=LLM, Teal=Traditional, Gray=Baseline)
 • Enhanced error bars with Wilson confidence intervals
 • Professional styling with improved readability
 • Color-coded model type labels
 • Clean single-metric visualization

📊 PERFORMANCE SUMMARY:
   Traditional Best: Top-1: {trad['best_performance']['top1_accuracy']:.1%}, Top-2: {trad['best_performance']['top2_accuracy']:.1%}
   LLM Overall:     Top-1: {corrected_llm_perf['top1_accuracy']:.1%}, Top-2: {corrected_llm_perf['top2_accuracy']:.1%}
"""
    )
)
