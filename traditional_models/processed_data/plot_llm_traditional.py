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
PALETTE = ["#386cb0", "#7fc97f", "#f0027f", "#fdc086", "#beaed4"]

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
    margin = (z * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))) / (1 + (z**2) / n)
    
    ci_low = max(0, center - margin)  # Ensure not below 0
    ci_high = min(1, center + margin)  # Ensure not above 1
    
    # Error bars relative to point estimate (ensure non-negative)
    error_below = max(0, p_hat - ci_low)
    error_above = max(0, ci_high - p_hat)
    
    return ci_low, ci_high, error_below, error_above

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


# Use the already-corrected overall LLM performance from JSON file
# (This data already includes all LLM configurations, excluding baselines)
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
            trad["best_performance"].get("total_scenarios", 500),  # Default to 500 if not specified
            corrected_llm_perf["total_scenarios"],
        ],
        "top1_successes": [
            int(trad["best_performance"]["top1_accuracy"] * trad["best_performance"].get("total_scenarios", 500)),
            corrected_llm_perf["total_top1_successes"],
        ],
        "top2_successes": [
            int(trad["best_performance"]["top2_accuracy"] * trad["best_performance"].get("total_scenarios", 500)),
            corrected_llm_perf["total_top2_successes"],
        ],
    }
)

# Sort by Top-1 accuracy in ascending order for consistency
overall_df = overall_df.sort_values("Top‑1", ascending=True)

# Get individual LLM model performance (using already-calculated best performance from JSON)
llm_models = []
for model_name, model_data in llm["all_models_individual_performance"].items():
    llm_models.append(
        {
            "Model": model_name,
            "Top‑1": model_data["best_top1_accuracy"],  # Already in decimal format
            "Top‑2": model_data["best_top2_accuracy"],  # Already in decimal format
            "Best_Config": model_data["best_configuration"],
            "n": model_data["best_total_scenarios"],
            "top1_successes": model_data["best_top1_successes"],
            "top2_successes": model_data["best_top2_successes"],
        }
    )

llm_df = pd.DataFrame(llm_models).sort_values("Top‑1", ascending=True)  # Changed to ascending

trad_df = pd.DataFrame(
    [
        {
            "Model": k, 
            "Top‑1": v["top1_accuracy"], 
            "Top‑2": v["top2_accuracy"],
            "n": v.get("total_scenarios", 500),  # Default to 500 if not specified
            "top1_successes": int(v["top1_accuracy"] * v.get("total_scenarios", 500)),
            "top2_successes": int(v["top2_accuracy"] * v.get("total_scenarios", 500)),
        }
        for k, v in trad["model_results"].items()
    ]
).sort_values("Top‑1", ascending=True)  # Changed to ascending

# Combined for full model comparison plot
df_all_models = pd.concat(
    [
        llm_df[["Model", "Top‑1", "Top‑2", "n", "top1_successes", "top2_successes"]].assign(Type="LLM"),
        trad_df[["Model", "Top‑1", "Top‑2", "n", "top1_successes", "top2_successes"]].assign(Type="Traditional"),
    ]
)
df_all_models["Model"] = df_all_models["Model"].astype(str)
df_all_models = df_all_models.sort_values("Top‑1", ascending=True)  # Changed to ascending


# ──────────────────────────── UTILITIES ──────────────────────────────
def barplot(df, index_col, title, fname, colors):
    """Create bar plot with Wilson CI error bars"""
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
    
    # Create plot
    ax = df_plot[["Top‑1", "Top‑2"]].plot(
        kind="bar",
        figsize=(8, 5),
        color=colors[:2],
        width=0.75,
        edgecolor="0.2",
        yerr=[
            [top1_errors_below, top2_errors_below], 
            [top1_errors_above, top2_errors_above]
        ],
        capsize=4,
        error_kw={'linewidth': 1.2}
    )
    
    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda y, pos: f"{y:.0%}")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Add value labels inside bars with appropriate text colors
    for i, (metric, values) in enumerate([("Top‑1", df_plot["Top‑1"]), ("Top‑2", df_plot["Top‑2"])]):
        for j, value in enumerate(values):
            x_pos = j + (i - 0.5) * 0.375  # Adjust for grouped bars
            y_pos = value / 2  # Center of bar
            # Use white text for better visibility inside colored bars
            text_color = 'white' if value > 0.15 else 'black'  # Switch to black for very short bars
            ax.text(x_pos, y_pos, f"{value:.1%}", ha='center', va='center', 
                   fontsize=9, color=text_color, weight='bold')

    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=300)
    plt.close()


def barplot_grouped(df, index_col, title, fname, colors):
    """Grouped bar chart for multiple models and metrics (Top‑1 / Top‑2) with Wilson CI error bars."""
    df = df.copy()
    df = df.set_index(index_col)
    df = df.sort_values("Top‑1", ascending=True)  # Changed to ascending

    # Calculate Wilson CIs
    top1_errors_below, top1_errors_above = [], []
    top2_errors_below, top2_errors_above = [], []
    
    for idx in df.index:
        row = df.loc[idx]
        
        # Top-1 Wilson CI
        _, _, err_below1, err_above1 = calculate_wilson_ci(row["Top‑1"], row["n"])
        top1_errors_below.append(err_below1)
        top1_errors_above.append(err_above1)
        
        # Top-2 Wilson CI  
        _, _, err_below2, err_above2 = calculate_wilson_ci(row["Top‑2"], row["n"])
        top2_errors_below.append(err_below2)
        top2_errors_above.append(err_above2)

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
        yerr=[top1_errors_below, top1_errors_above],
        capsize=4,
        error_kw={'linewidth': 1.2}
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        df["Top‑2"],
        width=width,
        label="Top‑2",
        color=colors[1],
        edgecolor="0.2",
        yerr=[top2_errors_below, top2_errors_above],
        capsize=4,
        error_kw={'linewidth': 1.2}
    )

    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda y, pos: f"{y:.0%}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False)

    # Add labels inside bars with appropriate text colors
    for i, value in enumerate(df["Top‑1"]):
        y_pos = value / 2  # Center of bar
        text_color = 'white' if value > 0.15 else 'black'  # Switch to black for very short bars
        ax.text(i - width/2, y_pos, f"{value:.1%}", ha='center', va='center', 
               fontsize=9, color=text_color, weight='bold')
    for i, value in enumerate(df["Top‑2"]):
        y_pos = value / 2  # Center of bar
        text_color = 'white' if value > 0.15 else 'black'  # Switch to black for very short bars
        ax.text(i + width/2, y_pos, f"{value:.1%}", ha='center', va='center', 
               fontsize=9, color=text_color, weight='bold')

    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=300)
    plt.close()


# ───────────────────────────── PLOTS ─────────────────────────────────
barplot(
    overall_df,
    "Group",
    "Overall Accuracy: LLM vs. Traditional (All LLM Configurations)",
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
    df_all_models,
    "Model",
    "Top‑1 and Top‑2 Accuracy: All Models (LLM + Traditional, All Configurations)",
    "plot_all_models_grouped_accuracy_corrected.png",
    PALETTE,
)


# NEW PLOT: Include baseline methods for context
def create_baseline_comparison_plot():
    """Create a plot that includes baseline methods alongside actual models with Wilson CI error bars"""
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
                "n": config["n"],
                "top1_successes": config["top1_successes"],
                "top2_successes": config["top2_successes"],
            }
            for config in baseline_configs
        ]
    )

    # Combine all models with baselines
    df_with_baselines = pd.concat(
        [
            llm_df[["Model", "Top‑1", "Top‑2", "n", "top1_successes", "top2_successes"]].assign(Type="LLM"),
            trad_df[["Model", "Top‑1", "Top‑2", "n", "top1_successes", "top2_successes"]].assign(Type="Traditional"),
            baseline_df,
        ]
    )

    # Sort by Top-1 accuracy for better visualization
    df_with_baselines = df_with_baselines.sort_values("Top‑1", ascending=True)  # Changed to ascending

    # Calculate Wilson CIs for each model
    df_with_baselines["top1_ci_low"] = 0.0
    df_with_baselines["top1_ci_high"] = 0.0
    df_with_baselines["top2_ci_low"] = 0.0
    df_with_baselines["top2_ci_high"] = 0.0
    
    for idx, row in df_with_baselines.iterrows():
        # Top-1 Wilson CI
        ci_low1, ci_high1, _, _ = calculate_wilson_ci(row["Top‑1"], row["n"])
        df_with_baselines.loc[idx, "top1_ci_low"] = ci_low1
        df_with_baselines.loc[idx, "top1_ci_high"] = ci_high1
        
        # Top-2 Wilson CI
        ci_low2, ci_high2, _, _ = calculate_wilson_ci(row["Top‑2"], row["n"])
        df_with_baselines.loc[idx, "top2_ci_low"] = ci_low2
        df_with_baselines.loc[idx, "top2_ci_high"] = ci_high2

    # Calculate error arrays and check for negative values
    top1_err_below = df_with_baselines["Top‑1"] - df_with_baselines["top1_ci_low"]
    top1_err_above = df_with_baselines["top1_ci_high"] - df_with_baselines["Top‑1"]
    top2_err_below = df_with_baselines["Top‑2"] - df_with_baselines["top2_ci_low"]
    top2_err_above = df_with_baselines["top2_ci_high"] - df_with_baselines["Top‑2"]
    
    # Ensure no negative values
    top1_err_below = np.maximum(0, top1_err_below)
    top1_err_above = np.maximum(0, top1_err_above)
    top2_err_below = np.maximum(0, top2_err_below)
    top2_err_above = np.maximum(0, top2_err_above)

    # Create the plot with error bars
    fig, ax = plt.subplots(figsize=(12, 7))
    
    width = 0.35
    x = np.arange(len(df_with_baselines))
    
    # Top-1 bars with error bars
    bars1 = ax.bar(
        x - width/2, 
        df_with_baselines["Top‑1"], 
        width, 
        label="Top‑1",
        color="#386cb0",
        edgecolor="0.2",
        yerr=[top1_err_below, top1_err_above],
        capsize=4,
        error_kw={'linewidth': 1.2}
    )
    
    # Top-2 bars with error bars
    bars2 = ax.bar(
        x + width/2, 
        df_with_baselines["Top‑2"], 
        width, 
        label="Top‑2",
        color="#7fc97f",
        edgecolor="0.2",
        yerr=[top2_err_below, top2_err_above],
        capsize=4,
        error_kw={'linewidth': 1.2}
    )

    # Color-code the x-axis labels by type (after setting labels)
    colors_by_type = {"LLM": "#2E8B57", "Traditional": "#B8860B", "Baseline": "#CD5C5C"}
    
    ax.set_title("Model Performance Comparison (Including Baselines)")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(df_with_baselines["Model"], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    
    for i, (_, row) in enumerate(df_with_baselines.iterrows()):
        ax.get_xticklabels()[i].set_color(colors_by_type[row["Type"]])

    # Add value labels inside bars with appropriate text colors
    for i, (_, row) in enumerate(df_with_baselines.iterrows()):
        # Top-1 bar
        y_pos1 = row["Top‑1"] / 2  # Center of bar
        text_color1 = 'white' if row["Top‑1"] > 0.15 else 'black'
        ax.text(i - width/2, y_pos1, f"{row['Top‑1']:.1%}", ha='center', va='center', 
               fontsize=8, color=text_color1, weight='bold')
        
        # Top-2 bar
        y_pos2 = row["Top‑2"] / 2  # Center of bar
        text_color2 = 'white' if row["Top‑2"] > 0.15 else 'black'
        ax.text(i + width/2, y_pos2, f"{row['Top‑2']:.1%}", ha='center', va='center', 
               fontsize=8, color=text_color2, weight='bold')

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

📊 PERFORMANCE SUMMARY (Corrected - All LLM Configurations):
   Traditional Best: Top-1: {trad['best_performance']['top1_accuracy']:.1%}, Top-2: {trad['best_performance']['top2_accuracy']:.1%}
   LLM Overall:     Top-1: {corrected_llm_perf['top1_accuracy']:.1%}, Top-2: {corrected_llm_perf['top2_accuracy']:.1%}
   Total LLM Scenarios: {corrected_llm_perf['total_scenarios']:,} (vs Traditional: {trad['best_performance'].get('total_scenarios', 500)})
   
🔧 EXCLUDED from LLM calculations: {', '.join(BASELINE_METHODS)}
🎯 INCLUDED in baseline plot: {', '.join(BASELINE_METHODS)}
"""
    )
)

# Show best configuration for each model
print("\n🏆 BEST CONFIGURATIONS PER MODEL:")
for _, row in llm_df.iterrows():
    model_data = llm["all_models_individual_performance"][row["Model"]]
    print(f"   {row['Model']}: {model_data['best_configuration']} (Top-1: {row['Top‑1']:.1%})")
