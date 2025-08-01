import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import norm
from statsmodels.stats.contingency_tables import mcnemar

# ─────────────────────────────────── CONFIG
MODELS = {
    "gpt-4.1 nano": "gpt-4.1-nano",
    "gpt-4.1 mini": "gpt-4.1-mini",
    "gpt-4.1": "gpt-4.1",
    "gpt-o4 mini": "o4-mini",
}

MODEL_ORDER = ["gpt-4.1 nano", "gpt-4.1 mini", "gpt-4.1", "gpt-o4 mini"]
MODEL_COLOURS = ["#440154", "#414487", "#2A788E", "#22A884"]
MODEL2COLOR = dict(zip(MODEL_ORDER, MODEL_COLOURS))

EXTRA_BAR_COLOR = {-2: "#999999"}  # random + TCD both grey
BLOCK_ID_MAP = {  # cat_id 0‑8 = feature sets
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
os.makedirs(os.path.join(SUMMARY_DIR, "best_category_by_model"), exist_ok=True)
os.makedirs(os.path.join(SUMMARY_DIR, "mcnemar_analysis"), exist_ok=True)

METRICS = ["top1_acc", "top2_acc"]

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "font.family": "monospace",
    }
)


# ─────────────────────────────────── HELPERS
def wilson_ci(successes: int, n: int, confidence=0.95) -> tuple[float, float]:
    """Wilson score interval (returned in %)"""
    if n == 0:
        return 0.0, 0.0
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return (centre - margin) * 100, (centre + margin) * 100


def load_detailed_results(model_key: str, metric: str) -> dict:
    """
    Load detailed per-sample results for McNemar test from predictions folder.
    The CSV files are in long format with columns: sample_id, block, correct_top1, correct_top2
    """
    detailed_path = os.path.join("predictions", f"{model_key}_preds.csv")

    if not os.path.exists(detailed_path):
        print(f"⚠️  Missing prediction file: {detailed_path}")
        return {}

    try:
        df = pd.read_csv(detailed_path)
        print(f"✓ Loaded {detailed_path} with {len(df)} rows")

        # Select the correct metric column
        correct_col = "correct_top1" if metric == "top1_acc" else "correct_top2"

        if correct_col not in df.columns:
            print(f"⚠️  Missing column {correct_col} in {detailed_path}")
            return {}

        # Get unique blocks in the data
        unique_blocks = df["block"].unique()
        print(f"✓ Found blocks: {unique_blocks}")

        results = {}

        # Process each block that we're interested in
        for block_name, cat_id in BLOCK_ID_MAP.items():
            if cat_id == -2:  # Skip random classifier
                continue

            # Check if this block exists in the data
            if block_name in unique_blocks:
                block_data = df[df["block"] == block_name]
                if len(block_data) > 0:
                    # Sort by sample_id to ensure consistent ordering
                    block_data = block_data.sort_values("sample_id")
                    results[block_name] = block_data[correct_col].values
                    print(f"✓ Loaded {block_name}: {len(block_data)} samples")
                else:
                    print(f"⚠️  No data found for block {block_name}")
            else:
                print(f"⚠️  Block {block_name} not found in data")

        # Check for Top Career Dismissal
        if "Top Career Dismissal" in unique_blocks:
            tcd_data = df[df["block"] == "Top Career Dismissal"].sort_values(
                "sample_id"
            )
            results["Top Career Dismissal"] = tcd_data[correct_col].values
            print(f"✓ Loaded Top Career Dismissal: {len(tcd_data)} samples")

        # Verify all methods have same number of samples
        if results:
            sample_counts = {name: len(data) for name, data in results.items()}
            unique_counts = set(sample_counts.values())
            if len(unique_counts) > 1:
                print(
                    f"⚠️  Warning: Different sample counts across methods: {sample_counts}"
                )
            else:
                print(f"✓ All methods have {list(unique_counts)[0]} samples")

        print(f"✓ Successfully loaded {len(results)} methods for {model_key}")
        return results

    except Exception as e:
        print(f"⚠️  Error loading {detailed_path}: {e}")
        return {}


def mcnemar_test_pairwise(results1: np.ndarray, results2: np.ndarray) -> float:
    """
    Perform McNemar's test between two sets of binary results.
    Returns p-value.
    """
    if len(results1) != len(results2):
        return np.nan

    # Create contingency table
    # [correct_both, incorrect_1_correct_2]
    # [correct_1_incorrect_2, incorrect_both]
    both_correct = np.sum((results1 == 1) & (results2 == 1))
    result1_correct_result2_incorrect = np.sum((results1 == 1) & (results2 == 0))
    result1_incorrect_result2_correct = np.sum((results1 == 0) & (results2 == 1))
    both_incorrect = np.sum((results1 == 0) & (results2 == 0))

    contingency_table = np.array(
        [
            [both_correct, result1_incorrect_result2_correct],
            [result1_correct_result2_incorrect, both_incorrect],
        ]
    )

    try:
        result = mcnemar(contingency_table, exact=False, correction=True)
        return result.pvalue
    except:
        return np.nan


def create_mcnemar_heatmap(model_label: str, model_key: str, metric: str):
    """Create beautiful McNemar test heatmap for a single model."""

    # Load detailed results
    detailed_results = load_detailed_results(model_key, metric)

    if not detailed_results:
        print(f"⚠️  No detailed results for {model_label}, skipping McNemar analysis")
        return

    # Get the methods we want to compare (excluding random classifier)
    methods = list(detailed_results.keys())
    n_methods = len(methods)

    if n_methods < 2:
        print(f"⚠️  Need at least 2 methods for comparison, got {n_methods}")
        return

    # Create p-value matrix
    p_matrix = np.ones((n_methods, n_methods))

    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i != j:
                p_val = mcnemar_test_pairwise(
                    detailed_results[method1], detailed_results[method2]
                )
                p_matrix[i, j] = p_val

    # Create cleaner method labels
    def clean_label(method_name):
        """Create shorter, cleaner labels for the heatmap"""
        label_map = {
            "Baseline": "Baseline",
            "General Stats": "General\nStats",
            "General Stats + Venue Stats": "General +\nVenue Stats",
            "General Stats + Venue Stats + Vs-Team Stats": "General + Venue\n+ Vs-Team Stats",
            "General Stats + Venue Stats + Vs-Team Stats + Commentary": "General + Venue\n+ Vs-Team + Commentary",
            "Venue Stats": "Venue\nStats",
            "Vs-Team Stats": "Vs-Team\nStats",
            "Venue Stats + Vs-Team Stats": "Venue +\nVs-Team Stats",
            "Venue Stats + Vs-Team Stats + Commentary": "Venue + Vs-Team\n+ Commentary",
        }
        return label_map.get(method_name, method_name)

    clean_labels = [clean_label(m) for m in methods]

    # Create the heatmap with same styling as original
    plt.figure(figsize=(12, 10))

    # Use exact same colormap as original
    colors = [
        "#1a9850",  # dark green
        "#66bd63",  # green
        "#a6d96a",  # yellow-green
        "#d9ef8b",  # yellow
        "#fee08b",  # light orange
        "#fdae61",  # orange
        "#f46d43",  # red-orange
        "#d73027",  # red
    ]

    boundaries = [0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    # Create custom colormap
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)

    # Create mask for diagonal (self-comparisons)
    mask = np.eye(n_methods, dtype=bool)

    # Create heatmap with original styling
    ax = sns.heatmap(
        p_matrix,
        annot=True,
        fmt=".4f",
        cmap=cmap,
        norm=norm,
        mask=mask,
        square=True,
        cbar_kws={
            "label": "McNemar p-value",
            "shrink": 0.8,
            "aspect": 30,
            "boundaries": boundaries,
            "ticks": boundaries,
        },
        linewidths=0.5,
        linecolor="white",
    )

    # Improved formatting - better spacing and readability
    ax.set_xticklabels(clean_labels, rotation=45, ha="right", fontsize=10, va="top")
    ax.set_yticklabels(clean_labels, rotation=0, ha="right", fontsize=10, va="center")
    ax.invert_yaxis()

    # Add extra padding for labels
    ax.tick_params(axis="x", which="major", pad=10)
    ax.tick_params(axis="y", which="major", pad=5)

    # Title and labels - same as original
    metric_name = "Top-1" if metric == "top1_acc" else "Top-2"
    plt.title(
        f"{model_label}: McNemar Test P-values\n{metric_name} Accuracy Comparisons",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    plt.xlabel("Category B", fontsize=12, fontweight="semibold")
    plt.ylabel("Category A", fontsize=12, fontweight="semibold")

    # Add significance indicators - same as original
    for i in range(n_methods):
        for j in range(n_methods):
            if i != j:
                p_val = p_matrix[i, j]
                if p_val < 0.001:
                    marker = "***"
                elif p_val < 0.01:
                    marker = "**"
                elif p_val < 0.05:
                    marker = "*"
                else:
                    marker = ""

                if marker:
                    ax.text(
                        j + 0.5,
                        i + 0.8,
                        marker,
                        ha="center",
                        va="center",
                        fontsize=14,
                        fontweight="bold",
                        color="white" if p_val < 0.05 else "black",
                    )

    # Add legend for significance levels - same position as original
    legend_text = "*** p < 0.001\n" "**  p < 0.01\n" "*   p < 0.05"

    plt.figtext(
        0.02,
        0.02,
        legend_text,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    # Better layout adjustment to prevent label cutoff
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.2)

    # Save
    out_path = os.path.join(
        SUMMARY_DIR, f"mcnemar_analysis/{model_key}_{metric}_mcnemar_heatmap.png"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved McNemar heatmap: {out_path}")

    # Save p-value matrix as CSV with clean labels
    p_df = pd.DataFrame(p_matrix, index=clean_labels, columns=clean_labels)
    csv_path = os.path.join(
        SUMMARY_DIR, f"mcnemar_analysis/{model_key}_{metric}_mcnemar_pvalues.csv"
    )
    p_df.to_csv(csv_path)
    print(f"✓ Saved p-value matrix: {csv_path}")


def create_combined_mcnemar_heatmap(metric: str):
    """Create a combined heatmap showing McNemar results across all models."""

    all_p_values = {}
    all_methods = set()

    # Helper function for clean labels
    def clean_label(method_name):
        """Create shorter, cleaner labels for the heatmap"""
        label_map = {
            "Baseline": "Base",
            "General Stats": "Gen",
            "General Stats + Venue Stats": "Gen+Ven",
            "General Stats + Venue Stats + Vs-Team Stats": "Gen+Ven+Vs",
            "General Stats + Venue Stats + Vs-Team Stats + Commentary": "All",
            "Venue Stats": "Ven",
            "Vs-Team Stats": "Vs",
            "Venue Stats + Vs-Team Stats": "Ven+Vs",
            "Venue Stats + Vs-Team Stats + Commentary": "Ven+Vs+Com",
        }
        return label_map.get(method_name, method_name[:8])

    # Collect p-values from all models
    for model_label, model_key in MODELS.items():
        detailed_results = load_detailed_results(model_key, metric)
        if not detailed_results:
            continue

        methods = list(detailed_results.keys())
        all_methods.update(methods)

        model_p_values = {}
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i != j:
                    p_val = mcnemar_test_pairwise(
                        detailed_results[method1], detailed_results[method2]
                    )
                    model_p_values[f"{method1} vs {method2}"] = p_val

        all_p_values[model_label] = model_p_values

    if not all_p_values:
        print("⚠️  No McNemar results to combine")
        return

    # Create combined visualization with better layout
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    axes = axes.flatten()

    metric_name = "Top-1" if metric == "top1_acc" else "Top-2"
    fig.suptitle(
        f"McNemar Test Results Across Models: {metric_name} Accuracy",
        fontsize=24,
        fontweight="bold",
        y=0.95,
    )

    # Better colormap
    colors = [
        "#b2182b",
        "#d6604d",
        "#f4a582",
        "#fddbc7",
        "#d1e5f0",
        "#92c5de",
        "#4393c3",
        "#2166ac",
    ]
    boundaries = [0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)

    for idx, (model_label, model_key) in enumerate(MODELS.items()):
        if idx >= 4:  # Only handle up to 4 models
            break

        ax = axes[idx]

        if model_label not in all_p_values:
            ax.text(
                0.5,
                0.5,
                f"No data for\n{model_label}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
                bbox=dict(boxstyle="round", facecolor="lightgray"),
            )
            ax.set_title(model_label, fontsize=16, fontweight="semibold")
            continue

        # Reconstruct matrix for this model
        detailed_results = load_detailed_results(model_key, metric)
        methods = list(detailed_results.keys())
        n_methods = len(methods)

        p_matrix = np.ones((n_methods, n_methods))
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i != j:
                    key = f"{method1} vs {method2}"
                    if key in all_p_values[model_label]:
                        p_matrix[i, j] = all_p_values[model_label][key]

        mask = np.eye(n_methods, dtype=bool)

        # Create heatmap for this model
        im = ax.imshow(p_matrix, cmap=cmap, norm=norm, aspect="equal")

        # Add text annotations
        for i in range(n_methods):
            for j in range(n_methods):
                if not mask[i, j]:
                    p_val = p_matrix[i, j]

                    # Add p-value text
                    ax.text(
                        j,
                        i,
                        f"{p_val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="semibold",
                        color="white" if p_val < 0.05 else "black",
                    )

                    # Add significance markers
                    if p_val < 0.001:
                        marker = "***"
                    elif p_val < 0.01:
                        marker = "**"
                    elif p_val < 0.05:
                        marker = "*"
                    else:
                        marker = ""

                    if marker:
                        ax.text(
                            j,
                            i + 0.25,
                            marker,
                            ha="center",
                            va="center",
                            fontsize=10,
                            fontweight="bold",
                            color="white",
                        )

        # Format labels with clean names
        clean_labels = [clean_label(m) for m in methods]
        ax.set_xticks(range(n_methods))
        ax.set_yticks(range(n_methods))
        ax.set_xticklabels(clean_labels, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(clean_labels, rotation=0, ha="right", fontsize=10)

        ax.set_title(model_label, fontsize=16, fontweight="semibold", pad=15)

        # Add grid
        ax.set_xlim(-0.5, n_methods - 0.5)
        ax.set_ylim(-0.5, n_methods - 0.5)
        ax.grid(True, color="white", linewidth=1)

    # Add a single colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(
        im, cax=cbar_ax, boundaries=boundaries, ticks=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    )
    cbar.set_label("McNemar p-value", fontsize=14, fontweight="semibold")
    cbar.ax.tick_params(labelsize=12)

    # Add overall legend
    legend_text = (
        "Significance Levels:\n" "*** p < 0.001\n" "**  p < 0.01\n" "*   p < 0.05"
    )

    fig.text(
        0.02,
        0.02,
        legend_text,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9),
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.9)

    # Save combined heatmap
    out_path = os.path.join(
        SUMMARY_DIR, f"mcnemar_analysis/combined_{metric}_mcnemar_heatmap.png"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✓ Saved combined McNemar heatmap: {out_path}")


# ─────────────────────────────────── MAIN LOOP
for metric in METRICS:
    rows = []

    for model_label, model_key in MODELS.items():
        csv_path = os.path.join(SUMMARY_DIR, f"{model_key}_summary.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️  Missing summary: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df["block"] = df["block"].str.strip()
        df = df[
            df["block"].isin(BLOCK_ID_MAP) | (df["block"] == "Top Career Dismissal")
        ]

        # ── baseline: random classifier
        rand_n = int(df["n"].iloc[0])
        rand_successes = int(rand_n * (1 / 8 if metric == "top1_acc" else 2 / 8))
        rand_acc = 12.5 if metric == "top1_acc" else 25.0
        low, high = wilson_ci(rand_successes, rand_n)
        print(f"[{model_label} | Random] CI: ({low:.2f}%, {high:.2f}%) | n={rand_n}")
        rows.append(
            dict(
                model=model_label,
                block="Random Classifier",
                accuracy=rand_acc,
                cat_id=-2,
                ci_low=low,
                ci_high=high,
            )
        )

        # ── baseline: TCD (if present)
        tcd = df[df["block"] == "Top Career Dismissal"]
        if not tcd.empty:
            row = tcd.iloc[0]
            succ = (
                row["top1_successes"] if metric == "top1_acc" else row["top2_successes"]
            )
            n = row["n"]
            acc = row[metric]
            low, high = wilson_ci(succ, n)
            print(f"[{model_label} | TCD] CI: ({low:.2f}%, {high:.2f}%) | n={n}")
            rows.append(
                dict(
                    model=model_label,
                    block="Top Career Dismissal",
                    accuracy=acc,
                    cat_id=-1,
                    ci_low=low,
                    ci_high=high,
                )
            )

        # ── best two feature blocks
        for _, row in df.nlargest(2, metric).iterrows():
            succ = (
                row["top1_successes"] if metric == "top1_acc" else row["top2_successes"]
            )
            n = row["n"]
            low, high = wilson_ci(succ, n)
            print(
                f"[{model_label} | {row['block']}] CI: ({low:.2f}%, {high:.2f}%) | n={n}"
            )
            rows.append(
                dict(
                    model=model_label,
                    block=row["block"],
                    accuracy=row[metric],
                    cat_id=BLOCK_ID_MAP[row["block"]],
                    ci_low=low,
                    ci_high=high,
                )
            )

        # ── Create McNemar heatmap for this model
        create_mcnemar_heatmap(model_label, model_key, metric)

    # ── Create combined McNemar heatmap
    create_combined_mcnemar_heatmap(metric)

    # ── persist tall table of intervals
    df_all = pd.DataFrame(rows)
    if df_all.empty:
        print(f"⚠️  No rows for metric {metric}")
        continue
    df_all.to_csv(
        os.path.join(
            SUMMARY_DIR, f"best_category_by_model/best_{metric}_top2_per_model.csv"
        ),
        index=False,
    )

    # ─────────────────────── plotting (unchanged except tiny print‑fix)
    pivot = df_all.pivot(index="model", columns="cat_id", values="accuracy").reindex(
        MODEL_ORDER
    )
    n_models = len(MODEL_ORDER)
    bar_w, gap = 0.6, 0.7
    centres = np.arange(n_models) * (pivot.shape[1] * bar_w + gap)

    fig, ax = plt.subplots(figsize=(1.3 * n_models + 4, 7), dpi=150)
    ax.set_facecolor("white")

    for i, model in enumerate(MODEL_ORDER):
        mdf = df_all[df_all["model"] == model]
        plot_df = pd.concat(
            [
                mdf[mdf["cat_id"].isin([-2, -1])],
                mdf[mdf["cat_id"] >= 0].nlargest(2, "accuracy"),
            ]
        )
        plot_df = plot_df.sort_values("accuracy").reset_index(drop=True)

        blocks = plot_df["cat_id"].to_list()
        heights = plot_df["accuracy"].to_list()
        xs = centres[i] + np.arange(len(blocks)) * bar_w
        colors = [
            EXTRA_BAR_COLOR[-2] if cid in (-2, -1) else MODEL2COLOR[model]
            for cid in blocks
        ]

        ax.bar(
            xs,
            heights,
            width=bar_w,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            label=model if metric == METRICS[0] else None,
        )

        # CI error bars
        for x, cid in zip(xs, blocks):
            r = plot_df[plot_df["cat_id"] == cid].iloc[0]
            err_low = r["accuracy"] - r["ci_low"]
            err_high = r["ci_high"] - r["accuracy"]
            ax.errorbar(
                x,
                r["accuracy"],
                yerr=[[err_low], [err_high]],
                fmt="none",
                ecolor="black",
                capsize=4,
                linewidth=1,
            )

        # per‑block labels (now using correct row each time)
        for x, cid in zip(xs, blocks):
            r = plot_df[plot_df["cat_id"] == cid].iloc[0]
            ax.text(x, -2, str(cid), ha="center", va="top", fontsize=9)
            ax.text(
                x,
                r["accuracy"] / 2 - 1.5,
                f"{r['accuracy']:.1f}%",
                ha="center",
                va="center",
                color="white" if r["accuracy"] > 10 else "black",
                fontsize=7,
                weight="bold",
            )

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
            handles=[
                plt.Line2D([0], [0], color=MODEL2COLOR[m], lw=8) for m in MODEL_ORDER
            ],
            labels=MODEL_ORDER,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.07),
            frameon=False,
            ncol=4,
        )

    ax.set_title(f"{ylab} Blocks per Model", pad=18, weight="semibold")

    # mapping guide
    id2block = {v: k.replace("\n", " ").strip() for k, v in BLOCK_ID_MAP.items()}
    id2block[-2] = "Random Classifier"
    id2block[-1] = "Top Career Dismissal"
    # legend_txt = "\n".join(f"{i:2} – {id2block[i]}" for i in sorted(id2block))

    # fig.subplots_adjust(bottom=0.25)
    # sub_ax = fig.add_axes([0, 0, 1, 0.2])
    # sub_ax.axis("off")
    # sub_ax.text(
    #     0.01,
    #     0.05,
    #     "Category ID Mapping:\n" + legend_txt,
    #     ha="left",
    #     va="bottom",
    #     fontsize=9,
    #     family="monospace",
    # )

    plt.tight_layout(rect=[0, 0.25, 1, 0.93])
    out = os.path.join(
        SUMMARY_DIR,
        f"best_category_by_model/best_{metric}_top2_by_model_grouped_clean.png",
    )
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {out}")

print("\n🎉 Analysis complete! Check the mcnemar_analysis/ folder for heatmaps.")
