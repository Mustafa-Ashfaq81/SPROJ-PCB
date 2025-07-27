import os
import pandas as pd
from collections import defaultdict

# Map filenames to their corresponding model keys
FILES = {
    "Babar Azam_prediction_results_gpt-4.1.csv": "gpt-4.1",
    "Babar Azam_prediction_results_gpt-4.1-mini.csv": "gpt-4.1-mini",
    "Babar Azam_prediction_results_gpt-4.1-nano.csv": "gpt-4.1-nano",
    "Babar Azam_prediction_results_o4-mini.csv": "o4-mini",
    "Charith Asalanka_prediction_results_gpt-4.1.csv": "gpt-4.1",
    "Charith Asalanka_prediction_results_gpt-4.1-mini.csv": "gpt-4.1-mini",
    "Charith Asalanka_prediction_results_gpt-4.1-nano.csv": "gpt-4.1-nano",
    "Charith Asalanka_prediction_results_o4-mini.csv": "o4-mini",
    "Daryl Mitchell_prediction_results_gpt-4.1.csv": "gpt-4.1",
    "Daryl Mitchell_prediction_results_gpt-4.1-mini.csv": "gpt-4.1-mini",
    "Daryl Mitchell_prediction_results_gpt-4.1-nano.csv": "gpt-4.1-nano",
    "Daryl Mitchell_prediction_results_o4-mini.csv": "o4-mini",
    "Harry Tector_prediction_results_gpt-4.1.csv": "gpt-4.1",
    "Harry Tector_prediction_results_gpt-4.1-mini.csv": "gpt-4.1-mini",
    "Harry Tector_prediction_results_gpt-4.1-nano.csv": "gpt-4.1-nano",
    "Harry Tector_prediction_results_o4-mini.csv": "o4-mini",
    "Ibrahim Zadran_prediction_results_gpt-4.1.csv": "gpt-4.1",
    "Ibrahim Zadran_prediction_results_gpt-4.1-mini.csv": "gpt-4.1-mini",
    "Ibrahim Zadran_prediction_results_gpt-4.1-nano.csv": "gpt-4.1-nano",
    "Ibrahim Zadran_prediction_results_o4-mini.csv": "o4-mini",
    "Shai Hope_prediction_results_gpt-4.1.csv": "gpt-4.1",
    "Shai Hope_prediction_results_gpt-4.1-mini.csv": "gpt-4.1-mini",
    "Shai Hope_prediction_results_gpt-4.1-nano.csv": "gpt-4.1-nano",
    "Shai Hope_prediction_results_o4-mini.csv": "o4-mini",
    "Shubman Gill_prediction_results_gpt-4.1.csv": "gpt-4.1",
    "Shubman Gill_prediction_results_gpt-4.1-mini.csv": "gpt-4.1-mini",
    "Shubman Gill_prediction_results_gpt-4.1-nano.csv": "gpt-4.1-nano",
    "Shubman Gill_prediction_results_o4-mini.csv": "o4-mini",
}

OUTPUT_DIR = "predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_block(row):
    flags = []
    if row["general_stats_flag"] == "on":
        flags.append("General Stats")
    if row["venuewise_stats_flag"] == "on":
        flags.append("Venue Stats")
    if row["against_teams_stats_flag"] == "on":
        flags.append("Vs-Team Stats")
    if row["commentary"] == "on":
        flags.append("Commentary")
    return " + ".join(flags) if flags else "Baseline"


grouped_dfs = defaultdict(list)

for filename, model_key in FILES.items():
    print(f"Processing: {filename}")
    df = pd.read_csv(filename)

    df["block"] = df.apply(get_block, axis=1)
    df["sample_id"] = range(len(df))

    df["correct_top1"] = (
        df["mode_of_dismissal_ground_truth"] == df["most_likely_mode"]
    ).astype(int)

    df["correct_top2"] = df.apply(
        lambda row: int(
            row["mode_of_dismissal_ground_truth"]
            in row["mode_of_dismissal_predicted"].split(", ")[:2]
        ),
        axis=1,
    )

    out_df = df[["sample_id", "block", "correct_top1", "correct_top2"]]
    grouped_dfs[model_key].append(out_df)

for model_key, dfs in grouped_dfs.items():
    combined_df = pd.concat(dfs, ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, f"{model_key}_preds.csv")
    combined_df.to_csv(out_path, index=False)
    print(f"✓ Saved {out_path}")
