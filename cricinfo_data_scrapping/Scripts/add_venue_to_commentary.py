import os
import json
import re

# Absolute or relative paths from the script's location
scorecards_folder = "../match_scorecards_json"
commentaries_folder = "../CommentaryData/Processed_commetaries"


def extract_match_id(filename):
    match = re.match(r"(\d+)_", filename)
    return match.group(1) if match else None


for fname in os.listdir(commentaries_folder):
    if not fname.lower().endswith(".txt"):
        continue

    match_id = extract_match_id(fname)
    if not match_id:
        continue

    json_filename = f"match_{match_id}.json"
    json_path = os.path.join(scorecards_folder, json_filename)

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        venue = data.get("match_details", {}).get("venue", "")
        if not venue:
            continue

        base, ext = os.path.splitext(fname)
        safe_venue = venue.replace("/", "-")  # in case there's a slash
        new_fname = f"{base}_{safe_venue}{ext}"

        old_path = os.path.join(commentaries_folder, fname)
        new_path = os.path.join(commentaries_folder, new_fname)

        os.rename(old_path, new_path)
        print(f"Renamed: {fname} --> {new_fname}")
    else:
        print(f"No matching JSON for {fname} (expected {json_filename})")
