import os
import json
import re

scorecards_folder = "../match_scorecards_json"
commentaries_folder = "../CommentaryData/Processed_commetaries/Raw"


def extract_match_id(filename):
    match = re.match(r"(\d+)_", filename)
    return match.group(1) if match else None


def extract_country(series_name):
    # First try: "in Country" at the beginning
    match1 = re.search(
        r"in ([A-Za-z\s]+?)\s(?:ODI|Test|T20|Series|Tour|Cup|Championship)", series_name
    )
    if match1:
        return match1.group(1).strip()

    # Second try: look inside parentheses e.g. "(in Country)"
    match2 = re.search(r"\(in ([^)]+?)\)", series_name)
    if match2:
        return match2.group(1).strip()

    return None


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

        series_name = data.get("series_name", "")
        country = extract_country(series_name)
        if not country:
            continue

        base, ext = os.path.splitext(fname)
        safe_country = country.replace("/", "-")
        new_fname = f"{base}_{safe_country}{ext}"

        old_path = os.path.join(commentaries_folder, fname)
        new_path = os.path.join(commentaries_folder, new_fname)

        os.rename(old_path, new_path)
        print(f"Renamed: {fname} --> {new_fname}")
    else:
        print(f"No matching JSON for {fname} (expected {json_filename})")
