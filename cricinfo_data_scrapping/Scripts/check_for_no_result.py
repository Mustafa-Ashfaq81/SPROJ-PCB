#!/usr/bin/env python3
import os
import csv
import json
import re

# Directory containing match JSON files
JSON_DIR = "../match_scorecards_json"

# CSV file to filter
FILTERED_CSV = "series_filteredbyTopCountries.csv"

# Regex to capture the final numeric ID before "/ball-by-ball-commentary"
MATCH_ID_PATTERN = re.compile(r"-(\d+)/ball-by-ball-commentary")


def extract_match_id_from_commentary_url(url: str) -> str:
    """
    Extract the numeric match ID from an ESPN commentary URL.
    Example: ...66274/ball-by-ball-commentary -> '66274'
    """
    match = MATCH_ID_PATTERN.search(url)
    if match:
        return match.group(1)  # just the digits
    return None


def main():
    print("===== STEP 1: Scanning JSON files for 'No result' or 'no toss' =====")

    no_result_or_no_toss_ids = set()

    # Gather all JSON files
    json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]

    for filename in json_files:
        filepath = os.path.join(JSON_DIR, filename)

        # Strip off ".json" and remove "match_" prefix if present
        raw_id = os.path.splitext(filename)[0]  # e.g. "match_66274"
        if raw_id.startswith("match_"):
            raw_id = raw_id[len("match_") :]

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"    [WARN] Could not load {filepath}. Error: {e}")
            continue

        result = data.get("result", "").strip().lower()
        toss = data.get("match_details", {}).get("toss", "").strip().lower()

        print(f"Scanning {filename}: result='{result}', toss='{toss}'")

        # FIXED condition: empty result counts as "no result"
        if result == "no result" or toss == "no toss":
            no_result_or_no_toss_ids.add(raw_id)
            print(
                f"    -> Found match_id={raw_id} with result='{result}' or toss='{toss}'"
            )

    print(f"\nTotal JSON files scanned: {len(json_files)}")
    print(f"Match IDs with 'No result' or 'no toss': {len(no_result_or_no_toss_ids)}\n")

    if not no_result_or_no_toss_ids:
        print(
            "No matches found with 'no result' or 'no toss'. No CSV rows will be removed."
        )
        return

    print("===== STEP 2: Removing those IDs from series_filtered.csv =====")

    rows_to_keep = []
    total_rows = 0
    removed_rows = 0

    # Read the CSV
    with open(FILTERED_CSV, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        for row_index, row in enumerate(reader, start=1):
            total_rows += 1
            commentary_url = row.get("Commentary URL", "")
            match_id = extract_match_id_from_commentary_url(commentary_url)

            print(f"Row {row_index}: URL={commentary_url} --> match_id={match_id}")

            # If we fail to parse ID, assume we keep the row (cannot confirm removal)
            if not match_id:
                rows_to_keep.append(row)
                continue

            # If this match_id is in no_result_or_no_toss_ids, remove it
            if match_id in no_result_or_no_toss_ids:
                removed_rows += 1
                print(f"    -> Removing Row {row_index} (match_id={match_id})")
            else:
                rows_to_keep.append(row)

    # Overwrite the CSV
    with open(FILTERED_CSV, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_keep)

    print("\n===== DONE =====")
    print(f"Total rows in {FILTERED_CSV} before: {total_rows}")
    print(f"Rows removed: {removed_rows}")
    print(f"Rows kept: {len(rows_to_keep)}")


if __name__ == "__main__":
    main()
