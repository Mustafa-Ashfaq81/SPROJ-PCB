import os
import json
from pathlib import Path

# Path to the directory containing the scorecards
SCORECARDS_DIR = "../cricinfo_data_scrapping/match_scorecards_json"

# Output file path
OUTPUT_FILE = "match_info_mapping.json"

# Initialize the final dictionary
match_info_map = {}

# Traverse the directory
for filename in os.listdir(SCORECARDS_DIR):
    if filename.endswith(".json") and filename.startswith("match_"):
        match_id = filename.replace("match_", "").replace(".json", "")
        file_path = os.path.join(SCORECARDS_DIR, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract required fields
            info = {
                "series_name": data.get("series_name"),
                "match_title": data.get("match_title"),
            }

            match_details = data.get("match_details", {})
            keys_to_extract = [
                "header", "toss", "series", "season",
                "player_of_the_match", "match_days", "venue", "date"
            ]

            for key in keys_to_extract:
                info[key] = match_details.get(key)

            # Add to final dictionary
            match_info_map[match_id] = info

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Save the final dictionary to a JSON file
with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_file:
    json.dump(match_info_map, out_file, indent=4, ensure_ascii=False)

print(f"✅ Extraction complete. Output written to {OUTPUT_FILE}")
