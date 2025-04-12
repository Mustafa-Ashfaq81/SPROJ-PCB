import os
import json

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

            # Extract basic info
            info = {
                "series_name": data.get("series_name"),
                "match_title": data.get("match_title"),
            }

            # Extract match_details subfields
            match_details = data.get("match_details", {})
            keys_to_extract = [
                "header", "toss", "series", "season",
                "player_of_the_match", "match_days", "venue", "date"
            ]
            for key in keys_to_extract:
                info[key] = match_details.get(key)

            # Add playing XI if available
            if "playing_xi" in data and isinstance(data["playing_xi"], dict):
                info["playing_xi"] = data["playing_xi"]

            # Save in final map
            match_info_map[match_id] = info

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

# Save the complete mapping to JSON
with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_file:
    json.dump(match_info_map, out_file, indent=4, ensure_ascii=False)

print(f"✅ Completed! Output written to {OUTPUT_FILE}")
