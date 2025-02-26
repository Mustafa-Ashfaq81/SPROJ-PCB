import os
import json
import re

# Directory containing JSON files
directory = "match_scorecards_json"

# Regex pattern to match "series_name" field with multiple countries separated by "/"
pattern = re.compile(r'in \b[A-Za-z]+/[A-Za-z]+(/[A-Za-z]+)*\b')

matching_files = []

for filename in os.listdir(directory):
    if filename.endswith(".json"):  # Only process JSON files
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)  # Load JSON
                if "series_name" in data and pattern.search(data["series_name"]):
                    matching_files.append(filename)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {filename}")

# Print matching files
print("Matching files:", matching_files)
print(f"Number of matches to correct: {len(matching_files)}")
