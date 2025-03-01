import os
import json

# Directory containing JSON files
directory = "match_scorecards_json"

# List of matching files from your previous script
matching_files = ['match_1466414.json', 'match_1466415.json', 'match_1466416.json', 'match_1466417.json', 'match_1466418.json', 'match_1466419.json', 'match_1466420.json', 'match_1466421.json', 'match_1468192.json', 'match_1468193.json', 'match_1468194.json', 'match_1468195.json']

for filename in matching_files:
    filepath = os.path.join(directory, filename)
    
    with open(filepath, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)  # Load JSON
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON: {filename}")
            continue

    # Extract necessary fields
    series_name = data.get("series_name", "N/A")
    date = data.get("match_details", {}).get("date", "N/A")

    print(f"\nEditing file: {filename}")
    print(f"📅 Match Date: {date}")
    print(f"🏆 Current Series Name: {series_name}")

    # Ask user for modification
    new_series_name = input("Enter new series_name (or press Enter to keep unchanged): ").strip()

    # Update and save if there's a change
    if new_series_name and new_series_name != series_name:
        data["series_name"] = new_series_name
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        print(f"✅ Updated and saved: {filename}")
    else:
        print("No changes made.")

print("\n🎉 All files processed!")
