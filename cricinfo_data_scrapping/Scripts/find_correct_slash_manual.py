import os
import json
import re

def main():
    # Directory containing JSON files
    directory = "../match_scorecards_json"

    # Updated Regex pattern to match "series_name" field with multiple countries in parentheses
    pattern = re.compile(
        # 1) Matches e.g. "in CountryA/CountryB[/CountryC...]"
        r"in \b[A-Za-z]+/[A-Za-z]+(/[A-Za-z]+)*\b"
        # 2) OR matches e.g. "... (Australia, South Africa, Zimbabwe), 2014"
        #    allowing multiple words in each country name: [A-Za-z ]+
        r"|.*\([A-Za-z ]+(,\s?[A-Za-z ]+)+\),\s*\d{4}.*"
    )
    matching_files = []

    # --- Part 1: Find all files that match the pattern ---
    for filename in os.listdir(directory):
        if filename.endswith(".json"):  # Only process JSON files
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    # Check if "series_name" exists and matches the regex pattern
                    if "series_name" in data and pattern.search(data["series_name"]):
                        matching_files.append(filename)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {filename}")

    # Print matching files
    print("Matching files:", matching_files)
    print(f"Number of matches to correct: {len(matching_files)}")

    # --- Part 2: Prompt user to correct the series_name for matching files ---
    for filename in matching_files:
        filepath = os.path.join(directory, filename)

        # Load JSON data
        with open(filepath, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {filename}")
                continue

        # Extract the current series_name and match date
        series_name = data.get("series_name", "N/A")
        date = data.get("match_details", {}).get("date", "N/A")

        # Display info to user
        print(f"\nEditing file: {filename}")
        print(f"📅 Match Date: {date}")
        print(f"🏆 Current Series Name: {series_name}")

        # Ask user for modification
        new_series_name = input("Enter new series_name (or press Enter to keep unchanged): ").strip()

        # Update and save if there's a change
        if new_series_name and new_series_name != series_name:
            data["series_name"] = new_series_name
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            print(f"✅ Updated and saved: {filename}")
        else:
            print("No changes made.")

    print("\n🎉 All files processed!")

if __name__ == "__main__":
    main()
