import os
import csv

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

# Path to your directory containing commentary text files
PROCESSED_DIR = "."

# Path to the input CSV
INPUT_CSV = "series_match_commentary_urls.csv"

# Path to the updated CSV (rows with processed IDs removed)
OUTPUT_CSV = "series_filtered_updated.csv"

# Path to the log file listing processed matches
PROCESSED_LOG = "processed_matches.log"

# -------------------------------------------------------------
# Step 1: Collect all match IDs from filenames in PROCESSED_DIR
# -------------------------------------------------------------
processed_ids = set()

# Go through each file in the processed_commentaries folder
for filename in os.listdir(PROCESSED_DIR):
    # We only care about .txt files that seem to have the ID first (e.g. "1457925_ABC_...txt")
    if filename.endswith(".txt"):
        # Extract match ID (the text before the first underscore)
        # Example filename: "1457925_AUS_commentary_....txt"
        match_id_str = filename.split("_")[0]
        if match_id_str.isdigit():
            processed_ids.add(match_id_str)

# -------------------------------------------------------------
# Step 2: Read the CSV and figure out which rows are processed
# -------------------------------------------------------------
rows_to_keep = []
processed_rows = []

# Read the existing CSV
with open(INPUT_CSV, mode="r", encoding="utf-8", newline="") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) < 2:
            # Skip malformed rows (or handle as appropriate)
            continue

        # The second column typically has something like: "Zimbabwe Vs Afghanistan 3Rd Odi 1457925"
        # The match ID is presumably the last 'word'
        second_col = row[1].strip()
        # We can safely split by space, get the last item if it’s numeric
        possible_id = second_col.split()[-1]

        # If the last item is a digit and in our processed_ids set, we consider it processed
        if possible_id.isdigit() and possible_id in processed_ids:
            processed_rows.append(possible_id)
        else:
            # If it's not in processed_ids, we keep the row in the updated CSV
            rows_to_keep.append(row)

# -------------------------------------------------------------
# Step 3: Write the updated CSV (unprocessed rows only)
# -------------------------------------------------------------
with open(OUTPUT_CSV, mode="w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(rows_to_keep)

# -------------------------------------------------------------
# Step 4: Write or print the log of processed matches
# -------------------------------------------------------------
with open(PROCESSED_LOG, mode="w", encoding="utf-8") as log_file:
    for match_id in processed_rows:
        log_file.write(f"Match {match_id} has been successfully processed.\n")

print("Done!")
print(f"Processed matches have been logged in '{PROCESSED_LOG}'.")
print(f"An updated CSV without processed matches has been written to '{OUTPUT_CSV}'.")
