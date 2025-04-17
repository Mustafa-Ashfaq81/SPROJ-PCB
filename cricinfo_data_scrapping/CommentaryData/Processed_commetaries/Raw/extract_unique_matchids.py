import os
from collections import defaultdict

# Directory containing your commentary files
folder_path = "."  # current directory, adjust if needed

# Dictionary to hold counts of match IDs
matchid_counts = defaultdict(int)

# Loop through all files
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        parts = filename.split("_")
        if parts[0].isdigit():
            matchid = parts[0]
            matchid_counts[matchid] += 1

# Filter match IDs that occur more than twice
repeated_matchids = [matchid for matchid, count in matchid_counts.items() if count > 2]

print("Repeated Match IDs (more than 2 occurrences):")
for matchid in repeated_matchids:
    print(matchid)
