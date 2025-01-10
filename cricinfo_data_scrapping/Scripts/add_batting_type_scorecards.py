import json
import os
import re

# Load the batters categorized JSON (the file that maps each batter's name 
# to "Right Handed", "Left Handed", or "Unknown")
with open('batters_categorized.json', 'r') as file:
    batters_categorized = json.load(file)

# Directory containing match scorecards
scorecards_dir = '../match_scorecards_json'

def add_batting_type(scorecard):
    """
    For each batter in the scorecard, remove any bracketed annotations 
    (like '(c)', '(wk)', etc.), then add a 'batting_type' key.
    """
    for inning in scorecard.get('innings', []):
        for batter in inning.get('batting', []):
            # Get the raw name from the JSON
            raw_name = batter.get('batsman', '')

            # Strip out anything in parentheses, e.g. (c), (wk), (capt), etc.
            cleaned_name = re.sub(r'\(.*?\)', '', raw_name).strip()

            # Look up the cleaned name in batters_categorized, defaulting to 'Unknown'
            batter['batting_type'] = batters_categorized.get(cleaned_name, 'Unknown')
    return scorecard

# Iterate over all scorecards and update each one
for filename in os.listdir(scorecards_dir):
    if filename.endswith('.json'):
        filepath = os.path.join(scorecards_dir, filename)

        # Read the scorecard JSON file
        with open(filepath, 'r', encoding='utf-8') as file:
            scorecard_data = json.load(file)

        # Update the scorecard with batting types
        updated_scorecard = add_batting_type(scorecard_data)

        # Write the updated scorecard back to file
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(updated_scorecard, file, indent=4)

print("All scorecards have been updated with batting types.")
