import json
import os

# Load the bowlers categorized JSON
with open('bowlers_categorized.json', 'r') as file:
    bowlers_categorized = json.load(file)

# Directory containing match scorecards
scorecards_dir = 'match_scorecards_json'

# Function to update bowlers in a scorecard
def add_bowler_type(scorecard):
    for inning in scorecard.get('innings', []):
        for bowler in inning.get('bowling', []):
            name = bowler.get('bowler', '')
            # Add bowling type if present in the categorized bowlers file, otherwise add "Unknown"
            bowler['b_type'] = bowlers_categorized.get(name, 'Unknown')
    return scorecard

# Iterate over all the scorecards and update each one
for filename in os.listdir(scorecards_dir):
    if filename.endswith('.json'):
        filepath = os.path.join(scorecards_dir, filename)
        
        # Read the scorecard JSON file
        with open(filepath, 'r') as file:
            scorecard = json.load(file)
        
        # Update the scorecard with bowling types
        updated_scorecard = add_bowler_type(scorecard)
        
        # Save the updated scorecard back to the file
        with open(filepath, 'w') as file:
            json.dump(updated_scorecard, file, indent=4)

print("All scorecards have been updated with bowler types.")