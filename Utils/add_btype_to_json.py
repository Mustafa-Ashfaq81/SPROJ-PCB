import os
import json
import pandas as pd

# Load the "bowler_list.csv" file
bowler_list_file = 'data/Dismissal Types ODI - Bowlers_list_cleaned.csv'
bowler_df = pd.read_csv(bowler_list_file)

# Create a dictionary that maps bowler names to their types
bowler_type_map = bowler_df.set_index('Player')['Bowler Type'].to_dict()

# Directory containing the match JSON files
json_dir = 'data/odis_json/'
output_json_dir = 'data/odis_json_with_bowler_type/'  # Directory for updated JSON files

# Ensure the output directory exists
if not os.path.exists(output_json_dir):
    os.makedirs(output_json_dir)

# Function to update a single match JSON file
def update_json_with_bowler_type(file_path, output_path):
    with open(file_path, 'r') as f:
        match_data = json.load(f)
    
    # Iterate over the innings and deliveries to update bowler_type
    for inning in match_data.get('innings', []):
        for over in inning.get('overs', []):
            for delivery in over.get('deliveries', []):
                bowler = delivery.get('bowler')

                # Find the bowler type using the bowler name
                bowler_type = bowler_type_map.get(bowler, 'unknown')
                
                # Add or update the "bowler_type" field in the delivery
                delivery['bowler_type'] = bowler_type

    # Save the updated JSON file
    with open(output_path, 'w') as f_out:
        json.dump(match_data, f_out, indent=4)

# Process all JSON files in the directory
def update_all_json_files():
    for file_name in os.listdir(json_dir):
        if file_name.endswith('.json'):
            input_file = os.path.join(json_dir, file_name)
            output_file = os.path.join(output_json_dir, file_name)
            
            # Update the JSON file with the bowler type
            update_json_with_bowler_type(input_file, output_file)
            print(f"Updated: {file_name}")

# Run the update process
update_all_json_files()
