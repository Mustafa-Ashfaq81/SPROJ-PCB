import pandas as pd
import re

# Load the dataset (replace 'your_file.csv' with the actual path to your CSV file)
file_path = 'data\Dismissal Types CT17 Squad ODI Career - Bowler_list.csv'
df = pd.read_csv(file_path)

# Function to extract the team name from the player field
def extract_team(player_name):
    # Use regex to extract the team in brackets
    match = re.search(r'\((.*?)\)', player_name)
    if match:
        # Extract the last component after "/" if multiple teams are listed
        return match.group(1).split('/')[-1]
    return 'Unknown'

# Apply the function to create a new column 'Team'
df['Team'] = df['Player'].apply(extract_team)

# Remove the team name from the 'Player' column (cleaning up the original player names)
df['Player'] = df['Player'].apply(lambda x: re.sub(r'\(.*?\)', '', x).strip())

# Save the updated DataFrame to a new CSV file
output_file = 'updated_bowler_data_with_team.csv'
df.to_csv(output_file, index=False)

print(f"Updated dataset saved to {output_file}")
