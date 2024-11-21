import os
import json
import csv

# Directory containing the updated match JSON files with bowler_type
json_dir = 'data/odis_json_with_bowler_type/'

# Initialize min and max years for span calculation
min_year = float('inf')
max_year = -float('inf')

# Function to track dismissal type
def track_dismissal_type(wicket, dismissal_stats):
    if wicket:
        dismissal_type = wicket[0].get('kind')
        if dismissal_type == 'bowled':
            dismissal_stats['bwd'] += 1
        elif dismissal_type == 'caught':
            dismissal_stats['caught'] += 1
        elif dismissal_type == 'stumped':
            dismissal_stats['st'] += 1
        elif dismissal_type == 'lbw':
            dismissal_stats['lbw'] += 1
        elif dismissal_type == 'hit wicket':
            dismissal_stats['hw'] += 1
        elif dismissal_type == 'run out':
            dismissal_stats['ro'] += 1
        else:
            dismissal_stats['oth'] += 1

# Function to process a single match JSON file
def process_match_json(file_path, player_stats):
    global min_year, max_year

    with open(file_path, 'r') as f:
        match_data = json.load(f)

    # Extract the match year for span calculation
    match_date = match_data.get('info', {}).get('dates', [])[0]
    match_year = int(match_date.split('-')[0])
    
    # Update the span (min/max year)
    min_year = min(min_year, match_year)
    max_year = max(max_year, match_year)

    # Extract team information
    teams = match_data.get('info', {}).get('teams', [])
    
    # Iterate over innings, overs, and deliveries to gather stats for all players
    for inning in match_data.get('innings', []):
        team = inning.get('team')  # The team of the batting side
        for over in inning.get('overs', []):
            for delivery in over.get('deliveries', []):
                bowler_type = delivery.get('bowler_type')  # Use the bowler_type from JSON
                batter = delivery.get('batter')
                runs = delivery.get('runs', {}).get('batter', 0)
                wicket = delivery.get('wickets', [])

                # Ensure player exists in stats dictionary
                if batter not in player_stats:
                    player_stats[batter] = {}

                # Ensure bowler type exists for this player
                if bowler_type not in player_stats[batter]:
                    player_stats[batter][bowler_type] = {
                        'team': team,
                        'balls_faced': 0,
                        'runs': 0,
                        'dismissals': 0,
                        'dot_balls': 0,  # Dot balls counter
                        'bwd': 0, 'caught': 0, 'st': 0, 'lbw': 0, 'hw': 0, 'ro': 0, 'oth': 0
                    }

                # Increment stats for the current bowler type
                player_stats[batter][bowler_type]['balls_faced'] += 1
                player_stats[batter][bowler_type]['runs'] += runs

                # Increment dot balls if no runs were scored
                if runs == 0:
                    player_stats[batter][bowler_type]['dot_balls'] += 1

                # Handle dismissals
                if wicket:
                    player_stats[batter][bowler_type]['dismissals'] += 1
                    # Track dismissal type
                    track_dismissal_type(wicket, player_stats[batter][bowler_type])

    return player_stats

# Function to process all JSON files and aggregate data for all players
def process_all_json_files():
    player_stats = {}

    # Iterate over all JSON files in the directory
    for file_name in os.listdir(json_dir):
        if file_name.endswith('.json'):
            match_file = os.path.join(json_dir, file_name)
            player_stats = process_match_json(match_file, player_stats)
    
    return player_stats

# Function to write player data to CSV
def write_player_data_to_csv(player_stats):
    global min_year, max_year
    output_file = 'all_players_batting_stats.csv'
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Player Name', 'Team', 'Bowler Type', 'Span', 'Dis', 'bwd', 'caught', 'st', 'lbw', 
                         'hw', 'ro', 'oth', 'dot_balls', 'Ave', 'BF', 'Runs'])
        
        for player, bowler_stats in player_stats.items():
            for bowler_type, stats in bowler_stats.items():
                # Calculate average
                ave = round(stats['runs'] / stats['dismissals'], 2) if stats['dismissals'] > 0 else 0
                
                writer.writerow([
                    player,  # Player name
                    stats['team'],  # Player's team
                    bowler_type,
                    f"{min_year}-{max_year}",  # Span
                    stats['dismissals'],
                    stats['bwd'], stats['caught'], stats['st'], stats['lbw'], 
                    stats['hw'], stats['ro'], stats['oth'],
                    stats['dot_balls'],  # New dot balls column
                    ave,
                    stats['balls_faced'],  # Balls faced
                    stats['runs']  # Runs
                ])
    
    print(f"Player data written to {output_file}")

# Run the script
player_stats = process_all_json_files()

# If data is found for players, write it to CSV
if player_stats:
    write_player_data_to_csv(player_stats)
