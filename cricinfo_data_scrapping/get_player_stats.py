import os
import json
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
import re

def safe_float_conversion(value, default=0.0):
    """
    Safely converts a value to float. Returns default if conversion fails.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int_conversion(value, default=0):
    """
    Safely converts a value to int. Returns default if conversion fails.
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def parse_match_date(match_details):
    """
    Parses the match date from match_details.
    """
    # Attempt to extract date from 'match_days' first
    match_days = match_details.get('match_days', '')
    date_pattern = r'(\d{1,2} \w+ \d{4})'
    match = re.search(date_pattern, match_days)
    if match:
        date_str = match.group(1)
        try:
            return datetime.strptime(date_str, '%d %B %Y')
        except ValueError:
            pass  # Continue to other fields if parsing fails

    # If not found in 'match_days', try 'date' field
    date_str = match_details.get('date', '')
    if date_str:
        # Assuming the year is present in 'season'
        season = match_details.get('season', '')
        full_date_str = f"{date_str} {season}"
        try:
            return datetime.strptime(full_date_str, '%B %d %Y')
        except ValueError:
            pass  # Continue to other fields if parsing fails

    # If not found in 'date', try to extract from 'header'
    header = match_details.get('header', '')
    date_pattern = r'(\d{1,2} \w+ \d{4})'
    match = re.search(date_pattern, header)
    if match:
        date_str = match.group(1)
        try:
            return datetime.strptime(date_str, '%d %B %Y')
        except ValueError:
            return None

    return None

def clean_name(name):
    """
    Removes annotations like (c), (wk), etc., from names and converts to lowercase.
    """
    return re.sub(r'\(.*?\)', '', name).strip().lower()

def parse_match_file(file_path, player_name, team_name):
    """
    Parses a single match JSON file and extracts batting, bowling, dismissal data, and match date for the specified player.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            match_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {file_path}")
            return None

    player_stats = {
        'batting': {},
        'bowling': {},
        'dismissals': [],
        'match_date': None
    }

    # Extract match date
    match_details = match_data.get('match_details', {})
    match_date = parse_match_date(match_details)
    if match_date:
        player_stats['match_date'] = match_date

    innings = match_data.get('innings', [])

    player_team = team_name.lower()
    player_name_lower = player_name.lower()

    for inning in innings:
        team = inning.get('team', '').lower()

        # Process Batting and Dismissal Data when player's team is batting
        if team == player_team:
            # Process Batting Data
            batting = inning.get('batting', [])
            for batsman in batting:
                batsman_name_clean = clean_name(batsman.get('batsman', ''))
                if batsman_name_clean == player_name_lower:
                    # Safely convert numeric fields
                    player_stats['batting'] = {
                        'runs': safe_int_conversion(batsman.get('runs', 0)),
                        'balls': safe_int_conversion(batsman.get('balls', 0)),
                        'fours': safe_int_conversion(batsman.get('fours', 0)),
                        'sixes': safe_int_conversion(batsman.get('sixes', 0)),
                        'strike_rate': safe_float_conversion(batsman.get('strike_rate', 0.0)),
                        'dismissal': batsman.get('dismissal', 'not out')
                    }

                    # Process Dismissal Data
                    dismissal = batsman.get('dismissal', '').lower()
                    if dismissal != 'not out':
                        # Initialize variables
                        bowler_type = 'unknown'
                        bowler_name = None
                        dismissal_method = 'oth'  # Default to 'oth' for other

                        # Define regex patterns for different dismissal types
                        dismissal_patterns = [
                            # Caught and Bowled: "c & b Bowler"
                            (r'c\s*&\s*b\s+([\w\s]+)', 'caught & bowled'),
                            # Caught: "c Fielder b Bowler" or "c Bowler b Bowler"
                            (r'c\b.*?b\s+([\w\s]+)', 'caught'),
                            # Stumped: "st [Fielder] b Bowler"
                            (r'st\s+[\w\s]+\s+b\s+([\w\s]+)', 'stumped'),
                            # Bowled: "b Bowler"
                            (r'\bb\s+([\w\s]+)', 'bowled'),
                            # LBW: "lbw b Bowler"
                            (r'lbw\s+b\s+([\w\s]+)', 'lbw'),
                            # Run Out: "run out"
                            (r'\brun out\b', 'run out'),
                        ]

                        # Attempt to match each pattern
                        for pattern, method in dismissal_patterns:
                            match = re.search(pattern, dismissal)
                            if match:
                                if method != 'run out':
                                    bowler_name = match.group(1).strip()
                                    dismissal_method = method
                                else:
                                    dismissal_method = method
                                break  # Stop after the first matching pattern

                        if bowler_name:
                            # To get bowler type, look up the bowler's type in the opponent's bowling data
                            opponent_inning = next(
                                (inn for inn in innings if inn.get('team', '').lower() != player_team), None)
                            if opponent_inning:
                                opponent_bowlers = opponent_inning.get('bowling', [])
                                # Prepare a list of bowler names
                                bowler_names = [clean_name(bowler.get('bowler', '')) for bowler in opponent_bowlers]
                                bowler_name_clean = clean_name(bowler_name)
                                # Try exact match
                                if bowler_name_clean in bowler_names:
                                    index = bowler_names.index(bowler_name_clean)
                                    bowler = opponent_bowlers[index]
                                    bowler_type = bowler.get('b_type', 'unknown').lower()
                                else:
                                    # Try matching last names
                                    bowler_last_names = {clean_name(bowler.get('bowler', '')).split()[-1]: bowler for bowler in opponent_bowlers}
                                    bowler_last_name = bowler_name_clean.split()[-1]
                                    bowler = bowler_last_names.get(bowler_last_name)
                                    if bowler:
                                        bowler_type = bowler.get('b_type', 'unknown').lower()
                                    else:
                                        # Try partial matching
                                        for b_name, bowler in zip(bowler_names, opponent_bowlers):
                                            if bowler_name_clean in b_name or b_name in bowler_name_clean:
                                                bowler_type = bowler.get('b_type', 'unknown').lower()
                                                break
                        else:
                            # For run outs or cases without a bowler
                            bowler_type = 'unknown'
                            bowler_name = None

                        dismissal_entry = {
                            'bowler_type': bowler_type,
                            'dismissal_method': dismissal_method,
                            'bowler_name': bowler_name
                        }
                        player_stats['dismissals'].append(dismissal_entry)
                    else:
                        player_stats['batting']['dismissal'] = 'not out'
                    break  # Assume player appears only once per batting innings

        # Process Bowling Data when opponent is batting
        else:
            # Process Bowling Data
            bowling = inning.get('bowling', [])
            for bowler in bowling:
                bowler_name_clean = clean_name(bowler.get('bowler', ''))
                if bowler_name_clean == player_name_lower:
                    # Safely convert numeric fields
                    player_stats['bowling'] = {
                        'overs': safe_float_conversion(bowler.get('overs', 0.0)),
                        'maidens': safe_int_conversion(bowler.get('maidens', 0)),
                        'runs_conceded': safe_int_conversion(bowler.get('runs_conceded', 0)),
                        'wickets': safe_int_conversion(bowler.get('wickets', 0)),
                        'economy': safe_float_conversion(bowler.get('economy', 0.0)),
                        'bowler_type': bowler.get('b_type', 'unknown').lower()
                    }
                    break  # Assume player appears only once per bowling innings

    # Only return stats if player participated in this match
    if player_stats['batting'] or player_stats['bowling'] or player_stats['dismissals']:
        return player_stats
    else:
        return None

def aggregate_stats(all_stats):
    """
    Aggregates batting, bowling, and dismissal data from all matches.
    Also calculates the span of the player's career based on match dates.
    """
    aggregated = {
        'batting': {
            'runs': 0,
            'balls': 0,
            'fours': 0,
            'sixes': 0,
            'strike_rate_total': 0.0,  # To calculate average strike rate later
            'dismissals': 0,
            'not_outs': 0
        },
        'bowling': {
            'overs': 0.0,
            'maidens': 0,
            'runs_conceded': 0,
            'wickets': 0,
            'economy_total': 0.0  # To calculate average economy rate later
        },
        'dismissals': defaultdict(lambda: {
            'Dis': 0
        }),
        'match_dates': [],
        'matches_played': 0,
        'batting_innings': 0,
        'bowling_innings': 0
    }

    for match in all_stats:
        player_batted = bool(match.get('batting'))
        player_bowled = bool(match.get('bowling'))
        player_dismissals = bool(match.get('dismissals'))

        if player_batted or player_bowled or player_dismissals:
            aggregated['matches_played'] += 1  # Increment matches played if player participated

        # Aggregate Match Dates
        match_date = match.get('match_date')
        if match_date:
            aggregated['match_dates'].append(match_date)

        # Aggregate Batting
        batting = match.get('batting', {})
        if batting:
            aggregated['batting_innings'] += 1  # Increment batting innings
            aggregated['batting']['runs'] += batting.get('runs', 0)
            aggregated['batting']['balls'] += batting.get('balls', 0)
            aggregated['batting']['fours'] += batting.get('fours', 0)
            aggregated['batting']['sixes'] += batting.get('sixes', 0)
            aggregated['batting']['strike_rate_total'] += batting.get('strike_rate', 0.0)
            if batting.get('dismissal', 'not out') != 'not out':
                aggregated['batting']['dismissals'] += 1
            else:
                aggregated['batting']['not_outs'] += 1

        # Aggregate Bowling
        bowling = match.get('bowling', {})
        if bowling:
            aggregated['bowling_innings'] += 1  # Increment bowling innings
            aggregated['bowling']['overs'] += bowling.get('overs', 0.0)
            aggregated['bowling']['maidens'] += bowling.get('maidens', 0)
            aggregated['bowling']['runs_conceded'] += bowling.get('runs_conceded', 0)
            aggregated['bowling']['wickets'] += bowling.get('wickets', 0)
            aggregated['bowling']['economy_total'] += bowling.get('economy', 0.0)

        # Aggregate Dismissals
        for dismissal in match.get('dismissals', []):
            bowler_type = dismissal.get('bowler_type', 'unknown').lower()
            aggregated['dismissals'][bowler_type]['Dis'] += 1

    # Calculate Batting Average and Strike Rate
    if aggregated['batting']['dismissals'] > 0:
        aggregated['batting']['bat_avg'] = round(aggregated['batting']['runs'] / aggregated['batting']['dismissals'], 2)
    else:
        aggregated['batting']['bat_avg'] = None  # Undefined if never dismissed

    if aggregated['batting']['balls'] > 0:
        aggregated['batting']['strike_rate'] = round((aggregated['batting']['runs'] / aggregated['batting']['balls']) * 100, 2)
    else:
        aggregated['batting']['strike_rate'] = None  # Undefined if no balls faced

    # Calculate Bowling Average and Economy
    if aggregated['bowling']['wickets'] > 0:
        aggregated['bowling']['bowl_avg'] = round(aggregated['bowling']['runs_conceded'] / aggregated['bowling']['wickets'], 2)
    else:
        aggregated['bowling']['bowl_avg'] = None  # Undefined if no wickets taken

    if aggregated['bowling']['overs'] > 0:
        aggregated['bowling']['economy'] = round(aggregated['bowling']['runs_conceded'] / aggregated['bowling']['overs'], 2)
    else:
        aggregated['bowling']['economy'] = None  # Undefined if no overs bowled

    # Calculate Span
    if aggregated['match_dates']:
        earliest_date = min(aggregated['match_dates'])
        latest_date = max(aggregated['match_dates'])
        span = f"{earliest_date.year} - {latest_date.year}"
    else:
        span = "Unknown"

    # Prepare final aggregated data
    final_aggregated = {
        'batting': aggregated['batting'],
        'bowling': aggregated['bowling'],
        'dismissal_data': [],
        'span': span,
        'matches_played': aggregated['matches_played'],
        'batting_innings': aggregated['batting_innings'],
        'bowling_innings': aggregated['bowling_innings']
    }

    # Populate dismissal data
    for bowler_type, data in aggregated['dismissals'].items():
        dismissal_entry = {
            'Bowler Type': bowler_type.replace('-', ' ').title(),
            'Span': span,
            'Dis': data['Dis'],
            # Additional fields like 'Ave', 'BF', 'Runs' would require more detailed data
        }
        final_aggregated['dismissal_data'].append(dismissal_entry)

    return final_aggregated

def process_file(args):
    """
    Wrapper function to unpack arguments for multiprocessing.
    """
    file_path, player_name, team_name = args
    return parse_match_file(file_path, player_name, team_name)

def generate_player_stats(player_name, team_name, scorecards_dir='match_scorecards_json'):
    """
    Generates a comprehensive statistics JSON file for the specified player.
    """
    all_stats = []

    # Prepare list of all JSON files
    json_files = [
        os.path.join(scorecards_dir, filename)
        for filename in os.listdir(scorecards_dir)
        if filename.endswith('.json')
    ]

    print(f"Processing {len(json_files)} match files...")

    # Prepare arguments for multiprocessing
    args = [(file_path, player_name, team_name) for file_path in json_files]

    # Use multiprocessing Pool to process files in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_file, args)

    # Filter out None results
    all_stats = [result for result in results if result is not None]

    if not all_stats:
        print(f"No data found for player '{player_name}' in team '{team_name}'.")
        return {}

    # Aggregate all stats
    aggregated_stats = aggregate_stats(all_stats)

    # Prepare final JSON structure
    player_stats_json = {
        'player': player_name,
        'team': team_name,
        'span': aggregated_stats['span'],
        'matches_played': aggregated_stats['matches_played'],
        'batting_innings': aggregated_stats['batting_innings'],
        'bowling_innings': aggregated_stats['bowling_innings'],
        'batting': {
            'Runs': aggregated_stats['batting']['runs'],
            'Balls Faced': aggregated_stats['batting']['balls'],
            'Fours': aggregated_stats['batting']['fours'],
            'Sixes': aggregated_stats['batting']['sixes'],
            'Strike Rate': aggregated_stats['batting']['strike_rate'],
            'Dismissals': aggregated_stats['batting']['dismissals'],
            'Not Outs': aggregated_stats['batting']['not_outs'],
            'Batting Average': aggregated_stats['batting']['bat_avg']
        },
        'bowling': {
            'Overs': aggregated_stats['bowling']['overs'],
            'Maidens': aggregated_stats['bowling']['maidens'],
            'Runs Conceded': aggregated_stats['bowling']['runs_conceded'],
            'Wickets': aggregated_stats['bowling']['wickets'],
            'Economy': aggregated_stats['bowling']['economy'],
            'Bowling Average': aggregated_stats['bowling']['bowl_avg']
        },
        'dismissal_data': aggregated_stats['dismissal_data']
    }

    return player_stats_json

def save_stats_to_json(player_stats, output_filename):
    """
    Saves the player statistics to a JSON file.
    """
    if not player_stats:
        print("No statistics to save.")
        return

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(player_stats, f, indent=4)
    print(f"Player statistics saved to {output_filename}")

def main():
    # Hard-coded Player and Team Name
    player_name = "Fakhar Zaman"  # Change this to the desired player
    team_name = "Pakistan"           # Change this to the desired team

    # Directory containing match scorecard JSON files
    scorecards_dir = "match_scorecards_json"  # Ensure this path is correct

    # Check if the directory exists
    if not os.path.isdir(scorecards_dir):
        print(f"Directory '{scorecards_dir}' does not exist. Please check the path.")
        return

    # Generate Player Stats
    player_stats = generate_player_stats(player_name, team_name, scorecards_dir)

    if not player_stats:
        print("No statistics generated.")
        return

    # Define Output Filename
    safe_player_name = player_name.lower().replace(' ', '_')
    output_filename = f"{safe_player_name}_stats.json"

    # Save to JSON
    save_stats_to_json(player_stats, output_filename)

if __name__ == "__main__":
    main()
