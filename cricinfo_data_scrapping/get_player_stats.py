import os
import json
import re
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count

def safe_float_conversion(value, default=0.0):
    """Safely converts a value to float. Returns default if conversion fails."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int_conversion(value, default=0):
    """Safely converts a value to int. Returns default if conversion fails."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def parse_match_date(match_details):
    """Parses the match date from match_details."""
    match_days = match_details.get('match_days', '')
    date_pattern = r'(\d{1,2} \w+ \d{4})'
    match = re.search(date_pattern, match_days)
    if match:
        date_str = match.group(1)
        try:
            return datetime.strptime(date_str, '%d %B %Y')
        except ValueError:
            pass

    date_str = match_details.get('date', '')
    if date_str:
        season = match_details.get('season', '')
        full_date_str = f"{date_str} {season}"
        try:
            return datetime.strptime(full_date_str, '%B %d %Y')
        except ValueError:
            pass

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
    """Removes annotations like (c), (wk), etc., from names and converts to lowercase."""
    return re.sub(r'\(.*?\)', '', name).strip()


def generate_name_variants(full_name):
    name_parts = full_name.strip().split()
    variants = set()
    variants.add(' '.join(name_parts).lower())
    variants.add(''.join(name_parts).lower())
    if len(name_parts) >= 2:
        first_initials = ''.join([part[0] for part in name_parts[:-1]])
        last_name = name_parts[-1]
        variants.add(f"{first_initials} {last_name}".lower())
        variants.add(f"{first_initials}{last_name}".lower())
        variants.add(last_name.lower())
    else:
        variants.add(name_parts[0].lower())
    return variants

def extract_dismissal_data(batsman, dismissal_text, innings, bowler_mapping):
    bowler_type = 'unknown'
    bowler_name = None
    dismissal_method = 'other'

    dismissal_patterns = [
        (r'c\s+\w+\s+b\s+([\w\s]+)', 'caught'),
        (r'b\s+([\w\s]+)', 'bowled'),
        (r'lbw\s+b\s+([\w\s]+)', 'lbw'),
        (r'st\s+\w+\s+b\s+([\w\s]+)', 'stumped'),
    ]

    for pattern, method in dismissal_patterns:
        match = re.search(pattern, dismissal_text)
        if match:
            bowler_name = match.group(1).strip()
            dismissal_method = method
            break

    if bowler_name in bowler_mapping:
        bowler_name_full = bowler_mapping[bowler_name]
    else:
        bowler_name_full = bowler_name

    for inning in innings:
        for bowler in inning.get('bowling', []):
            if bowler.get('bowler') == bowler_name_full:
                bowler_type = bowler.get('b_type', 'unknown').lower()
                break

    return {
        'bowler_type': bowler_type,
        'dismissal_method': dismissal_method,
        'bowler_name': bowler_name_full
    }
    
    
def parse_match_file(file_path, player_name, team_name):
    """Parses a single match JSON file and extracts data for the specified player."""
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
        'fielding': {
            'catches': 0,
            'stumpings': 0,
            'run_outs': 0
        },
        'match_date': None,
        'played': False  
    }

    
    match_details = match_data.get('match_details', {})
    match_date = parse_match_date(match_details)
    if match_date:
        player_stats['match_date'] = match_date


    player_name_lower = player_name.lower()
    team_name_lower = team_name.lower()
    
    player_name_lower = player_name.lower()
    team_name_lower = team_name.lower()
    playing_xi = match_data.get('playing_xi', {})
    player_team_xi = playing_xi.get(team_name, [])
    player_team_xi = [clean_name(name).lower() for name in player_team_xi]


    if player_name_lower in player_team_xi:
        player_stats['played'] = True
    else:
        return None

    # Proceed to extract batting, bowling, and fielding data
    innings = match_data.get('innings', [])
    for inning in innings:
        team = inning.get('team', '').lower()
        is_player_team = (team == team_name_lower)

        if is_player_team:
            batting = inning.get('batting', [])
            for batsman in batting:
                batsman_name_clean = clean_name(batsman.get('batsman', '')).lower()
                if batsman_name_clean == player_name_lower:
                    player_stats['batting'] = {
                        'runs': safe_int_conversion(batsman.get('runs', 0)),
                        'balls': safe_int_conversion(batsman.get('balls', 0)),
                        'fours': safe_int_conversion(batsman.get('fours', 0)),
                        'sixes': safe_int_conversion(batsman.get('sixes', 0)),
                        'strike_rate': safe_float_conversion(batsman.get('strike_rate', 0.0)),
                        'dismissal': batsman.get('dismissal', 'not out')
                    }

                    dismissal = batsman.get('dismissal', '')
                    if dismissal.lower() != 'not out':
                        dismissal_data = extract_dismissal_data(
                            batsman_name_clean, dismissal, innings, match_data.get('bowler_mapping', {})
                        )
                        player_stats['dismissals'].append(dismissal_data)
                    break

    return player_stats

def aggregate_stats(all_stats):
    """Aggregates batting, bowling, fielding, and dismissal data from all matches. Also calculates the span of the player's career based on match dates."""
    
    aggregated = {
        'batting': {
            'runs': 0,
            'balls': 0,
            'fours': 0,
            'sixes': 0,
            'strike_rate_total': 0.0,
            'dismissals': 0,
            'not_outs': 0
        },
        'bowling': {
            'overs': 0.0,
            'maidens': 0,
            'runs_conceded': 0,
            'wickets': 0,
            'economy_total': 0.0  
        },
        'fielding': {
            'catches': 0,
            'stumpings': 0,
            'run_outs': 0
        },
        "dismissals": defaultdict(lambda: {'Dis': 0}),
        "match_dates": [],
        "matches_played": 0,
        "batting_innings": 0,
        "bowling_innings": 0,
        "fielding_innings": 0 
    }
    
    for match in all_stats:
        # Check if the player played in the match 
        if match.get("played"):
            aggregated["matches_played"] += 1
            # Aggregate Match Dates 
            match_date = match["match_date"]
            if match_date: 
                aggregated["match_dates"].append(match_date)

        # Aggregate Batting 
        batting = match["batting"] 
        if batting: 
            aggregated["batting_innings"] += 1  
            aggregated["batting"]["runs"] += batting["runs"]
            aggregated["batting"]["balls"] += batting["balls"]
            aggregated["batting"]["fours"] += batting["fours"]
            aggregated["batting"]["sixes"] += batting["sixes"]
            aggregated["batting"]["strike_rate_total"] += batting["strike_rate"]
            if batting["dismissal"] != "not out":
                aggregated["batting"]["dismissals"] += 1 
            else: 
                aggregated["batting"]["not_outs"] += 1

        # Aggregate Bowling 
        bowling = match["bowling"]
        if bowling: 
            aggregated["bowling_innings"] += 1  
            aggregated["bowling"]["overs"] += bowling["overs"]
            aggregated["bowling"]["maidens"] += bowling["maidens"]
            aggregated["bowling"]["runs_conceded"] += bowling["runs_conceded"]
            aggregated["bowling"]["wickets"] += bowling["wickets"]
            aggregated["bowling"]["economy_total"] += bowling["economy"]

        # Aggregate Fielding 
        fielding = match["fielding"]
        if fielding: 
            aggregated["fielding_innings"] += 1  
            aggregated["fielding"]["catches"] += fielding["catches"]
            aggregated["fielding"]["stumpings"] += fielding["stumpings"]
            aggregated["fielding"]["run_outs"] += fielding["run_outs"]

        # Aggregate Dismissals 
        for dismissal in match.get("dismissals", []): 
            bowler_type = dismissal.get("bowler_type", "unknown").lower()  
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
        "batting": aggregated['batting'],  
        "bowling": aggregated['bowling'],  
        "fielding": aggregated['fielding'],  
        "dismissal_data": [],  
        "span": span,  
        "matches_played": aggregated['matches_played'],  
        "batting_innings": aggregated['batting_innings'],  
        "bowling_innings": aggregated['bowling_innings'],  
        "fielding_innings": aggregated['fielding_innings']   
    }  

    # Populate dismissal data 
    for bowler_type, data in aggregated['dismissals'].items():   
        dismissal_entry = {  
            "Bowler Type": bowler_type.replace('-', ' ').title(),   
            "Span": span,   
            "Dis": data['Dis'],   
            # Additional fields like ‘Ave’, ‘BF’, ‘Runs’ would require more detailed data   
        }   
        final_aggregated['dismissal_data'].append(dismissal_entry)   

    return final_aggregated  

def process_file(args):   
    """Wrapper function to unpack arguments for multiprocessing."""
    file_path, player_name, team_name = args     
    return parse_match_file(file_path, player_name, team_name)  

def generate_player_stats(player_name, team_name, scorecards_dir='match_scorecards_json'):
    """Generates a comprehensive statistics JSON file for the specified player."""
    
    all_stats = []
    
    # Prepare list of all JSON files
    json_files = [os.path.join(scorecards_dir, filename) for filename in os.listdir(scorecards_dir) if filename.endswith('.json')]
    
    print(f"Processing {len(json_files)} match files...")
    
    # Prepare arguments for multiprocessing
    args = [(file_path, player_name, team_name) for file_path in json_files]
    
    # Use multiprocessing Pool to process files in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_file, args)
        
        # Filter out None results
        all_stats = [result for result in results if result is not None]
    
    # Check results
    if not all_stats:
        print(f"No data found for player '{player_name}' in team '{team_name}'.")
        return {}
    
    # Aggregate all stats
    aggregated_stats = aggregate_stats(all_stats)
    
    # Prepare final JSON structure
    player_stats_json = {
        "player": player_name,
        "team": team_name,
        "span": aggregated_stats['span'],
        "matches_played": aggregated_stats['matches_played'],
        "batting_innings": aggregated_stats['batting_innings'],
        "bowling_innings": aggregated_stats['bowling_innings'],
        "fielding_innings": aggregated_stats['fielding_innings'],
        "batting": {
            "Runs": aggregated_stats['batting']['runs'],
            "Balls Faced": aggregated_stats['batting']['balls'],
            "Fours": aggregated_stats['batting']['fours'],
            "Sixes": aggregated_stats['batting']['sixes'],
            "Strike Rate": aggregated_stats['batting'].get("strike_rate"),
            "Dismissals": aggregated_stats['batting'].get("dismissals"),
            "Not Outs": aggregated_stats['batting'].get("not_outs"),
            "Batting Average": aggregated_stats['batting'].get("bat_avg")
        },
        "bowling": {
            "Overs": aggregated_stats['bowling']['overs'],
            "Maidens": aggregated_stats['bowling'].get("maidens"),
            "Runs Conceded": aggregated_stats['bowling'].get("runs_conceded"),
            "Wickets": aggregated_stats['bowling'].get("wickets"),
            "Economy Rate": aggregated_stats['bowling'].get("economy"),
            "Bowling Average": aggregated_stats['bowling'].get("bowl_avg")
        },
        "fielding": {
            "Catches": aggregated_stats['fielding'].get("catches"),
            "Stumpings": aggregated_stats['fielding'].get("stumpings"),
            "Run Outs": aggregated_stats['fielding'].get("run_outs")
        },
        "dismissals": aggregated_stats['dismissal_data']
    }

    # Save the JSON to file
    output_file = f"{player_name.lower().replace(' ', '_')}_stats.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(player_stats_json, f, indent=4, default=str)

    print(f"Player stats saved to {output_file}")
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
    player_name = "Virat Kohli"  # Change this to the desired player
    team_name = "India"      # Change this to the desired team

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