import os
import json
import re
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
import logging

# Configure logging to capture DEBUG and higher level logs for detailed tracing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define dismissal methods
DISMISSAL_METHODS = [
    "bowled",
    "caught-fielder",
    "caught wk",
    "stump",
    "lbw",
    "hit wicket",
    "caught and bowled"
]

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
    return re.sub(r'\(.*?\)', '', name).strip().lower()

def generate_name_variants(full_name):
    """Generates different name variants to enhance matching."""
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

def resolve_bowler_name(bowler_name, bowler_mapping):
    """
    Resolves the full bowler name using bowler_mapping.
    """
    bowler_name_lower = bowler_name.lower()
    partial_bowler_name = bowler_name_lower.split()[-1]
    possible_full_names = bowler_mapping.get(partial_bowler_name, [])
    if not possible_full_names:
        # Try to find using the entire bowler name
        possible_full_names = bowler_mapping.get(bowler_name_lower, [])
    if possible_full_names:
        return possible_full_names[0]  # Choose the first match
    else:
        return bowler_name  # Use as-is if no mapping found

def find_bowler_type(bowler_name_full, innings):
    """
    Finds the bowler type from the innings data.
    """
    for inning in innings:
        for bowler_entry in inning.get('bowling', []):
            if bowler_entry.get('bowler', '').lower() == bowler_name_full.lower():
                return bowler_entry.get('b_type', 'Unknown')
    return 'Unknown'

def extract_dismissal_data(batsman, dismissal_text, innings, bowler_mapping, wicket_keepers):
    """
    Extracts dismissal information including bowler type and dismissal method.
    """
    bowler_type = 'Unknown'
    dismissal_method = 'other'
    bowler_name_full = 'Unknown'
    fielder = 'Unknown'

    # Define enhanced dismissal patterns with optional spaces after annotations
    dismissal_patterns = [
        # Caught by wicket-keeper (e.g., "c (wk)Gilchrist b Warne" or "c (wk) Gilchrist b Warne")
        (re.compile(r'c\s+\(wk\)\s*([\w\s]+)\s+b\s+([\w\s]+)', re.IGNORECASE), 'caught wk'),
        
        # Caught by fielder (e.g., "c Dravid b Nehra")
        (re.compile(r'c\s+([\w\s]+)\s+b\s+([\w\s]+)', re.IGNORECASE), 'caught-fielder'),
        
        # Stumped by wicket-keeper (e.g., "st (wk)Gilchrist b Symonds" or "st (wk) Gilchrist b Symonds")
        (re.compile(r'st\s+\(wk\)\s*([\w\s]+)\s+b\s+([\w\s]+)', re.IGNORECASE), 'stump'),
        
        # Stumped (e.g., "st Dravid b Nehra")
        (re.compile(r'st\s+([\w\s]+)\s+b\s+([\w\s]+)', re.IGNORECASE), 'stump'),
        
        # LBW (e.g., "lbw b Nehra")
        (re.compile(r'lbw\s+b\s+([\w\s]+)', re.IGNORECASE), 'lbw'),
        
        # Caught and bowled (e.g., "c & b Nehra")
        (re.compile(r'c\s+&\s+b\s+([\w\s]+)', re.IGNORECASE), 'caught and bowled'),
        
        # Hit wicket (e.g., "hit wicket b Nehra")
        (re.compile(r'hit wicket\s+b\s+([\w\s]+)', re.IGNORECASE), 'hit wicket'),
        
        # Bowled (e.g., "b Nehra")
        (re.compile(r'^b\s+([\w\s]+)$', re.IGNORECASE), 'bowled'),
        
        # Run Out with fielder's name and bowler's name (if present)
        (re.compile(r'run out\s+\(([\w\s]+)\)\s+b\s+([\w\s]+)', re.IGNORECASE), 'run out'),
        
        # Run Out with fielder's name
        (re.compile(r'run out\s+\(([\w\s]+)\)', re.IGNORECASE), 'run out'),
        
        # Run Out without fielder's name
        (re.compile(r'run out', re.IGNORECASE), 'run out'),
        
        # Retired Hurt
        (re.compile(r'retired hurt', re.IGNORECASE), 'retired hurt'),
        
        # Retired Not Out
        (re.compile(r'retired not out', re.IGNORECASE), 'retired not out'),
        
        # Obstructing the Field
        (re.compile(r'obstructing the field', re.IGNORECASE), 'obstructing the field'),
        
        # Timed Out
        (re.compile(r'timed out', re.IGNORECASE), 'timed out'),
        
        # Handled the Ball
        (re.compile(r'handled the ball', re.IGNORECASE), 'handled the ball'),
        
        # Hit the Ball Twice
        (re.compile(r'hit the ball twice', re.IGNORECASE), 'hit the ball twice')
    ]

    for pattern, method in dismissal_patterns:
        match = pattern.search(dismissal_text)
        if match:
            if method in ['caught wk', 'caught-fielder']:
                if method == 'caught wk':
                    fielder = match.group(1).strip()
                    bowler = match.group(2).strip()
                else:
                    fielder = match.group(1).strip()
                    bowler = match.group(2).strip()
                # Check if fielder is a wicket-keeper
                if clean_name(fielder) in wicket_keepers:
                    dismissal_method = 'caught wk'
                else:
                    dismissal_method = 'caught-fielder'
                bowler_name = bowler
            elif method == 'caught and bowled':
                bowler = match.group(1).strip()
                dismissal_method = 'caught and bowled'
                bowler_name = bowler
            elif method == 'stump':
                fielder = match.group(1).strip()
                bowler = match.group(2).strip()
                dismissal_method = 'stump'
                bowler_name = bowler
            elif method in ['lbw', 'bowled', 'hit wicket']:
                bowler = match.group(1).strip()
                dismissal_method = method
                bowler_name = bowler
            elif method == 'run out':
                if len(match.groups()) == 2:
                    fielder = match.group(1).strip()
                    bowler = match.group(2).strip()
                    dismissal_method = 'run out'
                    bowler_name = bowler  # This may not always be present
                elif len(match.groups()) == 1:
                    fielder = match.group(1).strip()
                    dismissal_method = 'run out'
                    bowler_name = 'Unknown'  # Bowler not involved
                else:
                    dismissal_method = 'run out'
                    bowler_name = 'Unknown'
            elif method in ['retired hurt', 'retired not out', 'obstructing the field', 'timed out', 'handled the ball', 'hit the ball twice']:
                dismissal_method = method
                bowler_name = 'Unknown'
            else:
                bowler_name = 'Unknown'

            # Resolve full bowler name using bowler_mapping
            bowler_name_full = resolve_bowler_name(bowler_name, bowler_mapping)

            # Find bowler type
            bowler_type = find_bowler_type(bowler_name_full, innings)

            # Debugging logs for tracing
            logging.debug(f"Dismissal Text: '{dismissal_text}'")
            logging.debug(f"Matched Method: '{method}'")
            logging.debug(f"Fielder: '{fielder}', Bowler: '{bowler_name_full}', Bowler Type: '{bowler_type}'")

            break  # Exit after first matching pattern

    # Handle cases where dismissal method is not matched
    if dismissal_method == 'other':
        bowler_type = 'Unknown'
        bowler_name_full = 'Unknown'
        logging.debug(f"Unmatched dismissal pattern for dismissal_text: '{dismissal_text}'")

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
            logging.error(f"JSONDecodeError: Failed to parse {file_path}")
            return None

    player_stats = {
        'batting': {},
        'bowling': {
            'overs': 0.0,
            'maidens': 0,
            'runs_conceded': 0,
            'wickets': 0,
            'economy_total': 0.0,
            'dots': 0,
            'fours': 0,
            'sixes': 0,
            'wides': 0,
            'no_balls': 0
        },
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

    player_name_clean = clean_name(player_name)
    team_name_lower = team_name.lower()

    playing_xi = match_data.get('playing_xi', {})
    
    # Extract wicket-keepers from both teams
    wicket_keepers = set()
    for team, players in playing_xi.items():
        for name in players:
            if '(wk)' in name:
                cleaned_wk = clean_name(name.replace('(wk)', ''))
                wicket_keepers.add(cleaned_wk)

    # Check if player participated in the match
    player_team_xi = playing_xi.get(team_name, [])
    player_team_xi = [clean_name(name) for name in player_team_xi]

    if player_name_clean not in player_team_xi:
        return None
    else:
        player_stats['played'] = True

    # Build bowler_mapping: variant name to bowler's full name
    bowler_mapping = defaultdict(list)
    innings = match_data.get('innings', [])
    for inning in innings:
        for bowler in inning.get('bowling', []):
            full_bowler_name = bowler.get('bowler', '').lower()
            variants = generate_name_variants(bowler.get('bowler', ''))
            for variant in variants:
                bowler_mapping[variant].append(bowler.get('bowler', ''))

    # Proceed to extract batting, bowling, and fielding data
    for inning in innings:
        team = inning.get('team', '').lower()
        is_player_team = (team == team_name_lower)

        if is_player_team:
            # Batting
            batting = inning.get('batting', [])
            for batsman in batting:
                batsman_name_cleaned = clean_name(batsman.get('batsman', ''))
                if batsman_name_cleaned == player_name_clean:
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
                            batsman_name_cleaned, dismissal, innings, bowler_mapping, wicket_keepers
                        )
                        player_stats['dismissals'].append(dismissal_data)
                    break

            # Fielding (Extract player's fielding data if available)
            fielding = inning.get('fielding', [])
            for fielder in fielding:
                fielder_name_cleaned = clean_name(fielder.get('fielder', ''))
                if fielder_name_cleaned == player_name_clean:
                    player_stats['fielding']['catches'] += safe_int_conversion(fielder.get('catches', 0))
                    player_stats['fielding']['stumpings'] += safe_int_conversion(fielder.get('stumpings', 0))
                    player_stats['fielding']['run_outs'] += safe_int_conversion(fielder.get('run_outs', 0))
                    break

        else:
            # Process bowling data for player's team from opponent's innings
            bowling = inning.get('bowling', [])
            for bowler in bowling:
                bowler_name_cleaned = clean_name(bowler.get('bowler', ''))
                if bowler_name_cleaned == player_name_clean:
                    # Aggregate player's bowling data
                    player_stats['bowling']['overs'] += safe_float_conversion(bowler.get('overs', 0.0))
                    player_stats['bowling']['maidens'] += safe_int_conversion(bowler.get('maidens', 0))
                    player_stats['bowling']['runs_conceded'] += safe_int_conversion(bowler.get('runs_conceded', 0))
                    player_stats['bowling']['wickets'] += safe_int_conversion(bowler.get('wickets', 0))
                    player_stats['bowling']['economy_total'] += safe_float_conversion(bowler.get('economy', 0.0))
                    player_stats['bowling']['dots'] += safe_int_conversion(bowler.get('dots', 0))
                    player_stats['bowling']['fours'] += safe_int_conversion(bowler.get('fours', 0))
                    player_stats['bowling']['sixes'] += safe_int_conversion(bowler.get('sixes', 0))
                    player_stats['bowling']['wides'] += safe_int_conversion(bowler.get('wides', 0))
                    player_stats['bowling']['no_balls'] += safe_int_conversion(bowler.get('no_balls', 0))
                    break

    return player_stats

def aggregate_stats(all_stats):
    """Aggregates batting, bowling, fielding, and dismissal data from all matches. Also calculates the span of the player's career based on match dates."""
    
    # Define all bowler types
    bowler_types = [
        "Right arm Pace",
        "Right arm Spin",
        "Left arm Pace",
        "Left arm Spin",
        "Unknown"
    ]
    
    # Define all dismissal methods
    DISMISSAL_METHODS = [
        "bowled",
        "caught-fielder",
        "caught wk",
        "stump",
        "lbw",
        "hit wicket",
        "caught and bowled"
    ]
    
    # Initialize the aggregated data structure
    aggregated = {
        'batting': {
            'runs': 0,
            'balls': 0,
            'fours': 0,
            'sixes': 0,
            'strike_rate_total': 0.0,
            'dismissals': 0,
            'not_outs': 0,
            'bat_avg': 0.0,
            'strike_rate': 0.0
        },
        'bowling': {
            'overs': 0.0,
            'maidens': 0,
            'runs_conceded': 0,
            'wickets': 0,
            'economy_total': 0.0,
            'dots': 0,
            'fours': 0,
            'sixes': 0,
            'wides': 0,
            'no_balls': 0,
            'bowl_avg': 0.0,
            'economy': 0.0
        },
        'fielding': {
            'catches': 0,
            'stumpings': 0,
            'run_outs': 0
        },
        # Initialize 'dismissals' with 'Dis' and all dismissal methods set to 0
        "dismissals": {bowler_type: {"Dis": 0, **{method: 0 for method in DISMISSAL_METHODS}} for bowler_type in bowler_types},
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
            match_date = match.get("match_date")
            if match_date: 
                aggregated["match_dates"].append(match_date)

        # Aggregate Batting 
        batting = match.get("batting", {}) 
        if batting: 
            aggregated["batting_innings"] += 1  
            aggregated["batting"]["runs"] += batting.get("runs", 0)
            aggregated["batting"]["balls"] += batting.get("balls", 0)
            aggregated["batting"]["fours"] += batting.get("fours", 0)
            aggregated["batting"]["sixes"] += batting.get("sixes", 0)
            aggregated["batting"]["strike_rate_total"] += batting.get("strike_rate", 0.0)
            if batting.get("dismissal", "").lower() != "not out":
                aggregated["batting"]["dismissals"] += 1 
            else: 
                aggregated["batting"]["not_outs"] += 1

        # Aggregate Bowling 
        bowling = match.get("bowling", {}) 
        if bowling: 
            aggregated["bowling_innings"] += 1  
            aggregated["bowling"]["overs"] += bowling.get("overs", 0.0)
            aggregated["bowling"]["maidens"] += bowling.get("maidens", 0)
            aggregated["bowling"]["runs_conceded"] += bowling.get("runs_conceded", 0)
            aggregated["bowling"]["wickets"] += bowling.get("wickets", 0)
            aggregated["bowling"]["economy_total"] += bowling.get("economy", 0.0)
            aggregated["bowling"]["dots"] += bowling.get("dots", 0)
            aggregated["bowling"]["fours"] += bowling.get("fours", 0)
            aggregated["bowling"]["sixes"] += bowling.get("sixes", 0)
            aggregated["bowling"]["wides"] += bowling.get("wides", 0)
            aggregated["bowling"]["no_balls"] += bowling.get("no_balls", 0)

        # Aggregate Fielding 
        fielding = match.get("fielding", {})
        if fielding: 
            aggregated["fielding_innings"] += 1  
            aggregated["fielding"]["catches"] += fielding.get("catches", 0)
            aggregated["fielding"]["stumpings"] += fielding.get("stumpings", 0)
            aggregated["fielding"]["run_outs"] += fielding.get("run_outs", 0)

        # Aggregate Dismissals 
        for dismissal in match.get("dismissals", []): 
            bowler_type = dismissal.get("bowler_type", "Unknown")  # Keep case consistent
            dismissal_method = dismissal.get("dismissal_method", "other").lower()
            # Standardize dismissal method names
            if dismissal_method == "caught-fielder":
                dismissal_method_key = "caught-fielder"
            elif dismissal_method == "caught wk":
                dismissal_method_key = "caught wk"
            elif dismissal_method == "bowled":
                dismissal_method_key = "bowled"
            elif dismissal_method == "stump":
                dismissal_method_key = "stump"
            elif dismissal_method == "lbw":
                dismissal_method_key = "lbw"
            elif dismissal_method == "hit wicket":
                dismissal_method_key = "hit wicket"
            elif dismissal_method == "caught and bowled":
                dismissal_method_key = "caught and bowled"
            else:
                dismissal_method_key = "other"
            
            # Ensure bowler_type is recognized
            if bowler_type not in aggregated['dismissals']:
                bowler_type = "Unknown"

            # Increment specific dismissal method count
            if dismissal_method_key in DISMISSAL_METHODS:
                aggregated['dismissals'][bowler_type][dismissal_method_key] += 1
            else:
                # If dismissal method is 'other' or unrecognized, you can choose to handle it accordingly
                pass

            # Increment total dismissals
            aggregated['dismissals'][bowler_type]["Dis"] += 1

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
        "bowling": {
            "Overs": round(aggregated['bowling']['overs'], 2),
            "Maidens": aggregated['bowling']['maidens'],
            "Runs Conceded": aggregated['bowling']['runs_conceded'],
            "Wickets": aggregated['bowling']['wickets'],
            "Economy Rate": aggregated['bowling']['economy'],
            "Dots": aggregated['bowling']['dots'],
            "Fours": aggregated['bowling']['fours'],
            "Sixes": aggregated['bowling']['sixes'],
            "Wides": aggregated['bowling']['wides'],
            "No Balls": aggregated['bowling']['no_balls'],
            "Bowling Average": aggregated['bowling']['bowl_avg']
        },
        "fielding": aggregated['fielding'],
        "dismissal_data": [],  
        "span": span,  
        "matches_played": aggregated['matches_played'],  
        "batting_innings": aggregated['batting_innings'],  
        "bowling_innings": aggregated['bowling_innings'],  
        "fielding_innings": aggregated['fielding_innings']   
    }  

    # Populate dismissal data 
    for bowler_type in bowler_types:
        data = aggregated['dismissals'][bowler_type]
        dismissal_entry = {  
            "Bowler Type": bowler_type,   
            "Span": span,   
            "Dis": data['Dis'],
            "bowled": data['bowled'],
            "caught-fielder": data['caught-fielder'],
            "caught wk": data['caught wk'],
            "stump": data['stump'],
            "lbw": data['lbw'],
            "hit wicket": data['hit wicket'],
            "caught and bowled": data['caught and bowled']
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
    try:
        json_files = [os.path.join(scorecards_dir, filename) for filename in os.listdir(scorecards_dir) if filename.endswith('.json')]
    except FileNotFoundError:
        logging.error(f"The directory '{scorecards_dir}' does not exist.")
        return {}
    
    if not json_files:
        logging.warning(f"No JSON files found in the directory '{scorecards_dir}'.")
        return {}
    
    # Prepare arguments for multiprocessing
    args = [(file_path, player_name, team_name) for file_path in json_files]
    
    # Use multiprocessing Pool to process files in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_file, args)
        
        # Filter out None results
        all_stats = [result for result in results if result is not None]
    
    # Check results
    if not all_stats:
        logging.warning(f"No match data found for player '{player_name}' in team '{team_name}'.")
        return {}
    
    # Aggregate all stats
    aggregated_stats = aggregate_stats(all_stats)
    
    # Log the number of matches processed
    number_of_matches = len(all_stats)
    logging.info(f"Number of matches processed: {number_of_matches}")
    
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
            "Overs": aggregated_stats['bowling']['Overs'],
            "Maidens": aggregated_stats['bowling']['Maidens'],
            "Runs Conceded": aggregated_stats['bowling']['Runs Conceded'],
            "Wickets": aggregated_stats['bowling']['Wickets'],
            "Economy Rate": aggregated_stats['bowling'].get("Economy Rate"),
            "Bowling Average": aggregated_stats['bowling'].get("Bowling Average"),
            "Dots": aggregated_stats['bowling'].get("Dots"),
            "Fours": aggregated_stats['bowling'].get("Fours"),
            "Sixes": aggregated_stats['bowling'].get("Sixes"),
            "Wides": aggregated_stats['bowling'].get("Wides"),
            "No Balls": aggregated_stats['bowling'].get("No Balls")
        },
        "fielding": {
            "Catches": aggregated_stats['fielding'].get("catches"),
            "Stumpings": aggregated_stats['fielding'].get("stumpings"),
            "Run Outs": aggregated_stats['fielding'].get("run_outs")
        },
        "dismissals": aggregated_stats['dismissal_data']
    }

    # Save the JSON to file
    safe_player_name = player_name.lower().replace(' ', '_')
    output_file = f"{safe_player_name}_stats.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(player_stats_json, f, indent=4, default=str)
    
    # Log the output JSON produced
    logging.info(f"Output JSON produced: {output_file}")

    return player_stats_json

def save_stats_to_json(player_stats, output_filename):
    """
    Saves the player statistics to a JSON file.
    """
    if not player_stats:
        logging.warning("No player statistics to save.")
        return

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(player_stats, f, indent=4)

def main():
    # Hard-coded Player and Team Name
    player_name = "Azhar Mahmood"  # Change this to the desired player
    team_name = "Pakistan"          # Change this to the desired team

    # Directory containing match scorecard JSON files
    scorecards_dir = "match_scorecards_json_sample"  # Ensure this path is correct

    # Check if the directory exists
    if not os.path.isdir(scorecards_dir):
        logging.error(f"The directory '{scorecards_dir}' does not exist. Please provide a valid path.")
        return

    # Generate Player Stats
    player_stats = generate_player_stats(player_name, team_name, scorecards_dir)

    if not player_stats:
        logging.warning(f"No statistics found for player '{player_name}' in team '{team_name}'.")
        return

    # Define Output Filename
    safe_player_name = player_name.lower().replace(' ', '_')
    output_filename = f"{safe_player_name}_stats.json"

    # Save to JSON
    save_stats_to_json(player_stats, output_filename)

    # Optionally, log a success message
    logging.info(f"Player statistics successfully saved to '{output_filename}'.")

if __name__ == "__main__":
    main()
