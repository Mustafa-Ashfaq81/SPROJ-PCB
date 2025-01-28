import os
import json
import re
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
import logging

# Configure logging to capture INFO and higher-level logs by default
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
    """
    Removes annotations like (c), (wk), sub(...), etc., from names 
    and strips/trims the result, returning lowercase by default.
    """
    # Remove anything in parentheses, e.g. (wk), (c), sub(...)
    cleaned = re.sub(r'\(.*?\)', '', name)
    return cleaned.strip().lower()

def generate_case_variants(name_str: str) -> set:
    """
    Given a name like 'Kohli', produce variants in different letter cases:
    e.g. 'Kohli', 'kohli', 'KOHLI'.
    """
    # Trim and skip empty
    name_str = name_str.strip()
    if not name_str:
        return set()

    return {
        name_str,
        name_str.lower(),
        name_str.upper(),
        name_str.capitalize(),
        name_str.title()
    }

def generate_name_variants(full_name: str) -> set:
    """
    Generates a set of variants from the full_name for partial matching.
    
    For 'Virat Kohli', we produce:
       - "virat kohli", "viratkohli", "v kohli", "vkohli", "kohli"
       - Also case variants for each of the above.
    """
    full_name = full_name.strip()
    if not full_name:
        return set()

    name_parts = full_name.split()
    base_variants = set()

    # 1) The full name (lowercased)
    base_variants.add(' '.join(name_parts).lower())
    # 2) Full name joined
    base_variants.add(''.join(name_parts).lower())

    # 3) If multiple parts, add "v kohli", "vkohli", and "kohli"
    if len(name_parts) >= 2:
        first_initials = ''.join([part[0] for part in name_parts[:-1]]).lower()
        last_name = name_parts[-1].lower()
        base_variants.add(f"{first_initials} {last_name}")
        base_variants.add(f"{first_initials}{last_name}")
        base_variants.add(last_name)
    else:
        # Single name, e.g. "Sachin"
        base_variants.add(name_parts[0].lower())

    # Now, for each of these base variants (all are in lowercase),
    # generate additional case variants:
    final_variants = set()
    for variant in base_variants:
        final_variants.update(generate_case_variants(variant))

    return final_variants

def resolve_bowler_name(bowler_name, bowler_mapping):
    """
    Resolves the full bowler name using bowler_mapping, ignoring case.
    'bowler_mapping' is a dict where keys are partial-lowercase variants,
    e.g. "kumar" -> ["Bhuvneshwar Kumar"]
    """
    bowler_variants = generate_name_variants(bowler_name)
    possible_full_names = []
    for variant in bowler_variants:
        lookup_key = variant.lower()
        if lookup_key in bowler_mapping:
            possible_full_names.extend(bowler_mapping[lookup_key])

    if possible_full_names:
        # Just return the first match
        return possible_full_names[0]
    else:
        # Use as-is if no mapping found
        return bowler_name

def find_bowler_type(bowler_name_full, innings):
    """
    Finds the bowler type from the innings data by matching the bowler name 
    (ignoring case).
    """
    bowler_name_lower = bowler_name_full.lower()
    for inning in innings:
        for bowler_entry in inning.get('bowling', []):
            if bowler_entry.get('bowler', '').lower() == bowler_name_lower:
                return bowler_entry.get('b_type', 'Unknown')
    return 'Unknown'

def parse_fielding_contribution(dismissal_text: str, player_name_clean: str) -> dict:
    """
    Given a single dismissal text (e.g. "c Kohli b Kumar", "run out (Raina/ (wk)Dhoni)", 
    "c & b Jadeja"), determine whether 'player_name_clean' contributed as a fielder.
    
    Return increments in {"catches", "stumpings", "run_outs"}.
    """
    fielding_stats = {"catches": 0, "stumpings": 0, "run_outs": 0}
    dismissal_text_lower = dismissal_text.lower().strip()

    player_variants = generate_name_variants(player_name_clean)

    # 1) caught and bowled
    c_and_b_pattern = re.compile(r'c\s+&\s+b\s+([\w\s]+)', re.IGNORECASE)
    match_cab = c_and_b_pattern.search(dismissal_text_lower)
    if match_cab:
        bowler_fielder_raw = clean_name(match_cab.group(1))
        bowler_fielder_variants = generate_name_variants(bowler_fielder_raw)
        if player_variants & bowler_fielder_variants:
            fielding_stats["catches"] += 1
        return fielding_stats

    # 2) standard "caught": c FIELDER b BOWLER
    caught_pattern = re.compile(r'^c\s+(.*?)\s+b\s+([\w\s]+)$', re.IGNORECASE)
    match_caught = caught_pattern.search(dismissal_text_lower)
    if match_caught:
        fielder_raw = match_caught.group(1)
        fielder_raw = re.sub(r'\bsub\b', '', fielder_raw, flags=re.IGNORECASE)
        fielder_cleaned = clean_name(fielder_raw)
        fielder_variants = generate_name_variants(fielder_cleaned)
        if player_variants & fielder_variants:
            fielding_stats["catches"] += 1

    # 3) stumped: st FIELDER b BOWLER
    stumped_pattern = re.compile(r'st\s+(.*?)\s+b\s+([\w\s]+)', re.IGNORECASE)
    match_stumped = stumped_pattern.search(dismissal_text_lower)
    if match_stumped:
        fielder_raw = match_stumped.group(1)
        fielder_raw = re.sub(r'\bsub\b', '', fielder_raw, flags=re.IGNORECASE)
        fielder_cleaned = clean_name(fielder_raw)
        fielder_variants = generate_name_variants(fielder_cleaned)
        if player_variants & fielder_variants:
            fielding_stats["stumpings"] += 1

    # 4) run out
    run_out_pattern = re.compile(r'run out\s*\(([\w\s/]+)\)', re.IGNORECASE)
    match_runout = run_out_pattern.search(dismissal_text_lower)
    if match_runout:
        fielder_block = match_runout.group(1)
        potential_fielders = [clean_name(x) for x in fielder_block.split('/')]
        for f_name in potential_fielders:
            if player_variants & generate_name_variants(f_name):
                fielding_stats["run_outs"] += 1
                break

    return fielding_stats

def extract_dismissal_data(batsman, dismissal_text, innings, bowler_mapping, wicket_keepers):
    """
    Extract how the *batsman* got out (bowled, lbw, etc.), and find the bowler name + bowler type.
    """
    bowler_type = 'Unknown'
    dismissal_method = 'other'
    bowler_name_full = 'Unknown'
    fielder = 'Unknown'

    dismissal_patterns = [
        # Caught by wicket-keeper
        (re.compile(r'c\s+\(wk\)\s*([\w\s]+)\s+b\s+([\w\s]+)', re.IGNORECASE), 'caught wk'),
        # Caught by fielder
        (re.compile(r'c\s+([\w\s]+)\s+b\s+([\w\s]+)', re.IGNORECASE), 'caught-fielder'),
        # Stumped by wicket-keeper
        (re.compile(r'st\s+\(wk\)\s*([\w\s]+)\s+b\s+([\w\s]+)', re.IGNORECASE), 'stump'),
        # Stumped
        (re.compile(r'st\s+([\w\s]+)\s+b\s+([\w\s]+)', re.IGNORECASE), 'stump'),
        # LBW
        (re.compile(r'lbw\s+b\s+([\w\s]+)', re.IGNORECASE), 'lbw'),
        # Caught and bowled
        (re.compile(r'c\s+&\s+b\s+([\w\s]+)', re.IGNORECASE), 'caught and bowled'),
        # Hit wicket
        (re.compile(r'hit wicket\s+b\s+([\w\s]+)', re.IGNORECASE), 'hit wicket'),
        # Bowled
        (re.compile(r'^b\s+([\w\s]+)$', re.IGNORECASE), 'bowled'),
        # Run Out
        (re.compile(r'run out\s+\(([\w\s]+)\)\s+b\s+([\w\s]+)', re.IGNORECASE), 'run out'),
        (re.compile(r'run out\s+\(([\w\s/]+)\)', re.IGNORECASE), 'run out'),
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

    dismissal_text_lower = dismissal_text.lower().strip()

    for pattern, method in dismissal_patterns:
        match = pattern.search(dismissal_text_lower)
        if match:
            if method == 'caught wk':
                fielder = match.group(1).strip()
                bowler = match.group(2).strip()
                dismissal_method = 'caught wk'
                bowler_name = bowler

            elif method == 'caught-fielder':
                fielder = match.group(1).strip()
                bowler = match.group(2).strip()
                # Check if fielder is a wicketkeeper
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
                # match.groups() might have 1 or 2
                if len(match.groups()) == 2:
                    fielder = match.group(1).strip()
                    bowler = match.group(2).strip()
                    dismissal_method = 'run out'
                    bowler_name = bowler
                elif len(match.groups()) == 1:
                    fielder = match.group(1).strip()
                    dismissal_method = 'run out'
                    bowler_name = 'Unknown'
                else:
                    dismissal_method = 'run out'
                    bowler_name = 'Unknown'

            elif method in [
                'retired hurt', 'retired not out',
                'obstructing the field', 'timed out',
                'handled the ball', 'hit the ball twice'
            ]:
                dismissal_method = method
                bowler_name = 'Unknown'

            else:
                bowler_name = 'Unknown'

            bowler_name_full = resolve_bowler_name(bowler_name, bowler_mapping)
            bowler_type = find_bowler_type(bowler_name_full, innings)
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
            'no_balls': 0,
            'boundary_runs_this_innings': 0,
            'match_wickets': 0,
            'match_runs': 0
        },
        'fielding': {
            'catches': 0,
            'stumpings': 0,
            'run_outs': 0
        },
        'dismissals': [],
        'match_date': None,
        'played': False,

        # ADDED: Per-match breakdown of wickets vs. batting type
        'bowling_vs_batting_type': {
            'Right Handed': {
                'wickets': 0,
                'bowled': 0,
                'caught-fielder': 0,
                'caught wk': 0,
                'stump': 0,
                'lbw': 0,
                'hit wicket': 0,
                'caught and bowled': 0,
                'other': 0
            },
            'Left Handed': {
                'wickets': 0,
                'bowled': 0,
                'caught-fielder': 0,
                'caught wk': 0,
                'stump': 0,
                'lbw': 0,
                'hit wicket': 0,
                'caught and bowled': 0,
                'other': 0
            },
            'Unknown': {
                'wickets': 0,
                'bowled': 0,
                'caught-fielder': 0,
                'caught wk': 0,
                'stump': 0,
                'lbw': 0,
                'hit wicket': 0,
                'caught and bowled': 0,
                'other': 0
            }
        }
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

    # Check if player participated
    player_team_xi = playing_xi.get(team_name, [])
    player_team_xi_cleaned = [clean_name(x) for x in player_team_xi]
    if player_name_clean not in player_team_xi_cleaned:
        return None
    else:
        player_stats['played'] = True

    # Build bowler_mapping for partial name resolution
    bowler_mapping = defaultdict(list)
    innings = match_data.get('innings', [])
    for inning in innings:
        for bowler in inning.get('bowling', []):
            bowler_original = bowler.get('bowler', '')
            for variant in generate_name_variants(bowler_original):
                bowler_mapping[variant.lower()].append(bowler_original)

    # ----------------------------
    # MAIN LOGIC
    # ----------------------------
    for inning in innings:
        # 1) BATTING (if this inning belongs to the player's team)
        team = inning.get('team', '').lower()
        is_player_team = (team == team_name_lower)

        if is_player_team:
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
                    # Extract how *this player* got out
                    dismissal_text = batsman.get('dismissal', '')
                    if dismissal_text.lower() != 'not out':
                        dismissal_data = extract_dismissal_data(
                            batsman_name_cleaned,
                            dismissal_text,
                            innings,
                            bowler_mapping,
                            wicket_keepers
                        )
                        player_stats['dismissals'].append(dismissal_data)

        # 2) BOWLING (if the player's team is bowling)
        else:
            bowling_list = inning.get('bowling', [])
            for bowler in bowling_list:
                bowler_name_cleaned = clean_name(bowler.get('bowler', ''))
                if bowler_name_cleaned == player_name_clean:
                    overs_val = safe_float_conversion(bowler.get('overs', 0.0))
                    wickets_val = safe_int_conversion(bowler.get('wickets', 0))
                    runs_val = safe_int_conversion(bowler.get('runs_conceded', 0))
                    fours_val = safe_int_conversion(bowler.get('fours', 0))
                    sixes_val = safe_int_conversion(bowler.get('sixes', 0))
                    
                    player_stats['bowling']['overs'] += overs_val
                    player_stats['bowling']['maidens'] += safe_int_conversion(bowler.get('maidens', 0))
                    player_stats['bowling']['runs_conceded'] += runs_val
                    player_stats['bowling']['wickets'] += wickets_val
                    player_stats['bowling']['economy_total'] += safe_float_conversion(bowler.get('economy', 0.0))
                    player_stats['bowling']['dots'] += safe_int_conversion(bowler.get('dots', 0))
                    player_stats['bowling']['fours'] += fours_val
                    player_stats['bowling']['sixes'] += sixes_val
                    player_stats['bowling']['wides'] += safe_int_conversion(bowler.get('wides', 0))
                    player_stats['bowling']['no_balls'] += safe_int_conversion(bowler.get('no_balls', 0))

                    player_stats['bowling']['boundary_runs_this_innings'] += (fours_val * 4) + (sixes_val * 6)
                    player_stats['bowling']['match_wickets'] = wickets_val
                    player_stats['bowling']['match_runs'] = runs_val

        # 3) FIELDING + BOWLING DISMISSALS VS BATTING TYPE
        #    For fielding: parse all batsmen in this inning
        #    For bowling: check if our player is the credited bowler.
        batting_for_fielding = inning.get('batting', [])
        for batsman_data in batting_for_fielding:
            dismiss_text = batsman_data.get('dismissal', '').strip()
            if not dismiss_text:
                continue  # not out => no fielder/bowler credited

            # Fielding contribution
            fielding_contrib = parse_fielding_contribution(dismiss_text, player_name_clean)
            player_stats['fielding']['catches'] += fielding_contrib["catches"]
            player_stats['fielding']['stumpings'] += fielding_contrib["stumpings"]
            player_stats['fielding']['run_outs'] += fielding_contrib["run_outs"]

            # ADDED: Check if this dismissal credits the player as bowler
            dis_data = extract_dismissal_data(
                clean_name(batsman_data.get('batsman', '')),
                dismiss_text,
                innings,
                bowler_mapping,
                wicket_keepers
            )
            credited_bowler = clean_name(dis_data.get('bowler_name', ''))
            method = dis_data.get('dismissal_method', 'other').lower()

            if credited_bowler == player_name_clean:
                # The bowler is indeed "our player"
                batting_type_str = batsman_data.get('batting_type', '').strip().lower()
                if 'left' in batting_type_str:
                    style_key = 'Left Handed'
                elif 'right' in batting_type_str:
                    style_key = 'Right Handed'
                else:
                    style_key = 'Unknown'

                # Increment "wickets"
                player_stats['bowling_vs_batting_type'][style_key]['wickets'] += 1

                # Dismissal method check
                known_methods = {
                    'bowled', 'caught-fielder', 'caught wk', 'stump',
                    'lbw', 'hit wicket', 'caught and bowled'
                }
                if method in known_methods:
                    player_stats['bowling_vs_batting_type'][style_key][method] += 1
                else:
                    # e.g. "run out", "retired hurt", etc. => not a typical bowler wicket
                    # or an unmatched pattern => increment "other" 
                    # (only if the scoreboard incorrectly credits the bowler with that method)
                    player_stats['bowling_vs_batting_type'][style_key]['other'] += 1

    return player_stats

def correct_overs(overs):
    """Convert decimal overs into the standard X.Y overs format where Y < 6."""
    whole_overs = int(overs)
    balls = (overs - whole_overs) * 10
    if balls >= 6:
        whole_overs += 1
        balls -= 6
    return whole_overs + balls / 10

def aggregate_stats(all_stats):
    """Aggregates batting, bowling, fielding, and dismissal data from all matches."""
    
    bowler_types = [
        "Right arm Pace",
        "Right arm Spin",
        "Left arm Pace",
        "Left arm Spin",
        "Unknown"
    ]
    
    DISMISSAL_METHODS = [
        "bowled",
        "caught-fielder",
        "caught wk",
        "stump",
        "lbw",
        "hit wicket",
        "caught and bowled"
    ]
    
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
            'economy': 0.0,
            'total_boundary_runs': 0,
            'best_bowling_wickets': 0,
            'best_bowling_runs': 999999,
            'three_wicket_hauls': 0,
            'four_wicket_hauls': 0,
            'five_wicket_hauls': 0
        },
        'fielding': {
            'catches': 0,
            'stumpings': 0,
            'run_outs': 0
        },
        "dismissals": {
            bowler_type: {"Dis": 0, **{method: 0 for method in DISMISSAL_METHODS}}
            for bowler_type in bowler_types
        },
        "match_dates": [],
        "matches_played": 0,
        "batting_innings": 0,
        "bowling_innings": 0,
        "fielding_innings": 0
    }

    # ADDED: aggregator for bowling vs batting type
    style_keys = ["Right Handed", "Left Handed", "Unknown"]
    method_keys = DISMISSAL_METHODS + ["other"]  # includes "other"
    bowling_vs_batting_type_agg = {
        style: {
            "wickets": 0,
            **{m: 0 for m in method_keys}
        }
        for style in style_keys
    }

    for match in all_stats:
        if match.get("played"):
            aggregated["matches_played"] += 1
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

            # boundary runs
            aggregated["bowling"]['total_boundary_runs'] += bowling.get('boundary_runs_this_innings', 0)

            # best figures
            match_wickets = bowling.get('match_wickets', 0)
            match_runs = bowling.get('match_runs', 0)
            if match_wickets > 0:
                if ((match_wickets > aggregated["bowling"]['best_bowling_wickets']) or
                    (match_wickets == aggregated["bowling"]['best_bowling_wickets'] and match_runs < aggregated["bowling"]['best_bowling_runs'])):
                    aggregated["bowling"]['best_bowling_wickets'] = match_wickets
                    aggregated["bowling"]['best_bowling_runs'] = match_runs

            if match_wickets == 3:
                aggregated["bowling"]['three_wicket_hauls'] += 1
            elif match_wickets == 4:
                aggregated["bowling"]['four_wicket_hauls'] += 1
            elif match_wickets >= 5:
                aggregated["bowling"]['five_wicket_hauls'] += 1

        # Aggregate Fielding
        fielding = match.get("fielding", {})
        if fielding:
            aggregated["fielding_innings"] += 1
            aggregated["fielding"]["catches"] += fielding.get("catches", 0)
            aggregated["fielding"]["stumpings"] += fielding.get("stumpings", 0)
            aggregated["fielding"]["run_outs"] += fielding.get("run_outs", 0)

        # Aggregate "who got me out?" from 'dismissals'
        for dismissal in match.get("dismissals", []):
            bowler_type = dismissal.get("bowler_type", "Unknown")
            dismissal_method = dismissal.get("dismissal_method", "other").lower()
            if dismissal_method == "caught-fielder":
                key = "caught-fielder"
            elif dismissal_method == "caught wk":
                key = "caught wk"
            elif dismissal_method == "bowled":
                key = "bowled"
            elif dismissal_method == "stump":
                key = "stump"
            elif dismissal_method == "lbw":
                key = "lbw"
            elif dismissal_method == "hit wicket":
                key = "hit wicket"
            elif dismissal_method == "caught and bowled":
                key = "caught and bowled"
            else:
                key = "other"
            
            if bowler_type not in aggregated['dismissals']:
                bowler_type = "Unknown"

            if key in DISMISSAL_METHODS:
                aggregated['dismissals'][bowler_type][key] += 1
            aggregated['dismissals'][bowler_type]["Dis"] += 1

        # ADDED: Now combine the match-level "bowling_vs_batting_type"
        match_bvbt = match.get("bowling_vs_batting_type", {})
        for style in style_keys:
            style_dict = match_bvbt.get(style, {})
            # Wickets
            bowling_vs_batting_type_agg[style]["wickets"] += style_dict.get("wickets", 0)
            # Each method
            for m_key in method_keys:
                bowling_vs_batting_type_agg[style][m_key] += style_dict.get(m_key, 0)

    # Final batting average & SR
    if aggregated['batting']['dismissals'] > 0:
        aggregated['batting']['bat_avg'] = round(aggregated['batting']['runs'] / aggregated['batting']['dismissals'], 2)
    else:
        aggregated['batting']['bat_avg'] = None

    if aggregated['batting']['balls'] > 0:
        aggregated['batting']['strike_rate'] = round((aggregated['batting']['runs'] / aggregated['batting']['balls']) * 100, 2)
    else:
        aggregated['batting']['strike_rate'] = None

    # Bowling average & economy
    if aggregated['bowling']['wickets'] > 0:
        aggregated['bowling']['bowl_avg'] = round(aggregated['bowling']['runs_conceded'] / aggregated['bowling']['wickets'], 2)
    else:
        aggregated['bowling']['bowl_avg'] = None

    if aggregated['bowling']['overs'] > 0:
        aggregated['bowling']['economy'] = round(aggregated['bowling']['runs_conceded'] / aggregated['bowling']['overs'], 2)
    else:
        aggregated['bowling']['economy'] = None

    corrected_overs = correct_overs(aggregated['bowling']['overs'])
    whole_overs = int(corrected_overs)
    partial_balls = int(round((corrected_overs - whole_overs) * 10))
    total_balls_bowled = whole_overs * 6 + partial_balls

    # Bowling Strike Rate
    if aggregated['bowling']['wickets'] > 0 and total_balls_bowled > 0:
        bowling_strike_rate = round(total_balls_bowled / aggregated['bowling']['wickets'], 2)
    else:
        bowling_strike_rate = None

    # Dot-Ball Percentage
    if total_balls_bowled > 0:
        dot_ball_percentage = round((aggregated['bowling']['dots'] / total_balls_bowled) * 100, 2)
    else:
        dot_ball_percentage = None

    # Best Bowling Figures
    if (aggregated['bowling']['best_bowling_wickets'] > 0 and
        aggregated['bowling']['best_bowling_runs'] < 999999):
        best_bowling_figures = f"{aggregated['bowling']['best_bowling_wickets']}/{aggregated['bowling']['best_bowling_runs']}"
    else:
        best_bowling_figures = None

    # Career span
    if aggregated['match_dates']:
        earliest_date = min(aggregated['match_dates'])
        latest_date = max(aggregated['match_dates'])
        span = f"{earliest_date.year} - {latest_date.year}"
    else:
        span = "Unknown"

    final_aggregated = {
        "batting": aggregated['batting'],
        "bowling": {
            "Overs": round(corrected_overs, 2),
            "Maidens": aggregated['bowling']['maidens'],
            "Runs Conceded": aggregated['bowling']['runs_conceded'],
            "Wickets": aggregated['bowling']['wickets'],
            "Economy Rate": aggregated['bowling']['economy'],
            "Bowling Average": aggregated['bowling']['bowl_avg'],
            "Dots": aggregated['bowling']['dots'],
            "Fours": aggregated['bowling']['fours'],
            "Sixes": aggregated['bowling']['sixes'],
            "Wides": aggregated['bowling']['wides'],
            "No Balls": aggregated['bowling']['no_balls'],
            "Boundary Runs Conceded": aggregated['bowling']['total_boundary_runs'],
            "Bowling Strike Rate": bowling_strike_rate,
            "Dot Ball Percentage": dot_ball_percentage,
            "3wHauls": aggregated['bowling']['three_wicket_hauls'],
            "4wHauls": aggregated['bowling']['four_wicket_hauls'],
            "5wHauls": aggregated['bowling']['five_wicket_hauls'],
            "Best Bowling Figures": best_bowling_figures
        },
        "fielding": aggregated['fielding'],
        "dismissal_data": [],
        "span": span,
        "matches_played": aggregated['matches_played'],
        "batting_innings": aggregated['batting_innings'],
        "bowling_innings": aggregated['bowling_innings'],
        "fielding_innings": aggregated['fielding_innings'],

        # ADDED: the newly aggregated data for bowling vs batting type
        "bowling_vs_batting_type": bowling_vs_batting_type_agg
    }

    # Populate dismissal_data
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

def generate_player_stats(player_name, team_name, scorecards_dir='../match_scorecards_json'):
    """
    Generates a comprehensive statistics JSON file for the specified player 
    using the match scorecards found in 'scorecards_dir'.
    """
    all_stats = []
    
    # Prepare list of all JSON files
    try:
        json_files = [os.path.join(scorecards_dir, filename) 
                      for filename in os.listdir(scorecards_dir) if filename.endswith('.json')]
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
            "Overs": aggregated_stats['bowling']["Overs"],
            "Maidens": aggregated_stats['bowling']["Maidens"],
            "Runs Conceded": aggregated_stats['bowling']["Runs Conceded"],
            "Wickets": aggregated_stats['bowling']["Wickets"],
            "Economy Rate": aggregated_stats['bowling'].get("Economy Rate"),
            "Bowling Average": aggregated_stats['bowling'].get("Bowling Average"),
            "Dots": aggregated_stats['bowling'].get("Dots"),
            "Fours": aggregated_stats['bowling'].get("Fours"),
            "Sixes": aggregated_stats['bowling'].get("Sixes"),
            "Wides": aggregated_stats['bowling'].get("Wides"),
            "No Balls": aggregated_stats['bowling'].get("No Balls"),
            "Boundary Runs Conceded": aggregated_stats['bowling'].get("Boundary Runs Conceded"),
            "Bowling Strike Rate": aggregated_stats['bowling'].get("Bowling Strike Rate"),
            "Dot Ball Percentage": aggregated_stats['bowling'].get("Dot Ball Percentage"),
            "3wHauls": aggregated_stats['bowling'].get("3wHauls"),
            "4wHauls": aggregated_stats['bowling'].get("4wHauls"),
            "5wHauls": aggregated_stats['bowling'].get("5wHauls"),
            "Best Bowling Figures": aggregated_stats['bowling'].get("Best Bowling Figures")
        },
        "fielding": {
            "Catches": aggregated_stats['fielding'].get("catches"),
            "Stumpings": aggregated_stats['fielding'].get("stumpings"),
            "Run Outs": aggregated_stats['fielding'].get("run_outs")
        },
        "dismissals": aggregated_stats['dismissal_data'],

        # ADDED: The newly aggregated breakdown
        "bowling_vs_batting_type": aggregated_stats["bowling_vs_batting_type"]
    }

    # Save the JSON to ../aggregated_stats
    safe_player_name = player_name.lower().replace(' ', '_')
    output_dir = "../aggregated_stats"
    output_file = os.path.join(output_dir, f"{safe_player_name}_stats.json")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(player_stats_json, f, indent=4, default=str)
    
    logging.info(f"Output JSON produced: {output_file}")

    return player_stats_json

def save_stats_to_json(player_stats, output_filename):
    """Saves the player statistics to a JSON file."""
    if not player_stats:
        logging.warning("No player statistics to save.")
        return

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(player_stats, f, indent=4)

def main():
    # Example usage
    player_name = "Azhar Mahmood"
    team_name = "Pakistan"
    scorecards_dir = "../match_scorecards_json"

    if not os.path.isdir(scorecards_dir):
        logging.error(f"The directory '{scorecards_dir}' does not exist.")
        return

    player_stats = generate_player_stats(player_name, team_name, scorecards_dir)
    if not player_stats:
        logging.warning(f"No statistics found for player '{player_name}' in team '{team_name}'.")
        return

    logging.info("Player statistics successfully saved.")

if __name__ == "__main__":
    main()
