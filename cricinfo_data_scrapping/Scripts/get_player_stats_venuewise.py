#!/usr/bin/env python3

import os
import json
import re
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
import logging

# ---------------------------------------
# CONFIGURE LOGGING
# ---------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------
# DISMISSAL METHODS
# ---------------------------------------
DISMISSAL_METHODS = [
    "bowled",
    "caught-fielder",
    "caught wk",
    "stump",
    "lbw",
    "hit wicket",
    "caught and bowled"
]

# ---------------------------------------
# SAFE CONVERSIONS
# ---------------------------------------
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

# ---------------------------------------
# DATE PARSING
# ---------------------------------------
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

# ---------------------------------------
# NAME CLEANING AND VARIANTS
# ---------------------------------------
def clean_name(name):
    """
    Removes annotations like (c), (wk), sub(...), etc., from names 
    and strips/trims the result, returning lowercase by default.
    """
    cleaned = re.sub(r'\(.*?\)', '', name)
    return cleaned.strip().lower()

def generate_case_variants(name_str: str) -> set:
    """
    Given a name like 'Kohli', produce variants in different letter cases:
    e.g. 'Kohli', 'kohli', 'KOHLI', etc.
    """
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
       - plus case variants for each.
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

    # 3) If multiple parts, add "v kohli", "vkohli", "kohli"
    if len(name_parts) >= 2:
        first_initials = ''.join([part[0] for part in name_parts[:-1]]).lower()
        last_name = name_parts[-1].lower()
        base_variants.add(f"{first_initials} {last_name}")
        base_variants.add(f"{first_initials}{last_name}")
        base_variants.add(last_name)
    else:
        # Single name
        base_variants.add(name_parts[0].lower())

    # Generate case variants
    final_variants = set()
    for variant in base_variants:
        final_variants.update(generate_case_variants(variant))

    return final_variants

# ---------------------------------------
# BOWLER NAME AND TYPE RESOLUTION
# ---------------------------------------
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
        # Return the first match (could refine if needed)
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

# ---------------------------------------
# FIELDING CONTRIBUTION (CATCHES, STUMPINGS, RUN OUTS)
# ---------------------------------------
def parse_fielding_contribution(dismissal_text: str, player_name_clean: str) -> dict:
    """
    Determine whether 'player_name_clean' contributed as a fielder
    in the given dismissal_text.
    
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

# ---------------------------------------
# DISMISSAL DATA (HOW BATSMAN GOT OUT)
# ---------------------------------------
def extract_dismissal_data(batsman, dismissal_text, innings, bowler_mapping, wicket_keepers):
    """
    Extract how the *batsman* got out (method), and find the bowler name + bowler type.
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
        # Caught & bowled
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
                fielder = match.group(1).strip() if len(match.groups()) >= 2 else ''
                bowler = match.group(2).strip() if len(match.groups()) >= 2 else ''
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
                fielder = match.group(1).strip() if len(match.groups()) >= 2 else ''
                bowler = match.group(2).strip() if len(match.groups()) >= 2 else ''
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
                    bowler_name = bowler
                elif len(match.groups()) == 1:
                    fielder = match.group(1).strip()
                    dismissal_method = 'run out'
                    bowler_name = 'Unknown'
                else:
                    dismissal_method = 'run out'
                    bowler_name = 'Unknown'

            else:
                # For retired hurt, obstructing field, etc.
                dismissal_method = method
                bowler_name = 'Unknown'

            bowler_name_full = resolve_bowler_name(bowler_name, bowler_mapping)
            bowler_type = find_bowler_type(bowler_name_full, innings)
            break

    return {
        'bowler_type': bowler_type,
        'dismissal_method': dismissal_method,
        'bowler_name': bowler_name_full
    }

# ---------------------------------------
# OVERS -> TOTAL BALLS
# ---------------------------------------
def float_overs_to_balls(overs_val: float) -> int:
    """
    Convert a decimal overs value (e.g. 9.6 => 9.6 float) into total balls bowled.
    If 'overs_val' has decimal part 0.6 or more, that means 6 valid deliveries 
    => which is effectively the next full over.

    Examples:
      9.6 => 10 overs => 60 balls
      9.5 => 9 overs + 5 balls => 59 balls
    """
    whole_overs = int(overs_val)
    decimal_part = overs_val - whole_overs
    possible_balls = int(round(decimal_part * 10))  # e.g. 0.6 => 6

    if possible_balls >= 6:
        # e.g. 9.6 => 10 overs => 10*6 = 60
        return (whole_overs + 1) * 6
    else:
        # e.g. 9.5 => 9 overs, 5 balls => 9*6 + 5 = 59
        return whole_overs * 6 + possible_balls

# ---------------------------------------
# PARSE SINGLE MATCH FILE FOR A GIVEN PLAYER
# ---------------------------------------
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
            'balls_bowled': 0,  # <--- store total balls instead of raw float overs
            'maidens': 0,
            'runs_conceded': 0,
            'wickets': 0,
            'economy_total': 0.0,  # (still here if ever needed for something else)
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

        # Per-match breakdown of wickets vs. batting type
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
    
    # Identify wicket-keepers from both teams
    wicket_keepers = set()
    for team, players in playing_xi.items():
        for name in players:
            if '(wk)' in name:
                cleaned_wk = clean_name(name.replace('(wk)', ''))
                wicket_keepers.add(cleaned_wk)

    # Check if player participated (is in the playing XI of the given team)
    player_team_xi = playing_xi.get(team_name, [])
    player_team_xi_cleaned = [clean_name(x) for x in player_team_xi]
    if player_name_clean not in player_team_xi_cleaned:
        return None  # The player didn't appear in this match for that team
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

    # ---------------------------------------
    # MAIN LOGIC
    # ---------------------------------------
    for inning in innings:
        # 1) BATTING (if it's the player's team)
        team = inning.get('team', '').lower()
        is_player_team = (team == team_name_lower)
        if is_player_team:
            batting_list = inning.get('batting', [])
            for batsman in batting_list:
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
                    # Extract how the player got out
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

        # 2) BOWLING (if the other team is batting)
        else:
            bowling_list = inning.get('bowling', [])
            for bowler in bowling_list:
                bowler_name_cleaned = clean_name(bowler.get('bowler', ''))
                if bowler_name_cleaned == player_name_clean:
                    overs_val = safe_float_conversion(bowler.get('overs', 0.0))
                    total_balls_this_spell = float_overs_to_balls(overs_val)

                    wickets_val = safe_int_conversion(bowler.get('wickets', 0))
                    runs_val = safe_int_conversion(bowler.get('runs_conceded', 0))
                    fours_val = safe_int_conversion(bowler.get('fours', 0))
                    sixes_val = safe_int_conversion(bowler.get('sixes', 0))
                    
                    player_stats['bowling']['balls_bowled'] += total_balls_this_spell
                    player_stats['bowling']['maidens'] += safe_int_conversion(bowler.get('maidens', 0))
                    player_stats['bowling']['runs_conceded'] += runs_val
                    player_stats['bowling']['wickets'] += wickets_val
                    # economy_total is stored but never truly used for final; kept for consistency
                    player_stats['bowling']['economy_total'] += safe_float_conversion(bowler.get('economy', 0.0))
                    player_stats['bowling']['dots'] += safe_int_conversion(bowler.get('dots', 0))
                    player_stats['bowling']['fours'] += fours_val
                    player_stats['bowling']['sixes'] += sixes_val
                    player_stats['bowling']['wides'] += safe_int_conversion(bowler.get('wides', 0))
                    player_stats['bowling']['no_balls'] += safe_int_conversion(bowler.get('no_balls', 0))
                    player_stats['bowling']['boundary_runs_this_innings'] += (fours_val * 4) + (sixes_val * 6)
                    player_stats['bowling']['match_wickets'] = wickets_val
                    player_stats['bowling']['match_runs'] = runs_val

        # 3) FIELDING + (bowling dismissals vs batting type)
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

            # Check if our player was the credited bowler
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
                # The bowler is indeed our player
                batting_type_str = batsman_data.get('batting_type', '').strip().lower()
                if 'left' in batting_type_str:
                    style_key = 'Left Handed'
                elif 'right' in batting_type_str:
                    style_key = 'Right Handed'
                else:
                    style_key = 'Unknown'

                # Wickets
                player_stats['bowling_vs_batting_type'][style_key]['wickets'] += 1

                # Dismissal method
                known_methods = {
                    'bowled', 'caught-fielder', 'caught wk', 'stump',
                    'lbw', 'hit wicket', 'caught and bowled'
                }
                if method in known_methods:
                    player_stats['bowling_vs_batting_type'][style_key][method] += 1
                else:
                    player_stats['bowling_vs_batting_type'][style_key]['other'] += 1

    return player_stats

# ---------------------------------------
# INITIALIZE AN EMPTY AGGREGATOR STRUCTURE
# ---------------------------------------
def init_aggregator():
    bowler_types = [
        "Right arm Pace",
        "Right arm Spin",
        "Left arm Pace",
        "Left arm Spin",
        "Unknown"
    ]
    aggregator = {
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
            'balls_bowled': 0,  # now storing total balls
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
            bowler_type: {"Dis": 0, **{dm: 0 for dm in DISMISSAL_METHODS}}
            for bowler_type in bowler_types
        },
        "match_dates": [],
        "matches_played": 0,
        "batting_innings": 0,
        "bowling_innings": 0,
        "fielding_innings": 0
    }
    return aggregator

# ---------------------------------------
# MERGE SINGLE MATCH STATS INTO AGGREGATOR
# ---------------------------------------
def merge_single_match_stats(agg, match_stats):
    """Add the stats of a single match (match_stats) into the aggregator (agg)."""
    if match_stats.get("played"):
        agg["matches_played"] += 1
        match_date = match_stats.get("match_date")
        if match_date:
            agg["match_dates"].append(match_date)

    # Batting
    batting = match_stats.get("batting", {})
    if batting:
        agg["batting_innings"] += 1
        agg["batting"]["runs"] += batting.get("runs", 0)
        agg["batting"]["balls"] += batting.get("balls", 0)
        agg["batting"]["fours"] += batting.get("fours", 0)
        agg["batting"]["sixes"] += batting.get("sixes", 0)
        agg["batting"]["strike_rate_total"] += batting.get("strike_rate", 0.0)
        if batting.get("dismissal", "").lower() != "not out":
            agg["batting"]["dismissals"] += 1
        else:
            agg["batting"]["not_outs"] += 1

    # Bowling
    bowling = match_stats.get("bowling", {})
    if bowling:
        agg["bowling_innings"] += 1
        agg["bowling"]["balls_bowled"] += bowling.get("balls_bowled", 0)
        agg["bowling"]["maidens"] += bowling.get("maidens", 0)
        agg["bowling"]["runs_conceded"] += bowling.get("runs_conceded", 0)
        agg["bowling"]["wickets"] += bowling.get("wickets", 0)
        agg["bowling"]["economy_total"] += bowling.get("economy_total", 0.0)
        agg["bowling"]["dots"] += bowling.get("dots", 0)
        agg["bowling"]["fours"] += bowling.get("fours", 0)
        agg["bowling"]["sixes"] += bowling.get("sixes", 0)
        agg["bowling"]["wides"] += bowling.get("wides", 0)
        agg["bowling"]["no_balls"] += bowling.get("no_balls", 0)
        agg["bowling"]["total_boundary_runs"] += bowling.get('boundary_runs_this_innings', 0)

        # best figures
        match_wickets = bowling.get('match_wickets', 0)
        match_runs = bowling.get('match_runs', 0)
        if match_wickets > 0:
            if ((match_wickets > agg["bowling"]['best_bowling_wickets']) or
                (match_wickets == agg["bowling"]['best_bowling_wickets'] and match_runs < agg["bowling"]['best_bowling_runs'])):
                agg["bowling"]['best_bowling_wickets'] = match_wickets
                agg["bowling"]['best_bowling_runs'] = match_runs

        if match_wickets == 3:
            agg["bowling"]['three_wicket_hauls'] += 1
        elif match_wickets == 4:
            agg["bowling"]['four_wicket_hauls'] += 1
        elif match_wickets >= 5:
            agg["bowling"]['five_wicket_hauls'] += 1

    # Fielding
    fielding = match_stats.get("fielding", {})
    if fielding:
        agg["fielding_innings"] += 1
        agg["fielding"]["catches"] += fielding.get("catches", 0)
        agg["fielding"]["stumpings"] += fielding.get("stumpings", 0)
        agg["fielding"]["run_outs"] += fielding.get("run_outs", 0)

    # "Who got me out?" dismissals
    for dismissal in match_stats.get("dismissals", []):
        bowler_type = dismissal.get("bowler_type", "Unknown")
        if bowler_type not in agg['dismissals']:
            bowler_type = "Unknown"
        dismissal_method = dismissal.get("dismissal_method", "other").lower()
        if dismissal_method in DISMISSAL_METHODS:
            agg['dismissals'][bowler_type][dismissal_method] += 1
        agg['dismissals'][bowler_type]["Dis"] += 1

    # Bowling vs batting type
    match_bvbt = match_stats.get("bowling_vs_batting_type", {})
    if "bvbt" not in agg:  # lazy init for clarity
        agg["bvbt"] = {
            "Right Handed": {
                "wickets": 0,
                "bowled": 0,
                "caught-fielder": 0,
                "caught wk": 0,
                "stump": 0,
                "lbw": 0,
                "hit wicket": 0,
                "caught and bowled": 0,
                "other": 0
            },
            "Left Handed": {
                "wickets": 0,
                "bowled": 0,
                "caught-fielder": 0,
                "caught wk": 0,
                "stump": 0,
                "lbw": 0,
                "hit wicket": 0,
                "caught and bowled": 0,
                "other": 0
            },
            "Unknown": {
                "wickets": 0,
                "bowled": 0,
                "caught-fielder": 0,
                "caught wk": 0,
                "stump": 0,
                "lbw": 0,
                "hit wicket": 0,
                "caught and bowled": 0,
                "other": 0
            },
        }
    for style_key in ["Right Handed", "Left Handed", "Unknown"]:
        style_dict = match_bvbt.get(style_key, {})
        agg["bvbt"][style_key]["wickets"] += style_dict.get("wickets", 0)
        for mkey in ["bowled", "caught-fielder", "caught wk", "stump",
                     "lbw", "hit wicket", "caught and bowled", "other"]:
            agg["bvbt"][style_key][mkey] += style_dict.get(mkey, 0)

# ---------------------------------------
# FINALIZE AGGREGATOR -> MAKE JSON-FRIENDLY
# ---------------------------------------
def finalize_aggregator(agg):
    """
    After merging all single-match stats, compute final averages, SR, etc.
    Return a structure with the same fields you want in the final output.
    """
    # BATTING AVERAGE, STRIKE RATE
    if agg['batting']['dismissals'] > 0:
        agg['batting']['bat_avg'] = round(agg['batting']['runs'] / agg['batting']['dismissals'], 2)
    else:
        agg['batting']['bat_avg'] = None

    if agg['batting']['balls'] > 0:
        agg['batting']['strike_rate'] = round((agg['batting']['runs'] / agg['batting']['balls']) * 100, 2)
    else:
        agg['batting']['strike_rate'] = None

    # Convert total_balls to overs float X.Y
    total_balls_bowled = agg['bowling']['balls_bowled']
    overs_int = total_balls_bowled // 6
    leftover_balls = total_balls_bowled % 6
    corrected_overs = overs_int + leftover_balls / 10.0  # e.g. 43 overs, 2 balls => 43.2

    # BOWLING AVERAGE & ECONOMY
    if agg['bowling']['wickets'] > 0:
        agg['bowling']['bowl_avg'] = round(agg['bowling']['runs_conceded'] / agg['bowling']['wickets'], 2)
    else:
        agg['bowling']['bowl_avg'] = None

    if total_balls_bowled > 0:
        economy = agg['bowling']['runs_conceded'] / (total_balls_bowled / 6)
        agg['bowling']['economy'] = round(economy, 2)
    else:
        agg['bowling']['economy'] = None

    # Bowling Strike Rate
    if agg['bowling']['wickets'] > 0 and total_balls_bowled > 0:
        bowling_strike_rate = round(total_balls_bowled / agg['bowling']['wickets'], 2)
    else:
        bowling_strike_rate = None

    # Dot-ball percentage
    if total_balls_bowled > 0:
        dot_ball_percentage = round((agg['bowling']['dots'] / total_balls_bowled) * 100, 2)
    else:
        dot_ball_percentage = None

    # Best Bowling Figures
    if (agg['bowling']['best_bowling_wickets'] > 0 and
        agg['bowling']['best_bowling_runs'] < 999999):
        best_bowling_figures = f"{agg['bowling']['best_bowling_wickets']}/{agg['bowling']['best_bowling_runs']}"
    else:
        best_bowling_figures = None

    # Career span
    if agg['match_dates']:
        earliest_date = min(agg['match_dates'])
        latest_date = max(agg['match_dates'])
        span = f"{earliest_date.year} - {latest_date.year}"
    else:
        span = "Unknown"

    # Dismissal data
    bowler_types = ["Right arm Pace","Right arm Spin","Left arm Pace","Left arm Spin","Unknown"]
    dismissal_data = []
    for btype in bowler_types:
        data = agg['dismissals'][btype]
        dismissal_entry = {
            "Bowler Type": btype,
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
        dismissal_data.append(dismissal_entry)

    # Bowling vs batting type
    style_keys = ["Right Handed", "Left Handed", "Unknown"]
    final_bvbt = {}
    if "bvbt" in agg:
        for style in style_keys:
            style_dict = agg["bvbt"].get(style, {})
            final_bvbt[style] = {
                "wickets": style_dict.get("wickets", 0),
                "bowled": style_dict.get("bowled", 0),
                "caught-fielder": style_dict.get("caught-fielder", 0),
                "caught wk": style_dict.get("caught wk", 0),
                "stump": style_dict.get("stump", 0),
                "lbw": style_dict.get("lbw", 0),
                "hit wicket": style_dict.get("hit wicket", 0),
                "caught and bowled": style_dict.get("caught and bowled", 0),
                "other": style_dict.get("other", 0)
            }

    final_aggregated = {
        "span": span,
        "matches_played": agg['matches_played'],
        "batting_innings": agg['batting_innings'],
        "bowling_innings": agg['bowling_innings'],
        "fielding_innings": agg['fielding_innings'],
        "batting": {
            "Runs": agg['batting']['runs'],
            "Balls Faced": agg['batting']['balls'],
            "Fours": agg['batting']['fours'],
            "Sixes": agg['batting']['sixes'],
            "Strike Rate": agg['batting']['strike_rate'],
            "Dismissals": agg['batting']['dismissals'],
            "Not Outs": agg['batting']['not_outs'],
            "Batting Average": agg['batting']['bat_avg']
        },
        "bowling": {
            "Overs": round(corrected_overs, 2),
            "Maidens": agg['bowling']["maidens"],
            "Runs Conceded": agg['bowling']["runs_conceded"],
            "Wickets": agg['bowling']["wickets"],
            "Economy Rate": agg['bowling']['economy'],
            "Bowling Average": agg['bowling']['bowl_avg'],
            "Dots": agg['bowling']['dots'],
            "Fours": agg['bowling']['fours'],
            "Sixes": agg['bowling']['sixes'],
            "Wides": agg['bowling']['wides'],
            "No Balls": agg['bowling']['no_balls'],
            "Boundary Runs Conceded": agg['bowling']['total_boundary_runs'],
            "Bowling Strike Rate": bowling_strike_rate,
            "Dot Ball Percentage": dot_ball_percentage,
            "3wHauls": agg['bowling']['three_wicket_hauls'],
            "4wHauls": agg['bowling']['four_wicket_hauls'],
            "5wHauls": agg['bowling']['five_wicket_hauls'],
            "Best Bowling Figures": best_bowling_figures
        },
        "fielding": {
            "Catches": agg['fielding']["catches"],
            "Stumpings": agg['fielding']["stumpings"],
            "Run Outs": agg['fielding']["run_outs"]
        },
        "dismissals": dismissal_data,
        "bowling_vs_batting_type": final_bvbt
    }

    return final_aggregated

# ---------------------------------------
# PARSE COUNTRY FROM series_name
# ---------------------------------------
def parse_country_from_series_name(series_name: str) -> str:
    """
    Extract the country from 'series_name' by looking for a pattern "X in Y ...".
    If no match found, returns "Unknown".
    
    Example: "Zimbabwe in India ODI Series" => "India"
    """
    sn_lower = series_name.lower()
    match = re.search(r'\bin\s+([a-zA-Z\s]+)', sn_lower)
    if match:
        country_str = match.group(1).strip()
        # Remove common words: 'odi', 't20i', 'test', 'series', digits, etc.
        country_str = re.sub(r'(odi|t20i|test|\d+|series)', '', country_str, flags=re.IGNORECASE)
        country_str = country_str.strip()
        if not country_str:
            return "Unknown"
        return " ".join(word.capitalize() for word in country_str.split())
    return "Unknown"

# ---------------------------------------
# PROCESS SINGLE FILE (MULTIPROCESSING)
# ---------------------------------------
def process_file(args):
    """Wrapper function for multiprocessing pool."""
    file_path, player_name, team_name = args
    return parse_match_file(file_path, player_name, team_name)

# ---------------------------------------
# MAIN FUNCTION FOR VENUE-WISE AGGREGATION
# ---------------------------------------
def generate_player_stats_venuewise(player_name, team_name, scorecards_dir='../match_scorecards_json'):
    """
    Generates a *venue-wise* statistics JSON for the specified player
    using the match scorecards found in 'scorecards_dir'.
    
    The final JSON has the same structure of stats but grouped by 
    the country extracted from 'series_name'.
    """
    # 1) Collect all JSON files
    try:
        json_files = [
            os.path.join(scorecards_dir, filename)
            for filename in os.listdir(scorecards_dir)
            if filename.endswith('.json')
        ]
    except FileNotFoundError:
        logging.error(f"The directory '{scorecards_dir}' does not exist.")
        return {}

    if not json_files:
        logging.warning(f"No JSON files found in the directory '{scorecards_dir}'.")
        return {}

    # 2) Process all files in parallel
    args = [(fp, player_name, team_name) for fp in json_files]
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_file, args)

    # Filter out None results
    all_stats = [r for r in results if r is not None]
    if not all_stats:
        logging.warning(f"No match data found for '{player_name}' in team '{team_name}'.")
        return {}

    # 3) Group them *by venue*
    aggregator_by_venue = {}
    for i, match_stats in enumerate(all_stats):
        # read the series_name from the corresponding file to parse the country
        file_path = args[i][0]
        with open(file_path, 'r', encoding='utf-8') as f:
            match_data = json.load(f)

        series_name = match_data.get('series_name', '')
        country_venue = parse_country_from_series_name(series_name)

        # Initialize aggregator if not present
        if country_venue not in aggregator_by_venue:
            aggregator_by_venue[country_venue] = init_aggregator()

        # Merge this match's stats
        merge_single_match_stats(aggregator_by_venue[country_venue], match_stats)

    # 4) Finalize each aggregator => produce final JSON data
    final_output = {
        "player": player_name,
        "team": team_name,
        "venues": {}
    }

    for venue, agg_data in aggregator_by_venue.items():
        final_output["venues"][venue] = finalize_aggregator(agg_data)

    # 5) Save the JSON to ../venue_wise_aggregated_stats
    safe_player_name = player_name.lower().replace(' ', '_')
    output_dir = "../aggregated_stats_venue_wise"  # <--- new directory name
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            logging.error(f"Could not create directory '{output_dir}'. Error: {e}")

    output_file = os.path.join(output_dir, f"{safe_player_name}_stats_venuewise.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, default=str)

    logging.info(f"Venue-wise stats JSON produced: {output_file}")
    return final_output

# ---------------------------------------
# EXAMPLE MAIN
# ---------------------------------------
def main():
    # Example usage
    player_name = "Mitchell Starc"
    team_name = "Australia"
    scorecards_dir = "../match_scorecards_json"

    if not os.path.isdir(scorecards_dir):
        logging.error(f"The directory '{scorecards_dir}' does not exist.")
        return

    result_json = generate_player_stats_venuewise(player_name, team_name, scorecards_dir)
    if not result_json:
        logging.warning("No results or no matching data found.")
    else:
        logging.info("Venue-wise player statistics successfully computed.")

if __name__ == "__main__":
    main()
