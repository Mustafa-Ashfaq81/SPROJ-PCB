#!/usr/bin/env python3
# Usage: python get_player_stats_with_span.py <player_name> <team_name> <start_year> <end_year>
# Usage Example 1: python get_player_stats.py "Babar Azam" Pakistan 2015 2018
# Usage Example 2: python get_player_stats.py "Babar Azam" Pakistan all all

import os
import json
import re
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dismissal methods we consider
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
    """
    Attempts to parse a match date from various fields. Returns a datetime or None.
    Priority: match_days -> date+season -> header
    """
    match_days = match_details.get('match_days', '')
    date_pattern = r'(\d{1,2} \w+ \d{4})'
    match_obj = re.search(date_pattern, match_days)
    if match_obj:
        date_str = match_obj.group(1)
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
    match_obj = re.search(date_pattern, header)
    if match_obj:
        date_str = match_obj.group(1)
        try:
            return datetime.strptime(date_str, '%d %B %Y')
        except ValueError:
            return None

    return None

def clean_name(name):
    """
    Remove annotations like (c), (wk), sub(...), etc. Return a lowercase version.
    """
    cleaned = re.sub(r'\(.*?\)', '', name)
    return cleaned.strip().lower()

def generate_case_variants(name_str: str) -> set:
    """
    Produce variants in different letter cases for a given name.
    E.g. 'Kohli', 'kohli', 'KOHLI', 'Kohli', etc.
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
    Generates various partial/case variants from a full name for partial matching.
    E.g. 'Virat Kohli' -> {'virat kohli','viratkohli','v kohli','vkohli','kohli'} + case combos.
    """
    full_name = full_name.strip()
    if not full_name:
        return set()

    name_parts = full_name.split()
    base_variants = set()

    # full name (spaces), full joined
    base_variants.add(' '.join(name_parts).lower())
    base_variants.add(''.join(name_parts).lower())

    # If multiple parts, add "v kohli", "vkohli", "kohli"
    if len(name_parts) >= 2:
        first_initials = ''.join([part[0] for part in name_parts[:-1]]).lower()
        last_name = name_parts[-1].lower()
        base_variants.add(f"{first_initials} {last_name}")
        base_variants.add(f"{first_initials}{last_name}")
        base_variants.add(last_name)
    else:
        base_variants.add(name_parts[0].lower())

    final_variants = set()
    for variant in base_variants:
        final_variants.update(generate_case_variants(variant))

    return final_variants

def resolve_bowler_name(bowler_name, bowler_mapping):
    """
    Resolves the full bowler name using bowler_mapping, ignoring case.
    If multiple matches, returns first. Otherwise returns raw name.
    """
    bowler_variants = generate_name_variants(bowler_name)
    possible_full_names = []
    for variant in bowler_variants:
        lookup_key = variant.lower()
        if lookup_key in bowler_mapping:
            possible_full_names.extend(bowler_mapping[lookup_key])

    if possible_full_names:
        return possible_full_names[0]
    else:
        return bowler_name

def find_bowler_type(bowler_name_full, innings):
    """
    Searches across all innings->bowling for a matching bowler and returns bowler_type.
    """
    bowler_name_lower = bowler_name_full.lower()
    for inning in innings:
        for bowler_entry in inning.get('bowling', []):
            if bowler_entry.get('bowler', '').lower() == bowler_name_lower:
                return bowler_entry.get('b_type', 'Unknown')
    return 'Unknown'

def parse_fielding_contribution(dismissal_text: str, player_name_clean: str) -> dict:
    """
    Checks whether 'player_name_clean' contributed as a fielder in the given dismissal_text.
    Returns increments for 'catches', 'stumpings', 'run_outs'.
    """
    fielding_stats = {"catches": 0, "stumpings": 0, "run_outs": 0}
    dismissal_text_lower = dismissal_text.lower().strip()
    player_variants = generate_name_variants(player_name_clean)

    c_and_b_pattern = re.compile(r'c\s+&\s+b\s+([\w\s]+)', re.IGNORECASE)
    match_cab = c_and_b_pattern.search(dismissal_text_lower)
    if match_cab:
        bowler_fielder_raw = clean_name(match_cab.group(1))
        bowler_fielder_variants = generate_name_variants(bowler_fielder_raw)
        if player_variants & bowler_fielder_variants:
            fielding_stats["catches"] += 1
        return fielding_stats

    caught_pattern = re.compile(r'^c\s+(.*?)\s+b\s+([\w\s]+)$', re.IGNORECASE)
    match_caught = caught_pattern.search(dismissal_text_lower)
    if match_caught:
        fielder_raw = match_caught.group(1)
        fielder_raw = re.sub(r'\bsub\b', '', fielder_raw, flags=re.IGNORECASE)
        fielder_cleaned = clean_name(fielder_raw)
        fielder_variants = generate_name_variants(fielder_cleaned)
        if player_variants & fielder_variants:
            fielding_stats["catches"] += 1

    stumped_pattern = re.compile(r'st\s+(.*?)\s+b\s+([\w\s]+)', re.IGNORECASE)
    match_stumped = stumped_pattern.search(dismissal_text_lower)
    if match_stumped:
        fielder_raw = match_stumped.group(1)
        fielder_raw = re.sub(r'\bsub\b', '', fielder_raw, flags=re.IGNORECASE)
        fielder_cleaned = clean_name(fielder_raw)
        fielder_variants = generate_name_variants(fielder_cleaned)
        if player_variants & fielder_variants:
            fielding_stats["stumpings"] += 1

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

def extract_dismissal_data(batsman_name_cleaned, dismissal_text, innings, bowler_mapping, wicket_keepers):
    """
    Extract how the *batsman* got out: method, bowler name, bowler type, etc.
    """
    bowler_type = 'Unknown'
    dismissal_method = 'other'
    bowler_name_full = 'Unknown'
    fielder = 'Unknown'

    dismissal_patterns = [
        (re.compile(r'c\s+\(wk\)\s*([\w\s]+)\s+b\s+([\w\s]+)', re.IGNORECASE), 'caught wk'),
        (re.compile(r'c\s+([\w\s]+)\s+b\s+([\w\s]+)', re.IGNORECASE), 'caught-fielder'),
        (re.compile(r'st\s+\(wk\)\s*([\w\s]+)\s+b\s+([\w\s]+)', re.IGNORECASE), 'stump'),
        (re.compile(r'st\s+([\w\s]+)\s+b\s+([\w\s]+)', re.IGNORECASE), 'stump'),
        (re.compile(r'lbw\s+b\s+([\w\s]+)', re.IGNORECASE), 'lbw'),
        (re.compile(r'c\s+&\s+b\s+([\w\s]+)', re.IGNORECASE), 'caught and bowled'),
        (re.compile(r'hit wicket\s+b\s+([\w\s]+)', re.IGNORECASE), 'hit wicket'),
        (re.compile(r'^b\s+([\w\s]+)$', re.IGNORECASE), 'bowled'),
        (re.compile(r'run out\s+\(([\w\s]+)\)\s+b\s+([\w\s]+)', re.IGNORECASE), 'run out'),
        (re.compile(r'run out\s+\(([\w\s/]+)\)', re.IGNORECASE), 'run out'),
        (re.compile(r'run out', re.IGNORECASE), 'run out'),
        (re.compile(r'retired hurt', re.IGNORECASE), 'retired hurt'),
        (re.compile(r'retired not out', re.IGNORECASE), 'retired not out'),
        (re.compile(r'obstructing the field', re.IGNORECASE), 'obstructing the field'),
        (re.compile(r'timed out', re.IGNORECASE), 'timed out'),
        (re.compile(r'handled the ball', re.IGNORECASE), 'handled the ball'),
        (re.compile(r'hit the ball twice', re.IGNORECASE), 'hit the ball twice')
    ]

    dismissal_text_lower = dismissal_text.lower().strip()

    for pattern, method in dismissal_patterns:
        match_obj = pattern.search(dismissal_text_lower)
        if match_obj:
            if method == 'caught wk':
                fielder = match_obj.group(1).strip()
                bowler = match_obj.group(2).strip()
                dismissal_method = 'caught wk'
                bowler_name = bowler
            elif method == 'caught-fielder':
                fielder = match_obj.group(1).strip()
                bowler = match_obj.group(2).strip()
                if clean_name(fielder) in wicket_keepers:
                    dismissal_method = 'caught wk'
                else:
                    dismissal_method = 'caught-fielder'
                bowler_name = bowler
            elif method == 'caught and bowled':
                bowler = match_obj.group(1).strip()
                dismissal_method = 'caught and bowled'
                bowler_name = bowler
            elif method == 'stump':
                fielder = match_obj.group(1).strip()
                bowler = match_obj.group(2).strip()
                dismissal_method = 'stump'
                bowler_name = bowler
            elif method in ['lbw', 'bowled', 'hit wicket']:
                bowler = match_obj.group(1).strip()
                dismissal_method = method
                bowler_name = bowler
            elif method == 'run out':
                if len(match_obj.groups()) == 2:
                    fielder = match_obj.group(1).strip()
                    bowler = match_obj.group(2).strip()
                    dismissal_method = 'run out'
                    bowler_name = bowler
                elif len(match_obj.groups()) == 1:
                    fielder = match_obj.group(1).strip()
                    dismissal_method = 'run out'
                    bowler_name = 'Unknown'
                else:
                    dismissal_method = 'run out'
                    bowler_name = 'Unknown'
            elif method in [
                'retired hurt', 'retired not out', 'obstructing the field',
                'timed out', 'handled the ball', 'hit the ball twice'
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

def parse_years(start_year, end_year):
    """
    Decide whether we are filtering by year range or taking all data.
    If 'all' is given, we return (None, None).
    Otherwise, parse them as integers for filtering.
    """
    if isinstance(start_year, str) and start_year.lower() == "all":
        return (None, None)
    if isinstance(end_year, str) and end_year.lower() == "all":
        return (None, None)

    try:
        sy = int(start_year)
        ey = int(end_year)
        return (sy, ey)
    except (ValueError, TypeError):
        return (None, None)

def parse_match_file(file_path, player_name, team_name, start_year=None, end_year=None):
    """
    Parse a single match JSON. If start_year/end_year is None => no filter. 
    Else only include if match year is in [start_year, end_year].
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            match_data = json.load(f)
        except json.JSONDecodeError:
            logging.error(f"JSONDecodeError: Failed to parse {file_path}")
            return None

    match_details = match_data.get('match_details', {})
    match_date = parse_match_date(match_details)

    if start_year is not None and end_year is not None and match_date is not None:
        if match_date.year < start_year or match_date.year > end_year:
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
        'match_date': match_date,
        'played': False,
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

    player_name_clean = clean_name(player_name)
    team_name_lower = team_name.lower()

    playing_xi = match_data.get('playing_xi', {})
    wicket_keepers = set()
    for tm, players in playing_xi.items():
        for name in players:
            if '(wk)' in name:
                cleaned_wk = clean_name(name.replace('(wk)', ''))
                wicket_keepers.add(cleaned_wk)

    player_team_xi = playing_xi.get(team_name, [])
    player_team_xi_cleaned = [clean_name(x) for x in player_team_xi]
    if player_name_clean not in player_team_xi_cleaned:
        return None
    else:
        player_stats['played'] = True

    bowler_mapping = defaultdict(list)
    innings = match_data.get('innings', [])
    for inning in innings:
        for bowler in inning.get('bowling', []):
            bowler_original = bowler.get('bowler', '')
            for variant in generate_name_variants(bowler_original):
                bowler_mapping[variant.lower()].append(bowler_original)

    for inning in innings:
        team = inning.get('team', '').lower()
        is_player_team = (team == team_name_lower)

        if is_player_team:
            for batsman in inning.get('batting', []):
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
        else:
            for bowler in inning.get('bowling', []):
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

        for batsman_data in inning.get('batting', []):
            dismiss_text = batsman_data.get('dismissal', '').strip()
            if not dismiss_text:
                continue

            fielding_contrib = parse_fielding_contribution(dismiss_text, player_name_clean)
            player_stats['fielding']['catches'] += fielding_contrib["catches"]
            player_stats['fielding']['stumpings'] += fielding_contrib["stumpings"]
            player_stats['fielding']['run_outs'] += fielding_contrib["run_outs"]

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
                batting_type_str = batsman_data.get('batting_type', '').strip().lower()
                if 'left' in batting_type_str:
                    style_key = 'Left Handed'
                elif 'right' in batting_type_str:
                    style_key = 'Right Handed'
                else:
                    style_key = 'Unknown'

                player_stats['bowling_vs_batting_type'][style_key]['wickets'] += 1
                known_methods = {
                    'bowled', 'caught-fielder', 'caught wk', 'stump',
                    'lbw', 'hit wicket', 'caught and bowled'
                }
                if method in known_methods:
                    player_stats['bowling_vs_batting_type'][style_key][method] += 1
                else:
                    player_stats['bowling_vs_batting_type'][style_key]['other'] += 1

    return player_stats

def correct_overs(overs):
    """Convert decimal overs into standard X.Y where Y<6."""
    whole_overs = int(overs)
    balls = (overs - whole_overs) * 10
    if balls >= 6:
        whole_overs += 1
        balls -= 6
    return whole_overs + balls / 10

def aggregate_stats(all_stats):
    bowler_types = ["Right arm Pace", "Right arm Spin", "Left arm Pace", "Left arm Spin", "Unknown"]
    dismissal_methods = [
        "bowled", "caught-fielder", "caught wk", "stump",
        "lbw", "hit wicket", "caught and bowled"
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
            bt: {"Dis": 0, **{dm: 0 for dm in dismissal_methods}}
            for bt in bowler_types
        },
        "match_dates": [],
        "matches_played": 0,
        "batting_innings": 0,
        "bowling_innings": 0,
        "fielding_innings": 0
    }

    style_keys = ["Right Handed", "Left Handed", "Unknown"]
    method_keys = dismissal_methods + ["other"]
    bowling_vs_batting_type_agg = {
        style: {"wickets": 0, **{m: 0 for m in method_keys}}
        for style in style_keys
    }

    for match in all_stats:
        if match.get("played"):
            aggregated["matches_played"] += 1
            mdate = match.get("match_date")
            if mdate:
                aggregated["match_dates"].append(mdate)

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
            aggregated["bowling"]['total_boundary_runs'] += bowling.get('boundary_runs_this_innings', 0)

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

        fielding = match.get("fielding", {})
        if fielding:
            aggregated["fielding_innings"] += 1
            aggregated["fielding"]["catches"] += fielding.get("catches", 0)
            aggregated["fielding"]["stumpings"] += fielding.get("stumpings", 0)
            aggregated["fielding"]["run_outs"] += fielding.get("run_outs", 0)

        for dismissal in match.get("dismissals", []):
            bowler_type = dismissal.get("bowler_type", "Unknown")
            dismissal_method = dismissal.get("dismissal_method", "other").lower()
            if bowler_type not in aggregated['dismissals']:
                bowler_type = "Unknown"

            aggregated['dismissals'][bowler_type]["Dis"] += 1
            if dismissal_method in aggregated['dismissals'][bowler_type]:
                aggregated['dismissals'][bowler_type][dismissal_method] += 1

        match_bvbt = match.get("bowling_vs_batting_type", {})
        for style in style_keys:
            style_dict = match_bvbt.get(style, {})
            bowling_vs_batting_type_agg[style]["wickets"] += style_dict.get("wickets", 0)
            for m_key in method_keys:
                bowling_vs_batting_type_agg[style][m_key] += style_dict.get(m_key, 0)

    if aggregated['batting']['dismissals'] > 0:
        aggregated['batting']['bat_avg'] = round(aggregated['batting']['runs'] / aggregated['batting']['dismissals'], 2)
    else:
        aggregated['batting']['bat_avg'] = None

    if aggregated['batting']['balls'] > 0:
        aggregated['batting']['strike_rate'] = round((aggregated['batting']['runs']/aggregated['batting']['balls']) * 100, 2)
    else:
        aggregated['batting']['strike_rate'] = None

    if aggregated['bowling']['wickets'] > 0:
        aggregated['bowling']['bowl_avg'] = round(aggregated['bowling']['runs_conceded'] / aggregated['bowling']['wickets'], 2)
    else:
        aggregated['bowling']['bowl_avg'] = None

    if aggregated['bowling']['overs'] > 0:
        aggregated['bowling']['economy'] = round(aggregated['bowling']['runs_conceded']/aggregated['bowling']['overs'], 2)
    else:
        aggregated['bowling']['economy'] = None

    corrected_overs_val = correct_overs(aggregated['bowling']['overs'])
    wo = int(corrected_overs_val)
    pb = int(round((corrected_overs_val - wo)*10))
    total_balls_bowled = wo*6 + pb

    if aggregated['bowling']['wickets'] > 0 and total_balls_bowled > 0:
        bowling_strike_rate = round(total_balls_bowled / aggregated['bowling']['wickets'], 2)
    else:
        bowling_strike_rate = None

    if total_balls_bowled > 0:
        dot_ball_percentage = round((aggregated['bowling']['dots']/total_balls_bowled)*100, 2)
    else:
        dot_ball_percentage = None

    if (aggregated['bowling']['best_bowling_wickets'] > 0 and
        aggregated['bowling']['best_bowling_runs'] < 999999):
        best_figures = f"{aggregated['bowling']['best_bowling_wickets']}/{aggregated['bowling']['best_bowling_runs']}"
    else:
        best_figures = None

    if aggregated['match_dates']:
        earliest_date = min(aggregated['match_dates'])
        latest_date = max(aggregated['match_dates'])
        career_span = f"{earliest_date.year} - {latest_date.year}"
    else:
        career_span = "Unknown"

    final_aggregated = {
        "span": career_span,
        "batting": aggregated['batting'],
        "bowling": {
            "Overs": round(corrected_overs_val, 2),
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
            "Best Bowling Figures": best_figures
        },
        "fielding": aggregated['fielding'],
        "dismissals": [],
        "matches_played": aggregated['matches_played'],
        "batting_innings": aggregated['batting_innings'],
        "bowling_innings": aggregated['bowling_innings'],
        "fielding_innings": aggregated['fielding_innings'],
        "bowling_vs_batting_type": bowling_vs_batting_type_agg
    }

    for bt in bowler_types:
        data = aggregated['dismissals'][bt]
        final_aggregated['dismissals'].append({
            "Bowler Type": bt,
            "Span": career_span,
            "Dis": data["Dis"],
            "bowled": data["bowled"],
            "caught-fielder": data["caught-fielder"],
            "caught wk": data["caught wk"],
            "stump": data["stump"],
            "lbw": data["lbw"],
            "hit wicket": data["hit wicket"],
            "caught and bowled": data["caught and bowled"]
        })

    return final_aggregated

def process_file(args):
    file_path, player_name, team_name, sy, ey = args
    return parse_match_file(file_path, player_name, team_name, sy, ey)

def generate_player_stats_with_span(player_name, team_name, start_year, end_year, scorecards_dir='../cricinfo_data_scrapping/match_scorecards_json/'):
    sy, ey = parse_years(start_year, end_year)

    try:
        json_files = [
            os.path.join(scorecards_dir, fn)
            for fn in os.listdir(scorecards_dir)
            if fn.endswith('.json')
        ]
    except FileNotFoundError:
        logging.error(f"Directory {scorecards_dir} not found.")
        return {}

    if not json_files:
        logging.warning(f"No JSON files found in {scorecards_dir}.")
        return {}

    args_list = [(fp, player_name, team_name, sy, ey) for fp in json_files]

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_file, args_list)

    all_stats = [r for r in results if r is not None]
    if not all_stats:
        logging.warning(f"No match data found for {player_name} in {team_name} with the given span.")
        return {}

    aggregated_data = aggregate_stats(all_stats)
    total_processed = len(all_stats)
    logging.info(f"Processed {total_processed} matches for {player_name} ({team_name}).")

    player_stats_json = {
        "player": player_name,
        "team": team_name,
        "span": aggregated_data["span"],
        "matches_played": aggregated_data["matches_played"],
        "batting_innings": aggregated_data["batting_innings"],
        "bowling_innings": aggregated_data["bowling_innings"],
        "fielding_innings": aggregated_data["fielding_innings"],
        "batting": {
            "Runs": aggregated_data["batting"]["runs"],
            "Balls Faced": aggregated_data["batting"]["balls"],
            "Fours": aggregated_data["batting"]["fours"],
            "Sixes": aggregated_data["batting"]["sixes"],
            "Strike Rate": aggregated_data["batting"].get("strike_rate"),
            "Dismissals": aggregated_data["batting"].get("dismissals"),
            "Not Outs": aggregated_data["batting"].get("not_outs"),
            "Batting Average": aggregated_data["batting"].get("bat_avg")
        },
        "bowling": {
            "Overs": aggregated_data["bowling"]["Overs"],
            "Maidens": aggregated_data["bowling"]["Maidens"],
            "Runs Conceded": aggregated_data["bowling"]["Runs Conceded"],
            "Wickets": aggregated_data["bowling"]["Wickets"],
            "Economy Rate": aggregated_data["bowling"].get("Economy Rate"),
            "Bowling Average": aggregated_data["bowling"].get("Bowling Average"),
            "Dots": aggregated_data["bowling"].get("Dots"),
            "Fours": aggregated_data["bowling"].get("Fours"),
            "Sixes": aggregated_data["bowling"].get("Sixes"),
            "Wides": aggregated_data["bowling"].get("Wides"),
            "No Balls": aggregated_data["bowling"].get("No Balls"),
            "Boundary Runs Conceded": aggregated_data["bowling"].get("Boundary Runs Conceded"),
            "Bowling Strike Rate": aggregated_data["bowling"].get("Bowling Strike Rate"),
            "Dot Ball Percentage": aggregated_data["bowling"].get("Dot Ball Percentage"),
            "3wHauls": aggregated_data["bowling"].get("3wHauls"),
            "4wHauls": aggregated_data["bowling"].get("4wHauls"),
            "5wHauls": aggregated_data["bowling"].get("5wHauls"),
            "Best Bowling Figures": aggregated_data["bowling"].get("Best Bowling Figures")
        },
        "fielding": {
            "Catches": aggregated_data["fielding"].get("catches"),
            "Stumpings": aggregated_data["fielding"].get("stumpings"),
            "Run Outs": aggregated_data["fielding"].get("run_outs")
        },
        "dismissals": aggregated_data["dismissals"],
        "bowling_vs_batting_type": aggregated_data["bowling_vs_batting_type"]
    }

    safe_player_name = player_name.lower().replace(' ', '_')
    os.makedirs("../aggregated_stats", exist_ok=True)

    if sy is None and ey is None:
        out_filename = os.path.join(
            "../cricinfo_data_scrapping/aggregated_stats",
            f"{safe_player_name}_all_span_stats.json"
        )
    else:
        out_filename = os.path.join(
            "../cricinfo_data_scrapping/aggregated_stats",
            f"{safe_player_name}_{sy}_{ey}_span_stats.json"
        )

    with open(out_filename, 'w', encoding='utf-8') as f:
        json.dump(player_stats_json, f, indent=4, default=str)

    logging.info(f"Output written to: {out_filename}")
    return player_stats_json

def main():
    import sys
    if len(sys.argv) != 5:
        print("Usage: python get_player_stats_with_span.py <player_name> <team_name> <start_year> <end_year>")
        sys.exit(1)

    player_name = sys.argv[1]
    team_name = sys.argv[2]
    start_span = sys.argv[3]
    end_span = sys.argv[4]

    stats = generate_player_stats_with_span(
        player_name=player_name,
        team_name=team_name,
        start_year=start_span,
        end_year=end_span,
        scorecards_dir="../cricinfo_data_scrapping/match_scorecards_json/"
    )

    if stats:
        print(json.dumps(stats, indent=4, default=str))
    else:
        logging.warning("No stats generated with the given parameters.")

if __name__ == "__main__":
    main()
