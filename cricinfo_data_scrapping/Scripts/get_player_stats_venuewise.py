#!/usr/bin/env python3
# Usage: python get_player_stats_venuewise.py <player name> <team> <start_year> <end_date>
# Usage Example 1: python get_player_stats_venuewise.py "Babar Azam" Pakistan all all
# Usage Example 2: python get_player_stats_venuewise.py "Babar Azam" Pakistan 2015 "September 2 2024"

import os
import json
import re
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of recognized dismissal methods for counting
DISMISSAL_METHODS = [
    "bowled",
    "caught-fielder",
    "caught wk",
    "stump",
    "lbw",
    "hit wicket",
    "caught and bowled"
]

# -------------------------------------------------------------------
# Safe conversions
# -------------------------------------------------------------------
def safe_float_conversion(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int_conversion(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

# -------------------------------------------------------------------
# Date parsing
# -------------------------------------------------------------------
def parse_match_date(match_details):
    """
    Parses the match date from match_details
    using any of: 'match_days', 'date'+ 'season', 'header'.
    Returns a datetime or None if not found/parseable.
    """
    match_days = match_details.get('match_days', '')
    # e.g. "21 March 2023"
    date_pattern = r'(\d{1,2} \w+ \d{4})'
    found = re.search(date_pattern, match_days)
    if found:
        date_str = found.group(1)
        try:
            return datetime.strptime(date_str, '%d %B %Y')
        except ValueError:
            pass

    date_str = match_details.get('date', '')
    if date_str:
        # e.g. date = "March 21", season = "2023"
        season = match_details.get('season', '')
        full_date_str = f"{date_str} {season}".strip()
        try:
            return datetime.strptime(full_date_str, '%B %d %Y')
        except ValueError:
            pass

    header = match_details.get('header', '')
    found2 = re.search(date_pattern, header)
    if found2:
        date_str = found2.group(1)
        try:
            return datetime.strptime(date_str, '%d %B %Y')
        except ValueError:
            return None

    return None


# -------------------------------------------------------------------
# Robust date parsing (added)
# -------------------------------------------------------------------
def robust_parse_date(date_str):
    """
    Attempt to parse a textual date in a flexible manner, handling abbreviations
    (Sep, Sept), commas, etc. Returns a datetime or None if it can't parse.
    """
    # Remove commas, parentheses, etc. and go lowercase
    date_str = re.sub(r'[,\(\)]', '', date_str.strip().lower())

    # Common month abbreviations to unify
    month_map = {
        'jan': 'January', 'feb': 'February', 'mar': 'March', 'apr': 'April',
        'may': 'May', 'jun': 'June', 'jul': 'July', 'aug': 'August',
        'sep': 'September', 'sept': 'September', 'oct': 'October',
        'nov': 'November', 'dec': 'December'
    }

    parts = date_str.split()
    if len(parts) < 2:
        return None

    # If first part is a known abbreviation or partial, expand it
    maybe_month = parts[0][:3]
    if maybe_month in month_map:
        canonical = month_map[maybe_month]
        new_str = canonical + " " + " ".join(parts[1:])
        patterns = [
            "%B %d %Y",   # September 2 2024
            "%B %d %y",   # September 2 24
        ]
        for pat in patterns:
            try:
                return datetime.strptime(new_str, pat)
            except ValueError:
                pass

    # Fallback attempts
    fallback_patterns = [
        "%B %d %Y",  # September 2 2024
        "%b %d %Y",  # Sep 2 2024
        "%B %d %y",  # September 2 24
        "%b %d %y"   # Sep 2 24
    ]
    for pat in fallback_patterns:
        try:
            return datetime.strptime(" ".join(parts), pat)
        except ValueError:
            pass

    return None


def parse_date_range(start_year, end_date):
    """
    If either argument is 'all', return (None, None) => no filtering.
    Otherwise:
      - start_year is taken as an integer => Jan 1 of that year
      - end_date is parsed as a precise datetime
    If parsing fails => no filtering.
    """
    if isinstance(start_year, str) and start_year.lower() == "all":
        return (None, None)
    if isinstance(end_date, str) and end_date.lower() == "all":
        return (None, None)

    # Try parse start_year as int
    try:
        sy = int(start_year)
        start_dt = datetime(sy, 1, 1)
    except (ValueError, TypeError):
        return (None, None)

    # robust parse end_date
    ed = robust_parse_date(end_date)
    if not ed:
        return (None, None)

    return (start_dt, ed)


# -------------------------------------------------------------------
# Name cleaning & variants
# -------------------------------------------------------------------
def clean_name(name):
    """
    Removes annotations like (c), (wk), sub(...), etc.
    Then strip + lower. So "Babar Azam(c)" => "babar azam".
    """
    # Remove *anything* in parentheses:
    cleaned = re.sub(r'\(.*?\)', '', name)
    return cleaned.strip().lower()

def generate_case_variants(name_str: str) -> set:
    """
    E.g. 'Kohli' => 'Kohli','kohli','KOHLI','Kohli','KOHLI', etc.
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
    Generates partial/case variants for matching bowlers, e.g. "Virat Kohli"
    => 'virat kohli', 'viratkohli', 'v kohli', 'vkohli', 'kohli' + case variants
    """
    full_name = full_name.strip()
    if not full_name:
        return set()

    name_parts = full_name.split()
    base_variants = set()

    # (1) The full name (lowercased, spaced)
    base_variants.add(' '.join(name_parts).lower())
    # (2) The full name joined
    base_variants.add(''.join(name_parts).lower())

    # (3) If multiple parts, create "v kohli", etc.
    if len(name_parts) >= 2:
        first_initials = ''.join(part[0] for part in name_parts[:-1]).lower()
        last_name = name_parts[-1].lower()
        base_variants.add(f"{first_initials} {last_name}")
        base_variants.add(f"{first_initials}{last_name}")
        base_variants.add(last_name)
    else:
        base_variants.add(name_parts[0].lower())

    # Expand into case variants
    final_variants = set()
    for variant in base_variants:
        final_variants |= generate_case_variants(variant)

    return final_variants

# -------------------------------------------------------------------
# Bowler name + type resolution
# -------------------------------------------------------------------
def resolve_bowler_name(bowler_name, bowler_mapping):
    """
    Finds the full bowler name from 'bowler_mapping' (dict of partial-> list).
    If no match, returns the original name.
    """
    bowler_variants = generate_name_variants(bowler_name)
    possible = []
    for variant in bowler_variants:
        lk = variant.lower()
        if lk in bowler_mapping:
            possible.extend(bowler_mapping[lk])

    if possible:
        return possible[0]
    else:
        return bowler_name

def find_bowler_type(bowler_name_full, innings):
    """
    Look for bowler_name_full in each inning's 'bowling' list,
    returning 'b_type' if found. Else 'Unknown'.
    """
    b_lower = bowler_name_full.lower()
    for inning in innings:
        for binfo in inning.get('bowling', []):
            if binfo.get('bowler','').lower() == b_lower:
                return binfo.get('b_type', 'Unknown')
    return 'Unknown'

# -------------------------------------------------------------------
# Fielding contribution parsing
# -------------------------------------------------------------------
def parse_fielding_contribution(dismissal_text, player_name_clean):
    """
    Checks if 'player_name_clean' is the fielder (caught, stumped, run out).
    Returns increments in {"catches", "stumpings", "run_outs"}.
    """
    out_ = {"catches": 0, "stumpings": 0, "run_outs": 0}
    dt_low = dismissal_text.lower().strip()
    player_variants = generate_name_variants(player_name_clean)

    # 1) c & b <NAME>
    c_and_b = re.search(r'c\s+&\s+b\s+([\w\s]+)', dt_low, re.IGNORECASE)
    if c_and_b:
        fielder = clean_name(c_and_b.group(1))
        if generate_name_variants(fielder) & player_variants:
            out_["catches"] += 1
        return out_

    # 2) c FIELDER b BOWLER
    pat_caught = re.compile(r'^c\s+(.*?)\s+b\s+([\w\s]+)$', re.IGNORECASE)
    cmtch = pat_caught.search(dt_low)
    if cmtch:
        fielder_raw = re.sub(r'\bsub\b','', cmtch.group(1), flags=re.IGNORECASE)
        fielder_clean = clean_name(fielder_raw)
        if generate_name_variants(fielder_clean) & player_variants:
            out_["catches"] += 1

    # 3) st FIELDER b BOWLER
    pat_stump = re.compile(r'st\s+(.*?)\s+b\s+([\w\s]+)', re.IGNORECASE)
    st_match = pat_stump.search(dt_low)
    if st_match:
        st_fielder_raw = re.sub(r'\bsub\b','', st_match.group(1), flags=re.IGNORECASE)
        st_fielder_clean = clean_name(st_fielder_raw)
        if generate_name_variants(st_fielder_clean) & player_variants:
            out_["stumpings"] += 1

    # 4) run out( FIELDER ), possibly multiple
    pat_runout = re.compile(r'run out\s*\(([\w\s/]+)\)', re.IGNORECASE)
    ro = pat_runout.search(dt_low)
    if ro:
        block = ro.group(1)
        potential_fielders = [clean_name(x) for x in block.split('/')]
        for f_ in potential_fielders:
            if generate_name_variants(f_) & player_variants:
                out_["run_outs"] += 1
                break
    return out_

# -------------------------------------------------------------------
# Extract how a batsman got out (method + bowler)
# -------------------------------------------------------------------
def extract_dismissal_data(batsman_clean, dismissal_text, innings, bowler_mapping, wicket_keepers):
    """
    Returns { 'bowler_type':'...', 'dismissal_method':'...', 'bowler_name':'...'}.
    """
    bowler_type = 'Unknown'
    dismissal_method = 'other'
    bowler_name_full = 'Unknown'

    # Patterns for typical dismissals
    patterns = [
        (re.compile(r'c\s+\(wk\)\s*([\w\s]+)\s+b\s+([\w\s]+)', re.IGNORECASE), 'caught wk'),
        (re.compile(r'c\s+([\w\s]+)\s+b\s+([\w\s]+)', re.IGNORECASE), 'caught-fielder'),
        (re.compile(r'st\s+\(wk\)\s*([\w\s]+)\s+b\s+([\w\s]+)', re.IGNORECASE), 'stump'),
        (re.compile(r'st\s+([\w\s]+)\s+b\s+([\w\s]+)', re.IGNORECASE), 'stump'),
        (re.compile(r'lbw\s+b\s+([\w\s]+)', re.IGNORECASE), 'lbw'),
        (re.compile(r'c\s+&\s+b\s+([\w\s]+)', re.IGNORECASE), 'caught and bowled'),
        (re.compile(r'hit wicket\s+b\s+([\w\s]+)', re.IGNORECASE), 'hit wicket'),
        (re.compile(r'^b\s+([\w\s]+)$', re.IGNORECASE), 'bowled'),
        # Run out
        (re.compile(r'run out\s+\(([\w\s]+)\)\s+b\s+([\w\s]+)', re.IGNORECASE), 'run out'),
        (re.compile(r'run out\s+\(([\w\s/]+)\)', re.IGNORECASE), 'run out'),
        (re.compile(r'run out', re.IGNORECASE), 'run out'),
        (re.compile(r'retired hurt', re.IGNORECASE), 'retired hurt'),
        (re.compile(r'retired not out', re.IGNORECASE), 'retired not out'),
        (re.compile(r'obstructing the field', re.IGNORECASE), 'obstructing the field'),
        (re.compile(r'timed out', re.IGNORECASE), 'timed out'),
        (re.compile(r'handled the ball', re.IGNORECASE), 'handled the ball'),
        (re.compile(r'hit the ball twice', re.IGNORECASE), 'hit the ball twice'),
    ]

    dt_low = dismissal_text.lower().strip()
    for pat, method in patterns:
        found = pat.search(dt_low)
        if found:
            if method == 'caught wk':
                # c (wk) X b Y or c(wk) Y b Z
                # group(1) = fielder, group(2)= bowler
                bowler = found.group(2).strip() if len(found.groups()) >= 2 else ''
                dismissal_method = 'caught wk'
                bowler_name_full = resolve_bowler_name(bowler, bowler_mapping)

            elif method == 'caught-fielder':
                # c FIELDER b BOWLER
                fielder = found.group(1).strip()
                bowler = found.group(2).strip()
                # If fielder is a recognized wk => 'caught wk'
                if clean_name(fielder) in wicket_keepers:
                    dismissal_method = 'caught wk'
                else:
                    dismissal_method = 'caught-fielder'
                bowler_name_full = resolve_bowler_name(bowler, bowler_mapping)

            elif method == 'caught and bowled':
                # c & b BOWLER
                bowler = found.group(1).strip()
                dismissal_method = 'caught and bowled'
                bowler_name_full = resolve_bowler_name(bowler, bowler_mapping)

            elif method == 'stump':
                # st FIELDER b BOWLER
                fielder = found.group(1).strip() if len(found.groups())>=2 else ''
                bowler = found.group(2).strip() if len(found.groups())>=2 else ''
                dismissal_method = 'stump'
                bowler_name_full = resolve_bowler_name(bowler, bowler_mapping)

            elif method in ['lbw','bowled','hit wicket']:
                bowler = found.group(1).strip()
                dismissal_method = method
                bowler_name_full = resolve_bowler_name(bowler, bowler_mapping)

            elif method == 'run out':
                if len(found.groups()) == 2:
                    # run out(X) b Y
                    # group(1)=X, group(2)=Y
                    bowler = found.group(2).strip()
                    dismissal_method = 'run out'
                    bowler_name_full = resolve_bowler_name(bowler, bowler_mapping)
                elif len(found.groups()) == 1:
                    # run out(X)
                    dismissal_method = 'run out'
                    bowler_name_full = 'Unknown'
                else:
                    dismissal_method = 'run out'
                    bowler_name_full = 'Unknown'

            else:
                # retired hurt, obstructing, etc.
                dismissal_method = method
                bowler_name_full = 'Unknown'

            # Once matched, get bowler_type
            bowler_type = find_bowler_type(bowler_name_full, innings)
            break

    return {
        'bowler_type': bowler_type,
        'dismissal_method': dismissal_method,
        'bowler_name': bowler_name_full
    }

# -------------------------------------------------------------------
# Convert overs float => total balls
# -------------------------------------------------------------------
def float_overs_to_balls(overs_val):
    """
    E.g. 9.5 => 59 balls, 9.6 => 60 balls.
    """
    whole = int(overs_val)
    decimal_part = overs_val - whole
    possible_balls = int(round(decimal_part * 10))
    if possible_balls >= 6:
        return (whole+1)*6
    return whole*6 + possible_balls

# -------------------------------------------------------------------
# parse_match_file (updated with fine-grained date range)
# -------------------------------------------------------------------
def parse_match_file(file_path, player_name, team_name, start_dt=None, end_dt=None):
    """
    Returns stats for that match if the given player was in the playing XI
    for 'team_name' AND match_date is in [start_dt, end_dt], if those are not None.
    If 'all' => no date filtering. If match_date is outside range => skip.
    """

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            match_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        logging.error(f"Failed to parse {file_path}")
        return None

    match_details = match_data.get('match_details', {})
    match_date = parse_match_date(match_details)

    # -----------------------------
    # NEW: date-range filtering
    # -----------------------------
    if start_dt is not None and end_dt is not None:
        # If match_date is None, or out of range => skip
        if (not match_date) or (match_date < start_dt or match_date > end_dt):
            return None

    # Prepare the aggregator for a single match
    player_stats = {
        'batting': {},
        'bowling': {
            'balls_bowled': 0,
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

    # Clean the player's name
    player_name_clean = clean_name(player_name)
    team_name_lower = team_name.lower()

    playing_xi = match_data.get('playing_xi', {})
    # Also clean playing XI so if the scorecard has 'Babar Azam(c)' in playing_xi, we remove (c).
    for tkey in list(playing_xi.keys()):
        playing_xi[tkey] = [clean_name(x) for x in playing_xi[tkey]]

    # Check if player in that team's XI
    if team_name not in playing_xi:
        return None
    if player_name_clean not in playing_xi[team_name]:
        return None

    player_stats['played'] = True

    # Identify wicketkeepers (we can't reliably do it if parentheses are removed,
    # but we keep the logic for consistency)
    wicket_keepers = set()
    for t, p_list in playing_xi.items():
        for name_ in p_list:
            pass  # In some data, we might detect WK info. If not, we skip.

    # Build a bowler mapping for partial name => full
    bowler_mapping = defaultdict(list)
    innings = match_data.get('innings', [])
    for inn in innings:
        for binfo in inn.get('bowling', []):
            original_bowler = binfo.get('bowler','')
            for variant in generate_name_variants(original_bowler):
                bowler_mapping[variant.lower()].append(original_bowler)

    # Parsing the innings
    for inn in innings:
        inn_team = inn.get('team','').lower()
        # If inn_team == player's team => batting
        if inn_team == team_name_lower:
            for batsman in inn.get('batting', []):
                raw_name = batsman.get('batsman','')
                bname_clean = clean_name(raw_name)
                if bname_clean == player_name_clean:
                    # capture batting
                    b_runs = safe_int_conversion(batsman.get('runs',0))
                    b_balls= safe_int_conversion(batsman.get('balls',0))
                    b_fours= safe_int_conversion(batsman.get('fours',0))
                    b_sixes= safe_int_conversion(batsman.get('sixes',0))
                    b_sr   = safe_float_conversion(batsman.get('strike_rate',0.0))
                    dis_text = batsman.get('dismissal','not out')

                    player_stats['batting'] = {
                        'runs': b_runs,
                        'balls': b_balls,
                        'fours': b_fours,
                        'sixes': b_sixes,
                        'strike_rate': b_sr,
                        'dismissal': dis_text
                    }
                    if dis_text.lower() != 'not out':
                        dd = extract_dismissal_data(bname_clean, dis_text, innings, bowler_mapping, wicket_keepers)
                        player_stats['dismissals'].append(dd)
        else:
            # bowling
            for binfo in inn.get('bowling', []):
                bclean = clean_name(binfo.get('bowler',''))
                if bclean == player_name_clean:
                    overs_val = safe_float_conversion(binfo.get('overs',0.0))
                    total_balls = float_overs_to_balls(overs_val)
                    wicks = safe_int_conversion(binfo.get('wickets',0))
                    runs_c = safe_int_conversion(binfo.get('runs_conceded',0))
                    fours_ = safe_int_conversion(binfo.get('fours',0))
                    sixes_ = safe_int_conversion(binfo.get('sixes',0))
                    
                    ps_bowl = player_stats['bowling']
                    ps_bowl['balls_bowled'] += total_balls
                    ps_bowl['maidens'] += safe_int_conversion(binfo.get('maidens',0))
                    ps_bowl['runs_conceded'] += runs_c
                    ps_bowl['wickets'] += wicks
                    ps_bowl['economy_total'] += safe_float_conversion(binfo.get('economy',0.0))
                    ps_bowl['dots'] += safe_int_conversion(binfo.get('dots',0))
                    ps_bowl['fours'] += fours_
                    ps_bowl['sixes'] += sixes_
                    ps_bowl['wides'] += safe_int_conversion(binfo.get('wides',0))
                    ps_bowl['no_balls'] += safe_int_conversion(binfo.get('no_balls',0))
                    ps_bowl['boundary_runs_this_innings'] += 4*fours_ + 6*sixes_
                    ps_bowl['match_wickets'] = wicks
                    ps_bowl['match_runs'] = runs_c

        # fielding + (bowling vs batting type)
        for bat_data in inn.get('batting', []):
            dtext = bat_data.get('dismissal','').strip()
            if dtext:
                # fielding
                fc = parse_fielding_contribution(dtext, player_name_clean)
                player_stats['fielding']['catches'] += fc['catches']
                player_stats['fielding']['stumpings'] += fc['stumpings']
                player_stats['fielding']['run_outs'] += fc['run_outs']

                # was our player the credited bowler?
                dd = extract_dismissal_data(clean_name(bat_data.get('batsman','')),
                                            dtext,
                                            innings, bowler_mapping, wicket_keepers)
                credited_bowler_clean = clean_name(dd['bowler_name'])
                method_ = dd['dismissal_method'].lower()
                if credited_bowler_clean == player_name_clean:
                    # we add a wicket vs batting type
                    bat_type = bat_data.get('batting_type','').lower()
                    if 'left' in bat_type:
                        style_key = 'Left Handed'
                    elif 'right' in bat_type:
                        style_key = 'Right Handed'
                    else:
                        style_key = 'Unknown'

                    player_stats['bowling_vs_batting_type'][style_key]['wickets'] += 1
                    if method_ in {
                        'bowled','caught-fielder','caught wk','stump',
                        'lbw','hit wicket','caught and bowled'
                    }:
                        player_stats['bowling_vs_batting_type'][style_key][method_] += 1
                    else:
                        player_stats['bowling_vs_batting_type'][style_key]['other'] += 1

    return player_stats

# -------------------------------------------------------------------
# Initialize aggregator for a venue
# -------------------------------------------------------------------
def init_aggregator():
    bowler_types = ["Right arm Pace","Right arm Spin","Left arm Pace","Left arm Spin","Unknown"]
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
            'balls_bowled': 0,
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
            bt: {"Dis":0, **{dm:0 for dm in DISMISSAL_METHODS}}
            for bt in bowler_types
        },
        "match_dates": [],
        "matches_played": 0,
        "batting_innings": 0,
        "bowling_innings": 0,
        "fielding_innings": 0
    }
    return aggregator

# -------------------------------------------------------------------
# Merge single-match stats into aggregator
# -------------------------------------------------------------------
def merge_single_match_stats(agg, match_stats):
    if match_stats.get('played'):
        agg["matches_played"] += 1
        md = match_stats.get('match_date')
        if md:
            agg['match_dates'].append(md)

    # batting
    bdict = match_stats.get('batting', {})
    if bdict:
        agg['batting_innings'] += 1
        agg['batting']['runs'] += bdict.get('runs',0)
        agg['batting']['balls']+= bdict.get('balls',0)
        agg['batting']['fours']+= bdict.get('fours',0)
        agg['batting']['sixes']+= bdict.get('sixes',0)
        agg['batting']['strike_rate_total'] += bdict.get('strike_rate',0.0)
        if bdict.get('dismissal','').lower() != 'not out':
            agg['batting']['dismissals'] += 1
        else:
            agg['batting']['not_outs'] += 1

    # bowling
    bowld = match_stats.get('bowling', {})
    if bowld:
        agg['bowling_innings'] += 1
        agg['bowling']['balls_bowled'] += bowld.get('balls_bowled',0)
        agg['bowling']['maidens'] += bowld.get('maidens',0)
        agg['bowling']['runs_conceded'] += bowld.get('runs_conceded',0)
        agg['bowling']['wickets'] += bowld.get('wickets',0)
        agg['bowling']['economy_total'] += bowld.get('economy_total',0.0)
        agg['bowling']['dots'] += bowld.get('dots',0)
        agg['bowling']['fours'] += bowld.get('fours',0)
        agg['bowling']['sixes'] += bowld.get('sixes',0)
        agg['bowling']['wides'] += bowld.get('wides',0)
        agg['bowling']['no_balls'] += bowld.get('no_balls',0)
        agg['bowling']['total_boundary_runs'] += bowld.get('boundary_runs_this_innings',0)

        # best figures
        mw = bowld.get('match_wickets',0)
        mr = bowld.get('match_runs',999999)
        if mw>0:
            cur_best_w = agg['bowling']['best_bowling_wickets']
            cur_best_r = agg['bowling']['best_bowling_runs']
            if (mw>cur_best_w) or (mw==cur_best_w and mr<cur_best_r):
                agg['bowling']['best_bowling_wickets'] = mw
                agg['bowling']['best_bowling_runs'] = mr

        if mw==3:
            agg['bowling']['three_wicket_hauls']+=1
        elif mw==4:
            agg['bowling']['four_wicket_hauls']+=1
        elif mw>=5:
            agg['bowling']['five_wicket_hauls']+=1

    # fielding
    fd = match_stats.get('fielding',{})
    if fd:
        agg['fielding_innings'] += 1
        agg['fielding']['catches'] += fd.get('catches',0)
        agg['fielding']['stumpings'] += fd.get('stumpings',0)
        agg['fielding']['run_outs'] += fd.get('run_outs',0)

    # who got me out?
    for d_ in match_stats.get('dismissals', []):
        bt = d_.get('bowler_type','Unknown')
        if bt not in agg['dismissals']:
            bt = 'Unknown'
        dm_ = d_.get('dismissal_method','other').lower()
        if dm_ in DISMISSAL_METHODS:
            agg['dismissals'][bt][dm_]+=1
        agg['dismissals'][bt]["Dis"]+=1

    # bowling vs batting type
    mvbt = match_stats.get('bowling_vs_batting_type',{})
    if 'bvbt' not in agg:
        agg['bvbt'] = {
            'Right Handed': {
                'wickets': 0,
                'bowled': 0,
                'caught-fielder':0,
                'caught wk':0,
                'stump':0,
                'lbw':0,
                'hit wicket':0,
                'caught and bowled':0,
                'other':0
            },
            'Left Handed': {
                'wickets': 0,
                'bowled': 0,
                'caught-fielder':0,
                'caught wk':0,
                'stump':0,
                'lbw':0,
                'hit wicket':0,
                'caught and bowled':0,
                'other':0
            },
            'Unknown': {
                'wickets': 0,
                'bowled': 0,
                'caught-fielder':0,
                'caught wk':0,
                'stump':0,
                'lbw':0,
                'hit wicket':0,
                'caught and bowled':0,
                'other':0
            }
        }
    for style_key in ['Right Handed','Left Handed','Unknown']:
        st_dict = mvbt.get(style_key,{})
        agg['bvbt'][style_key]['wickets'] += st_dict.get('wickets',0)
        for meth in ['bowled','caught-fielder','caught wk','stump',
                     'lbw','hit wicket','caught and bowled','other']:
            agg['bvbt'][style_key][meth] += st_dict.get(meth,0)

# -------------------------------------------------------------------
# Finalize aggregator -> JSON-friendly
# -------------------------------------------------------------------
def finalize_aggregator(agg):
    # batting average
    if agg['batting']['dismissals']>0:
        agg['batting']['bat_avg'] = round(agg['batting']['runs']/agg['batting']['dismissals'],2)
    else:
        agg['batting']['bat_avg'] = None

    # strike rate
    if agg['batting']['balls']>0:
        sr_ = (agg['batting']['runs']/agg['batting']['balls'])*100
        agg['batting']['strike_rate'] = round(sr_,2)
    else:
        agg['batting']['strike_rate'] = None

    # overs
    tb = agg['bowling']['balls_bowled']
    ov_int = tb//6
    leftover = tb%6
    corrected_overs = ov_int + leftover/10.0

    # bowling average + economy
    w_ = agg['bowling']['wickets']
    r_ = agg['bowling']['runs_conceded']
    if w_>0:
        agg['bowling']['bowl_avg'] = round(r_/w_,2)
    else:
        agg['bowling']['bowl_avg'] = None

    if tb>0:
        eco = r_/(tb/6)
        agg['bowling']['economy'] = round(eco,2)
    else:
        agg['bowling']['economy'] = None

    # bowling strike rate
    if w_>0 and tb>0:
        bsr = round(tb/w_,2)
    else:
        bsr = None

    # dot ball %
    if tb>0:
        dbp = round((agg['bowling']['dots']/tb)*100,2)
    else:
        dbp = None

    # best figures
    if agg['bowling']['best_bowling_wickets']>0 and agg['bowling']['best_bowling_runs']<999999:
        best_fig = f"{agg['bowling']['best_bowling_wickets']}/{agg['bowling']['best_bowling_runs']}"
    else:
        best_fig = None

    # career span
    if agg['match_dates']:
        earliest = min(agg['match_dates'])
        latest  = max(agg['match_dates'])
        span_str= f"{earliest.year} - {latest.year}"
    else:
        span_str= "Unknown"

    # dismissal data
    bowler_types = ["Right arm Pace","Right arm Spin","Left arm Pace","Left arm Spin","Unknown"]
    disp_list = []
    for btype in bowler_types:
        ddata = agg['dismissals'][btype]
        disp_list.append({
            "Bowler Type": btype,
            "Span": span_str,
            "Dis": ddata["Dis"],
            "bowled": ddata["bowled"],
            "caught-fielder": ddata["caught-fielder"],
            "caught wk": ddata["caught wk"],
            "stump": ddata["stump"],
            "lbw": ddata["lbw"],
            "hit wicket": ddata["hit wicket"],
            "caught and bowled": ddata["caught and bowled"],
        })

    # bowling_vs_batting_type
    final_bvbt = {}
    if 'bvbt' in agg:
        for skey in ['Right Handed','Left Handed','Unknown']:
            sdict = agg['bvbt'][skey]
            final_bvbt[skey] = {
                "wickets": sdict.get("wickets",0),
                "bowled": sdict.get("bowled",0),
                "caught-fielder": sdict.get("caught-fielder",0),
                "caught wk": sdict.get("caught wk",0),
                "stump": sdict.get("stump",0),
                "lbw": sdict.get("lbw",0),
                "hit wicket": sdict.get("hit wicket",0),
                "caught and bowled": sdict.get("caught and bowled",0),
                "other": sdict.get("other",0)
            }

    final_json = {
        "span": span_str,
        "matches_played": agg["matches_played"],
        "batting_innings": agg["batting_innings"],
        "bowling_innings": agg["bowling_innings"],
        "fielding_innings": agg["fielding_innings"],
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
            "Overs": round(corrected_overs,2),
            "Maidens": agg['bowling']['maidens'],
            "Runs Conceded": agg['bowling']['runs_conceded'],
            "Wickets": agg['bowling']['wickets'],
            "Economy Rate": agg['bowling']['economy'],
            "Bowling Average": agg['bowling']['bowl_avg'],
            "Dots": agg['bowling']['dots'],
            "Fours": agg['bowling']['fours'],
            "Sixes": agg['bowling']['sixes'],
            "Wides": agg['bowling']['wides'],
            "No Balls": agg['bowling']['no_balls'],
            "Boundary Runs Conceded": agg['bowling']['total_boundary_runs'],
            "Bowling Strike Rate": bsr,
            "Dot Ball Percentage": dbp,
            "3wHauls": agg['bowling']['three_wicket_hauls'],
            "4wHauls": agg['bowling']['four_wicket_hauls'],
            "5wHauls": agg['bowling']['five_wicket_hauls'],
            "Best Bowling Figures": best_fig
        },
        "fielding": {
            "Catches": agg['fielding']['catches'],
            "Stumpings": agg['fielding']['stumpings'],
            "Run Outs": agg['fielding']['run_outs']
        },
        "dismissals": disp_list,
        "bowling_vs_batting_type": final_bvbt
    }
    return final_json

# -------------------------------------------------------------------
# parse_country_from_series_name
# -------------------------------------------------------------------
def parse_country_from_series_name(series_name):
    """
    For "Pakistan in South Africa ODI Series" => "South Africa".
    If nothing found => "Unknown".
    """
    sn_low = series_name.lower()
    match_ = re.search(r'\bin\s+([a-zA-Z\s]+)', sn_low)
    if match_:
        country_str = match_.group(1).strip()
        # remove e.g. "odi", "t20i", "test", "series" etc
        country_str = re.sub(r'(odi|t20i|test|\d+|series|match)','', country_str, flags=re.IGNORECASE)
        country_str = country_str.strip()
        if country_str.lower() == "st":
            return "West Indies"
        if country_str:
            return " ".join(word.capitalize() for word in country_str.split())
    return "Unknown"

# -------------------------------------------------------------------
# process_file => for multiprocessing
# -------------------------------------------------------------------
def process_file(args):
    fp, player_name, team_name, start_dt, end_dt = args
    stats = parse_match_file(fp, player_name, team_name, start_dt, end_dt)
    return (fp, stats)

# -------------------------------------------------------------------
# Main function: generate_player_stats_venuewise_span
# -------------------------------------------------------------------
def generate_player_stats_venuewise_span(
    player_name,
    team_name,
    start_year="all",
    end_date="all",
    # scorecards_dir="../cricinfo_data_scrapping/match_scorecards_json/"
    scorecards_dir="../match_scorecards_json/"
):
    """
    Exactly like your original aggregator, but with fine-grained date filtering:
      - <start_year> is either "all" or an integer year => (Year, Jan 1).
      - <end_date> is either "all" or a textual date => the precise end date.
    If either is "all", we skip date filtering.
    """

    # (1) parse the date range
    start_dt, end_dt = parse_date_range(start_year, end_date)

    # collect all JSON
    if not os.path.isdir(scorecards_dir):
        logging.error(f"Directory not found: {scorecards_dir}")
        return {}

    json_files = [
        os.path.join(scorecards_dir, f)
        for f in os.listdir(scorecards_dir) if f.endswith('.json')
    ]
    if not json_files:
        logging.warning(f"No JSON files found in {scorecards_dir}")
        return {}

    # parallel parse
    args_list = []
    for f_ in json_files:
        args_list.append((f_, player_name, team_name, start_dt, end_dt))

    with Pool(processes=cpu_count()) as p:
        results = p.map(process_file, args_list)

    # keep only non-None
    all_stats = [(fp, stats) for fp, stats in results if stats is not None]

    if not all_stats:
        logging.warning(
            f"No match data found for {player_name} in {team_name} "
            f"within the date range {start_year}-{end_date} (or no data)."
        )
        return {}

    # group by parse_country_from_series_name
    aggregator_by_venue = {}
    for fp, ms in all_stats:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                mdata = json.load(f)
        except Exception:
            continue
        sname = mdata.get('series_name', '')
        venue = parse_country_from_series_name(sname)
        if venue not in aggregator_by_venue:
            aggregator_by_venue[venue] = init_aggregator()
        merge_single_match_stats(aggregator_by_venue[venue], ms)

    # finalize
    final_out = {
        "player": player_name,
        "team": team_name,
        "venues": {}
    }
    for ven,agg in aggregator_by_venue.items():
        final_out["venues"][ven] = finalize_aggregator(agg)

    # optionally write out JSON
    safe_pname = player_name.lower().replace(' ','_')
    out_dir = "../aggregated_stats_venue_wise"
    # out_dir = "../cricinfo_data_scrapping/aggregated_stats_venue_wise"
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir)
        except:
            pass

    if start_dt is None and end_dt is None:
        fname = f"{safe_pname}_all_venuewise.json"
    else:
        # embed the start year and the Y-m-d of end_dt in the file name
        fname = (
            f"{safe_pname}_{start_dt.year}_"
            f"{end_dt.strftime('%Y_%m_%d')}_venuewise.json"
        )

    outfile = os.path.join(out_dir, fname)
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(final_out, f, indent=4, default=str)

    logging.info(f"Venue-wise stats (with fine-grained date filter) => {outfile}")
    return final_out

# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
def main():
    import sys
    if len(sys.argv) != 5:
        print("Usage: python get_player_stats_venuewise.py <player_name> <team> <start_year> <end_date>")
        sys.exit(1)
    player = sys.argv[1]
    team = sys.argv[2]
    start_year = sys.argv[3]
    end_date = sys.argv[4]

    result = generate_player_stats_venuewise_span(player, team, start_year, end_date)
    if not result:
        logging.warning("No results or empty data.")
    else:
        logging.info("Finished generating venue-wise stats with fine-grained date filter.")

if __name__ == "__main__":
    main()
