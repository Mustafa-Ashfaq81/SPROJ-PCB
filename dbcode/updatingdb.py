import json
import psycopg2
import re
import traceback
import os
import sys
import time

# Database Connection
conn = psycopg2.connect(
    dbname="cricket_db",
    user="postgres",
    password="Csmajor@lums25",
    host="localhost",
    port="5432",
)
cursor = conn.cursor()


def log_error(json_file, error_message, exception=None):
    print(f"\u274c ERROR: {json_file} - {error_message}")
    with open("failed_files.log", "a") as log_file:
        log_file.write(f"{json_file} - {error_message}\n")
        if exception:
            log_file.write(f"{traceback.format_exc()}\n")
    with open("failed_matches.txt", "a") as failed_matches:
        failed_matches.write(f"{json_file} - {error_message}\n")


def log_dropped_match(json_file, reason):
    print(f"⚠️ DROPPED: {json_file} - {reason}")
    with open("dropped_matches.log", "a") as drop_file:
        drop_file.write(f"{json_file} - {reason}\n")
    with open("dropped_files.txt", "a") as file_log:
        file_log.write(f"{json_file}\n")


def process_new_match(json_file):
    file_name = os.path.basename(json_file)
    if not (file_name.startswith("match_") and file_name.endswith(".json")):
        reason = "Filename not in expected format (match_xxxx.json)"
        log_error(json_file, reason)
        log_dropped_match(json_file, reason)
        return

    match_id_str = file_name[len("match_") : -5]
    if not match_id_str.isdigit():
        reason = "Invalid match id in filename"
        log_error(json_file, reason)
        log_dropped_match(json_file, reason)
        return

    match_id = int(match_id_str)
    try:
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        log_error(json_file, "Malformed JSON", exception=True)
        log_dropped_match(json_file, "Malformed JSON")
        return

    # === NEW: Skip cancelled matches ===
    result = data.get("result", "").lower()
    if "cancelled" in result:
        log_dropped_match(json_file, f"Match cancelled: {data.get('result')}")
        return

    if "match_details" not in data:
        reason = "Missing key: match_details"
        log_error(json_file, reason)
        log_dropped_match(json_file, reason)
        return

    if not insert_match(data, match_id):
        log_dropped_match(json_file, f"Match {match_id} not inserted")
        return

    insert_players(data)
    insert_performances(data, match_id)
    conn.commit()
    print(f"\u2705 Successfully committed match data from {json_file}")


def clean_name(name):
    """
    Remove annotations like (c), (wk), sub(...), etc. Return a lowercase version.
    """
    cleaned = re.sub(r"\(.*?\)", "", name)
    return cleaned.strip().lower()


def generate_case_variants(name_str: str) -> set:
    """
    Produce variants in different letter cases for a given name.
    """
    name_str = name_str.strip()
    if not name_str:
        return set()
    return {
        name_str,
        name_str.lower(),
        name_str.upper(),
        name_str.capitalize(),
        name_str.title(),
    }


def generate_name_variants(full_name: str) -> set:
    """
    Generates various partial/case variants from a full name for partial matching.
    """
    full_name = full_name.strip()
    if not full_name:
        return set()

    name_parts = full_name.split()
    base_variants = set()

    base_variants.add(" ".join(name_parts).lower())
    base_variants.add("".join(name_parts).lower())

    if len(name_parts) >= 2:
        first_initials = "".join([part[0] for part in name_parts[:-1]]).lower()
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


def extract_fielder_and_bowler(dismissal_text, playing_xi):
    """
    Check the dismissal text to identify if it was 'caught by sub(... ) b ...',
    normal 'c ... b ...', stumping, run out, etc.
    Return: (fielder_name, dismissal_type, is_sub)
    """
    dismissal_text = dismissal_text.strip().lower()

    # c sub (Fielder) b Bowler
    m = re.match(r"c\s*sub\s*\((.+?)\)\s*b\s*(.+)", dismissal_text)
    if m:
        # This is a sub fielder
        return get_full_fielder_name(m.group(1), playing_xi), "caught", True

    caught = re.match(r"c\s+(.+?)\s+b\s+(.+)", dismissal_text)
    if caught:
        return get_full_fielder_name(caught.group(1), playing_xi), "caught", False

    stumped = re.match(r"st\s+(.+?)\s+b\s+(.+)", dismissal_text)
    if stumped:
        return get_full_fielder_name(stumped.group(1), playing_xi), "stumped", False

    runout = re.match(r"run out\s*\(?([^)]*)\)?", dismissal_text)
    if runout:
        f = runout.group(1).strip()
        return (
            (get_full_fielder_name(f, playing_xi), "run out", False)
            if f
            else (None, "run out", False)
        )

    if "c & b" in dismissal_text:
        # e.g. 'c & b Maxwell'
        bowler = dismissal_text.replace("c & b", "").strip()
        return bowler, "caught & bowled", False

    return None, None, False


def get_full_fielder_name(fielder_raw, playing_xi):
    """
    Finds and returns the *full name* of the fielder from playing XI, if partial match found;
    otherwise returns the raw text.
    """
    fielder_raw = fielder_raw.strip()
    for team in playing_xi.values():
        for player in team:
            if fielder_raw.lower() in player.lower():
                return player
    return fielder_raw


def insert_match(match_data, match_id):
    """Insert match details into matches table if it doesn't already exist.
    Returns True if the match is inserted, False otherwise."""
    cursor.execute("SELECT match_id FROM matches WHERE match_id = %s", (match_id,))
    if cursor.fetchone():
        return True  # Already exists; treat as successful

    if "playing_xi" not in match_data:
        print(f"❌ Missing 'playing_xi' in match {match_id}")
        log_error(f"match_{match_id}.json", "Missing key: playing_xi")
        return False

    team_names = list(match_data["playing_xi"].keys())
    if len(team_names) < 2:
        message = f"Not enough teams in playing XI for match {match_id}"
        print(f"⚠️ Skipping match {match_id} ({message})")
        log_dropped_match(f"match_{match_id}.json", message)
        return False

    team_1, team_2 = team_names[0], team_names[1]

    # Extract year from "header"
    header = match_data["match_details"].get("header", "")
    found_year = re.search(r"\b(\d{4})\b", header)
    year = int(found_year.group(1)) if found_year else None

    match_info = (
        match_id,
        match_data.get("series_name", ""),
        match_data.get("match_title", ""),
        match_data.get("match_url", ""),
        match_data["match_details"].get("venue", ""),
        match_data["match_details"].get("match_days", "").split(" ")[0],
        match_data["match_details"].get("toss", "").split(",")[0],
        match_data["match_details"].get("toss", "").split(",")[-1].strip(),
        match_data["match_details"].get("player_of_the_match", None),
        match_data.get("result", "Unknown"),
        team_1,
        team_2,
        json.dumps(match_data["playing_xi"]),
        year,  # new year column
    )

    cursor.execute(
        """
        INSERT INTO matches (match_id, series_name, match_title, match_url, venue, match_date, 
                             toss_winner, elected_to, player_of_match, match_result, 
                             team_1, team_2, playing_xi, year)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """,
        match_info,
    )

    print(f"✅ Inserted match {match_id}")
    return True


def insert_players(match_data):
    """Insert new players into players table if they don't exist."""
    players = set()

    for team in match_data["playing_xi"].values():
        players.update(team)

    for player in players:
        cleaned_player_name = re.sub(r"\s*\(wk\)|\s*\(c\)", "", player).strip()

        cursor.execute(
            "SELECT player_id FROM players WHERE player_name = %s",
            (cleaned_player_name,),
        )
        if cursor.fetchone():
            continue

        cursor.execute(
            "INSERT INTO players (player_name, teams) VALUES (%s, %s)",
            (cleaned_player_name, json.dumps([])),
        )


def safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def insert_performances(match_data, match_id):
    """Insert or update player performances for batting, bowling, and fielding."""
    playing_xi = match_data["playing_xi"]
    teams = list(playing_xi.keys())
    if len(teams) != 2:
        print("❌ ERROR: Could not determine both teams from playing XI!")
        return

    team_1, team_2 = teams

    for innings in match_data["innings"]:
        batting_team = innings["team"]
        bowling_team = team_2 if batting_team == team_1 else team_1

        # Process Batting Performances
        for player in innings["batting"]:
            batsman_raw = player["batsman"].strip()
            batsman_name = clean_name(batsman_raw)

            is_captain = "(c)" in batsman_raw
            is_wk = "(wk)" in batsman_raw

            try:
                cursor.execute(
                    "SELECT player_id FROM players WHERE LOWER(player_name) = %s",
                    (batsman_name,),
                )
                result = cursor.fetchone()

                if not result:
                    print(
                        f"❌ ERROR: Player '{batsman_name}' not found in players table!"
                    )
                    continue

                player_id = result[0]
                fielder_name, dismissal_type, is_sub = extract_fielder_and_bowler(
                    player["dismissal"], playing_xi
                )

                if fielder_name:
                    fielder_name = clean_name(fielder_name)

                performance_data = (
                    match_id,
                    player_id,
                    bowling_team,
                    is_captain,
                    is_wk,
                    safe_int(player.get("runs", 0)),
                    safe_int(player.get("balls", 0)),
                    round(safe_float(player.get("strike_rate", 0.0)), 2),
                    safe_int(player.get("fours", 0)),
                    safe_int(player.get("sixes", 0)),
                    player.get("dismissal", ""),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )

                cursor.execute(
                    """
                    INSERT INTO player_performances (match_id, player_id, team, is_captain, is_wk, runs, balls, strike_rate, 
                                                     fours, sixes, dismissal, overs, maidens, runs_conceded, wickets, economy, 
                                                     catches, stumpings, run_outs)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (match_id, player_id)
                    DO UPDATE SET 
                        runs = player_performances.runs + EXCLUDED.runs,
                        balls = player_performances.balls + EXCLUDED.balls,
                        strike_rate = CASE 
                            WHEN (player_performances.balls + EXCLUDED.balls) > 0 
                            THEN ROUND(100.0 * (player_performances.runs + EXCLUDED.runs) / (player_performances.balls + EXCLUDED.balls), 2) 
                            ELSE 0 
                        END,
                        fours = player_performances.fours + EXCLUDED.fours,
                        sixes = player_performances.sixes + EXCLUDED.sixes,
                        is_captain = EXCLUDED.is_captain,
                        is_wk = EXCLUDED.is_wk;
                """,
                    performance_data,
                )

                # Update fielding stats if we have a valid fielder
                if fielder_name:
                    cursor.execute(
                        "SELECT player_id, player_name FROM players WHERE LOWER(player_name) LIKE %s",
                        (f"%{fielder_name}%",),
                    )
                    fielder_results = cursor.fetchall()

                    if fielder_results:
                        for fielder_id, fielder_full_name in fielder_results:
                            # Ensure a row for the fielder exists
                            cursor.execute(
                                """
                                INSERT INTO player_performances (match_id, player_id, team, runs, balls, strike_rate, fours, sixes, dismissal,
                                                                overs, maidens, runs_conceded, wickets, economy, catches, stumpings, run_outs)
                                VALUES (%s, %s, %s, 0, 0, 0.0, 0, 0, '', 0, 0, 0, 0, 0, 0, 0, 0)
                                ON CONFLICT (match_id, player_id)
                                DO NOTHING;
                                """,
                                (match_id, fielder_id, batting_team),
                            )

                            # Now increment the specific fielding count
                            cursor.execute(
                                """
                                INSERT INTO player_performances (match_id, player_id, team, catches, stumpings, run_outs, is_sub)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (match_id, player_id)
                                DO UPDATE SET
                                    catches = player_performances.catches + EXCLUDED.catches,
                                    stumpings = player_performances.stumpings + EXCLUDED.stumpings,
                                    run_outs = player_performances.run_outs + EXCLUDED.run_outs,
                                    is_sub = EXCLUDED.is_sub;
                                """,
                                (
                                    match_id,
                                    fielder_id,
                                    batting_team,
                                    1 if dismissal_type == "caught" else 0,
                                    1 if dismissal_type == "stumped" else 0,
                                    1 if "run out" in dismissal_type else 0,
                                    is_sub,
                                ),
                            )

            except Exception as e:
                print(f"❌ ERROR processing batsman {batsman_raw}: {e}")

        # Process Bowling Performances
        for player in innings["bowling"]:
            bowler_name = clean_name(player["bowler"])

            try:
                cursor.execute(
                    "SELECT player_id FROM players WHERE LOWER(player_name) = %s",
                    (bowler_name,),
                )
                result = cursor.fetchone()

                if not result:
                    print(
                        f"❌ ERROR: Bowler '{bowler_name}' not found in players table!"
                    )
                    continue

                player_id = result[0]
                performance_data = (
                    match_id,
                    player_id,
                    batting_team,
                    False,
                    False,
                    0,
                    0,
                    0.0,
                    0,
                    0,
                    "",
                    float(player["overs"]),
                    int(player["maidens"]),
                    int(player["runs_conceded"]),
                    int(player["wickets"]),
                    round(float(player["economy"]), 2),
                    0,
                    0,
                    0,
                )

                cursor.execute(
                    """
                    INSERT INTO player_performances (match_id, player_id, team, is_captain, is_wk, runs, balls, strike_rate, 
                                                     fours, sixes, dismissal, overs, maidens, runs_conceded, wickets, economy, 
                                                     catches, stumpings, run_outs)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (match_id, player_id)
                    DO UPDATE SET 
                        overs = player_performances.overs + EXCLUDED.overs,
                        maidens = player_performances.maidens + EXCLUDED.maidens,
                        runs_conceded = player_performances.runs_conceded + EXCLUDED.runs_conceded,
                        wickets = player_performances.wickets + EXCLUDED.wickets;
                """,
                    performance_data,
                )

            except Exception as e:
                print(f"❌ ERROR processing bowler {bowler_name}: {e}")


# Directory containing the JSON files
MATCH_DIRECTORY = "match_scorecards_json"


def print_progress(current, total):
    bar_length = 40
    progress = current / total
    filled = int(bar_length * progress)
    sys.stdout.write(
        f"\r🔄 Progress: |{'█'*filled}{'-'*(bar_length-filled)}| {current}/{total}"
    )
    sys.stdout.flush()


def process_files():
    files = [f for f in os.listdir(MATCH_DIRECTORY) if f.endswith(".json")]
    total = len(files)
    if total == 0:
        print("⚠️ No JSON files found.")
        return

    print(f"📂 Found {total} files. Starting...")
    for i, fname in enumerate(files, 1):
        path = os.path.join(MATCH_DIRECTORY, fname)

        # -- Removed the old "if 'sub' in file content, skip" lines -- #
        # We now always process the match file, because we handle sub fielders.

        process_new_match(path)
        print_progress(i, total)
        time.sleep(0.1)

    print("\n🎉 All done!")


if __name__ == "__main__":
    process_files()
    cursor.close()
    conn.close()
