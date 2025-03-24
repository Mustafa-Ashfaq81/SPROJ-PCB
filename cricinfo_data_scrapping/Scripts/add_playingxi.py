import os
import json
import re
from multiprocessing import Pool, cpu_count

def clean_name(name):
    """Removes unwanted annotations from names but keeps (c) and (wk)."""
    # Define a pattern to match parentheses that are not (c) or (wk)
    return re.sub(r'\((?!c\)|wk\)).*?\)', '', name).strip()

def process_match_file(file_path):
    """Processes a single match JSON file to extract and append (or merge) the playing XI for each team."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            match_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return

    # If 'innings' is missing or empty, skip
    innings = match_data.get('innings', [])
    if not innings:
        return

    # This dict will collect the new or updated playing XI data for all teams in this match
    new_playing_xi = {}

    for inning in innings:
        team_name = inning.get('team', '').strip()
        if not team_name:
            continue

        # Get batsmen names
        batting = inning.get('batting', [])
        batsmen = [clean_name(batsman.get('batsman', '')) for batsman in batting]

        # Get 'did_not_bat' players
        did_not_bat = inning.get('did_not_bat', '')
        did_not_bat_players = [
            clean_name(player) for player in did_not_bat.split(',')
        ] if did_not_bat else []

        # Combine to get playing XI (remove empty names)
        combined_xi = [p for p in (batsmen + did_not_bat_players) if p]

        # Keep them unique, preserving order
        seen = set()
        ordered_xi = []
        for player in combined_xi:
            # Strip trailing commas/spaces for cleanliness
            clean_player = player.rstrip(',').strip()
            if clean_player not in seen and clean_player != '':
                seen.add(clean_player)
                ordered_xi.append(clean_player)

        new_playing_xi[team_name] = ordered_xi

    # Now merge with existing 'playing_xi' (if any) in the JSON
    existing_playing_xi = match_data.setdefault('playing_xi', {})

    for team, xi_list in new_playing_xi.items():
        if team in existing_playing_xi:
            # If team already in existing_playing_xi, only add missing players
            current_xi = existing_playing_xi[team]
            # Normalize trailing commas/spaces
            current_xi_clean = [p.rstrip(',').strip() for p in current_xi]

            for player in xi_list:
                if player not in current_xi_clean:
                    current_xi_clean.append(player)

            existing_playing_xi[team] = current_xi_clean
        else:
            # Team is missing entirely; add it
            existing_playing_xi[team] = xi_list

    # Save the modified JSON data back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(match_data, f, indent=4)

def main():
    scorecards_dir = '../match_scorecards_json'  # Directory containing your JSON files
    if not os.path.isdir(scorecards_dir):
        print(f"Directory '{scorecards_dir}' does not exist.")
        return

    json_files = [
        os.path.join(scorecards_dir, filename)
        for filename in os.listdir(scorecards_dir)
        if filename.endswith('.json')
    ]

    print(f"Processing {len(json_files)} match files...")

    # Use multiprocessing to speed up processing
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_match_file, json_files)

    print("All files processed. The playing XI has been appended or merged for each JSON file.")

if __name__ == '__main__':
    main()
