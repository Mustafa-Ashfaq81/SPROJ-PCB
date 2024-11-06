import os
import json
import re
from multiprocessing import Pool, cpu_count

def clean_name(name):
    """ Removes annotations like (c), (wk), etc., from names and strips whitespace. """
    return re.sub(r'\(.*?\)', '', name).strip()

def process_match_file(file_path):
    """ Processes a single match JSON file to extract and append the playing XI for each team. """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            match_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return

    playing_xi = {}
    innings = match_data.get('innings', [])
    for inning in innings:
        team_name = inning.get('team', '')
        if not team_name:
            continue

        # Get batsmen names
        batting = inning.get('batting', [])
        batsmen = [clean_name(batsman.get('batsman', '')) for batsman in batting]

        # Get did_not_bat players
        did_not_bat = inning.get('did_not_bat', '')
        did_not_bat_players = [clean_name(name) for name in did_not_bat.split(',')] if did_not_bat else []

        # Combine to get playing XI
        team_playing_xi = batsmen + did_not_bat_players

        # Remove empty strings and duplicates
        team_playing_xi = list(set([name for name in team_playing_xi if name]))

        # Add to playing_xi dictionary
        playing_xi[team_name] = team_playing_xi

    # Append the playing XI to the match data
    match_data['playing_xi'] = playing_xi

    # Save the modified JSON data back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(match_data, f, indent=4)

def main():
    scorecards_dir = 'match_scorecards_json'  # Directory containing your JSON files
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

    print("All files processed. The playing XI has been appended to each JSON file.")

if __name__ == '__main__':
    main()
