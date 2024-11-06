import os
import json
import re
from collections import defaultdict
from tqdm import tqdm  # For progress bar
import logging

# Configure logging
logging.basicConfig(
    filename='bowler_mapping.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_bowler_full_name(dismissal):
    """
    Extracts the bowler's full name from the dismissal string.
    Examples:
        "c Dhoni b Yadav" -> "Yadav"
        "c sub (AM Phangiso) b Duminy" -> "Duminy"
        "c Raina b I Sharma" -> "I Sharma"
        "run out (Jadeja/Dhoni)" -> None
    """
    if ' b ' in dismissal.lower():
        match = re.search(r'\bb\s+([A-Za-z\.\-]+\s*[A-Za-z\.\-]*)', dismissal, re.IGNORECASE)
        if match:
            bowler_full_name = match.group(1).strip()
            return bowler_full_name
    return None

def extract_bowler_last_name(full_bowler_name):
    """
    Extracts the bowler's last name from the full bowler name.
    """
    if not full_bowler_name:
        return None
    name_parts = full_bowler_name.strip().split()
    if not name_parts:
        return None
    return name_parts[-1]

def create_bowler_lastname_mapping(bowling_data):
    """
    Creates a mapping from bowler last names to full bowler names.
    """
    mapping = defaultdict(list)
    for bowler in bowling_data:
        full_name = bowler.get('bowler', '').strip()
        if not full_name:
            continue
        last_name = extract_bowler_last_name(full_name)
        if last_name:
            mapping[last_name.lower()].append(full_name)
    return mapping

def process_json_file(file_path):
    """
    Processes a single JSON file to append bowler mappings.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logging.error(f"JSONDecodeError: Could not decode JSON from file {file_path}")
        return
    except Exception as e:
        logging.error(f"Error: {e} while reading file {file_path}")
        return

    bowler_mapping = {}
    unmatched_bowlers = set()

    innings = data.get('innings', [])
    for inning in innings:
        batting_team = inning.get('team', '').strip()
        bowling_data = inning.get('bowling', [])
        bowler_lastname_map = create_bowler_lastname_mapping(bowling_data)

        batting = inning.get('batting', [])
        for batsman in batting:
            dismissal = batsman.get('dismissal', '').strip()
            bowler_full_name = extract_bowler_full_name(dismissal)
            if bowler_full_name:
                bowler_last_name = extract_bowler_last_name(bowler_full_name)
                if bowler_last_name:
                    bowler_last_name_lower = bowler_last_name.lower()
                    if bowler_last_name_lower in bowler_lastname_map:
                        full_bowler_names = bowler_lastname_map[bowler_last_name_lower]
                        if len(full_bowler_names) == 1:
                            bowler_full_name_matched = full_bowler_names[0]
                            bowler_mapping[bowler_last_name] = bowler_full_name_matched
                        else:
                            bowler_full_name_matched = full_bowler_names[0]
                            bowler_mapping[bowler_last_name] = bowler_full_name_matched
                            logging.warning(f"Multiple bowlers with last name '{bowler_last_name}' in file {file_path}. Mapped to '{bowler_full_name_matched}'.")
                    else:
                        bowler_mapping[bowler_last_name] = None
                        unmatched_bowlers.add(bowler_last_name)
                else:
                    continue
            else:
                continue

    if unmatched_bowlers:
        for lname in unmatched_bowlers:
            logging.info(f"Unmatched bowler last name '{lname}' in file {file_path}.")

    data['bowler_mapping'] = bowler_mapping

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Updated bowler_mapping appended to file {file_path}.")
    except Exception as e:
        logging.error(f"Error writing updated JSON to file {file_path}: {e}")

def main():
    scorecards_dir = 'match_scorecards_json'

    if not os.path.isdir(scorecards_dir):
        print(f"Directory '{scorecards_dir}' does not exist. Please create it and add your JSON files.")
        return

    json_files = [os.path.join(scorecards_dir, file) for file in os.listdir(scorecards_dir) if file.endswith('.json')]

    if not json_files:
        print(f"No JSON files found in directory '{scorecards_dir}'.")
        return

    print(f"Found {len(json_files)} JSON files in directory '{scorecards_dir}'. Processing...")

    for file_path in tqdm(json_files, desc="Processing JSON files"):
        process_json_file(file_path)

    print("Bowler mapping extraction and appending completed. Check 'bowler_mapping.log' for details.")

if __name__ == "__main__":
    main()
