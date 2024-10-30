# Usage: python get_scorecard.py, input: series_match_scorecard_urls.csv

import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import re
import logging
import os
import unicodedata
import time
import random
from tqdm import tqdm

# =========================
# Configuration and Setup
# =========================

# Configure logging
logging.basicConfig(
    filename='scorecard_scraper.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

# Create output directory if it doesn't exist
output_dir = 'match_scorecards_json'
os.makedirs(output_dir, exist_ok=True)

# =========================
# Helper Functions
# =========================

def extract_match_id(url):
    """
    Extract match ID from the URL using regex.
    """
    match = re.search(r'-([0-9]+)/full-scorecard', url)
    return match.group(1) if match else "unknown_match_id"

def save_json(data, filename):
    """
    Save the extracted data as a JSON file.
    """
    filepath = os.path.join(output_dir, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.info(f"Saved JSON data to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save JSON data to {filepath}: {e}")

def clean_text(text):
    """
    Clean the input text by removing unwanted Unicode characters and extra spaces.
    """
    if text:
        try:
            # Normalize unicode characters
            text = unicodedata.normalize('NFKD', text)
            # Remove non-ASCII characters
            text = text.encode('ascii', 'ignore').decode('ascii')
            # Replace multiple spaces with single space
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        except Exception as e:
            logging.error(f"Error cleaning text: {e}")
            return text.strip()
    return text

def parse_match_details(soup):
    """
    Extract match details from the provided HTML snippet.
    """
    match_details = {}

    # Locate the 'Match Details' section by finding the span with text 'Match Details'
    match_details_span = soup.find('span', string=re.compile('Match Details', re.I))
    if match_details_span:
        # The table is likely the next sibling after the parent div
        table = match_details_span.find_parent('div').find_next('table')
        if table:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if not cols:
                    continue  # Skip rows without <td>

                # Determine the type of row based on the first <td>
                label = clean_text(cols[0].get_text(strip=True).replace(':', ''))
                key_clean = label.lower().replace(' ', '_')

                # Handle different row structures based on label
                if label == 'Extras' and len(cols) >= 3:
                    # Extract extras total from the third <td>
                    extras_total = clean_text(cols[2].get_text(strip=True))
                    match_details['extras'] = extras_total
                    logging.debug(f"Extracted Extras: {extras_total}")

                elif label == 'Total' and len(cols) >= 3:
                    # Extract overs from the second <td>
                    overs_div = cols[1].find('div')
                    if overs_div:
                        overs_text = clean_text(overs_div.get_text(separator=' ', strip=True))
                        match_details['overs'] = overs_text
                        logging.debug(f"Extracted Overs: {overs_text}")

                    # Extract runs from the third <td>
                    runs_text = clean_text(cols[2].get_text(strip=True))
                    match_details['runs'] = runs_text
                    logging.debug(f"Extracted Runs: {runs_text}")

                elif label == 'Toss':
                    toss_text = clean_text(cols[1].get_text(separator=' ', strip=True))
                    match_details['toss'] = toss_text
                    logging.debug(f"Extracted Toss: {toss_text}")

                elif label == 'Series':
                    series_links = cols[1].find_all('a')
                    series = [clean_text(link.get_text(strip=True)) for link in series_links]
                    match_details['series'] = series
                    logging.debug(f"Extracted Series: {series}")

                elif label == 'Season':
                    season_text = clean_text(cols[1].get_text(strip=True))
                    match_details['season'] = season_text
                    logging.debug(f"Extracted Season: {season_text}")

                elif label == 'Player Of The Match':
                    potm_tag = cols[1].find('a')
                    if potm_tag:
                        potm_text = clean_text(potm_tag.get_text(strip=True))
                        match_details['player_of_the_match'] = potm_text
                        logging.debug(f"Extracted Player Of The Match: {potm_text}")
                    else:
                        potm_text = clean_text(cols[1].get_text(strip=True))
                        match_details['player_of_the_match'] = potm_text
                        logging.debug(f"Extracted Player Of The Match (No <a> tag): {potm_text}")

                elif label == 'Match number':
                    match_number_text = clean_text(cols[1].get_text(strip=True))
                    match_details['match_number'] = match_number_text
                    logging.debug(f"Extracted Match Number: {match_number_text}")

                elif label == 'Match days':
                    match_days_text = clean_text(cols[1].get_text(strip=True))
                    match_details['match_days'] = match_days_text
                    logging.debug(f"Extracted Match Days: {match_days_text}")

                elif label == 'ODI debut':
                    odi_debut_tags = cols[1].find_all('a')
                    odi_debut_players = [clean_text(tag.get_text(strip=True)) for tag in odi_debut_tags]
                    match_details['odi_debut'] = odi_debut_players
                    logging.debug(f"Extracted ODI Debut: {odi_debut_players}")

                elif label == 'Umpires':
                    umpire_tags = cols[1].find_all('a')
                    umpires = [clean_text(ump.get_text(strip=True)) for ump in umpire_tags]
                    match_details['umpires'] = umpires
                    logging.debug(f"Extracted Umpires: {umpires}")

                elif label == 'TV Umpire':
                    tv_umpire_tag = cols[1].find('a')
                    if tv_umpire_tag:
                        tv_umpire_text = clean_text(tv_umpire_tag.get_text(strip=True))
                        match_details['tv_umpire'] = tv_umpire_text
                        logging.debug(f"Extracted TV Umpire: {tv_umpire_text}")
                    else:
                        tv_umpire_text = clean_text(cols[1].get_text(strip=True))
                        match_details['tv_umpire'] = tv_umpire_text
                        logging.debug(f"Extracted TV Umpire (No <a> tag): {tv_umpire_text}")

                elif label == 'Match Referee':
                    referee_tag = cols[1].find('a')
                    if referee_tag:
                        referee_text = clean_text(referee_tag.get_text(strip=True))
                        match_details['match_referee'] = referee_text
                        logging.debug(f"Extracted Match Referee: {referee_text}")
                    else:
                        referee_text = clean_text(cols[1].get_text(strip=True))
                        match_details['match_referee'] = referee_text
                        logging.debug(f"Extracted Match Referee (No <a> tag): {referee_text}")

                elif label == 'Points':
                    points_text = clean_text(cols[1].get_text(strip=True))
                    match_details['points'] = points_text
                    logging.debug(f"Extracted Points: {points_text}")

                # Add more elif blocks here if there are additional fields to extract

    else:
        logging.warning("Match Details section not found.")

    return match_details

def parse_scorecard_page(page_source, series_name, match_title, match_url):
    """
    Parse the scorecard page and extract match details, batting, and bowling data.
    """
    soup = BeautifulSoup(page_source, 'html.parser')

    match_data = {
        "series_name": series_name,
        "match_title": match_title,
        "match_url": match_url,
        "match_details": {},
        "innings": []
    }

    # Extract match header
    match_header = soup.find('h1', class_='ds-text-title-xs ds-font-bold ds-mb-2 ds-m-1')
    if match_header:
        match_data['match_details']['header'] = clean_text(match_header.get_text(separator=' ', strip=True))
        logging.debug(f"Match Header: {match_data['match_details']['header']}")

    # Extract match info (venue, date, result)
    match_info_div = soup.find('div', class_='ds-w-full ds-bg-fill-content-prime ds-overflow-hidden ds-rounded-xl ds-border ds-border-line')
    if match_info_div:
        # Venue and date
        description_div = match_info_div.find('div', class_='ds-text-tight-m ds-font-regular ds-text-typo-mid3')
        if description_div:
            description_text = clean_text(description_div.get_text(separator=' ', strip=True))
            logging.debug(f"Description Text: {description_text}")
            # The description might be in the format: '1st Match (D/N), Birmingham, June 07, 2001, NatWest Series'
            parts = [part.strip() for part in description_text.split(',')]
            if len(parts) >= 3:
                match_data['match_details']['venue'] = parts[1]
                match_data['match_details']['date'] = parts[2]
                logging.debug(f"Venue: {match_data['match_details']['venue']}, Date: {match_data['match_details']['date']}")
            else:
                match_data['match_details']['venue'] = description_text
                logging.debug(f"Venue (alternative): {match_data['match_details']['venue']}")

        # Result
        result_p = match_info_div.find('p', class_='ds-text-tight-s ds-font-medium ds-truncate ds-text-typo')
        if result_p:
            match_data['result'] = clean_text(result_p.get_text(strip=True))
            logging.debug(f"Result: {match_data['result']}")

    # Extract Player of the Match
    potm_label = soup.find('span', string=re.compile('Player Of The Match', re.I))
    if potm_label:
        potm_parent = potm_label.find_parent('div').find_next_sibling('div')
        if potm_parent:
            player_name_tag = potm_parent.find('a')
            if player_name_tag:
                potm_text = clean_text(player_name_tag.get_text(strip=True))
                match_data['player_of_match'] = potm_text
                logging.debug(f"Player of Match: {match_data['player_of_match']}")

    # Extract Match Details section
    match_details = parse_match_details(soup)
    if match_details:
        match_data['match_details'].update(match_details)
        logging.debug(f"Match Details: {match_details}")

    # Extract innings data
    # Find all batting tables
    batting_tables = soup.find_all('table', class_='ci-scorecard-table')

    if not batting_tables:
        logging.error("No batting tables found.")
        return match_data

    for batting_table in batting_tables:
        innings_data = {
            "team": "",
            "innings_info": "",
            "runs": "",
            "overs": "",
            "extras": "",
            "batting": [],
            "did_not_bat": "",
            "fall_of_wickets": "",
            "bowling": []
        }

        # Find team name and innings info
        team_span = batting_table.find_previous('span', class_='ds-text-title-xs ds-font-bold ds-capitalize')
        if team_span:
            innings_data['team'] = clean_text(team_span.get_text(strip=True))
            innings_info_span = team_span.find_next_sibling('span', class_='ds-text-compact-xs ds-font-regular')
            if innings_info_span:
                innings_data['innings_info'] = clean_text(innings_info_span.get_text(strip=True))
            logging.debug(f"Innings Team: {innings_data['team']}, Innings Info: {innings_data['innings_info']}")
        else:
            logging.warning("Team name not found for an innings.")
            continue  # Skip this innings if team name is not found

        # Extract batting data
        batting_rows = batting_table.find('tbody').find_all('tr', recursive=False)
        for row in batting_rows:
            if 'ds-hidden' in row.get('class', []):
                continue  # Skip hidden rows
            cols = row.find_all('td')
            if not cols:
                continue  # Skip rows without <td>

            # Get the text in the first column
            first_col_text = clean_text(cols[0].get_text(strip=True))
            num_cols = len(cols)

            if num_cols == 8:
                # Batsman rows
                batsman_name = clean_text(cols[0].get_text(strip=True))
                dismissal = clean_text(cols[1].get_text(separator=' ', strip=True))
                runs = clean_text(cols[2].get_text(strip=True))
                balls = clean_text(cols[3].get_text(strip=True))
                minutes = clean_text(cols[4].get_text(strip=True))
                fours = clean_text(cols[5].get_text(strip=True))
                sixes = clean_text(cols[6].get_text(strip=True))
                strike_rate = clean_text(cols[7].get_text(strip=True))
                innings_data['batting'].append({
                    "batsman": batsman_name,
                    "dismissal": dismissal,
                    "runs": runs,
                    "balls": balls,
                    "minutes": minutes,
                    "fours": fours,
                    "sixes": sixes,
                    "strike_rate": strike_rate
                })
                logging.debug(f"Batting: {batsman_name}, Dismissal: {dismissal}, Runs: {runs}, Balls: {balls}, Strike Rate: {strike_rate}")

            elif first_col_text == 'Extras':
                # Handle "Extras" row
                if len(cols) >= 3:
                    extras_total = clean_text(cols[2].get_text(strip=True))
                    innings_data['extras'] = extras_total
                    logging.debug(f"Extras: {extras_total}")

            elif first_col_text == 'Total':
                # Handle "Total" row
                if len(cols) >= 3:
                    # Extract overs from the second <td>
                    overs_div = cols[1].find('div')
                    if overs_div:
                        overs_text = clean_text(overs_div.get_text(separator=' ', strip=True))
                    else:
                        overs_text = clean_text(cols[1].get_text(strip=True))
                    innings_data['overs'] = overs_text
                    logging.debug(f"Overs: {overs_text}")

                    # Extract runs from the third <td>
                    runs_text = clean_text(cols[2].get_text(strip=True))
                    innings_data['runs'] = runs_text
                    logging.debug(f"Total Runs: {runs_text}")

            elif first_col_text.startswith('Did not bat'):
                # Handle "Did not bat" row
                did_not_bat_text = clean_text(cols[0].get_text(separator=' ', strip=True))
                # Remove the 'Did not bat' prefix
                did_not_bat_text = did_not_bat_text.replace('Did not bat:', '').strip()
                innings_data['did_not_bat'] = did_not_bat_text
                logging.debug(f"Did Not Bat: {did_not_bat_text}")

            elif first_col_text.startswith('Fall of wickets'):
                # Handle "Fall of wickets" row
                fall_of_wickets_text = clean_text(cols[0].get_text(separator=' ', strip=True))
                # Remove the 'Fall of wickets' prefix
                fall_of_wickets_text = fall_of_wickets_text.replace('Fall of wickets:', '').strip()
                innings_data['fall_of_wickets'] = fall_of_wickets_text
                logging.debug(f"Fall of Wickets: {fall_of_wickets_text}")

            else:
                # Skip any other rows or handle additional cases if needed
                continue

        # Find the corresponding bowling table
        # Assuming bowling table immediately follows the batting table
        bowling_table = batting_table.find_next('table', class_=lambda x: x and 'ci-scorecard-table' not in x)
        if bowling_table:
            bowling_rows = bowling_table.find('tbody').find_all('tr', recursive=False)
            for row in bowling_rows:
                cols = row.find_all('td')
                if len(cols) == 11:
                    bowler_name = clean_text(cols[0].get_text(strip=True))
                    overs = clean_text(cols[1].get_text(strip=True))
                    maidens = clean_text(cols[2].get_text(strip=True))
                    runs_conceded = clean_text(cols[3].get_text(strip=True))
                    wickets = clean_text(cols[4].get_text(strip=True))
                    economy = clean_text(cols[5].get_text(strip=True))
                    dots = clean_text(cols[6].get_text(strip=True))
                    fours = clean_text(cols[7].get_text(strip=True))
                    sixes = clean_text(cols[8].get_text(strip=True))
                    wides = clean_text(cols[9].get_text(strip=True))
                    no_balls = clean_text(cols[10].get_text(strip=True))
                    innings_data['bowling'].append({
                        "bowler": bowler_name,
                        "overs": overs,
                        "maidens": maidens,
                        "runs_conceded": runs_conceded,
                        "wickets": wickets,
                        "economy": economy,
                        "dots": dots,
                        "fours": fours,
                        "sixes": sixes,
                        "wides": wides,
                        "no_balls": no_balls
                    })
                    logging.debug(f"Bowling: {bowler_name}, Overs: {overs}, Runs Conceded: {runs_conceded}, Wickets: {wickets}")
        else:
            logging.warning(f"Bowling table not found for team {innings_data['team']}.")

        # Append the innings data to match_data
        match_data['innings'].append(innings_data)
        logging.info(f"Completed data extraction for innings: {innings_data['team']}")

    return match_data

# =========================
# Main Scraping Logic
# =========================

def main():
    try:
        csv_file_path = 'series_match_scorecard_urls.csv'
        if not os.path.exists(csv_file_path):
            logging.critical(f"CSV file not found at path: {csv_file_path}")
            print(f"CSV file not found at path: {csv_file_path}")
            return

        data = pd.read_csv(csv_file_path)
        matches = data  # Process all matches in the CSV

        total_matches = len(matches)
        logging.info(f"Total matches to process: {total_matches}")
        print(f"Total matches to process: {total_matches}")

        session = requests.Session()  # Reuse TCP connections for efficiency

        # Iterate over matches with a progress bar
        for index, row in tqdm(matches.iterrows(), total=total_matches, desc="Processing Matches"):
            series_name = row['Series Name']
            match_title = row['Match Title']
            match_url = row['Match URL']

            logging.info(f"Processing match: {match_title} - {match_url}")

            try:
                # Rotate User-Agent for respectful scraping
                user_agents = [
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
                    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36',
                    # Add more user agents as needed
                ]
                headers = {'User-Agent': random.choice(user_agents)}
                response = session.get(match_url, headers=headers)

                if response.status_code == 200:
                    page_source = response.text
                    match_data = parse_scorecard_page(page_source, series_name, match_title, match_url)
                    match_id = extract_match_id(match_url)
                    filename = f"match_{match_id}.json"
                    save_json(match_data, filename)
                    print(f"Match data saved for {match_title}")
                else:
                    logging.error(f"Failed to retrieve page for {match_title}: Status code {response.status_code}")
                    print(f"Failed to retrieve page for {match_title}: Status code {response.status_code}")
                    continue

                time.sleep(1)  # Respectful scraping: sleep for 1 second between requests

            except Exception as e:
                logging.error(f"Error processing match {match_title} at {match_url}: {e}")
                logging.exception("Exception occurred during parsing.")
                print(f"Error processing match {match_title}: {e}")
                continue

    except Exception as e:
        logging.critical(f"Critical error in main execution: {e}")
        logging.exception("Critical exception occurred.")
        print(f"Critical error in main execution: {e}")

if __name__ == "__main__":
    main()
