import pandas as pd
import json
import re
import logging
import os
import unicodedata
import time
import random
from tqdm import tqdm

# -- Selenium Imports --
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# -- BeautifulSoup --
from bs4 import BeautifulSoup


# =========================
# Configuration and Setup
# =========================

# We only log failures in a concise way
logging.basicConfig(
    filename='scorecard_scraper.log',
    filemode='a',
    format='%(asctime)s - [SCRAPE FAILURE] %(message)s',  # custom format
    level=logging.ERROR  # only log at ERROR or above
)

output_dir = '../match_scorecards_json'
os.makedirs(output_dir, exist_ok=True)


# =========================
# Helper Functions
# =========================

def extract_match_id(url):
    match = re.search(r'-([0-9]+)/full-scorecard', url)
    return match.group(1) if match else "unknown_match_id"

def save_json(data, filename):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def clean_text(text):
    if not text:
        return ""
    text = text.replace('†', ' (wk)')
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii', errors='ignore')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def postprocess_did_not_bat(raw_text):
    if not raw_text:
        return ""
    raw_text = re.sub(r'(?i)^did not bat\s*:\s*', '', raw_text.strip())
    raw_text = re.sub(r'\s*,\s*', ' , ', raw_text)
    return raw_text.strip()

def postprocess_fow(raw_text):
    if not raw_text:
        return ""
    raw_text = re.sub(r'(?i)^fall of wickets\s*:\s*', '', raw_text.strip())
    # remove " • DRS" if present
    raw_text = raw_text.replace(' • DRS', '')
    raw_text = re.sub(r'(\d)-(\d)', r'\1 - \2', raw_text)
    raw_text = re.sub(r'\(', r'( ', raw_text)
    raw_text = re.sub(r'\)', r' )', raw_text)
    raw_text = re.sub(r',', r' , ', raw_text)
    raw_text = re.sub(r'\s+', ' ', raw_text).strip()
    return f"Fall of wickets : {raw_text}"

def parse_match_details(soup):
    details = {}
    heading = soup.find('span', string=re.compile('Match Details', re.IGNORECASE))
    if not heading:
        return details

    parent_div = heading.find_parent('div', class_=re.compile(r'ds-flex ds-px-4 ds-border-b'))
    if not parent_div:
        return details

    table_container = parent_div.find_next_sibling('div', class_='ds-p-0')
    if not table_container:
        return details

    table = table_container.find('table')
    if not table:
        return details

    rows = table.find_all('tr')
    for row in rows:
        tds = row.find_all('td')
        if len(tds) != 2:
            continue
        label = clean_text(tds[0].get_text(strip=True))
        value_td = tds[1]
        value = clean_text(value_td.get_text(separator=' ', strip=True))

        lower_label = label.lower()
        if 'toss' in lower_label:
            details['toss'] = value
        elif 'series' in lower_label:
            as_links = value_td.find_all('a')
            details['series'] = [clean_text(a.get_text(strip=True)) for a in as_links] if as_links else [value]
        elif 'season' in lower_label:
            details['season'] = value
        elif 'player of the match' in lower_label:
            potm_a = value_td.find('a')
            potm_name = clean_text(potm_a.get_text(strip=True)) if potm_a else value
            details['player_of_the_match'] = potm_name
        elif 'match number' in lower_label:
            details['match_number'] = value
        elif 'match days' in lower_label:
            details['match_days'] = value
        elif 'odi debut' in lower_label:
            debut_as = value_td.find_all('a')
            details['odi_debut'] = [clean_text(a.get_text(strip=True)) for a in debut_as] if debut_as else [value]
        elif 'umpires' == lower_label:
            umpire_as = value_td.find_all('a')
            details['umpires'] = [clean_text(a.get_text(strip=True)) for a in umpire_as]
        elif 'tv umpire' in lower_label:
            tv_ump_a = value_td.find('a')
            details['tv_umpire'] = clean_text(tv_ump_a.get_text(strip=True)) if tv_ump_a else value
        elif 'reserve umpire' in lower_label:
            reserve_a = value_td.find('a')
            details['reserve_umpire'] = clean_text(reserve_a.get_text(strip=True)) if reserve_a else value
        elif 'match referee' in lower_label:
            ref_a = value_td.find('a')
            details['match_referee'] = clean_text(ref_a.get_text(strip=True)) if ref_a else value
        elif 'points' in lower_label:
            details['points'] = value

    return details

def parse_match_info_for_venue_date(soup, match_data):
    match_info_div = soup.find('div', class_='ds-w-full ds-bg-fill-content-prime ds-overflow-hidden ds-rounded-xl ds-border ds-border-line')
    if not match_info_div:
        return

    description_div = match_info_div.find('div', class_='ds-text-tight-m ds-font-regular ds-text-typo-mid3')
    if not description_div:
        return

    description_text = clean_text(description_div.get_text(separator=', ', strip=True))
    parts = [p.strip() for p in description_text.split(',') if p.strip()]

    if len(parts) >= 2:
        match_data['match_details']['venue'] = parts[1]
    if len(parts) >= 3:
        match_data['match_details']['date'] = parts[2]

def parse_final_result(soup, match_data):
    no_result_strong = soup.find('strong', class_='ds-uppercase ds-text-tight-m', string=re.compile('NO RESULT', re.IGNORECASE))
    if no_result_strong:
        match_data['result'] = "No result"
        return

    possible_p = soup.find('p', class_='ds-text-tight-s ds-font-medium ds-truncate ds-text-typo')
    if possible_p:
        text = clean_text(possible_p.get_text(strip=True))
        if re.search(r'(win by|won by)', text, re.IGNORECASE):
            text = re.sub(r'win by', 'won by', text, flags=re.IGNORECASE)
            match_data['result'] = text
            return

    hidden_divs = soup.find_all('div', class_='ds-mb-2')
    for div in hidden_divs:
        text_all = clean_text(div.get_text(separator=' ', strip=True))
        match_result = re.search(r'(.*) win by (\d+) (runs|wickets)', text_all, re.IGNORECASE)
        if match_result:
            result_line = match_result.group(0)
            result_line = re.sub(r' win by ', ' won by ', result_line, flags=re.IGNORECASE)
            match_data['result'] = result_line
            return

def parse_no_result_playing_xi(soup, match_data):
    match_data["playing_xi"] = {}
    all_tds = soup.find_all('td', class_='!ds-py-2', colspan='9')
    for td in all_tds:
        strong_tag = td.find('strong', string=re.compile(r"Playing XI", re.IGNORECASE))
        if strong_tag:
            heading_text = clean_text(strong_tag.get_text(strip=True))
            team_name = re.sub(r'\s*Playing XI\s*:\s*', '', heading_text, flags=re.IGNORECASE).strip()

            a_tags = td.find_all('a')
            players = []
            for a in a_tags:
                player_name = clean_text(a.get_text(strip=True))
                player_name = player_name.rstrip(',')  # remove trailing comma if present
                players.append(player_name)

            match_data["playing_xi"][team_name] = players

def parse_scorecard_page(page_source, series_name, match_title, match_url):
    soup = BeautifulSoup(page_source, 'html.parser')

    match_data = {
        "series_name": series_name,
        "match_title": match_title,
        "match_url": match_url,
        "match_details": {},
        "innings": []
    }
    match_data['result'] = ""

    h1 = soup.find('h1')
    if h1:
        match_data['match_details']['header'] = clean_text(h1.get_text(separator=' ', strip=True))
    else:
        title_tag = soup.find('title')
        if title_tag:
            title_text = clean_text(title_tag.get_text(separator=' ', strip=True))
            match_data['match_details']['header'] = title_text
        else:
            match_data['match_details']['header'] = ""

    details = parse_match_details(soup)
    match_data['match_details'].update(details)

    parse_match_info_for_venue_date(soup, match_data)

    scoreboard_tables = soup.find_all('table', class_=re.compile(r'ci-scorecard-table'))
    if not scoreboard_tables:
        return match_data

    for table in scoreboard_tables:
        container_div = table.find_previous('div', class_=re.compile(r'ds-flex ds-px-4 ds-border-b'))
        team_name = ""
        innings_info = ""
        if container_div:
            team_span = container_div.find('span', class_='ds-text-title-xs ds-font-bold ds-capitalize')
            if team_span:
                team_name = clean_text(team_span.get_text(strip=True))
            info_span = container_div.find('span', class_='ds-text-compact-xs ds-font-regular')
            if info_span:
                innings_info = clean_text(info_span.get_text(strip=True))

        innings_data = {
            "team": team_name,
            "innings_info": innings_info,
            "runs": "",
            "overs": "",
            "extras": "",
            "batting": [],
            "did_not_bat": "",
            "fall_of_wickets": "",
            "bowling": []
        }

        body = table.find('tbody')
        if not body:
            continue
        rows = body.find_all('tr', recursive=False)
        for row in rows:
            if 'ds-hidden' in (row.get('class') or []):
                continue
            cols = row.find_all('td', recursive=False)
            if not cols:
                continue
            first_col_text = clean_text(cols[0].get_text(strip=True))
            num_cols = len(cols)

            if num_cols == 8:
                batsman = clean_text(cols[0].get_text(strip=True))
                dismissal = clean_text(cols[1].get_text(separator=' ', strip=True))
                runs = clean_text(cols[2].get_text(strip=True))
                balls = clean_text(cols[3].get_text(strip=True))
                mins = clean_text(cols[4].get_text(strip=True))
                fours = clean_text(cols[5].get_text(strip=True))
                sixes = clean_text(cols[6].get_text(strip=True))
                sr = clean_text(cols[7].get_text(strip=True))
                innings_data['batting'].append({
                    "batsman": batsman,
                    "dismissal": dismissal,
                    "runs": runs,
                    "balls": balls,
                    "minutes": mins,
                    "fours": fours,
                    "sixes": sixes,
                    "strike_rate": sr
                })
            elif first_col_text.lower().startswith('extras'):
                if len(cols) >= 3:
                    extras_total = clean_text(cols[2].get_text(strip=True))
                    innings_data['extras'] = extras_total
            elif first_col_text.lower().startswith('total'):
                if len(cols) >= 3:
                    overs_text = clean_text(cols[1].get_text(separator=' ', strip=True))
                    runs_text = clean_text(cols[2].get_text(strip=True))
                    innings_data['overs'] = overs_text
                    innings_data['runs'] = runs_text
            elif first_col_text.lower().startswith('did not bat'):
                dnb_whole = row.get_text(separator=' ', strip=True)
                dnb_whole = re.sub(r'(?i)^did not bat\s*:\s*', '', dnb_whole).strip()
                innings_data['did_not_bat'] = dnb_whole
            elif first_col_text.lower().startswith('fall of wickets'):
                fow_whole = row.get_text(separator=' ', strip=True)
                fow_whole = re.sub(r'(?i)^fall of wickets\s*:\s*', '', fow_whole).strip()
                innings_data['fall_of_wickets'] = fow_whole

        next_table = table.find_next('table', class_=lambda x: x and 'ci-scorecard-table' not in x)
        if next_table:
            body_bowl = next_table.find('tbody')
            if body_bowl:
                bowl_rows = body_bowl.find_all('tr', recursive=False)
                for b_row in bowl_rows:
                    b_cols = b_row.find_all('td', recursive=False)
                    if len(b_cols) == 11:
                        bowler = clean_text(b_cols[0].get_text(strip=True))
                        overs_ = clean_text(b_cols[1].get_text(strip=True))
                        maidens = clean_text(b_cols[2].get_text(strip=True))
                        runs_c = clean_text(b_cols[3].get_text(strip=True))
                        wickets_ = clean_text(b_cols[4].get_text(strip=True))
                        economy = clean_text(b_cols[5].get_text(strip=True))
                        dots = clean_text(b_cols[6].get_text(strip=True))
                        fours_ = clean_text(b_cols[7].get_text(strip=True))
                        sixes_ = clean_text(b_cols[8].get_text(strip=True))
                        wides_ = clean_text(b_cols[9].get_text(strip=True))
                        noballs_ = clean_text(b_cols[10].get_text(strip=True))
                        innings_data['bowling'].append({
                            "bowler": bowler,
                            "overs": overs_,
                            "maidens": maidens,
                            "runs_conceded": runs_c,
                            "wickets": wickets_,
                            "economy": economy,
                            "dots": dots,
                            "fours": fours_,
                            "sixes": sixes_,
                            "wides": wides_,
                            "no_balls": noballs_
                        })

        innings_data['did_not_bat'] = postprocess_did_not_bat(innings_data['did_not_bat'])
        innings_data['fall_of_wickets'] = postprocess_fow(innings_data['fall_of_wickets'])

        match_data['innings'].append(innings_data)

    parse_final_result(soup, match_data)

    if match_data['result'].lower().startswith("no result"):
        parse_no_result_playing_xi(soup, match_data)

    return match_data


def main():
    try:
        csv_file_path = 'series_match_scorecard_urls.csv'
        if not os.path.exists(csv_file_path):
            print(f"CSV file not found at path: {csv_file_path}")
            return

        df = pd.read_csv(csv_file_path)
        total_matches = len(df)
        print(f"Total matches to process: {total_matches}")

        chrome_options = Options()
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/98.0.4758.102 Safari/537.36"
        )
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(options=chrome_options)

        for index, row in tqdm(df.iterrows(), total=total_matches, desc="Processing Matches"):
            series_name = row.get("Series Name", "")
            match_title = row.get("Match Title", "")
            match_url = row.get("Match URL", "")

            try:
                driver.get(match_url)
                time.sleep(3)

                try:
                    accept_btn = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(),'Accept')]"))
                    )
                    accept_btn.click()
                except TimeoutException:
                    pass

                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)

                page_source = driver.page_source
                match_data = parse_scorecard_page(page_source, series_name, match_title, match_url)

                match_id = extract_match_id(match_url)
                filename = f"match_{match_id}.json"
                save_json(match_data, filename)
                print(f"Match data saved for {match_title}")

                time.sleep(random.uniform(1.0, 2.0))

            except WebDriverException as e:
                # Only log errors: short reason
                logging.error(f"Failed for {match_title} => {match_url}. Reason: {e}")
                print(f"[ERROR] Selenium error for {match_title}: {e}")
                continue
            except Exception as e:
                logging.error(f"Failed for {match_title} => {match_url}. Reason: {e}")
                print(f"[ERROR] Error processing match {match_title}: {e}")
                continue

        driver.quit()

    except Exception as e:
        logging.error(f"CRITICAL startup failure: {e}")
        print(f"Critical error in main execution: {e}")


if __name__ == "__main__":
    main()
