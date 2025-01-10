import os
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import time
import logging
from tqdm import tqdm
import sys
import re

# Configure logging
logging.basicConfig(
    filename='batter_scraper.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Wikipedia constants
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
BASE_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/"
HEADERS = {
    "User-Agent": "BatterStyleScraper/1.0 (contact: your_email@example.com)"
}

# Cache file
CACHE_FILE = 'batter_cache.json'

# How many search results to consider from Wikipedia
SEARCH_RESULTS_LIMIT = 5

# Cricket categories: used to check that the page is about a cricketer
CRICKET_CATEGORIES = [
    'Cricketers',
    'Cricket players',
    'West Indian cricketers',
    'Kenyan cricketers',
    'Indian cricketers',
    'Australian cricketers',
    'English cricketers',
    'South African cricketers',
    'Pakistani cricketers',
    'New Zealand cricketers',
    'Sri Lankan cricketers',
    'Bangladeshi cricketers',
    'Afghan cricketers',
    'Irish cricketers',
    'Zimbabwean cricketers',
    'Namibian cricketers',
    'United States cricketers',
    'Canadian cricketers',
    'Emirati cricketers',
    'Dutch cricketers',
    'Hong Kong cricketers',
    'Singapore cricketers',
    'Nepalese cricketers',
    'Malaysian cricketers',
    'South African cricketers',
    # ... add more as needed
]

def load_cache():
    """
    Load the cache from the JSON file if it exists.
    """
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            logging.info("Loaded cache successfully.")
            return cache
        except Exception as e:
            logging.error(f"Error loading cache: {e}")
    return {}

def save_cache(cache):
    """
    Save the cache to the JSON file.
    """
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=4, ensure_ascii=False)
        logging.info("Cache saved successfully.")
    except Exception as e:
        logging.error(f"Error saving cache: {e}")

def get_batter_names(directory):
    """
    Traverse all JSON files in the directory and extract unique batter names.
    This assumes your match JSON files have a structure like:
    {
        "innings": [
            {
                "batting": [
                    {"batsman": "John Doe", ...}, ...
                ]
            }
        ]
    }

    Adjust key names if your JSON structure differs.
    """
    batters = set()
    file_count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                file_count += 1
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        match_data = json.load(f)
                    
                    for inning in match_data.get('innings', []):
                        # Grab batting info
                        for batter in inning.get('batting', []):
                            batter_name = batter.get('batsman', '').strip()
                            if batter_name:
                                # Remove any annotations like (c), (wk), etc.
                                batter_name = re.sub(r'\(.*?\)', '', batter_name).strip()
                                batters.add(batter_name)
                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {e}")

    logging.info(f"Processed {file_count} files.")
    logging.info(f"Found {len(batters)} unique batters.")
    print(f"Total unique batters found: {len(batters)}")
    return batters

def search_wikipedia_pages(name):
    """
    Use the Wikipedia Search API to find relevant page titles.
    Appends "cricketer" to the search for better results.
    """
    search_query = f"{name} cricketer"
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': search_query,
        'format': 'json',
        'srlimit': SEARCH_RESULTS_LIMIT
    }
    try:
        response = requests.get(WIKIPEDIA_API_URL, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        search_results = data.get('query', {}).get('search', [])
        page_titles = [result['title'] for result in search_results]
        if page_titles:
            return page_titles
        else:
            logging.warning(f"No search results found for: {name}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error searching Wikipedia for {name}: {e}")
    return []

def get_page_categories(wikipedia_title):
    """
    Retrieve the categories of a Wikipedia page by title.
    """
    params = {
        'action': 'query',
        'prop': 'categories',
        'titles': wikipedia_title,
        'format': 'json',
        'cllimit': 'max'
    }
    try:
        response = requests.get(WIKIPEDIA_API_URL, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        pages = data.get('query', {}).get('pages', {})
        for page_id, page_data in pages.items():
            if 'categories' in page_data:
                return [cat['title'] for cat in page_data['categories']]
        logging.warning(f"No categories found for page: {wikipedia_title}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching categories for {wikipedia_title}: {e}")
    return []

def is_cricketer_page(categories):
    """
    Check whether page categories indicate this is a cricketer.
    """
    for category in categories:
        for keyword in CRICKET_CATEGORIES:
            if keyword.lower() in category.lower():
                return True
    return False

def get_batting_style(wikipedia_title):
    """
    Scrape the batting style (e.g., 'Right-handed', 'Left-handed', etc.)
    from a Wikipedia page infobox.
    """
    if not wikipedia_title:
        return "Unknown"
    
    wikipedia_url = BASE_WIKIPEDIA_URL + quote(wikipedia_title.replace(' ', '_'))
    try:
        response = requests.get(wikipedia_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Check for disambiguation
        if soup.find('table', {'id': 'disambigbox'}) or 'may refer to' in soup.get_text():
            logging.warning(f"Disambiguation page encountered: {wikipedia_title}")
            return "Unknown"
        
        # Find the infobox
        infobox = soup.find('table', class_='infobox')
        if not infobox:
            logging.warning(f"No infobox found on page: {wikipedia_url}")
            return "Unknown"
        
        # Search rows for "Batting" label
        for row in infobox.find_all('tr'):
            header = row.find('th')
            if header and 'Batting' in header.text:
                data = row.find('td')
                if data:
                    batting_style = data.get_text(separator=' ', strip=True)
                    return batting_style
        logging.warning(f"'Batting' field not found on page: {wikipedia_url}")
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error for URL {wikipedia_url}: {http_err}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error for URL {wikipedia_url}: {req_err}")
    except Exception as e:
        logging.error(f"Error scraping {wikipedia_url}: {e}")
    
    return "Unknown"

def scrape_batting_styles(batters, cache):
    """
    Main scraping loop: for each batter, find a Wikipedia page, confirm 
    they are a cricketer, then extract the batting style.
    Uses a progress bar and respects Wikipedia's server with a delay.
    """
    batter_styles = {}
    total_batters = len(batters)
    
    with tqdm(total=total_batters, desc="Scraping Batting Styles", unit="batter") as pbar:
        for batter in sorted(batters):
            if batter in cache:
                batter_styles[batter] = cache[batter]
                pbar.update(1)
                continue

            # Search for Wikipedia pages
            wikipedia_titles = search_wikipedia_pages(batter)
            if not wikipedia_titles:
                batter_styles[batter] = "Unknown"
                logging.warning(f"No Wikipedia pages found for: {batter}")
                pbar.update(1)
                time.sleep(1)
                continue

            # Look for a valid cricketer page among the search results
            found_cricketer = False
            for title in wikipedia_titles:
                categories = get_page_categories(title)
                if is_cricketer_page(categories):
                    style = get_batting_style(title)
                    batter_styles[batter] = style
                    cache[batter] = style  # Update cache
                    logging.info(f"Batter: {batter}, Style: {style}")
                    found_cricketer = True
                    break

            if not found_cricketer:
                batter_styles[batter] = "Unknown"
                logging.warning(f"No valid cricketer page found for: {batter}")
            
            pbar.update(1)
            # Be polite to Wikipedia servers
            time.sleep(1)
    
    return batter_styles

def save_to_json(data, filename):
    """
    Save data to a JSON file.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.info(f"Successfully saved data to {filename}")
    except Exception as e:
        logging.error(f"Error saving data to {filename}: {e}")

def main():

    parent_directory = os.path.dirname(os.path.abspath(__file__))

    directory = os.path.join(parent_directory, '..', 'match_scorecards_json')
        
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' does not exist. Please check the path.")
        logging.error(f"Directory '{directory}' does not exist.")
        sys.exit(1)
    
    # Load existing cache
    cache = load_cache()
    
    # 1. Extract unique batter names
    batters = get_batter_names(directory)
    
    if not batters:
        print("No batters found in the provided JSON files.")
        logging.info("No batters found. Exiting script.")
        sys.exit(0)
    
    # 2. Scrape batting styles with caching
    batter_styles = scrape_batting_styles(batters, cache)
    
    # Update cache
    save_cache(cache)
    
    # 3. Save results to JSON
    output_file = 'batters_with_styles.json'
    save_to_json(batter_styles, output_file)
    
    print(f"Scraping complete. Results saved to '{output_file}'.")

if __name__ == "__main__":
    main()
