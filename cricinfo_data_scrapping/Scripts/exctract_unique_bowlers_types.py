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
    filename='bowler_scraper.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
BASE_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/"
HEADERS = {
    "User-Agent": "BowlerStyleScraper/2.0 (contact: your_email@example.com)"
}
CACHE_FILE = 'bowler_cache.json'
SEARCH_RESULTS_LIMIT = 5  # Number of search results to consider

# Comprehensive list of cricket-related categories for validation
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
    # Add more as needed
]

def load_cache():
    """
    Load the cache from the cache file if it exists.

    Returns:
        dict: The cached data.
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
    Save the cache to the cache file.

    Args:
        cache (dict): The cache data to save.
    """
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=4, ensure_ascii=False)
        logging.info("Cache saved successfully.")
    except Exception as e:
        logging.error(f"Error saving cache: {e}")

def get_bowler_names(directory):
    """
    Traverse through all JSON files in the given directory and extract unique bowler names.

    Args:
        directory (str): Path to the directory containing match JSON files.

    Returns:
        set: A set of unique bowler names.
    """
    bowlers = set()
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
                        for bowler in inning.get('bowling', []):
                            bowler_name = bowler.get('bowler', '').strip()
                            if bowler_name:
                                # Remove annotations like (c), (wk), etc.
                                bowler_name = re.sub(r'\(.*?\)', '', bowler_name).strip()
                                bowlers.add(bowler_name)
                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {e}")

    logging.info(f"Processed {file_count} files.")
    logging.info(f"Found {len(bowlers)} unique bowlers.")
    print(f"Total unique bowlers found: {len(bowlers)}")
    return bowlers

def search_wikipedia_pages(bowler_name):
    """
    Use the Wikipedia Search API to find relevant pages for the bowler.

    Args:
        bowler_name (str): The name of the bowler.

    Returns:
        list: A list of page titles from the search results.
    """
    search_query = f"{bowler_name} cricketer"  # Refined search query
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
            logging.warning(f"No search results found for bowler: {bowler_name}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error searching Wikipedia for {bowler_name}: {e}")
    return []

def get_page_categories(wikipedia_title):
    """
    Retrieve the categories of a Wikipedia page.

    Args:
        wikipedia_title (str): The title of the Wikipedia page.

    Returns:
        list: A list of category titles associated with the page.
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
                categories = [cat['title'] for cat in page_data['categories']]
                return categories
        logging.warning(f"No categories found for page: {wikipedia_title}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching categories for {wikipedia_title}: {e}")
    return []

def is_cricketer_page(categories):
    """
    Determine if a Wikipedia page is about a cricketer based on its categories.

    Args:
        categories (list): A list of category titles.

    Returns:
        bool: True if the page is about a cricketer, False otherwise.
    """
    for category in categories:
        for keyword in CRICKET_CATEGORIES:
            if keyword.lower() in category.lower():
                return True
    return False

def get_bowling_style(wikipedia_title):
    """
    Scrape the bowling style from a Wikipedia page given its title.

    Args:
        wikipedia_title (str): The title of the Wikipedia page.

    Returns:
        str: Bowling style if found, else "Unknown".
    """
    if not wikipedia_title:
        return "Unknown"
    
    wikipedia_url = BASE_WIKIPEDIA_URL + quote(wikipedia_title.replace(' ', '_'))
    try:
        response = requests.get(wikipedia_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Handle disambiguation pages
        if soup.find('table', {'id': 'disambigbox'}) or 'may refer to' in soup.text:
            logging.warning(f"Disambiguation page encountered for: {wikipedia_title}")
            return "Unknown"
        
        # Find the infobox
        infobox = soup.find('table', class_='infobox')
        if not infobox:
            logging.warning(f"No infobox found on page: {wikipedia_url}")
            return "Unknown"
        
        # Find the 'Bowling' row
        for row in infobox.find_all('tr'):
            header = row.find('th')
            if header and 'Bowling' in header.text:
                data = row.find('td')
                if data:
                    bowling_style = data.get_text(separator=' ', strip=True)
                    return bowling_style
        logging.warning(f"'Bowling' field not found on page: {wikipedia_url}")
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error for URL {wikipedia_url}: {http_err}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request exception for URL {wikipedia_url}: {req_err}")
    except Exception as e:
        logging.error(f"Error scraping URL {wikipedia_url}: {e}")
    
    return "Unknown"

def scrape_bowling_styles(bowlers, cache):
    """
    Scrape bowling styles for all bowlers using caching and progress bar.

    Args:
        bowlers (set): A set of unique bowler names.
        cache (dict): Cache dictionary to store previously fetched results.

    Returns:
        dict: A dictionary mapping bowler names to their bowling styles.
    """
    bowler_styles = {}
    total_bowlers = len(bowlers)
    
    with tqdm(total=total_bowlers, desc="Scraping Bowling Styles", unit="bowler") as pbar:
        for bowler in sorted(bowlers):
            if bowler in cache:
                bowler_styles[bowler] = cache[bowler]
                pbar.update(1)
                continue

            # Search for the Wikipedia pages
            wikipedia_titles = search_wikipedia_pages(bowler)
            if not wikipedia_titles:
                bowler_styles[bowler] = "Unknown"
                logging.warning(f"No Wikipedia pages found for bowler: {bowler}")
                pbar.update(1)
                time.sleep(1)
                continue

            # Iterate through search results to find a cricketer page
            found_cricketer = False
            for title in wikipedia_titles:
                categories = get_page_categories(title)
                if is_cricketer_page(categories):
                    style = get_bowling_style(title)
                    bowler_styles[bowler] = style
                    cache[bowler] = style  # Update cache
                    logging.info(f"Bowler: {bowler}, Style: {style}")
                    found_cricketer = True
                    break  # Stop after finding the first valid cricketer page

            if not found_cricketer:
                bowler_styles[bowler] = "Unknown"
                logging.warning(f"No valid cricketer page found for bowler: {bowler}")
            
            pbar.update(1)
            # Respect Wikipedia's servers by adding a delay
            time.sleep(1)  # 1 second delay
    
    return bowler_styles

def save_to_json(data, filename):
    """
    Save the data to a JSON file.

    Args:
        data (dict): The data to save.
        filename (str): The filename for the JSON file.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.info(f"Successfully saved data to {filename}")
    except Exception as e:
        logging.error(f"Error saving data to {filename}: {e}")

def main():
    # Directory containing the match JSON files
    directory = 'match_scorecards_json'  # Update this path if necessary
    
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' does not exist. Please check the path.")
        logging.error(f"Directory '{directory}' does not exist.")
        sys.exit(1)
    
    # Load cache
    cache = load_cache()
    
    # Step 1: Extract unique bowler names
    bowlers = get_bowler_names(directory)
    
    if not bowlers:
        print("No bowlers found in the provided JSON files.")
        logging.info("No bowlers found. Exiting script.")
        sys.exit(0)
    
    # Step 2: Scrape bowling styles from Wikipedia with caching and progress bar
    bowler_styles = scrape_bowling_styles(bowlers, cache)
    
    # Save updated cache
    save_cache(cache)
    
    # Step 3: Save the result to a JSON file
    save_to_json(bowler_styles, 'bowlers_with_styles.json')
    
    print("Scraping and data extraction completed. Results saved to 'bowlers_with_styles.json'.")

if __name__ == "__main__":
    main()
