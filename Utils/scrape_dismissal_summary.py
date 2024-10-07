import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import re
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Function to get player IDs from the match scorecard using Selenium
def get_player_ids_from_scorecard(scorecard_url):
    # Set up Selenium WebDriver
    options = Options()
    options.headless = True  # Run in headless mode (no browser window)
    driver = webdriver.Chrome(options=options)

    # Load the scorecard page
    driver.get(scorecard_url)

    # Wait until the player links are present
    try:
        WebDriverWait(driver, 30).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'a.ds-inline-flex.ds-items-start.ds-leading-none'))
        )
    except Exception as e:
        print("Error waiting for page to load:", e)
        driver.quit()
        return []

    # Get the page source
    page_source = driver.page_source

    # Close the driver
    driver.quit()

    # Parse the page source with BeautifulSoup
    soup = BeautifulSoup(page_source, 'html.parser')

    # Save the page source for debugging
    with open('scorecard_page.html', 'w', encoding='utf-8') as f:
        f.write(page_source)

    # Now find the player links within the scorecard
    player_links = soup.find_all('a', class_='ds-inline-flex ds-items-start ds-leading-none')
    print(f"Number of player links found: {len(player_links)}")

    player_ids = []
    player_names = []

    for link in player_links:
        href = link['href']
        text = link.text.strip()
        # For debugging
        # print(f"Processing link: {href}, text: {text}")
        # Use regex to find hrefs that match the player profile pattern
        match = re.search(r'/cricketers/[^/]+-(\d+)', href)
        if match:
            player_id = match.group(1)
            player_name = text.strip(',')  # Remove any trailing commas
            if player_id not in player_ids:
                print(f"Found player link: {href} with name: {player_name}")
                player_ids.append(player_id)
                player_names.append(player_name)

    # Combine player IDs and names into a list of tuples
    players = list(zip(player_ids, player_names))
    return players

# Function to get dismissal summary for a player
def get_player_dismissal_summary(player_id):
    url = f"https://stats.espncricinfo.com/ci/engine/player/{player_id}.html?class=2;template=results;type=batting;view=dismissal_summary"
    print(f"Fetching data from URL: {url}")

    # Set up requests session
    import requests
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to retrieve data for player ID {player_id}, Status Code: {response.status_code}")
        return []

    # Parse the page source with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all tables with class 'engineTable'
    tables = soup.find_all('table', class_='engineTable')

    # Identify the dismissal summary table
    dismissal_table = None
    for table in tables:
        if 'Dismissal summary' in table.text:
            dismissal_table = table
            break

    if not dismissal_table:
        print(f"No dismissal table found for player ID {player_id}")
        return []

    # Extract the table headers to determine the number of columns
    headers = dismissal_table.find('tr', class_='headlinks').find_all('th')
    header_texts = [header.text.strip() for header in headers]
    num_columns = len(header_texts)
    print(f"Number of columns in dismissal table: {num_columns}")
    # For debugging
    # print("Table headers:", header_texts)

    # Extract the relevant rows from the dismissal table
    dismissal_data = []
    rows = dismissal_table.find_all('tr', class_='data1')
    if not rows:
        print(f"No data rows found in dismissal table for player ID {player_id}")
        return []

    for row in rows:
        cols = row.find_all('td')
        if len(cols) == num_columns:
            row_data = [col.text.strip() for col in cols]
            dismissal_data.append(row_data)
        else:
            print(f"Row with unexpected number of columns ({len(cols)}): {row}")

    return dismissal_data, header_texts

# Main script execution
if __name__ == "__main__":
    # URL of the CT17 Final match scorecard
    scorecard_url = "https://www.espncricinfo.com/series/ct-2017-1022345/india-vs-pakistan-final-1022375/full-scorecard"

    # Get player IDs and names from the scorecard
    players = get_player_ids_from_scorecard(scorecard_url)

    # Print player IDs and names for verification
    print("\nPlayers found:")
    for player_id, player_name in players:
        print(f"{player_name}: {player_id}")

    # Prepare to collect all player data
    all_player_data = []
    columns = None  # We'll set this later based on the table headers

    # Collect dismissal data for each player
    for player_id, player_name in players:
        print(f"\nScraping data for {player_name} (ID: {player_id})")
        try:
            player_dismissal_summary, header_texts = get_player_dismissal_summary(player_id)
            for row in player_dismissal_summary:
                # Prepend player name and ID to each row
                all_player_data.append([player_name, player_id] + row)
            if not columns:
                # Set the columns with 'Player Name', 'Player ID', and the headers from the table
                columns = ['Player Name', 'Player ID'] + header_texts
        except Exception as e:
            print(f"Error fetching data for player ID {player_id}: {e}")
        time.sleep(1)  # Be respectful by adding delay to prevent being blocked

    # Convert the data to a Pandas DataFrame
    if all_player_data and columns:
        df = pd.DataFrame(all_player_data, columns=columns)
        # Save to CSV
        df.to_csv('ct17_final_squad_dismissal_summary.csv', index=False)
        print("\nData scraping completed and saved to 'ct17_final_squad_dismissal_summary.csv'")
    else:
        print("\nNo data was scraped. Please check the script and try again.")