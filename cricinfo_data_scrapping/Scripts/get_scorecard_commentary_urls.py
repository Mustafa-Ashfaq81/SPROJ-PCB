# Usage: python get_scorecard_commentary_urls.py, input: odi_series_urls.csv

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import re

# Define the filtering function before using it
def filter_titles(title):
    """
    Returns False if the title contains any unwanted keywords or patterns.
    """
    # Convert title to uppercase for case-insensitive matching
    title_upper = title.upper()
    
    # Define keywords to exclude
    exclude_keywords = [
        'TEST', 'T20I', 'A TEAM', 'B TEAM', 
        'INDIA A', 'PAKISTAN SHAHEENS', 'U19', 'U23', 'U25',
        'SENIOR CHALLENGER', 'RESERVE', 'SECOND XI', 'EMERGING TEAMS', 'EMERGING'
    ]
    
    # Exclude matches with specific keywords
    for keyword in exclude_keywords:
        if keyword in title_upper:
            print(f"Excluding match due to keyword '{keyword}': {title}")
            return False

    # Specific regex patterns to exclude unwanted standalone 'A' and 'B'
    patterns = [
        r'\bRESERVE\b',               # Exclude 'RESERVE'
        r'\bEMERGING\b',              # Exclude 'EMERGING'
        r'\bU\d{2}\b',                # Exclude 'U19', 'U23', etc.
        r'\b(?<!POOL )(?<!GROUP )A\b',  # Exclude standalone 'A' not preceded by 'POOL' or 'GROUP'
        r'\b(?<!POOL )(?<!GROUP )B\b',  # Exclude standalone 'B' not preceded by 'POOL' or 'GROUP'
        r'\b(?<!POOL )(?<!GROUP )A\s',  # Exclude 'A ' not preceded by 'POOL' or 'GROUP'
        r'\b(?<!POOL )(?<!GROUP )B\s',  # Exclude 'B ' not preceded by 'POOL' or 'GROUP'
    ]
    
    for pattern in patterns:
        if re.search(pattern, title_upper):
            print(f"Excluding match due to pattern '{pattern}': {title}")
            return False
    
    return True


# Load the CSV
csv_file_path = 'odi_series_urls.csv'
data = pd.read_csv(csv_file_path)

# Set up Selenium with ChromeDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Lists to store the extracted data
full_scorecard_data = []
ball_by_ball_data = []

try:
    # Iterate through all the series URLs
    for index, row in data.iterrows():
        series_name = row['Series/Tournament']
        series_url = row['URL']
        
        print(f"Processing series: {series_name}")
        
        # Extract the year from the series URL
        series_year_match = re.search(r'(\d{4})', series_url)
        if series_year_match:
            series_year = series_year_match.group(1)
        else:
            print(f"Could not extract year from URL: {series_url}")
            continue  # Skip this series if year extraction fails
        
        # Extract the series identifier from the URL (if available)
        series_id_match = re.search(r'/series/([^/]+)/', series_url)
        if series_id_match:
            series_identifier = series_id_match.group(1)
        else:
            print(f"Could not extract series identifier from URL: {series_url}")
            continue  # Skip this series if identifier extraction fails
        
        # Navigate to the series URL
        driver.get(series_url)
        time.sleep(3)  # Wait for the page to load

        # Use XPath to find all elements that link to match scorecards
        match_elements = driver.find_elements(By.XPATH, '//a[contains(@href, "/full-scorecard")]')
        
        # Extract the URLs and clean match titles from URLs
        for match in match_elements:
            match_url = match.get_attribute('href')
            
            # Verify that the match URL is from the same year as the series
            if series_year not in match_url:
                print(f"Skipping unrelated match URL: {match_url}")
                continue  # Skip matches that don't belong to the current year
            
            # Extract the match title from the URL
            match_title_match = re.search(r'series/[^/]+/([^/]+)/full-scorecard', match_url)
            if match_title_match:
                match_title = match_title_match.group(1).replace('-', ' ').title()
            else:
                match_title = "Unknown Match"
            
            # Apply the filtering function
            if not filter_titles(match_title):
                # If the title doesn't pass the filter, skip it
                continue
            
            # Add the full-scorecard URL to one list
            full_scorecard_data.append([series_name, match_title, match_url])
            print(f"Match found: {match_title} - {match_url}")
            
            # Generate the ball-by-ball-commentary URL and add to another list
            commentary_url = match_url.replace('/full-scorecard', '/ball-by-ball-commentary')
            ball_by_ball_data.append([series_name, match_title, commentary_url])

finally:
    driver.quit()

# Save the scraped data to DataFrames
full_scorecard_df = pd.DataFrame(full_scorecard_data, columns=["Series Name", "Match Title", "Match URL"])
ball_by_ball_df = pd.DataFrame(ball_by_ball_data, columns=["Series Name", "Match Title", "Commentary URL"])

# Apply the cleaning function
# Use .copy() to avoid SettingWithCopyWarning
full_scorecard_df_cleaned = full_scorecard_df.copy()
ball_by_ball_df_cleaned = ball_by_ball_df.copy()

full_scorecard_df_cleaned = full_scorecard_df_cleaned[full_scorecard_df_cleaned['Match Title'].apply(filter_titles)]
ball_by_ball_df_cleaned = ball_by_ball_df_cleaned[ball_by_ball_df_cleaned['Match Title'].apply(filter_titles)]

# Remove potential duplicates
full_scorecard_df_cleaned = full_scorecard_df_cleaned.drop_duplicates()
ball_by_ball_df_cleaned = ball_by_ball_df_cleaned.drop_duplicates()

# Save the cleaned data to two separate CSV files
full_scorecard_df_cleaned.to_csv("series_match_scorecard_urls.csv", index=False)
ball_by_ball_df_cleaned.to_csv("series_match_commentary_urls.csv", index=False)

print("Cleaned and deduplicated match URLs for all series extracted and saved in two CSV files!")