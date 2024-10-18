import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import re

# Load the CSV
csv_file_path = 'Series and Their URLS - Sheet1.csv'
data = pd.read_csv(csv_file_path)

# Set up Selenium with ChromeDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# List to store the extracted data
output_data = []

try:
    # Iterate through all the series URLs
    for index, row in data.iterrows():
        series_name = row['Series/Tournament']
        series_url = row['URL']
        
        print(f"Processing series: {series_name}")
        
        # Navigate to the series URL
        driver.get(series_url)
        time.sleep(3)  # Wait for the page to load

        # Use XPath to find all elements that link to match scorecards
        match_elements = driver.find_elements(By.XPATH, '//a[contains(@href, "/full-scorecard")]')
        
        # Extract the URLs and clean match titles from URLs
        for match in match_elements:
            match_url = match.get_attribute('href')
            # Extract the match title from the URL
            match_title = re.search(r'series/[^/]+/([^/]+)/full-scorecard', match_url)
            if match_title:
                match_title = match_title.group(1).replace('-', ' ').title()
            else:
                match_title = "Unknown Match"
            
            output_data.append([series_name, match_title, match_url])
            print(f"Match found: {match_title} - {match_url}")

finally:
    driver.quit()

# Save the scraped data to a DataFrame
output_df = pd.DataFrame(output_data, columns=["Series Name", "Match Title", "Match URL"])

# Cleaning function to remove unwanted matches
def filter_titles(title):
    return not any(keyword in title.upper() for keyword in ['TEST', 'T20I', 'BANGLADESH A'])

# Apply the cleaning function
output_df_cleaned = output_df[output_df['Match Title'].apply(filter_titles)]

# Remove duplicate match titles
output_df_cleaned = output_df_cleaned.drop_duplicates(subset=['Match Title'])

# Save the cleaned data to a CSV file
output_df_cleaned.to_csv("series_match_urls.csv", index=False)

print("Cleaned and deduplicated match URLs for all series extracted and saved!")