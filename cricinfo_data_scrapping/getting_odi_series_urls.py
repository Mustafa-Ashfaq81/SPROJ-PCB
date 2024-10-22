import csv
import re  
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import time
import logging

# Configure logging to save to a txt file
logging.basicConfig(filename='series_results_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set up Chrome WebDriver
service = ChromeService()
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)

try:
    # Navigate to the target URL
    url = 'https://www.espncricinfo.com/records/list-of-series-results-335432'
    driver.get(url)

    # Use explicit wait to ensure the page and the table are fully loaded
    try:
        # Wait for the table element to become visible
        table = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, '//table'))
        )
        
        # Locate all rows within the table body
        rows = table.find_elements(By.XPATH, './/tbody/tr')

        # Prepare the CSV file with the new "URL" column
        with open('odi_series_urls.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=["Series/Tournament", "Season", "Winner", "Margin", "URL"])
            writer.writeheader()  # Write header

            # Iterate through rows and collect data
            for row in rows:
                try:
                    # Locate the Series/Tournament cell and extract the name and URL
                    series_cell = row.find_element(By.XPATH, './td[1]')
                    series_link = series_cell.find_element(By.TAG_NAME, 'a')
                    series_name = series_link.text.strip()
                    series_url = series_link.get_attribute('href').strip()

                    # Extract other relevant columns
                    season = row.find_element(By.XPATH, './td[2]').text.strip()
                    winner = row.find_element(By.XPATH, './td[3]').text.strip()
                    margin = row.find_element(By.XPATH, './td[4]').text.strip()

                    # Use regex to extract the first 4-digit year from the season
                    match = re.search(r'(\d{4})', season)
                    if match:
                        year = int(match.group(1))
                        if year < 2001:
                            continue  # Skip rows where the season is before 2001
                    else:
                        continue  # Skip rows where no year could be found in the season

                    # Write the data into the CSV file
                    writer.writerow({
                        "Series/Tournament": series_name,
                        "Season": season,
                        "Winner": winner,
                        "Margin": margin,
                        "URL": series_url
                    })

                except NoSuchElementException:
                    # Handle missing elements and skip this row
                    continue

    except TimeoutException:
        print("Timed out waiting for the table to load.")
    except NoSuchElementException:
        print("Could not locate the table containing series information.")

finally:
    # Close the WebDriver after completion
    driver.quit()