import csv
import time
import re
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
)


###############################################################################
# Helper function to handle overlays/popups
###############################################################################
def handle_popups(driver):
    """
    Attempt to remove or click away any pop-ups that can intercept clicks.
    """
    # Example pop-ups; adapt to your site’s actual pop-ups:
    try:
        overlay = driver.find_element(By.CSS_SELECTOR, "div.wzrk-overlay")
        driver.execute_script("arguments[0].remove();", overlay)
        time.sleep(0.5)
    except:
        pass

    try:
        close_button = driver.find_element(By.CSS_SELECTOR, "div.wzrk-button")
        close_button.click()
        time.sleep(0.5)
    except:
        pass

    try:
        cookie_button = driver.find_element(
            By.CSS_SELECTOR, "button[id*='onetrust-accept']"
        )
        cookie_button.click()
        time.sleep(0.5)
    except:
        pass


###############################################################################
# Wait for the ESPN "loader" to disappear, or until timeout.
###############################################################################
def wait_for_loader_to_disappear(driver, timeout=15):
    """
    Blocks up to 'timeout' seconds until the loader element is not displayed.
    Adjust the loader selector if it changes on ESPN.
    """
    start_time = time.time()
    while (time.time() - start_time) < timeout:
        try:
            loader_elem = driver.find_element(
                By.CSS_SELECTOR, "div.ds-flex.ds-justify-center.ds-my-4 span.ds-spinner"
            )
            if loader_elem.is_displayed():
                time.sleep(0.5)
            else:
                break  # Found but not displayed
        except:
            # Not found in DOM => presumably gone
            break


###############################################################################
# Scroll the window without passing the "Sponsored Headlines" section
###############################################################################
def scroll_and_wait_for_loader(driver, scroll_amount=500):
    """
    Scrolls the window by scroll_amount pixels, but never goes below
    the 'Sponsored Headlines' element (tbl-feed-header-text).

    Returns:
       True  if we successfully scrolled
       False if we are already at/above sponsor area and cannot scroll further
    """
    current_y = driver.execute_script("return window.pageYOffset;")

    # Find the sponsor element’s absolute Y position
    try:
        sponsored_elem = driver.find_element(
            By.CSS_SELECTOR, "div.tbl-feed-header-text"
        )
        sponsored_y = driver.execute_script(
            "return arguments[0].getBoundingClientRect().top + window.pageYOffset;",
            sponsored_elem,
        )
    except:
        sponsored_elem = None
        sponsored_y = float("inf")  # If not found, no sponsor limit

    proposed_scroll = current_y + scroll_amount

    # If proposed scroll would go below sponsor, clamp
    if proposed_scroll >= sponsored_y:
        new_scroll = sponsored_y - 100
        if new_scroll <= (current_y + 10):
            print("Cannot scroll further without passing sponsor section.")
            return False
        driver.execute_script(f"window.scrollTo(0, {new_scroll});")
    else:
        driver.execute_script(f"window.scrollTo(0, {proposed_scroll});")

    time.sleep(1)
    wait_for_loader_to_disappear(driver)
    return True


###############################################################################
# Extract match_id from commentary URL
###############################################################################
def extract_match_id_from_url(commentary_url):
    parts = commentary_url.rstrip("/").split("/")
    if len(parts) < 2:
        return "UNKNOWN"
    last_part = parts[-2]
    match = re.search(r"(\d+)$", last_part)
    if match:
        return match.group(1)
    return "UNKNOWN"


###############################################################################
# Parse a match date from the page (minor helper to add date in file name)
###############################################################################
def parse_match_date(driver):
    """
    Attempts to parse a date from the top match-info area.
    Returns a string like "January_05_2025" or "UNKNOWN_DATE".
    """
    date_str = "UNKNOWN_DATE"
    try:
        info_el = driver.find_element(
            By.CSS_SELECTOR, "div.ds-text-tight-m.ds-font-regular.ds-text-typo-mid3"
        )
        text = info_el.text  # e.g. "1st ODI, Wellington, January 05, 2025, ..."
        m = re.search(r"([A-Za-z]+\s+\d{1,2},\s+\d{4})", text)
        if m:
            raw_date = m.group(1)  # e.g. "January 05, 2025"
            date_str = raw_date.replace(",", "").replace(
                " ", "_"
            )  # => "January_05_2025"
    except:
        pass
    return date_str


###############################################################################
# Scrape commentary for the currently selected "country"
###############################################################################
def scrape_commentary(driver, target_ball, output_file):
    found_target = False
    seen_entries = set()

    with open(output_file, "w", encoding="utf-8") as f:
        while True:
            handle_popups(driver)

            # NOTE: Same CSS selector for commentary blocks:
            commentary_blocks = driver.find_elements(
                By.CSS_SELECTOR,
                (
                    "div.ds-text-tight-m.ds-font-regular.ds-flex.ds-px-3.ds-py-2,"
                    "div.lg\\:hover\\:ds-bg-ui-fill-translucent.ds-hover-parent.ds-relative,"
                    "div.ds-text-compact-s"
                ),
            )

            if not commentary_blocks:
                print(
                    "No commentary blocks found. Possibly at bottom or overlay blocking."
                )
                break

            for block in commentary_blocks:
                # Attempt to find ball number
                try:
                    ball_num_elem = block.find_element(
                        By.CSS_SELECTOR,
                        "span.ds-text-tight-s.ds-font-regular.ds-mb-1."
                        "lg\\:ds-mb-0.lg\\:ds-mr-3.ds-block."
                        "ds-text-center.ds-text-typo-mid1",
                    )
                    ball_num = ball_num_elem.text.strip()
                except:
                    ball_num = "MISC"

                # ------------------------------------------------------------------
                # THIS IS THE KEY CHANGE: we capture all text in the block
                # so that we also include "Simi Singh to Morgan, SIX runs" etc.
                # ------------------------------------------------------------------
                commentary_text = block.text.strip()
                if not commentary_text:
                    continue

                # Deduplicate by storing in a set
                entry_key = (ball_num, commentary_text)
                if entry_key not in seen_entries:
                    f.write(f"Ball {ball_num}:\n{commentary_text}\n\n")
                    seen_entries.add(entry_key)

                # If we've hit the target ball, stop scraping further
                if ball_num == target_ball:
                    print(f"[{output_file}] Found ball {target_ball}. Stopping.")
                    found_target = True
                    break

            if found_target:
                break

            success = scroll_and_wait_for_loader(driver, scroll_amount=500)
            if not success:
                # We can’t scroll further without passing sponsor => done
                print(
                    "Stopping because no more scrolling is possible above the sponsor."
                )
                break

    return found_target


###############################################################################
# Check if a given file contains 'Ball 0.1'
###############################################################################
def file_contains_ball_0_1(filepath):
    """Return True if 'Ball 0.1' is found in the text file, else False."""
    if not os.path.exists(filepath):
        return False
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if "Ball 0.1" in line:
                return True
    return False


###############################################################################
# Process one CSV row: scrape commentary for both countries and check
# if 'Ball 0.1' was found. Return True if found, False otherwise.
###############################################################################
def process_row(row, target_ball="0.1"):
    """Scrape the match described by row. Return True if we found 'Ball 0.1'."""
    series_name = row["Series Name"].strip()
    match_title = row["Match Title"].strip()
    commentary_url = row["Commentary URL"].strip()

    if not commentary_url or "ball-by-ball-commentary" not in commentary_url:
        print(f"Invalid commentary URL for row: {series_name}, {match_title}")
        return False

    match_id = extract_match_id_from_url(commentary_url)
    print(f"\n=== Processing: {series_name} | {match_title} | match_id={match_id} ===")

    driver = webdriver.Chrome()
    try:
        driver.get(commentary_url)
        time.sleep(2)
        handle_popups(driver)

        # --- MINOR CHANGE: Parse match date so we can incorporate it in filename ---
        match_date = parse_match_date(driver)

        # Identify the currently selected "country"
        try:
            selected_country_span = driver.find_element(
                By.CSS_SELECTOR,
                "div.ds-flex.ds-items-center.ds-border-ui-stroke.ds-h-6.ds-px-4."
                "ds-border.ds-bg-ui-fill.ds-rounded-full.ds-w-full.ds-min-w-max.ds-cursor-pointer "
                "span.ds-text-tight-s.ds-font-regular.ds-text-typo",
            )
            country1 = selected_country_span.text.strip()
        except:
            country1 = "UNKNOWN"

        # Incorporate the parsed date into the output filename:
        output_file_1 = (
            f"../CommentaryData/raw_commentary/{match_id}_{country1}_commentary_{match_date}.txt"
        )
        print(f"Scraping first country '{country1}' => {output_file_1}")
        found_ball_country1 = scrape_commentary(driver, target_ball, output_file_1)

        # Attempt toggle to second country
        print("Attempting to toggle to the other country...")
        handle_popups(driver)

        try:
            toggle_div = driver.find_element(
                By.CSS_SELECTOR,
                "div.ds-flex.ds-items-center.ds-border-ui-stroke.ds-h-6.ds-px-4."
                "ds-border.ds-bg-ui-fill.ds-rounded-full.ds-w-full.ds-min-w-max.ds-cursor-pointer",
            )
            ActionChains(driver).move_to_element(toggle_div).click().perform()
            time.sleep(1)
        except Exception as e:
            print("Error clicking toggle div:", e)
            # Try JS click
            handle_popups(driver)
            try:
                driver.execute_script("arguments[0].click();", toggle_div)
                time.sleep(1)
            except:
                print("Still couldn't click toggle. Skipping second country.")
                if not found_ball_country1:
                    return False
                else:
                    return True

        other_country = "UNKNOWN"
        try:
            all_country_items = driver.find_elements(
                By.CSS_SELECTOR, "ul.ds-flex.ds-flex-col li"
            )
            for item in all_country_items:
                c_text = item.text.strip()
                if c_text and c_text != country1:
                    other_country = c_text
                    print(f"Clicking other country item with text: {c_text}")
                    item.click()
                    time.sleep(2)
                    break
        except Exception as e:
            print("Error picking the other country:", e)

        other_country = other_country.strip()
        if not other_country:
            other_country = "UNKNOWN"

        # Incorporate match_date into the second file’s name:
        output_file_2 = f"../CommentaryData/raw_commentary/{match_id}_{other_country}_commentary_{match_date}.txt"
        print(f"Scraping second country '{other_country}' => {output_file_2}")

        handle_popups(driver)
        found_ball_country2 = scrape_commentary(driver, target_ball, output_file_2)

    except Exception as e:
        print("Error processing row:", e)
        driver.quit()
        return False
    finally:
        driver.quit()

    # Now check if "Ball 0.1" is in either file
    success_country1 = file_contains_ball_0_1(output_file_1)
    success_country2 = file_contains_ball_0_1(output_file_2)

    if success_country1 and success_country2:
        print(f"SUCCESS: Found 'Ball 0.1' in row for {series_name} | {match_title}.")
        return True
    else:
        print(
            f"FAIL: Did NOT find 'Ball 0.1' in row for {series_name} | {match_title}."
        )
        return False


###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    csv_filename = "BabarAzamRecentOdis.csv"
    target_ball = "0.1"
    MAX_RETRIES = 2  # how many times we try each row

    # 1) Read the CSV rows into a list
    rows = []
    with open(csv_filename, "r", encoding="utf-8") as csvfile:
        fieldnames = ["Series Name", "Match Title", "Commentary URL"]
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)

        # If your CSV has a header row, uncomment this to skip it:
        # next(reader, None)  # skip header row if needed

        for row in reader:
            rows.append(row)

    # 2) Attempt to process each row
    remaining_rows = []
    for row in rows:
        attempts = 0
        success = False
        while attempts < MAX_RETRIES and not success:
            attempts += 1
            print(f"\n===== Attempt #{attempts} for row: {row} =====")
            success = process_row(row, target_ball=target_ball)
            if not success:
                print(f"Retrying row after failure... (attempt {attempts})")

        if not success:
            # after max attempts, we keep it in leftover
            remaining_rows.append(row)

    # 3) Rewrite the CSV with leftover rows that never succeeded
    if remaining_rows:
        print("\nRewriting CSV with leftover rows that did NOT find Ball 0.1...")
        with open(csv_filename, "w", encoding="utf-8", newline="") as outcsv:
            writer = csv.DictWriter(
                outcsv,
                fieldnames=["Series Name", "Match Title", "Commentary URL"],
                quoting=csv.QUOTE_MINIMAL,
            )
            for r in remaining_rows:
                writer.writerow(r)
    else:
        print("\nAll rows processed successfully. Removing all from CSV.")
        open(csv_filename, "w").close()  # empty file or you could delete entirely

    print("\n=== Done. ===")