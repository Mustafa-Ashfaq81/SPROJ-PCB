import os
import csv
import time
import re

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
)

###############################################################################
# 1) Utility to remove or close known pop-ups/cookie banners
###############################################################################
def handle_popups(driver):
    """Attempt to remove overlays, pop-ups, cookie banners that block the page."""
    try:
        overlay = driver.find_element(By.CSS_SELECTOR, "div.wzrk-overlay")
        driver.execute_script("arguments[0].remove();", overlay)
        print("[POPUP] Removed 'wzrk-overlay'")
        time.sleep(0.5)
    except:
        pass

    try:
        close_btn = driver.find_element(By.CSS_SELECTOR, "div.wzrk-button")
        close_btn.click()
        print("[POPUP] Clicked 'wzrk-button' close")
        time.sleep(0.5)
    except:
        pass

    try:
        cookie_btn = driver.find_element(By.CSS_SELECTOR, "button[id*='onetrust-accept']")
        cookie_btn.click()
        print("[POPUP] Clicked 'accept cookies' button")
        time.sleep(0.5)
    except:
        pass


###############################################################################
# 2) Parse numeric match_id from commentary URL
###############################################################################
def extract_match_id_from_url(commentary_url):
    """
    Example: 
      https://www.espncricinfo.com/series/blah-66274/ball-by-ball-commentary
    we parse '66274' from second-to-last path segment.
    """
    parts = commentary_url.rstrip("/").split("/")
    if len(parts) < 2:
        return "UNKNOWN"
    second_last = parts[-2]
    m = re.search(r"(\d+)$", second_last)
    if m:
        return m.group(1)
    return "UNKNOWN"


###############################################################################
# 3) Check if an element is currently in the browser's visible viewport
###############################################################################
def is_in_viewport(driver, element):
    """
    Return True if 'element' is (at least partially) visible in the current window.
    We'll do a basic boundingRect check vs. window.innerHeight.
    """
    rect = driver.execute_script("return arguments[0].getBoundingClientRect();", element)
    top_ = rect.get("top", 999999)
    bottom_ = rect.get("bottom", 999999)
    window_h = driver.execute_script("return window.innerHeight;")
    return (bottom_ > 0) and (top_ < window_h)


###############################################################################
# 4) Make sure "Full commentary" is selected if that button or link is present
###############################################################################
def ensure_full_commentary(driver):
    """
    On Cricinfo, there's often an option for "Full commentary" vs "Key events" vs "Latest".
    We want to be sure we see all deliveries from ball 0.1 onward.
    We'll attempt to find a button or link that says exactly "Full commentary" and click it.
    """
    try:
        full_comm = driver.find_element(By.XPATH, "//*[text()='Full commentary']")
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", full_comm)
        time.sleep(0.5)
        full_comm.click()
        time.sleep(1.5)
        print("[INFO] Clicked 'Full commentary' from top bar.")
    except NoSuchElementException:
        print("[INFO] 'Full commentary' button not found; possibly already in full mode.")


###############################################################################
# 5) Parse all commentary blocks currently loaded in the container
###############################################################################
def parse_commentary_blocks(driver):
    """
    Return a list of (ball_num, text) from the main commentary container.
    """
    results = []
    seen = set()

    try:
        container = driver.find_element(
            By.CSS_SELECTOR,
            "div.ds-w-full.ds-bg-fill-content-prime.ds-overflow-hidden.ds-rounded-xl."
            "ds-mb-4.ds-border-y.lg\\:ds-border.ds-border-line"
        )
    except NoSuchElementException:
        print("[ERROR] Commentary container not found. Returning empty.")
        return results

    blocks = container.find_elements(
        By.CSS_SELECTOR,
        "div.ds-text-tight-m.ds-font-regular.ds-flex.ds-px-3.ds-py-2, div.ds-text-compact-s"
    )

    for b in blocks:
        # Attempt to get ball number
        try:
            ball_el = b.find_element(
                By.CSS_SELECTOR,
                "span.ds-text-tight-s.ds-font-regular.ds-mb-1."
                "lg\\:ds-mb-0.lg\\:ds-mr-3.ds-block.ds-text-center.ds-text-typo-mid1"
            )
            ball_num = ball_el.text.strip()
        except NoSuchElementException:
            ball_num = "MISC"

        paras = b.find_elements(By.CSS_SELECTOR, "p.ci-html-content")
        combined_text = " ".join(p.text.strip() for p in paras if p.text.strip())
        if not combined_text:
            continue

        key = (ball_num, combined_text)
        if key not in seen:
            seen.add(key)
            results.append(key)

    return results


###############################################################################
# 6) "Ping-pong" scrolling
###############################################################################
def ping_pong_scroll(
    driver,
    target_ball="0.1",
    max_loops=60,
    scroll_step=400,
    max_no_new=8
):
    """
    We do a series of steps:
      - parse commentary
      - if "Sponsored Headlines" is visible => scroll up
      - if near the top => scroll down
      - each iteration, if no new lines appear multiple times => break
      - if spinner found => wait for vanish
      - if ball=0.1 found => break
    Returns final list of (ball, text).
    """
    commentary_data = []
    seen = set()
    no_new_count = 0

    spinner_selector = "span.ds-spinner"
    ads_heading_sel = "div.tbl-feed-header-text"

    direction = "down"

    for iteration in range(1, max_loops+1):
        handle_popups(driver)

        new_items = 0
        blocks = parse_commentary_blocks(driver)
        for (bnum, text) in blocks:
            key = (bnum, text)
            if key not in seen:
                seen.add(key)
                commentary_data.append(key)
                new_items += 1

        if new_items > 0:
            no_new_count = 0
        else:
            no_new_count += 1
            print(f"[DEBUG] No new commentary lines on iteration={iteration}, no_new_count={no_new_count}")

        # check if found 0.1
        if any(b == target_ball for (b,_) in commentary_data):
            print(f"[INFO] Found ball '{target_ball}' on iteration={iteration}. Stopping ping-pong.")
            break

        if no_new_count >= max_no_new:
            print("[WARN] No new lines for multiple loops, stopping ping-pong scroll.")
            break

        # see if we overshot into ads
        overshot = False
        try:
            ads_heading = driver.find_element(By.CSS_SELECTOR, ads_heading_sel)
            if is_in_viewport(driver, ads_heading):
                overshot = True
        except NoSuchElementException:
            pass

        current_offset = driver.execute_script("return window.pageYOffset;")
        if overshot:
            direction = "up"
        else:
            if current_offset < 200:
                direction = "down"

        # do the scroll
        if direction == "down":
            driver.execute_script(f"window.scrollBy(0,{scroll_step});")
        else:
            driver.execute_script(f"window.scrollBy(0,-{scroll_step});")
        time.sleep(1.5)

        # check spinner
        spinner_present = False
        try:
            driver.find_element(By.CSS_SELECTOR, spinner_selector)
            spinner_present = True
        except NoSuchElementException:
            pass

        if spinner_present:
            print("[INFO] Spinner found, waiting up to 10s for vanish.")
            try:
                WebDriverWait(driver, 10).until(
                    EC.invisibility_of_element_located((By.CSS_SELECTOR, spinner_selector))
                )
                print("[INFO] Spinner gone. Possibly loaded more commentary.")
                time.sleep(1.0)
            except TimeoutException:
                print("[WARN] Spinner still visible after 10s, continuing anyway...")

    return commentary_data


###############################################################################
# 7) Gather commentary for the currently selected side
###############################################################################
def gather_commentary_for_side(driver, side_label):
    """
    1) Try 'Full commentary' mode
    2) Ping-pong scroll
    3) Return the final list
    """
    print(f"\n[INFO] Gathering commentary for side: {side_label}")
    handle_popups(driver)
    ensure_full_commentary(driver)
    data = ping_pong_scroll(driver, target_ball="0.1", max_loops=70, scroll_step=400)
    print(f"[INFO] Collected {len(data)} lines for side '{side_label}'")
    return data


###############################################################################
# 8) Parse the date from the top "match info" area
#    e.g. "1st ODI, Wellington, January 05, 2025, ...
###############################################################################
def parse_match_date(driver):
    """
    We'll look for the top info element typically found in
      <div class="ds-text-tight-m ds-font-regular ds-text-typo-mid3">
      e.g. "1st ODI, Wellington, January 05, 2025, ...
    and extract "January_05_2025" via a regex (replacing commas/spaces).
    If not found, return "UNKNOWN_DATE".
    """
    date_str = "UNKNOWN_DATE"
    try:
        info_el = driver.find_element(By.CSS_SELECTOR, "div.ds-text-tight-m.ds-font-regular.ds-text-typo-mid3")
        text = info_el.text  # e.g. "1st ODI, Wellington, January 05, 2025, ..."
        # We'll do a pattern for Month dd, yyyy
        m = re.search(r"([A-Za-z]+\s+\d{1,2},\s+\d{4})", text)
        if m:
            # e.g. "January 05, 2025"
            raw_date = m.group(1)
            # replace commas and spaces with underscores => "January_05_2025"
            date_str = raw_date.replace(",", "").replace(" ", "_")
    except NoSuchElementException:
        print("[WARN] Could not locate top info div for match date.")
    return date_str


###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    import os

    # directory for raw commentary
    raw_commentary_dir = "../CommentaryData/raw_commentary"
    # ensure it exists
    os.makedirs(raw_commentary_dir, exist_ok=True)

    csv_filename = "series_match_commentary_urls.csv"
    target_ball = "0.1"

    # We'll keep a list to record any failures
    failures = []  # each entry: (match_id, reason)

    with open(csv_filename, "r", encoding="utf-8") as cf:
        fieldnames = ["Series Name", "Match Title", "Commentary URL"]
        reader = csv.DictReader(cf, fieldnames=fieldnames)

        for idx, row in enumerate(reader):
            series = row["Series Name"].strip()
            match_title = row["Match Title"].strip()
            url = row["Commentary URL"].strip()

            print(f"\n==============================\nRow #{idx}")
            print(f"Series: {series}")
            print(f"Match: {match_title}")
            print(f"Commentary URL: {url}")

            if not url or "ball-by-ball-commentary" not in url:
                print("[SKIP] invalid commentary URL.")
                continue

            match_id = extract_match_id_from_url(url)
            print(f"[INFO] match_id => {match_id}")

            driver = webdriver.Chrome()
            driver.maximize_window()
            driver.get(url)
            time.sleep(2)
            handle_popups(driver)

            # Parse date
            match_date = parse_match_date(driver)
            print(f"[INFO] Extracted match date => {match_date}")

            # side1
            try:
                side_span = driver.find_element(
                    By.CSS_SELECTOR,
                    "div.ds-flex.ds-items-center.ds-border-ui-stroke.ds-h-6.ds-px-4.ds-border."
                    "ds-bg-ui-fill.ds-rounded-full.ds-w-full.ds-min-w-max.ds-cursor-pointer "
                    "span.ds-text-tight-s.ds-font-regular.ds-text-typo"
                )
                side1 = side_span.text.strip()
            except:
                side1 = "UNKNOWN"

            side1_data = gather_commentary_for_side(driver, side1)
            found_01_side1 = any(b == target_ball for (b, _) in side1_data)

            # Attempt toggle to side2
            side2_data = []
            side2 = "UNKNOWN"
            toggled_ok = False
            print("\n[INFO] Attempting to toggle to second side…")
            try:
                toggle_div = driver.find_element(
                    By.CSS_SELECTOR,
                    "div.ds-flex.ds-items-center.ds-border-ui-stroke.ds-h-6.ds-px-4."
                    "ds-border.ds-bg-ui-fill.ds-rounded-full.ds-w-full.ds-min-w-max.ds-cursor-pointer"
                )
                ActionChains(driver).move_to_element(toggle_div).click().perform()
                time.sleep(1)
                toggled_ok = True
            except Exception as e:
                print(f"[ERROR] normal toggle click failed: {e}")
                handle_popups(driver)
                try:
                    driver.execute_script("arguments[0].click();", toggle_div)
                    time.sleep(1)
                    toggled_ok = True
                except Exception as e2:
                    print(f"[ERROR] JS click also failed => skipping second side. {e2}")

            if toggled_ok:
                # pick from dropdown
                try:
                    li_elems = driver.find_elements(By.CSS_SELECTOR, "ul.ds-flex.ds-flex-col li")
                    for li in li_elems:
                        label = li.text.strip()
                        if label and label != side1:
                            side2 = label
                            li.click()
                            time.sleep(2)
                            break
                except Exception as e:
                    print(f"[ERROR] picking second side: {e}")

                side2_data = gather_commentary_for_side(driver, side2)

            driver.quit()

            # Decide output filenames
            # e.g. "1443546_NZ_commentary_January_05_2025.txt"
            out1_name = f"{match_id}_{side1}_commentary_{match_date}.txt"
            out1_path = os.path.join(raw_commentary_dir, out1_name)

            out2_name = f"{match_id}_{side2}_commentary_{match_date}.txt"
            out2_path = os.path.join(raw_commentary_dir, out2_name)

            # side1 check
            if found_01_side1:
                print(f"[SAVE] Writing side1 => {out1_path}")
                with open(out1_path, "w", encoding="utf-8") as f1:
                    for (bn, txt) in side1_data:
                        f1.write(f"Ball {bn}:\n{txt}\n\n")
            else:
                reason = f"side1 '{side1}' missing ball {target_ball}"
                print(f"[WARN] {reason}. Skipping write.")
                failures.append((match_id, reason))

            # side2 check
            if toggled_ok:
                found_01_side2 = any(b == target_ball for (b, _) in side2_data)
                if found_01_side2:
                    print(f"[SAVE] Writing side2 => {out2_path}")
                    with open(out2_path, "w", encoding="utf-8") as f2:
                        for (bn, txt) in side2_data:
                            f2.write(f"Ball {bn}:\n{txt}\n\n")
                else:
                    reason = f"side2 '{side2}' missing ball {target_ball}"
                    print(f"[WARN] {reason}. Skipping write.")
                    failures.append((match_id, reason))
            else:
                # toggling side2 failed => log that
                reason = "toggling to second side failed"
                print(f"[WARN] {reason}.")
                failures.append((match_id, reason))

    print("[INFO] Done all rows. Exiting.")

    # Finally, produce the log file of failures if any
    if failures:
        log_fname = "scrape_failures.log"
        print(f"[INFO] Writing failures log => {log_fname}")
        with open(log_fname, "w", encoding="utf-8") as lf:
            lf.write("Scraping Failures:\n")
            for (mid, rsn) in failures:
                lf.write(f"MatchID={mid} => {rsn}\n")
        print(f"[INFO] Wrote {len(failures)} failure entries to {log_fname}")
    else:
        print("[INFO] No failures encountered.")
