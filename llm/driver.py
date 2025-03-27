#!/usr/bin/env python3
"""
driver.py

An end-to-end script that:

1) Logs each step so you can see the status.
2) Calls:
   - get_player_stats.py (General Stats),
   - get_player_stats_venuewise.py (Venue-wise Stats),
   - get_player_stats_against_team.py (Stats vs. Opponent).
3) Has two possible commentary modes:
   a) Commentary via script: calls filter_comms.py
   b) Commentary via txt: reads a local file
   - If BOTH are enabled, defaults to the txt file.
   - If commentary_on is False, no commentary is included at all.
4) Builds the MASTER PROMPT (exactly as requested) with:
   - General Stats
   - Venue-wise Data
   - Against-Teams snippet
   - Commentary (if on, in whichever mode)
5) Sends that prompt to an LLM (real openai.ChatCompletion call).
6) Parses the LLM’s textual output to extract:
   - Probability array (Bowled, Caught-Fielder, Caught-Wicketkeeper, Stumped, LBW, Hit wicket, Caught and Bowled, Not-Out),
   - Final predicted mode,
   - Ground truth (if any or "N/A").
7) Prints a single CSV row with columns:

   Batsman, train_data_span, #_commentaries, prediction_match,
   mode_of_dissmisal_predicted_probability,
   mode_of_dissmisal_predicted,
   mode_of_dissmisal_ground_truth

Usage:
------
python driver.py --player_name "Steven Smith" --team_name "Australia" --opponent_team "SL" --start_year 2019 --end_year 2024 --commentary_on True --use_script False --use_txt True --k 5

If you want to use commentary via script only:
 --commentary_on True --use_script True --use_txt False
If you want to use commentary via txt only:
 --commentary_on True --use_script False --use_txt True
If both script & txt are True => txt is used.
If commentary_on is False => no commentary.

Logs go to stderr, the final CSV row goes to stdout.
Redirect stdout if you want to save the CSV row to a file.

"""

import os
import json
import argparse
import subprocess
import shlex
import re
import logging

# We use dotenv to load OPENAI_API_KEY from .env if present
from dotenv import load_dotenv

logging.basicConfig(
    format='[%(levelname)s] %(message)s',
    level=logging.INFO
)

def run_subprocess(cmd):
    """
    Helper function to run a shell command, log it,
    and raise RuntimeError if it fails.
    """
    logging.info(f"Running command: {cmd}")
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    if proc.returncode != 0:
        logging.warning(f"Command stderr: {err}")
        raise RuntimeError(f"Command failed: {cmd}")
    return out, err

def load_json_file(filepath):
    """
    Safely loads a JSON file. Returns {} on error or if file not found.
    """
    if not os.path.isfile(filepath):
        logging.warning(f"JSON file not found: {filepath}")
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Could not parse JSON from {filepath}: {e}")
        return {}

def call_llm_api(master_prompt_text, api_key=None):
    """
    Real LLM API call using openai.ChatCompletion (like filter_comms.py).
    If api_key is None or empty, we log a warning and return a fallback text.
    Otherwise, we do a ChatCompletion call with model="o3-mini" or similar.
    """
    if not api_key:
        logging.warning("[WARNING] OPENAI_API_KEY not found. Returning fallback LLM text.")
        return (
            "LLM Fallback Response:\n"
            "Could not call the real LLM because no API key was provided."
        )

    import openai
    openai.api_key = api_key

    try:
        response = openai.ChatCompletion.create(
            model="o3-mini",  # or another model name you have access to
            messages=[{"role": "user", "content": master_prompt_text}],
        )
        final_text = response["choices"][0]["message"]["content"]
    except Exception as e:
        logging.warning(f"[WARNING] LLM call failed: {e}")
        final_text = (
            "LLM Fallback Response:\n"
            "Encountered an error during openai.ChatCompletion."
        )

    return final_text

def parse_llm_analysis(llm_text, player_name):
    """
    Given the LLM's final analysis text, parse out:
      - The final predicted mode of dismissal (string)
      - Probabilities in the order:
        Bowled, Caught-Fielder, Caught-Wicketkeeper, Stumped,
        LBW, Hit-Wicket, Caught-and-Bowled, Not-Out
      - Ground truth => "N/A" (unless you have other logic).

    We look for lines like:
      "Bowled: 21%"
      "Caught by a fielder: 32%"
      ...
    and also for "most likely to be dismissed ... <mode>".
    """

    # 1) Probability breakdown lines
    pattern = re.compile(r'(?i)^\s*(bowled|caught\s+by\s+a?\s?fielder|caught\s+by\s+the\s+wicket[\-–]?keeper|caught\s+wk|stumped|lbw|hit\s*wicket|caught\s*and\s*bowled|not[-\s]*out)\s*:\s*(\d+)\%', re.MULTILINE)

    # We'll map textual forms -> (standard_label, array index)
    mode_map = {
        "bowled": ("Bowled", 0),
        "caught by a fielder": ("Caught-Fielder", 1),
        "caught by the wicket-keeper": ("Caught-Wicketkeeper", 2),
        "caught wk": ("Caught-Wicketkeeper", 2),
        "stumped": ("Stumped", 3),
        "lbw": ("LBW", 4),
        "hit wicket": ("Hit-Wicket", 5),
        "caught and bowled": ("Caught-and-Bowled", 6),
        "not-out": ("Not-Out", 7),
        "not out": ("Not-Out", 7),
    }

    prob_array = [0]*8
    label_array = ["Bowled","Caught-Fielder","Caught-Wicketkeeper","Stumped","LBW","Hit-Wicket","Caught-and-Bowled","Not-Out"]

    for match_ in pattern.finditer(llm_text):
        raw_mode = match_.group(1).lower().strip()
        raw_prob = match_.group(2)
        if raw_mode in mode_map:
            _, idx = mode_map[raw_mode]
            prob_array[idx] = int(raw_prob)

    # 2) Final predicted mode
    # We'll look for: "most likely to be dismissed <some mode>"
    pred_pattern = re.compile(
        rf"most\s+likely\s+(?:mode\s+of\s+dismissal|to\s+be\s+dismissed)\s+(?:for\s+{player_name}\s+is\s+|is\s+|:?\s+)(?:“|\"|')?([^\"”'\n]+)",
        re.IGNORECASE
    )
    pred_mode = "N/A"
    pmatch = pred_pattern.search(llm_text)
    if pmatch:
        raw_pred = pmatch.group(1).lower().strip()
        matched_label = None
        for k_ in mode_map:
            if k_ in raw_pred:
                matched_label = mode_map[k_][0]
                break
        if matched_label:
            pred_mode = matched_label
        else:
            pred_mode = raw_pred.title()  # fallback

    # 3) Ground truth => "N/A"
    ground_truth = "N/A"

    return prob_array, label_array, pred_mode, ground_truth


def main():
    # Load environment (.env) so that OPENAI_API_KEY is available
    load_dotenv()
    final_api_key = os.environ.get("OPENAI_API_KEY", None)
    if not final_api_key:
        logging.warning("[WARNING] Environment variable OPENAI_API_KEY not found. LLM call may use fallback text.")

    parser = argparse.ArgumentParser(description="Majestic driver script that logs steps, gathers stats & commentary, builds MASTER PROMPT, calls LLM, parses final text, and outputs one CSV row.")
    parser.add_argument("--player_name", required=True, help="Player name, e.g. 'Steven Smith'")
    parser.add_argument("--team_name", required=True, help="Team name, e.g. 'Australia'")
    parser.add_argument("--opponent_team", required=True, help="Opponent team code or name, e.g. 'SL' or 'Sri Lanka'")
    parser.add_argument("--start_year", default="all", help="Start year for stats or 'all'")
    parser.add_argument("--end_year", default="all", help="End year for stats or 'all'")

    parser.add_argument("--commentary_on", default="False", choices=["True","False"],
                        help="Whether to embed commentary at all. Default False.")
    parser.add_argument("--use_script", default="False", choices=["True","False"],
                        help="If True, commentary is gathered by calling filter_comms.py.")
    parser.add_argument("--use_txt", default="False", choices=["True","False"],
                        help="If True, commentary is read from a local txt file.")
    parser.add_argument("--k", type=int, default=5, help="Number of commentary files to filter. Default 5.")

    args = parser.parse_args()

    player_name = args.player_name
    team_name = args.team_name
    opponent_team = args.opponent_team
    start_year = args.start_year
    end_year = args.end_year
    commentary_on = (args.commentary_on.lower() == "true")
    use_script_flag = (args.use_script.lower() == "true")
    use_txt_flag = (args.use_txt.lower() == "true")
    k = args.k

    logging.info("Starting driver process...")

    # --------------------------------------------------
    # 1) Run get_player_stats.py => general stats
    # --------------------------------------------------
    logging.info("Running get_player_stats.py ...")
    safe_player_name = player_name.lower().replace(' ', '_')
    try:
        cmd_general = f"python ../cricinfo_data_scrapping/Scripts/get_player_stats.py \"{player_name}\" \"{team_name}\" {start_year} {end_year}"
        run_subprocess(cmd_general)
    except RuntimeError:
        logging.warning("[WARNING] get_player_stats script may have failed.")

    # Load general stats => get 'span'
    if start_year.isdigit() and end_year.isdigit():
        general_stats_filename = f"{safe_player_name}_{start_year}_{end_year}_span_stats.json"
    else:
        general_stats_filename = f"{safe_player_name}_all_span_stats.json"
    general_stats_path = general_stats_path = os.path.join("..", "cricinfo_data_scrapping", "aggregated_stats", general_stats_filename)
    general_stats = load_json_file(general_stats_path)

    raw_span = general_stats.get("span","Unknown")
    train_data_span = raw_span.replace(" ", "").replace("–","-").replace("--","-")

    # --------------------------------------------------
    # 2) Run get_player_stats_venuewise.py => venue data
    # --------------------------------------------------
    logging.info("Running get_player_stats_venuewise.py ...")
    try:
        cmd_venue = f"python ../cricinfo_data_scrapping/Scripts/get_player_stats_venuewise.py \"{player_name}\" \"{team_name}\" {start_year} {end_year}"
        run_subprocess(cmd_venue)
    except RuntimeError:
        logging.warning("[WARNING] get_player_stats_venuewise script may have failed.")

    # Load the venue-wise JSON
    if start_year.isdigit() and end_year.isdigit():
        venue_filename = f"{safe_player_name}_{start_year}_{end_year}_venuewise.json"
    else:
        venue_filename = f"{safe_player_name}_all_venuewise.json"
    venue_stats_path = os.path.join("..", "cricinfo_data_scrapping", "aggregated_stats_venue_wise", venue_filename)
    venue_stats = load_json_file(venue_stats_path)

    # --------------------------------------------------
    # 3) Run get_player_stats_against_team.py => opponent block
    # --------------------------------------------------
    logging.info("Running get_player_stats_against_team.py ...")
    try:
        cmd_against = f"python ../cricinfo_data_scrapping/Scripts/get_player_stats_against_team.py \"{player_name}\" \"{team_name}\" {start_year} {end_year}"
        run_subprocess(cmd_against)
    except RuntimeError:
        logging.warning("[WARNING] get_player_stats_against_team script may have failed.")

    # Load the "against teams" JSON and find snippet
    if start_year.isdigit() and end_year.isdigit():
        against_filename = f"{safe_player_name}_{start_year}_{end_year}_against_teams.json"
    else:
        against_filename = f"{safe_player_name}_all_against_teams.json"
    against_stats_path = os.path.join("..", "cricinfo_data_scrapping", "aggregated_stats_against_teams", against_filename)
    against_stats = load_json_file(against_stats_path)

    snippet_block = {}
    if "oppositions" in against_stats:
        for k_ in against_stats["oppositions"].keys():
            # If "SL" in "Sri Lanka".lower() or direct match
            if opponent_team.lower() in k_.lower():
                snippet_block = {k_: against_stats["oppositions"][k_]}
                break

    # --------------------------------------------------
    # 4) Decide how to gather commentary
    # --------------------------------------------------
    commentary_text = ""
    if not commentary_on:
        commentary_text = "[Commentary OFF]"
    else:
        # commentary_on = True => choose script, txt, or both
        # If both script & txt => use txt
        # If only txt => use txt
        # If only script => run filter_comms
        # If none => no commentary
        if use_txt_flag and use_script_flag:
            logging.info("Both script & txt commentary modes are enabled. Defaulting to txt approach.")
            # read from default path or a path you'd like
            txt_commentary_path = "filtered_outputs/filtered_commentaries_o3mini.txt"
            if os.path.isfile(txt_commentary_path):
                with open(txt_commentary_path, 'r', encoding='utf-8') as ftxt:
                    commentary_text = ftxt.read()
            else:
                commentary_text = "[No commentary found at filtered_outputs/filtered_commentaries_o3mini.txt]"
        elif use_txt_flag:
            logging.info("Using commentary from local txt file (no script).")
            txt_commentary_path = "filtered_outputs/filtered_commentaries_o3mini.txt"
            if os.path.isfile(txt_commentary_path):
                with open(txt_commentary_path, 'r', encoding='utf-8') as ftxt:
                    commentary_text = ftxt.read()
            else:
                commentary_text = "[No commentary found at filtered_commentaries_o3mini.txt]"
        elif use_script_flag:
            logging.info("Using commentary from filter_comms.py script ...")
            player_team_code = team_name[:3].upper()
            opponent_team_code = opponent_team[:3].upper()
            cmd_filter = (
                f"python filter_comms.py "
                f"--player_name \"{player_name}\" "
                f"--player_team_code \"{player_team_code}\" "
                f"--opponent_team_code \"{opponent_team_code}\" "
                f"--is_batsman True "
                f"--k {k} "
                f"--base_dir \"../cricinfo_data_scrapping/CommentaryData/raw_commentary\" "
                f"--output_dir \"./filtered_outputs\""
            )
            try:
                run_subprocess(cmd_filter)
                # Load commentary from filtered_commentaries.txt
                filtered_path = "./filtered_outputs/filtered_commentaries.txt"
                if os.path.isfile(filtered_path):
                    with open(filtered_path, 'r', encoding='utf-8') as fc:
                        commentary_text = fc.read()
                else:
                    commentary_text = "[No commentary output found]"
            except RuntimeError:
                logging.warning("[WARNING] filter_comms script may have failed.")
                commentary_text = "[filter_comms failure]"
        else:
            logging.info("commentary_on=True but neither script nor txt chosen => no commentary used.")
            commentary_text = "[No commentary selected]"

    # --------------------------------------------------
    # 5) Build the MASTER PROMPT
    # --------------------------------------------------
    logging.info("Building MASTER PROMPT ...")

    master_prompt = f"""\
## **Prompt Title**  
**Cricket Dismissal Prediction for {player_name}: Balancing Commentary and Statistical Data**

---

## **Persona**  
You are a **seasoned cricket analysis expert** with over 25 years of experience in coaching and scrutinizing international cricket. You specialize in synthesizing **both qualitative insights from recent match commentaries** and **quantitative player metrics** to generate accurate, context-rich predictions. Your goal is to determine the **most likely mode of dismissal** for {player_name} in upcoming or ongoing matches.

---

## **Instructional Outline**

1. **Objective**  
   - **Match**: {team_name} vs {opponent_team} in Sri Lanka  
   - **Primary Task**: Predict the most probable mode of dismissal for {player_name}.  
   - **Key Emphasis**: Pay special attention to **recent commentary(if available)** detailing {player_name}’s current form, pitch conditions, and how other batters have been dismissed in the same or comparable conditions.

2. **Data Sources**  
   - **A. Statistical Data (JSON Files)**  
     - Career-long batting averages, strike rates, typical modes of dismissal, venue-wise and opposition-wise splits, etc.  
     - These metrics serve as a baseline and a long-term perspective on {player_name}’s batting profile.  
   - **B. Recent Match Commentary(if available)** (Crucial for capturing short-term context)  
     1. **Venue and Match Context**  
     2. **Pitch Conditions**  
     3. **Comments/Quotes by or about {player_name}**  
     4. **How All Batters in the Innings Got Out**  
     5. **All Ball-by-Ball Entries Involving {player_name}**  
     6. **{player_name}’s Mode of Dismissal** (if it already happened)

3. **Analysis Approach (Chain-of-Thought Reasoning)**  
   - **Step 1: Data Examination (Long-Term Stats)**  
   - **Step 2: Commentary Examination (Recent Context)**  
   - **Step 3: Integrate and Prioritize**  

4. **Prediction and Justification**  
   - **Predict the Most Likely Dismissal Method** (bowled, caught-fielder, caught-wk, stumped, lbw, hit wicket, caught and bowled, not-out).  
   - **Provide a probability breakdown** for all eight modes that sums to 100%.  
   - **Justify** the Prediction (cite commentary lines & key stats).

5. **Actionable Insights** (Optional)

6. **Important Prompt Requirements**  
   - **Clarity of Reasoning**  
   - **No Contradictions**  
   - **Concise Final Statement**  

---

## ** General Stats Data: **
{json.dumps(general_stats, indent=4)}

---

## **Venue Wise Data:**
{json.dumps(venue_stats, indent=4)}

---

## **Against Teams Data:**
{json.dumps(snippet_block, indent=4) if snippet_block else "(No relevant data for this opponent)"}

---

## **Commentary Data(For k recent matches prior to the predicted match)**:
{commentary_text}

---
Please produce your integrated analysis with step-by-step reasoning as described,
and end with a "Probability Breakdown" plus a final statement of the predicted mode of dismissal.
"""

    # Get the directory where the current script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the full path for the output text file
    prompt_file_path = os.path.join(script_dir, "master_prompt_output.txt")

    # Write the master_prompt to the file
    with open(prompt_file_path, "w", encoding="utf-8") as f:
        f.write(master_prompt)

    logging.info(f"Master prompt written to {prompt_file_path}")

    # --------------------------------------------------
    # 6) LLM call => real openai ChatCompletion
    # --------------------------------------------------
    logging.info("Sending MASTER PROMPT to the real LLM ...")
    llm_response_text = call_llm_api(master_prompt, api_key=final_api_key)

    # --- 1. Save LLM response to a .txt file ---
    llm_txt_path = os.path.join(script_dir, "llm_response_output.txt")
    with open(llm_txt_path, "w", encoding="utf-8") as f:
        f.write(llm_response_text)
    logging.info(f"LLM response saved to: {llm_txt_path}")

    logging.info("Parsing the LLM response for probabilities & final predicted mode ...")
    prob_array, label_array, predicted_mode, ground_truth = parse_llm_analysis(llm_response_text, player_name)

    # Build the probability string
    prob_str = ",".join(str(x) for x in prob_array)
    label_str = ",".join(label_array)

    # #_commentaries is k (the user-specified or default)
    num_commentaries = k
    # We'll guess a match name => e.g. "Australia vs SL ODI 2024"
    if end_year.isdigit():
        prediction_match = f"{team_name} vs {opponent_team} ODI {end_year}"
    else:
        prediction_match = f"{team_name} vs {opponent_team}"

    # CSV row structure:
    # Batsman,train_data_span,#_commentaries,prediction_match,
    # mode_of_dissmisal_predicted_probability,
    # mode_of_dissmisal_predicted,
    # mode_of_dissmisal_ground_truth
    csv_fields = [
        player_name,
        train_data_span,
        str(num_commentaries),
        prediction_match,
        f"\"{prob_str}\"",
        f"\"{label_str}\"",
        ground_truth
    ]
    csv_line = ",".join(csv_fields)

    # --- 2. Save prediction row to a CSV file ---
    csv_output_path = os.path.join(script_dir, "llm_prediction_results.csv")

    # Check if file exists to write header only once
    write_header = not os.path.exists(csv_output_path)

    with open(csv_output_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("Batsman,train_data_span,#_commentaries,prediction_match,"
                    "mode_of_dismissal_predicted_probability,"
                    "mode_of_dismissal_predicted,"
                    "mode_of_dismissal_ground_truth\n")
        f.write(csv_line + "\n")

    logging.info(f"Prediction row appended to: {csv_output_path}")

    # --------------------------------------------------
    # 7) Print the CSV row
    # --------------------------------------------------
    logging.info("Final CSV row:")
    print(csv_line)

    logging.info("Driver process complete.")


if __name__ == "__main__":
    main()
