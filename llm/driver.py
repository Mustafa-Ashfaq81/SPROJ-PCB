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
6) Parses the LLM's textual output to extract:
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
python driver.py --player_name "Steven Smith" --team_name "Australia" --opponent_team "Pakistan" --start_year 2015 --end_date "December 23 2020" --k 5 --venue_country "Australia" --venue_ground "MCG" --match_context "1st ODI Series Match at Melbourne 4th Nov" --is_batsman True --chunk_size 100 --chunk_overlap 20 --ground_truth "Caught-Fielder" --include_general_stats False --include_venue_stats False --include_against_stats False --commentary_on False

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
import openai
from datetime import datetime
from dateutil.parser import parse
import sys

# We use dotenv to load OPENAI_API_KEY from .env if present
from dotenv import load_dotenv

logging.basicConfig(
    format='[%(levelname)s] %(message)s',
    level=logging.INFO
)

def run_subprocess(cmd):
    """
    Runs the shell command in unbuffered mode and streams its stdout to the driver in real time.
    Raises RuntimeError if the command fails.
    """

    logging.info(f"Running command: {cmd}")

    # 1) Force unbuffered (-u) if command starts with 'python'
    if isinstance(cmd, str):
        parts = shlex.split(cmd)
    else:
        parts = cmd

    if parts and parts[0] == "python":
        parts[0] = "python"
        parts.insert(1, "-u")  # Force unbuffered

    proc = subprocess.Popen(
        parts,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # line-buffered
    )

    all_output = []
    for line in iter(proc.stdout.readline, ''):
        if line:
            print(line, end='')
            all_output.append(line)

    proc.stdout.close()
    return_code = proc.wait()

    if return_code != 0:
        raise RuntimeError(f"Command failed (exit code={return_code}): {parts}")

    return "".join(all_output), ""

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

    openai.api_key = api_key

    try:
        response = openai.ChatCompletion.create(
            model="o3-mini",
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
      - prob_array (in this fixed order):
          [ Bowled, Caught-Fielder, Caught-Wicketkeeper, Stumped,
            LBW, Hit-Wicket, Caught-and-Bowled, Not-Out ]
      - label_array (same order, just strings)
      - predicted_mode (the single final mode from LLM text)
      - ground truth => "N/A"
      - reasoning => everything before "probability breakdown" (if that phrase exists),
        or the entire text otherwise.

    IMPORTANT: We match lines like:
        Bowled: 15%
        LBW: 20%
        Caught by a fielder: 40%
        ...
    exactly. Then we match:
        most likely mode of dismissal for <player_name> is <mode>
    to get the final predicted_mode.
    """

    label_array = [
        "Bowled",
        "Caught-Fielder",
        "Caught-Wicketkeeper",
        "Stumped",
        "LBW",
        "Hit-Wicket",
        "Caught-and-Bowled",
        "Not-Out"
    ]
    prob_array = [0] * 8  # 8 possible dismissal modes

    # 1) Create a lookup for synonyms.
    #    (Everything on the left is lowercase; everything on the right is the final label.)
    synonyms = {
        "bowled": "Bowled",
        "caught by a fielder": "Caught-Fielder",
        "caught by the wicketkeeper": "Caught-Wicketkeeper",
        "caught wk": "Caught-Wicketkeeper",
        "stumped": "Stumped",
        "lbw": "LBW",
        "hit wicket": "Hit-Wicket",
        "caught and bowled": "Caught-and-Bowled",
        "caught & bowled": "Caught-and-Bowled",
        "not-out": "Not-Out",
        "not out": "Not-Out"
    }

    # 2) Regex to match lines like:  "Bowled: 10%"
    #    NOTE: Adjusted to allow bullet/dash prefix, e.g. "• Bowled: 13%"
    prob_line_pattern = re.compile(
        r'^\s*(?:[*\-•]+\s*)?(?P<mode>[^:：]+?)\s*[:：]\s*(?P<pct>\d+)\%',
        re.IGNORECASE | re.MULTILINE
    )

    # 3) Go through each line that fits "<something>: <digits>%"
    for match in prob_line_pattern.finditer(llm_text):
        raw_mode = match.group("mode").strip().lower()
        pct_str = match.group("pct")
        pct_val = int(pct_str)

        # Normalize the mode via synonyms
        if raw_mode in synonyms:
            final_label = synonyms[raw_mode]
            idx = label_array.index(final_label)
            prob_array[idx] = pct_val
        else:
            # Fallback attempt for partial matches, if needed
            if "caught by a fielder" in raw_mode:
                prob_array[label_array.index("Caught-Fielder")] = pct_val
            elif "bowled" in raw_mode:
                prob_array[label_array.index("Bowled")] = pct_val
            elif "wicketkeeper" in raw_mode or "caught wk" in raw_mode:
                prob_array[label_array.index("Caught-Wicketkeeper")] = pct_val
            elif "stumped" in raw_mode:
                prob_array[label_array.index("Stumped")] = pct_val
            elif "lbw" in raw_mode:
                prob_array[label_array.index("LBW")] = pct_val
            elif "hit wicket" in raw_mode:
                prob_array[label_array.index("Hit-Wicket")] = pct_val
            elif "caught and bowled" in raw_mode or "caught & bowled" in raw_mode:
                prob_array[label_array.index("Caught-and-Bowled")] = pct_val
            elif "not" in raw_mode:
                prob_array[label_array.index("Not-Out")] = pct_val

    # 4) Find final predicted mode.
    pred_pattern = re.compile(
        rf"most\s+likely\s+mode\s+of\s+dismissal\s+for\s+{re.escape(player_name)}\s+is\s+(?P<pmode>.+)$",
        re.IGNORECASE | re.MULTILINE
    )
    pred_mode = "N/A"
    pm = pred_pattern.search(llm_text)
    if pm:
        raw_pred = pm.group("pmode").strip().lower()
        raw_pred = raw_pred.strip('"\'')
        if raw_pred in synonyms:
            pred_mode = synonyms[raw_pred]
        else:
            if "caught by a fielder" in raw_pred:
                pred_mode = "Caught-Fielder"
            elif "bowled" in raw_pred:
                pred_mode = "Bowled"
            elif "wicketkeeper" in raw_pred or "caught wk" in raw_pred:
                pred_mode = "Caught-Wicketkeeper"
            elif "stumped" in raw_pred:
                pred_mode = "Stumped"
            elif "lbw" in raw_pred:
                pred_mode = "LBW"
            elif "hit wicket" in raw_pred:
                pred_mode = "Hit-Wicket"
            elif "caught and bowled" in raw_pred or "caught & bowled" in raw_pred:
                pred_mode = "Caught-and-Bowled"
            elif "not" in raw_pred:
                pred_mode = "Not-Out"
            else:
                pred_mode = raw_pred.title()

    ground_truth = "N/A"

    reasoning_marker = re.search(r'probability breakdown', llm_text, re.IGNORECASE)
    if reasoning_marker:
        reasoning = llm_text[:reasoning_marker.start()].strip()
    else:
        reasoning = llm_text.strip()

    return prob_array, label_array, pred_mode, ground_truth, reasoning

def parse_end_date(end_date_str):
    """
    Parse 'end_date_str' to produce 'YYYY_MM_DD' with zero-padded month and day.
    If end_date_str.lower() == 'all', we will return 'all'.
    Otherwise, we attempt to parse it into year, month, day.
    """
    if end_date_str.lower() == 'all':
        return 'all'
    try:
        dt = parse(end_date_str)
    except Exception:
        return 'all'
    return dt.strftime("%Y_%m_%d")  # zero-padded month, zero-padded day

def main():
    load_dotenv()
    final_api_key = os.environ.get("OPENAI_API_KEY", None)
    if not final_api_key:
        logging.warning("[WARNING] Environment variable OPENAI_API_KEY not found. LLM call may use fallback text.")

    parser = argparse.ArgumentParser(
        description="Majestic driver script that logs steps, gathers stats & commentary, "
                    "builds MASTER PROMPT, calls LLM, parses final text, and outputs one CSV row."
    )
    parser.add_argument("--player_name", required=True, help="Player name, e.g. 'Steven Smith'")
    parser.add_argument("--team_name", required=True, help="Team name, e.g. 'Australia'")
    parser.add_argument("--opponent_team", required=True, help="Opponent team code or name, e.g. 'SL' or 'Sri Lanka'")
    parser.add_argument("--start_year", default="all", help="Start year for stats or 'all'")
    parser.add_argument("--end_date", default="all", help="End date for stats or 'all'")

    parser.add_argument("--match_context", required=True, help="Context for the match, e.g. 'ICC World Cup group match 2'")

    parser.add_argument("--commentary_on", default="False", choices=["True", "False"],
                        help="Whether to embed commentary at all. Default False.")
    parser.add_argument("--use_script", default="True", choices=["True", "False"],
                        help="If True, commentary is gathered by calling filter_comms.py.")
    parser.add_argument("--use_txt", default="False", choices=["True", "False"],
                        help="If True, commentary is read from a local txt file.")
    parser.add_argument("--k", type=int, default=5, help="Number of commentary files to filter. Default 5.")

    parser.add_argument("--is_batsman", default="True", choices=["True", "False"],
                        help="Whether the player is batting (for filter_comms). Default True.")
    parser.add_argument("--chunk_size", type=int, default=100,
                        help="Chunk size for commentary splitting in filter_comms. Default 100.")
    parser.add_argument("--chunk_overlap", type=int, default=20,
                        help="Overlap size for commentary splitting in filter_comms. Default 20.")

    # MODIFICATION #2: Add ground_truth argument
    parser.add_argument("--ground_truth", default="N/A",
                        help="(Optional) Provide the ground truth mode of dismissal, e.g. 'Caught-Fielder'")

    # ----------- NEW ARGS (needed for filter_comms_chunks.py) -----------
    parser.add_argument("--venue_country", required=True, help="Country filter for venuewise commentary")
    parser.add_argument("--venue_ground", default="", help="Ground filter for venuewise commentary")

    # ----------- NEW COMPONENT TOGGLE ARGS -----------
    parser.add_argument("--include_general_stats", default="True", choices=["True", "False"],
                        help="Include General Stats section in master prompt. Default True.")
    parser.add_argument("--include_venue_stats", default="True", choices=["True", "False"],
                        help="Include Venue Wise Stats section in master prompt. Default True.")
    parser.add_argument("--include_against_stats", default="True", choices=["True", "False"],
                        help="Include Against Teams Stats section in master prompt. Default True.")

    args = parser.parse_args()

    player_name = args.player_name
    team_name = args.team_name
    opponent_team = args.opponent_team
    start_year = args.start_year
    raw_end_date = args.end_date
    end_date_parsed = parse_end_date(raw_end_date)  # 'YYYY_MM_DD' or 'all'
    match_context = args.match_context
    commentary_on = (args.commentary_on.lower() == "true")
    use_script_flag = (args.use_script.lower() == "true")
    use_txt_flag = (args.use_txt.lower() == "true")
    k = args.k
    cli_ground_truth = args.ground_truth

    include_general_stats = (args.include_general_stats.lower() == "true")
    include_venue_stats = (args.include_venue_stats.lower() == "true")
    include_against_stats = (args.include_against_stats.lower() == "true")

    # NEW: expose for later
    venue_country_arg = args.venue_country
    venue_ground_arg = args.venue_ground

    logging.info("Starting driver process...")

    if not commentary_on:
        k = 0

    # 1) Run get_player_stats.py
    logging.info("Running get_player_stats.py ...")
    safe_player_name = player_name.lower().replace(' ', '_')
    try:
        cmd_general = (
            f"python ../cricinfo_data_scrapping/Scripts/get_player_stats.py "
            f"\"{player_name}\" \"{team_name}\" {start_year} \"{raw_end_date}\""
        )
        run_subprocess(cmd_general)
    except RuntimeError:
        logging.warning("[WARNING] get_player_stats script may have failed.")

    # Build general stats filename
    if start_year.lower() == 'all' or end_date_parsed == 'all':
        general_stats_filename = f"{safe_player_name}_all_span_stats.json"
    else:
        general_stats_filename = f"{safe_player_name}_{start_year}_{end_date_parsed}_stats.json"

    general_stats_path = os.path.join(
        "..", "cricinfo_data_scrapping", "aggregated_stats", general_stats_filename
    )
    general_stats = load_json_file(general_stats_path)

    raw_span = general_stats.get("span", "Unknown")
    train_data_span = raw_span.replace(" ", "").replace("–", "-").replace("--", "-")

    # 2) Run get_player_stats_venuewise.py
    logging.info("Running get_player_stats_venuewise.py ...")
    try:
        cmd_venue = (
            f"python ../cricinfo_data_scrapping/Scripts/get_player_stats_venuewise.py "
            f"\"{player_name}\" \"{team_name}\" {start_year} \"{raw_end_date}\""
        )
        run_subprocess(cmd_venue)
    except RuntimeError:
        logging.warning("[WARNING] get_player_stats_venuewise script may have failed.")

    if start_year.lower() == 'all' or end_date_parsed == 'all':
        venue_filename = f"{safe_player_name}_all_venuewise.json"
    else:
        venue_filename = f"{safe_player_name}_{start_year}_{end_date_parsed}_venuewise.json"

    venue_stats_path = os.path.join(
        "..", "cricinfo_data_scrapping", "aggregated_stats_venue_wise", venue_filename
    )
    venue_stats = load_json_file(venue_stats_path)

    # retrieve venue snippet
    venue_snippet_block = {}
    if "venues" in venue_stats:
        for k_ in venue_stats["venues"].keys():
            if venue_country_arg.lower() in k_.lower():
                venue_snippet_block = {k_: venue_stats["venues"][k_]}
                break

    # 3) Run get_player_stats_against_teams.py
    logging.info("Running get_player_stats_against_teams.py ...")
    try:
        cmd_against = (
            f"python ../cricinfo_data_scrapping/Scripts/get_player_stats_against_teams.py "
            f"\"{player_name}\" \"{team_name}\" {start_year} \"{raw_end_date}\""
        )
        run_subprocess(cmd_against)
    except RuntimeError:
        logging.warning("[WARNING] get_player_stats_against_teams script may have failed.")

    if start_year.lower() == 'all' or end_date_parsed == 'all':
        against_filename = f"{safe_player_name}_all_against_teams.json"
    else:
        against_filename = f"{safe_player_name}_{start_year}_{end_date_parsed}_against_teams.json"

    against_stats_path = os.path.join(
        "..", "cricinfo_data_scrapping", "aggregated_stats_against_teams", against_filename
    )
    against_stats = load_json_file(against_stats_path)

    snippet_block = {}
    if "oppositions" in against_stats:
        for k_ in against_stats["oppositions"].keys():
            if opponent_team.lower() in k_.lower():
                snippet_block = {k_: against_stats["oppositions"][k_]}
                break

    # 4) commentary
    commentary_text = ""
    if not commentary_on:
        commentary_text = "[Commentary OFF]"
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        team_maps_path = os.path.join(script_dir, "team_maps.json")
        team_map = load_json_file(team_maps_path)
        opponent_short = team_map.get(opponent_team, opponent_team)
        safe_player_name_for_commentary = player_name.replace(" ", "")

        if use_txt_flag and use_script_flag:
            logging.info("Both script & txt commentary modes are enabled. Defaulting to txt approach.")
            txt_commentary_path = (
                f"filtered_outputs/filtered_commentaries_"
                f"{safe_player_name_for_commentary}_{opponent_short}_{k}.txt"
            )
            if os.path.isfile(txt_commentary_path):
                with open(txt_commentary_path, 'r', encoding='utf-8') as ftxt:
                    commentary_text = ftxt.read()
            else:
                commentary_text = f"[No commentary found at {txt_commentary_path}]"
        elif use_txt_flag:
            logging.info("Using commentary from local txt file (no script).")
            txt_commentary_path = (
                f"filtered_outputs/filtered_commentaries_"
                f"{safe_player_name_for_commentary}_{opponent_short}_{k}.txt"
            )
            if os.path.isfile(txt_commentary_path):
                with open(txt_commentary_path, 'r', encoding='utf-8') as ftxt:
                    commentary_text = ftxt.read()
            else:
                commentary_text = f"[No commentary found at {txt_commentary_path}]"
        elif use_script_flag:
            logging.info("Using commentary from filter_comms_chunks.py script ...")
            comm_player_name = args.player_name.replace(" ", "")
            player_team_code = team_map.get(team_name, team_name)
            opponent_team_code = opponent_short
            cmd_filter = (
                f"python filter_comms_chunks.py "
                f"--player_name \"{comm_player_name}\" "
                f"--player_team_code \"{player_team_code}\" "
                f"--opponent_team_code \"{opponent_team_code}\" "
                f"--is_batsman {args.is_batsman} "
                f"--k {k} "
                f"--against_comms 0 "
                f"--venue_comms {k} "
                f"--venue_country \"{venue_country_arg}\" "
                f"--venue_ground \"{venue_ground_arg}\" "
                f"--end_date \"{args.end_date}\" "
                f"--base_dir \"../cricinfo_data_scrapping/CommentaryData/processed_commetaries/raw\" "
                f"--venue_dir \"../cricinfo_data_scrapping/CommentaryData/processed_commetaries/venuewise\" "
                f"--chunk_size {args.chunk_size} "
                f"--chunk_overlap {args.chunk_overlap} "
                f"--output_dir \"./filtered_outputs\" "
            )
            try:
                run_subprocess(cmd_filter)
                filtered_path = (
                    f"filtered_outputs/filtered_commentaries_"
                    f"{safe_player_name_for_commentary}_{opponent_short}_{k}.txt"
                )
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

    # 5) Build MASTER PROMPT
    logging.info("Building MASTER PROMPT ...")

    # ----- Prompt header through formatting requirements -----
    master_prompt_header = f"""\
## **Prompt Title**  
**Cricket Dismissal Prediction for {player_name}: Balancing Commentary and Statistical Data**

---

## **Persona**  
You are a **seasoned cricket analysis expert** with over 25 years of experience in coaching and scrutinizing international cricket. You specialize in synthesizing **both qualitative insights from recent match commentaries** and **quantitative player metrics** to generate accurate, context-rich predictions. Your goal is to determine the **most likely mode of dismissal** for {player_name} in upcoming or ongoing matches.

---

## **Instructional Outline**

1. **Objective**  
   - **Match**: {match_context}, {team_name} vs {opponent_team} in {venue_country_arg}  
   - **Primary Task**: Predict the most probable mode of dismissal for {player_name}.  
   - **Key Emphasis**: Pay special attention to: **Venue Stats of {player_name}** because it will tell you how this player performs in the venue the prediction match will be played, **{player_name} Stats against the Opponent** because it will tell you how the player performs against the specific team, and most important **recent commentary (if available)** detailing {player_name}'s current form, pitch conditions, and how other batters have been dismissed in the same or comparable conditions.

2. **Data Sources**  
   - **A. Statistical Data (JSON Files)**  
     - Career-long batting averages, strike rates, typical modes of dismissal, venue-wise and opposition-wise splits, etc.  
   - **B. Recent Match Commentary (if available)** (Crucial for capturing short-term context)  
     1. **Venue and Match Context**  
     2. **Pitch Conditions**  
     3. **Comments/Quotes by or about {player_name}**  
     4. **How All Batters in the Innings Got Out**  
     5. **All Ball-by-Ball Entries Involving {player_name}**  
     6. **{player_name}'s Mode of Dismissal** (if it already happened)

3. **Analysis Approach (Chain-of-Thought Reasoning)**  
   - **Step 1:** Data Examination (Long-Term Stats)  
   - **Step 2:** Commentary Examination (Recent Context)  
   - **Step 3:** Integrate and Prioritize  

   **Detailed Reasoning**  
   Show how you arrive at your final predicted mode of dismissal by summarizing:  
   1) Player's inherent weaknesses and past patterns.  
   2) Conditions of the pitch or environment favoring certain modes of dismissal.  
   3) Any relevant points from the commentary or recent form.

4. **Prediction and Justification**  
   - **Predict the Most Likely Dismissal Method** (bowled, caught-fielder, caught-wk, stumped, lbw, hit wicket, caught and bowled, not-out).  
   - **Provide a probability breakdown** for all eight modes that sums to 100 %.  

5. **Important Prompt Requirements**  
   - **Clarity of Reasoning**  
   - **No Contradictions**  
   - **Concise Final Statement**  

6. **Exact Output Formatting Requirements**  
   - Present your probability breakdown EXACTLY as follows (one entry per line, preserving the exact text and punctuation so it can be parsed automatically):  
     Bowled: <number>%  
     Caught by a fielder: <number>%  
     Caught by the wicketkeeper: <number>%  
     Stumped: <number>%  
     LBW: <number>%  
     Hit wicket: <number>%  
     Caught and bowled: <number>%  
     Not-out: <number>%  
   - Then, finish with this exact text (again, to be parsed automatically):  
     most likely mode of dismissal for {player_name} is <mode>\
"""

    # ----- Conditional sections -----
    # General Stats
    general_stats_content = (
        json.dumps(general_stats, indent=4)
        if include_general_stats else "{General Stats Not Available}"
    )
    general_stats_section = f"""\


---

## ** General Stats Data of {player_name}: **
{general_stats_content}\
"""

    # Venue-wise Stats
    venue_stats_content = (
        json.dumps(venue_snippet_block, indent=4) if venue_snippet_block else "(No relevant data for this venue)"
    ) if include_venue_stats else "{Venue Wise Stats Not Available}"
    venue_stats_section = f"""\


---

## **Venue Wise Stats Data of {player_name} in {venue_country_arg}:**
{venue_stats_content}\
"""

    # Against-Teams Stats
    against_stats_content = (
        json.dumps(snippet_block, indent=4) if snippet_block else "(No relevant data for this opponent)"
    ) if include_against_stats else "{Against Teams Stats Not Available}"
    against_stats_section = f"""\


---

## **Against Teams Stats Data of {player_name} against {opponent_team}:**
{against_stats_content}\
"""

    # Commentary
    commentary_content = commentary_text if commentary_on else "{Commentary Data Not Available}"
    commentary_section = f"""\


---

## **Commentary Data (for k recent matches prior to the predicted match) of {player_name}:**
{commentary_content}\
"""

    master_prompt_footer = """\


---
Please produce your integrated analysis with step-by-step reasoning as described,
and then follow the **Exact Output Formatting Requirements** above.
"""

    master_prompt = (
        master_prompt_header
        + general_stats_section
        + venue_stats_section
        + against_stats_section
        + commentary_section
        + master_prompt_footer
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file_path = os.path.join(script_dir, "master_prompt_output.txt")
    with open(prompt_file_path, "w", encoding="utf-8") as f:
        f.write(master_prompt)

    logging.info(f"Master prompt written to {prompt_file_path}")

    logging.info(
        "Exiting now so you can review master_prompt_output.txt. "
        "Remove the sys.exit(0) call below once satisfied with the prompt."
    )
    sys.exit(0)

    # 6) LLM call
    logging.info("Sending MASTER PROMPT to the real LLM ...")
    llm_response_text = call_llm_api(master_prompt, api_key=final_api_key)

    llm_txt_path = os.path.join(script_dir, "llm_response_output.txt")
    with open(llm_txt_path, "w", encoding="utf-8") as f:
        f.write(llm_response_text)
    logging.info(f"LLM response saved to: {llm_txt_path}")

    # parse
    logging.info("Parsing the LLM response for probabilities & final predicted mode ...")
    prob_array, label_array, predicted_mode, parsed_ground_truth, reasoning_text = parse_llm_analysis(llm_response_text, player_name)

    # MODIFICATION #2 => use the user-provided ground_truth, else "N/A"
    final_ground_truth = cli_ground_truth if cli_ground_truth else parsed_ground_truth

    # MODIFICATION #1 => sort by probability descending
    zipped = list(zip(prob_array, label_array))
    zipped.sort(key=lambda x: x[0], reverse=True)
    sorted_prob_array = [str(item[0]) for item in zipped]
    sorted_label_array = [item[1] for item in zipped]

    prob_str = ", ".join(sorted_prob_array)
    label_str = ", ".join(sorted_label_array)

    num_commentaries = k
    prediction_match = f"{match_context} {team_name} vs {opponent_team} in {venue_country_arg} {venue_ground_arg}"
    commentary_flag = "on" if commentary_on else "off"

    # sanitize reasoning
    reasoning_sanitized = reasoning_text.replace("\r", " ")
    reasoning_sanitized = reasoning_sanitized.replace("\n", " ")
    reasoning_sanitized = re.sub(r"\s+", " ", reasoning_sanitized).strip()
    reasoning_sanitized = reasoning_sanitized.replace("\"", "'")
    reasoning_sanitized = reasoning_sanitized.replace(",", ";")
    reasoning_sanitized = reasoning_sanitized.encode("ascii", "ignore").decode("ascii")

    general_stats_flag        = "on" if include_general_stats   else "off"
    venuewise_stats_flag      = "on" if include_venue_stats     else "off"
    against_teams_stats_flag  = "on" if include_against_stats   else "off"
    commentary_flag           = "on" if commentary_on           else "off"

    csv_fields = [
        player_name,
        prediction_match,
        train_data_span,
        general_stats_flag,
        venuewise_stats_flag,
        against_teams_stats_flag,
        commentary_flag,
        str(num_commentaries),
        f"\"{prob_str}\"",
        f"\"{label_str}\"",
        predicted_mode,
        final_ground_truth,
        f"\"{reasoning_sanitized}\""
    ]
    csv_line = ",".join(csv_fields)

    csv_output_path = os.path.join(script_dir, "llm_prediction_results.csv")
    write_header = not os.path.exists(csv_output_path)

    with open(csv_output_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(
                "player_name,prediction_match,train_data_span,"
                "general_stats_flag,venuewise_stats_flag,against_teams_stats_flag,commentary,"
                "#_commentaries,"
                "mode_of_dismissal_predicted_probability,"
                "mode_of_dismissal_predicted,"
                "most_likely_mode,"
                "mode_of_dismissal_ground_truth,"
                "reasoning\n"
            )
        f.write(csv_line + "\n")

    logging.info(f"Prediction row appended to: {csv_output_path}")

    logging.info("Final CSV row:")
    print(csv_line)
    logging.info("Driver process complete.")

if __name__ == "__main__":
    main()
