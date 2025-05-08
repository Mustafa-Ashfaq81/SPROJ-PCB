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
python driver.py --player_name "Steven Smith" --team_name "Australia" --opponent_team "Pakistan" --start_year 2015 --end_date "December 23 2020" --k 5 --venue_country "Australia" --venue_ground "Melbourne" --match_context "1st ODI Series Match at Melbourne 4th Nov" --is_batsman True --ground_truth "Caught-Fielder" --include_general_stats False --include_venue_stats False --include_against_stats False --commentary_on False

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
            model="gpt-4o-mini",
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

def parse_llm_analysis(llm_text: str, player_name: str):
    """
    Parse the LLM's answer and return:
        prob_map ........ dict {label → float %}
        pred_mode ....... string (from the “most likely …” sentence or "")
        reasoning ....... string (everything before “probability breakdown”)
    The 8 canonical labels are fixed; any missing entry gets 0 %.
    """

    CANON = [
        "Bowled",
        "Caught-Fielder",
        "Caught-Wicketkeeper",
        "Stumped",
        "LBW",
        "Hit-Wicket",
        "Caught-and-Bowled",
        "Not-Out",
    ]

    # ------------------------- regexes & look‑ups -------------------------
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
        "not out": "Not-Out",
    }

    prob_line_pattern = re.compile(
        r"""
        ^[ \t\u00A0\u200B]*                                    # leading whitespace
        (?:[*\-–—‣●▪•\u2022\u2023\u25E6\u2043]+\s*)?           # bullet glyphs
        (?P<mode>[^:：–—\n]+?)                                 # mode text
        \s*[:：–—]\s*                                          # separator
        (?P<pct>\d+(?:\.\d+)?)\s*                              # number (int/float)
        (?:%|％|percent)?\s*$                                  # percent token
        """,
        re.IGNORECASE | re.MULTILINE | re.VERBOSE | re.UNICODE,
    )

    pred_sentence = re.compile(
        rf"""most\s+likely\s+mode\s+of\s+dismissal\s+for\s+{re.escape(player_name)}\s+is\s+(?P<pmode>.+?)\s*$""",
        re.IGNORECASE | re.MULTILINE,
    )

    # ----------------------------- parsing -------------------------------
    prob_map = {lbl: 0.0 for lbl in CANON}

    for m in prob_line_pattern.finditer(llm_text):
        mode_raw = m.group("mode").strip().lower()
        pct_val = float(m.group("pct"))

        # canonicalise mode name
        mode_key = None
        if mode_raw in synonyms:
            mode_key = synonyms[mode_raw]
        else:  # fuzzy contains
            for k, v in synonyms.items():
                if k in mode_raw:
                    mode_key = v
                    break
        if mode_key:
            prob_map[mode_key] = pct_val

    # ---------- predicted mode from narrative sentence ----------
    pm = pred_sentence.search(llm_text)
    predicted_mode = ""
    if pm:
        pmode_raw = pm.group("pmode").strip().lower().strip("\"'")
        predicted_mode = synonyms.get(pmode_raw, "")
        if not predicted_mode:  # fuzzy fallback
            for k, v in synonyms.items():
                if k in pmode_raw:
                    predicted_mode = v
                    break

    # ---------- reasoning ----------
    reasoning_split = re.search(r"probability breakdown", llm_text, re.IGNORECASE)
    reasoning = llm_text[: reasoning_split.start()].strip() if reasoning_split else llm_text.strip()

    return prob_map, predicted_mode, reasoning

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

    # MODIFICATION #2: Add ground_truth argument
    parser.add_argument("--ground_truth", default="N/A",
                        help="(Optional) Provide the ground truth mode of dismissal, e.g. 'Caught-Fielder'")

    # ----------- NEW ARGS (needed for filter_comms.py) -----------
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
            logging.info("Using commentary from filter_recent_comms.py script ...")
            comm_player_name = args.player_name.replace(" ", "")
            player_team_code = team_map.get(team_name, team_name)
            opponent_team_code = opponent_short
            cmd_filter = (
                f"python filter_recent_embeddings_comms.py "
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
                f"--output_dir \"./filtered_outputs\" "
            )
            try:
                run_subprocess(cmd_filter)
                filtered_path = (
                    f"filtered_outputs/filtered_commentaries_"
                    f"{safe_player_name_for_commentary}_{player_team_code}_{k}.txt"
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
   - **Match**: {match_context}
   - **Primary Task**: Predict the most probable mode of dismissal for {player_name} in this match.  
   - **Key Emphasis**: Pay special attention to: 
        1. **Venue Stats of {player_name} (if available)** because it will tell you how this player performs in the venue the prediction match will be played.
        2. **{player_name} Stats against the Opponent (if available)** because it will tell you how the player performs against the specific team.
        3. **recent commentary (if available)** detailing {player_name}'s current form, pitch conditions, and how other batters have been dismissed in the same or comparable conditions.
        4. **Recent Commentary as Primary Driver**: treat every dismissal or form signal in the commentaries as your highest-signal evidence—if it conflicts with older stats, let it override those lower-signal sources.

2. **Data Sources**  
   - **A. General Career Stats (JSON Files)**  
     • Career-long batting averages, strike rates, typical modes of dismissal.  
   - **B. Venue-Wise Stats (JSON Files)**  
     • Venue splits (e.g. performance in {venue_country_arg}), Bowled/Caught/… rates.  
   - **C. Opponent-Wise Stats (JSON Files)**  
     • Performance vs {opponent_team}, specific dismissal breakdown.  
   - **D. Recent Match Commentary (if available)**  
     • Ball-by-ball notes, pitch conditions, quotes, and dismissal events.

> **Stat Heading Guide (read carefully before analysis)**  
> • The **“batting”** object contains the core performance metrics for {player_name} and should drive most of your reasoning.  
> • The **“dismissals”** array shows how those wickets were taken by bowler type — consider this *high-signal* information.  
> • The **“bowling”** and **“bowling_vs_batting_type”** objects are included for completeness; for a specialist batter they are usually *low-signal* and may be weighted lightly.  
> • Treat any `null` values as “not applicable / data unavailable.”

3. **Analysis Approach (Chain-of-Thought Reasoning)**  
   - **Step 1:** Examine General Career Stats  
   - **Step 2:** Examine Venue-Wise Stats (note any deviations in dismissal patterns)  
   - **Step 3:** Examine Opponent-Wise Stats (note any deviations in dismissal patterns)  
   - **Step 4:** Examine Recent Commentary (context, conditions, specific dismissals)  
   - **Step 5:** Integrate & Prioritize All Data Sources Which Are Made Available
   - **Step 6:** Final Prediction & Justification

   **Detailed Reasoning** 
   Show how you arrive at your final predicted mode of dismissal by summarizing each data source, **and for each probability line in your breakdown include a parenthetical that cites raw counts from career, venue-wise, and opponent-wise stats.** 
   Also show how you learn:  
   1) Player's inherent weaknesses and past patterns gathered from recent commentaries.  
   2) Conditions of the pitch or environment favoring certain modes of dismissal.  
   3) Any relevant points from the commentary or recent form.

4. **Prediction and Justification**  
   - **Predict the Most Likely Dismissal Method** (bowled, caught-fielder, caught-wk, stumped, lbw, hit wicket, caught and bowled, not-out).  
   - **Provide a probability breakdown** for all eight modes that sums to 100 %.  

5. **Important Prompt Requirements**  
   - **Use Only Provided Data** - Do **not** consult the web or any external knowledge; rely **exclusively** on the statistics and commentary contained in this prompt.  
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

## **Commentary Data (for k recent matches prior to the predicted match) of {player_name} to get a concrete idea of the recent form and the modes of dismissals of {player_name}. This is very crucial to analyse for this task. :**
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

    # logging.info(
    #     "Exiting now so you can review master_prompt_output.txt. "
    #     "Remove the sys.exit(0) call below once satisfied with the prompt."
    # )
    # sys.exit(0)

    # 6) LLM call
    logging.info("Sending MASTER PROMPT to the real LLM ...")
    llm_response_text = call_llm_api(master_prompt, api_key=final_api_key)

    llm_txt_path = os.path.join(script_dir, "llm_response_output.txt")
    with open(llm_txt_path, "w", encoding="utf-8") as f:
        f.write(llm_response_text)
    logging.info(f"LLM response saved to: {llm_txt_path}")

    # parse
    logging.info("Parsing the LLM response for probabilities & final predicted mode ...")
    prob_map, narrative_mode, reasoning_text = parse_llm_analysis(
        llm_response_text, player_name
    )

    # Sort descending by probability
    sorted_items = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)

    prob_str   = ", ".join(
        f"{int(p) if p.is_integer() else p:g}" for _, p in sorted_items
    )
    label_str  = ", ".join(lbl for lbl, _ in sorted_items)

    # Ensure “most_likely_mode” == top label (unless all zeros)
    top_label = sorted_items[0][0] if sorted_items and sorted_items[0][1] > 0 else "N/A"
    if not narrative_mode or narrative_mode.lower() != top_label.lower():
        logging.warning(
            f"[Mismatch] Narrative said '{narrative_mode}' but highest‑probability is '{top_label}'. "
            "Using the highest probability."
        )
    predicted_mode = top_label

    num_commentaries = k
    prediction_match = f"{match_context}"
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
        cli_ground_truth,
        f"\"{reasoning_sanitized}\""
    ]
    csv_line = ",".join(csv_fields)

    csv_output_path = os.path.join(script_dir, f"llm_prediction_results_{player_name}.csv")
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
