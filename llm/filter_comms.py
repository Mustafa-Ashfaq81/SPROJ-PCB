"""
filter_comms.py

Example Usage (Windows or Linux shell):
--------------------------------------
python filter_comms.py --player_name "StevenSmith" --player_team_code "AUS" --opponent_team_code "SL" --is_batsman True --k 5 --base_dir "../cricinfo_data_scrapping/CommentaryData/raw_commentary" --output_dir "./filtered_outputs"

Overview:
---------
1. This script locates up to k=25 relevant commentary files for a given player from a directory structure like:
   "../CommentaryData/raw_commentary/{PlayerName}".
2. It prints which files are selected, so you can verify.
3. In batches of 5 files, it calls GPT-03-mini (or a similar model) with the filtering prompt (below).
4. It saves each batch’s filtered commentary output for subsequent usage in driver.py.

API Key Handling:
-----------------
- The script reads the environment variable "OPENAI_API_KEY" for authentication.
- If "OPENAI_API_KEY" is not found, it warns you that LLM calls may fail
  and uses a simulated response in the `call_llm_filtering_api` function.

"""

import os
import re
import argparse
from typing import List
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# 1) The filtering prompt (exact lines from your specified requirements)
# ------------------------------------------------------------------------------
FILTERING_PROMPT_TEMPLATE = """
You are given the **complete match commentary text** for a cricket match. Your task is to **extract and return only the exact lines** (verbatim, without paraphrasing) from the commentary that meet the following criteria related to **{PLAYER_NAME}**. Specifically, search the entire commentary and gather lines that include:

1. **Venue and Match Context**
- Any lines explicitly stating the venue or location of the match, crowd attendance, or references to the stadium conditions—especially if they refer to how it might affect batting or bowling.

2. **Pitch Conditions**
- Any lines (including quotations) that describe the pitch behavior, pace, bounce, slowness, or anything else about the surface that could be relevant to how batters, especially {PLAYER_NAME}, perform or get out.

3. **Comments/Quotes by or about {PLAYER_NAME}**
- Any direct quotes where {PLAYER_NAME} discusses how they felt at the crease, their form, the difficulty of the pitch, or mentions of their batting approach and mindset.

4. **How All Batters in the Innings Got Out**
- Extract lines describing dismissals of any batter in the innings. Include the exact commentary lines for each such dismissal.

5. **All Ball-by-Ball Entries Involving {PLAYER_NAME}**
- Every commentary line that mentions {PLAYER_NAME}’s deliveries faced, runs scored, boundaries, near-dismissals, or any relevant event while {PLAYER_NAME} was batting.
- If a line references {PLAYER_NAME}’s partnership, shot selection, or a bowler’s tactics against them, include it exactly as it appears.

6. **{PLAYER_NAME}’s Mode of Dismissal**
- The precise commentary line that states how {PLAYER_NAME} got out (caught, bowled, run out, etc.). This is **crucial**. Extract it word-for-word from the commentary.

**Important Requirements**:
- **Return only the exact lines from the original commentary**; do not paraphrase or summarize.
- If a particular commentary snippet is relevant to more than one of the above categories, simply include the full line once in the final output.
- Maintain the **original order** in which these lines appear in the commentary.
- Do not include any commentary text that is irrelevant to the six points above.
"""

# ------------------------------------------------------------------------------
# 2) File selection logic
# ------------------------------------------------------------------------------
def get_commentary_files_for_player(
    base_dir: str,
    player_name: str,
    player_team_code: str,
    opponent_team_code: str,
    is_batsman: bool,
    k: int = 25
) -> List[str]:
    """
    1) Looks in the directory: base_dir/player_name (e.g. ../CommentaryData/raw_commentary/StevenSmith)
    2) Collects commentary files matching the naming convention:
         {matchID}_{TeamThatIsBatting}_commentary_*.txt
    3) If is_batsman=True, prefer files where batting_team == player_team_code;
       If is_batsman=False, prefer files where batting_team == opponent_team_code.
    4) Sort by descending matchID. Then pick up to k files.

    Returns:
        A list of file paths to the selected commentary files.
    """
    # Normalize path to avoid confusion with different OS path formats
    player_dir = os.path.normpath(os.path.join(base_dir, player_name))

    if not os.path.isdir(player_dir):
        path_print = player_dir.replace("\\", "/")
        print(f"[WARNING] Directory not found: {path_print}")
        return []

    all_files = os.listdir(player_dir)
    pattern = re.compile(r"(\d+)_(\w+)_commentary_.*\.txt$")
    matched_files = []

    for filename in all_files:
        match = pattern.match(filename)
        if not match:
            continue
        match_id_str, batting_team = match.groups()
        match_id = int(match_id_str)

        # Batsman => batting_team == player_team_code
        # Bowler  => batting_team == opponent_team_code
        if is_batsman and (batting_team == player_team_code):
            matched_files.append((match_id, filename))
        elif not is_batsman and (batting_team == opponent_team_code):
            matched_files.append((match_id, filename))

    # Sort by match_id descending => most recent first
    matched_files.sort(key=lambda x: x[0], reverse=True)

    # Take up to k
    top_k_files = [os.path.join(player_dir, mf[1]) for mf in matched_files[:k]]
    return top_k_files

# ------------------------------------------------------------------------------
# 3) LLM call to filter commentary
# ------------------------------------------------------------------------------
def call_llm_filtering_api(
    raw_commentary: str,
    player_name: str,
    api_key: str = None
) -> str:
    """
    Sends the raw_commentary text to GPT-03-mini (or any other LLM)
    with the filtering prompt. Receives a filtered text response.

    If no valid API key is found, returns a simulated response.
    """
    # Build the full user prompt
    filtering_prompt = FILTERING_PROMPT_TEMPLATE.format(PLAYER_NAME=player_name)
    user_prompt = (
        f"{filtering_prompt}\n\n"
        f"---\n\n"
        f"Below is the complete commentary:\n"
        f"{raw_commentary}\n\n"
        f"---\n\n"
        f"Return only the relevant lines in the original order, as per instructions."
    )

    # If no API key, use a simulated response
    if not api_key:
        print("[WARNING] OPENAI_API_KEY not found. Using a simulated LLM response.\n")
        return (
            "SIMULATED LLM RESPONSE:\n"
            "This is where the LLM would return the verbatim lines meeting the criteria."
        )

    # Actual GPT-03-mini call (example)
    import openai
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # or another model name you have access to
        messages=[{"role": "user", "content": user_prompt}],
    )
    filtered_text = response["choices"][0]["message"]["content"]

    return filtered_text

# ------------------------------------------------------------------------------
# 4) Main routine: (a) gather files, (b) process them individually, (c) call LLM
# ------------------------------------------------------------------------------
def main():
    # Load environment variables from .env (if available)
    load_dotenv()

    parser = argparse.ArgumentParser(description="Filter commentary files for a given player.")
    parser.add_argument("--player_name", type=str, required=True,
                        help="Name of the player, e.g. 'StevenSmith'")
    parser.add_argument("--player_team_code", type=str, required=True,
                        help="Team code for the player, e.g. 'AUS'")
    parser.add_argument("--opponent_team_code", type=str, required=True,
                        help="Team code for the opponent, e.g. 'SL'")
    parser.add_argument("--is_batsman", type=str, choices=["True","False"], default="True",
                        help="Whether the player is a batsman (True) or bowler (False). Default is True.")
    parser.add_argument("--k", type=int, default=25,
                        help="Maximum number of commentary files to gather. Default = 25.")
    parser.add_argument("--base_dir", type=str, default="../CommentaryData/raw_commentary",
                        help="Base directory of raw commentary data.")
    parser.add_argument("--output_dir", type=str, default="./filtered_outputs",
                        help="Directory to store the filtered commentary outputs.")
    args = parser.parse_args()

    # Convert string to boolean
    is_batsman_bool = (args.is_batsman.lower() == "true")

    # Retrieve the API key from environment
    final_api_key = os.environ.get("OPENAI_API_KEY", None)
    if not final_api_key:
        print("[WARNING] Environment variable OPENAI_API_KEY not set. Calls will use a simulated LLM response.")

    # 1) Get relevant commentary files
    selected_files = get_commentary_files_for_player(
        base_dir=args.base_dir,
        player_name=args.player_name,
        player_team_code=args.player_team_code,
        opponent_team_code=args.opponent_team_code,
        is_batsman=is_batsman_bool,
        k=args.k
    )

    if not selected_files:
        print(f"[INFO] No commentary files selected for '{args.player_name}'. Exiting.")
        return

    # Print selected files for verification
    print("[INFO] The following files have been selected for filtering:")
    for fpath in selected_files:
        formatted_path = fpath.replace('\\', '/')
        print(f"   {formatted_path}")

    # 2) Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 3) Process each file individually
    final_filtered_output = []
    for i, file_path in enumerate(selected_files, start=1):
        with open(file_path, 'r', encoding='utf-8') as f:
            file_text = f.read()

        filtered_response = call_llm_filtering_api(
            raw_commentary=file_text,
            player_name=args.player_name,
            api_key=final_api_key
        )

        # Append in the required format
        final_filtered_output.append(f"commentary {i} filtered:\n{filtered_response}")

    # 4) Save the final output in one file
    output_filename = "filtered_commentaries.txt"
    output_path = os.path.join(args.output_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write("\n".join(final_filtered_output))

    clean_path = output_path.replace("\\", "/")
    print(f"[INFO] Filtered commentary saved to: {clean_path}")

if __name__ == "__main__":
    main()