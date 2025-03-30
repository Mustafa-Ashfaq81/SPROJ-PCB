"""
filter_comms.py

Usage (example):
----------------
python filter_comms.py --player_name "StevenSmith" --player_team_code "AUS" --opponent_team_code "SL" --is_batsman True --k 5 --base_dir "../cricinfo_data_scrapping/CommentaryData/raw_commentary" --output_dir "./filtered_outputs" --chunk_size 100 --chunk_overlap 20

overview

This script filters raw cricket match commentary to extract only the lines relevant to a specific player using an LLM(4o-mini).

Input: Raw `.txt` commentary files from `base_dir/<player_name>/`.
Selection Criteria: Picks up to `k` latest files where the player’s team was batting (`is_batsman=True`) or bowling (`is_batsman=False`).

How it works: Each file is split into overlapping chunks (`chunk_size`, `chunk_overlap`). Each chunk is passed to an LLM with a structured prompt to extract lines about venue, pitch, quotes, dismissals, and player-specific actions. Overlapping lines are deduplicated.

Output: A single `.txt` file combining all filtered lines, saved as `filtered_commentaries_playerName_opposition.txt` in the specified `output_dir`.

"""

import os
import re
import argparse
from typing import List
from dotenv import load_dotenv


FILTERING_PROMPT_TEMPLATE = """
You are given a **chunk** of the complete match commentary text for a cricket match. This chunk might be a continuation from a previous portion, so it may reference events or context from earlier lines not shown here.

Your task is to **extract and return only the exact lines** (verbatim, without paraphrasing) from this chunk that meet the following criteria related to **{PLAYER_NAME}**. Specifically, search this chunk and gather lines that include:

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
- **Return only the exact lines from this chunk**; do not paraphrase or summarize.
- If a line is relevant to more than one category, include it once in full.
- Maintain the **exact order** in which these lines appear in this chunk.
- Do not include any commentary text that is irrelevant to the six points above.
"""

# ----------------------------
# 2) File selection logic
# ----------------------------
def get_commentary_files_for_player(
    base_dir: str,
    player_name: str,
    player_team_code: str,
    opponent_team_code: str,
    is_batsman: bool,
    k: int = 25
) -> List[str]:
    """
    1) Looks in base_dir/player_name for commentary files matching:
         {matchID}_{TeamThatIsBatting}_commentary_*.txt
    2) If is_batsman=True, we pick files with batting_team == player_team_code
       If is_batsman=False, we pick files with batting_team == opponent_team_code.
    3) Sort by matchID descending, pick up to k files.
    """
    player_dir = os.path.normpath(os.path.join(base_dir, player_name))
    if not os.path.isdir(player_dir):
        print(f"[WARNING] Directory not found: {player_dir}")
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

        if is_batsman and (batting_team == player_team_code):
            matched_files.append((match_id, filename))
        elif not is_batsman and (batting_team == opponent_team_code):
            matched_files.append((match_id, filename))

    matched_files.sort(key=lambda x: x[0], reverse=True)
    top_k_files = [os.path.join(player_dir, mf[1]) for mf in matched_files[:k]]
    return top_k_files

# ----------------------------
# 3) LLM call for a single chunk
# ----------------------------
def call_llm_filtering_api_for_chunk(
    chunk_text: str,
    player_name: str,
    api_key: str = None
) -> str:
    """
    Sends this chunk of commentary text to the 4o-mini model with the filtering prompt.
    Returns the filtered text from the model.
    If no valid API key is found, returns a simulated response.
    """
    # Build the user prompt
    filtering_prompt = FILTERING_PROMPT_TEMPLATE.format(PLAYER_NAME=player_name)
    user_prompt = (
        f"{filtering_prompt}\n\n"
        f"---\n\n"
        f"Below is a **chunk** of the commentary. "
        f"Return only the relevant lines in their original order, as demonstrated.\n\n"
        f"{chunk_text}\n\n"
        f"---\n\n"
        f"Remember to follow the example style and instructions closely."
    )

    if not api_key:
        print("[WARNING] OPENAI_API_KEY not found. Using a simulated LLM response.\n")
        return (
            "SIMULATED LLM RESPONSE:\n"
            "This is where the model would return the verbatim lines meeting the criteria.\n"
        )

    import openai
    openai.api_key = api_key

    print("[INFO] Calling 4o-mini API for this chunk...")
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_prompt}],
    )
    filtered_text = response["choices"][0]["message"]["content"]
    print("[INFO] API call complete.")
    return filtered_text

# ----------------------------
# 4) Process a single commentary file in chunks with overlap
# ----------------------------
def filter_commentary_in_chunks(
    file_text: str,
    player_name: str,
    api_key: str = None,
    chunk_size: int = 300,
    overlap: int = 10
) -> str:
    """
    1. Split the commentary into lines.
    2. Because it's reversed chronologically, we chunk from bottom (earliest ball) up to the top.
    3. Each chunk has chunk_size lines, with an overlap of ~10 lines for the next chunk.
    4. If we hit the start of the file (chunk_start=0), we process that chunk then break.
    5. We do a final dedup pass on the combined output to remove overlap duplicates.
    6. Return the chunked LLM outputs reversed so earliest lines appear at top in final output.
    """
    lines = file_text.splitlines()
    total_lines = len(lines)

    chunk_results = []
    current_end = total_lines
    chunk_index = 0

    while current_end > 0:
        chunk_index += 1
        chunk_start = max(0, current_end - chunk_size)
        num_lines_in_chunk = current_end - chunk_start

        print(f"[INFO] Processing chunk #{chunk_index}: lines {chunk_start}..{current_end-1} "
              f"({num_lines_in_chunk} lines total)")

        # Instead of adding "CHUNK START/END" lines, we just feed the raw lines to the model
        chunk_lines = lines[chunk_start:current_end]
        chunk_block = "\n".join(chunk_lines)

        # Call the LLM
        filtered_chunk = call_llm_filtering_api_for_chunk(
            chunk_block,
            player_name=player_name,
            api_key=api_key
        )
        chunk_results.append(filtered_chunk)

        if chunk_start == 0:
            print("[INFO] Reached the start of the file. No more lines above.")
            break
        else:
            next_end = chunk_start + overlap
            print(f"[INFO] Setting up for next chunk. new current_end = {next_end}")
            current_end = next_end

    # chunk_results[0] is earliest chunk, chunk_results[-1] is latest
    chunk_results_reversed = chunk_results[::-1]
    combined_text = "\n".join(chunk_results_reversed)

    # 5) Deduplicate lines (due to overlap)
    # We'll keep the first occurrence of each line and remove repeats
    final_lines = []
    seen = set()
    for line in combined_text.splitlines():
        line_stripped = line.strip()
        if line_stripped and (line_stripped not in seen):
            final_lines.append(line)
            seen.add(line_stripped)

    final_output = "\n".join(final_lines)
    return final_output

# ----------------------------
# 5) Main routine
# ----------------------------
def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Filter commentary files in overlapping chunks for a given player.")
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
    parser.add_argument("--chunk_size", type=int, default=300,
                        help="Number of lines per chunk (before overlap). Default = 300.")
    parser.add_argument("--chunk_overlap", type=int, default=10,
                        help="Number of overlapping lines between chunks. Default = 10.")
    args = parser.parse_args()

    is_batsman_bool = (args.is_batsman.lower() == "true")

    final_api_key = os.environ.get("OPENAI_API_KEY", None)
    if not final_api_key:
        print("[WARNING] Environment variable OPENAI_API_KEY not set. Using a simulated LLM response.\n")

    # 1) Gather relevant commentary files
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

    print("[INFO] The following files have been selected for filtering (chunk + overlap):")
    for fpath in selected_files:
        print(f"   {fpath}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 2) Process each file
    all_filtered_outputs = []
    for i, file_path in enumerate(selected_files, start=1):
        print(f"\n[INFO] ====== Processing Commentary File #{i} ======")
        with open(file_path, 'r', encoding='utf-8') as f:
            file_text = f.read()

        # Filter with chunk-based approach
        filtered_for_this_file = filter_commentary_in_chunks(
            file_text=file_text,
            player_name=args.player_name,
            api_key=final_api_key,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap
        )

        commentary_label = f"Commentary {i}:"
        section_text = f"{commentary_label}\n{filtered_for_this_file}"
        all_filtered_outputs.append(section_text)

    # 3) Save final output to a single file
    output_filename = f"filtered_commentaries_{args.player_name}_{args.opponent_team_code}_{args.k}.txt"
    output_path = os.path.join(args.output_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write("\n\n".join(all_filtered_outputs))

    print(f"\n[INFO] All done! Filtered commentary saved to: {output_path}\n")


if __name__ == "__main__":
    main()
