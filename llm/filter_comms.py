#!/usr/bin/env python3
"""
filter_comms.py

Usage (example):
----------------
python filter_comms.py --player_name "StevenSmith" --player_team_code "AUS" --opponent_team_code "IND" --is_batsman True --k 5 --against_comms 0 --venue_comms 5 --venue_country "India" --venue_ground "Mohali" --end_date "December 23 2025" --base_dir "../cricinfo_data_scrapping/CommentaryData/processed_commetaries/raw" --venue_dir "../cricinfo_data_scrapping/CommentaryData/processed_commetaries/venuewise" --output_dir "./filtered_outputs"

Overview
---------------------------------------------------------------
1. Try to fill `against_comms` from the **raw vs-opponent** bucket.
2. If that fails, try to fill **all `k`** from **venuewise**.
3. If still short, **top-up vs-opponent again** (new step) before falling back.
4. Pick the remainder from **raw (recency)**, then (if required) the blind
   team-code fallback sorted by match-id.
5. When step-1 succeeds but `venue_comms` cannot be met, the same extra
   **top-up vs-opponent** pass is run before generic fallback.

Thus the script **always** returns *some* final commentary set, ensuring a valid output in all scenarios.
"""

import os
import re
import sys
import json
import argparse
from typing import List, Optional, Tuple, Set
from dotenv import load_dotenv
from datetime import datetime
import openai

# -------------------------------------------------------------------------
#                         TEMPLATE & DEBUG HELPER
# -------------------------------------------------------------------------

FILTERING_PROMPT_TEMPLATE = """
You are given multiple entire cricket match commentaries. Each is labeled as "Commentary X" with its full text. 
Your task is to filter out and return **only** those lines that satisfy **all** of the following:

1) Relevance to **{PLAYER_NAME}** or essential match context. Specifically, lines must meet at least one of these:

   a) **Venue and Match Context**:
      - Lines mentioning match type, significance (e.g., ICC tournament, final, etc.), series name or standing, 
        or explicit venue/stadium references (including anything about how the stadium or location may affect batting/bowling).
   
   b) **Pitch Conditions**:
      - Any line describing the pitch (pace, bounce, spin, slowness, dryness, etc.), or how it may affect batting or bowling.

   c) **Comments/Quotes by or about {PLAYER_NAME}**:
      - Direct quotes from {PLAYER_NAME} about form, mindset, approach.
      - Observations from commentators about {PLAYER_NAME}'s batting, style, or readiness.

   d) **Dismissals in the Innings**:
      - Any and all lines describing how *any* batter got dismissed (not just {PLAYER_NAME}). This gives us the context of how other batters are getting dismissed.

   e) **Ball-by-Ball Entries Involving {PLAYER_NAME}**:
      - Lines describing deliveries faced by {PLAYER_NAME}, runs scored, boundaries, edges, near-misses, bowler tactics, 
        or partnerships involving {PLAYER_NAME}.

   f) **{PLAYER_NAME}'s Mode of Dismissal**:
      - The exact commentary line stating how {PLAYER_NAME} was dismissed (if applicable).

2) For **each** commentary, return those exact lines in the **original order**. 
   - Do not paraphrase or summarize or give any additional quotes from yourself. 
   - Do not include irrelevant lines that do not match these criteria.

3) **Be extremely thorough**. Carefully check each commentary to avoid missing any relevant line. 
   Sometimes relevant lines appear unexpectedly, e.g., referencing {PLAYER_NAME} in passing.

---

Return the final result in the following structured format:

Commentary 1:
<filtered lines, in original order, verbatim>

Commentary 2:
<filtered lines, in original order, verbatim>

...

Commentary k:
<filtered lines, in original order, verbatim>

"""

def debug_print(msg: str):
    """Print debug messages with a unified prefix."""
    print(f"[DEBUG] {msg}")

# -------------------------------------------------------------------------
#                              PARSING FUNCS
# -------------------------------------------------------------------------

def parse_date_str(date_str: str) -> datetime:
    """Convert a string like 'December 23 2025' to a datetime."""
    return datetime.strptime(date_str, "%B %d %Y")

def date_is_on_or_before(d: datetime, end_d: datetime) -> bool:
    """Return True if d <= end_d."""
    return d <= end_d

def load_team_map() -> dict:
    """
    Load team_maps.json from the same directory as this script.
    Return a dict mapping, e.g. {"India": "IND", "Australia": "AUS", ...}
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    map_path = os.path.join(script_dir, "team_maps.json")
    if not os.path.isfile(map_path):
        print(f"[WARN] Could not find {map_path}. Returning empty map.")
        return {}
    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def parse_date_from_text(txt: str) -> Optional[datetime]:
    """
    Attempt to find a date in txt (e.g. "Sep 22 2023" or "March 5 2024").
    Return datetime or None if not found.
    """
    pattern = re.compile(r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b")
    match = pattern.search(txt)
    if match:
        month_str = match.group(1)
        day_str = match.group(2)
        year_str = match.group(3)
        for fmt in ["%b %d %Y", "%B %d %Y"]:
            try:
                return datetime.strptime(f"{month_str} {day_str} {year_str}", fmt)
            except ValueError:
                pass
    return None

def parse_ground_country(header_text: str) -> Tuple[str, str]:
    """
    Attempt to parse ' at SOMESTADIUM,' from the header text.
    Return (ground, country='') for now.
    """
    ground = ""
    country = ""
    at_idx = header_text.lower().find(" at ")
    if at_idx >= 0:
        sub = header_text[at_idx+4:]
        comma_idx = sub.find(",")
        if comma_idx > 0:
            ground = sub[:comma_idx].strip()
        else:
            ground = sub.strip()
    return (ground, country)

def parse_recent100_file(player_name: str, team_map: dict) -> List[dict]:
    """
    Read player_matches/{player_name}Recent100.txt, parse lines into:
      { match_id, date, team1_code, team2_code, ground, country }
    sorted by date descending.
    """
    folder = "player_matches"
    filename = f"{player_name}Recent100.txt"
    path = os.path.join(folder, filename)
    if not os.path.isfile(path):
        print(f"[WARN] Could not find {path}. Returning empty list.")
        return []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    records = []
    mid_pat = re.compile(r"Match ID:\s*(\d+)")
    head_pat = re.compile(r"Header:\s*(.*)")

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        mid_match = mid_pat.search(line_str)
        if not mid_match:
            continue
        try:
            match_id = int(mid_match.group(1))
        except ValueError:
            continue

        head_match = head_pat.search(line_str)
        if head_match:
            header_text = head_match.group(1).strip()
        else:
            header_text = line_str

        match_date = parse_date_from_text(header_text) or parse_date_from_text(line_str)
        if not match_date:
            continue

        ground, country = parse_ground_country(header_text)

        comma_idx = header_text.find(",")
        if comma_idx == -1:
            teams_str = header_text
        else:
            teams_str = header_text[:comma_idx].strip()

        team1_code = ""
        team2_code = ""
        if " vs " in teams_str:
            t1, t2 = teams_str.split(" vs ", 1)
            t1 = t1.strip()
            t2 = t2.strip()
            team1_code = team_map.get(t1, t1.upper())
            team2_code = team_map.get(t2, t2.upper())

        rec = {
            "match_id": match_id,
            "date": match_date,
            "team1_code": team1_code,
            "team2_code": team2_code,
            "ground": ground,
            "country": country,
        }
        records.append(rec)

    records.sort(key=lambda x: x["date"], reverse=True)
    debug_print(f"Sorted records by date descending. Total valid: {len(records)}")
    return records

# -------------------------------------------------------------------------
#                           FILE FINDERS
# -------------------------------------------------------------------------

def find_commentary_file_raw(
    match_id: int,
    is_batsman: bool,
    player_team: str,
    opponent_team: str,
    base_dir: str
) -> Optional[str]:
    """
    In 'raw' folder, look for {match_id}_{TEAM}_commentary_*.txt
    Where TEAM = player_team if is_batsman=True, else opponent_team.
    """
    if not os.path.isdir(base_dir):
        return None

    desired_team = player_team if is_batsman else opponent_team
    pattern_str = rf"^{match_id}_{desired_team}_commentary_.*\.txt$"
    pattern = re.compile(pattern_str)

    all_files = os.listdir(base_dir)
    for fn in all_files:
        if pattern.match(fn):
            full_path = os.path.join(base_dir, fn)
            return full_path
    return None

def find_commentary_file_venuewise(
    match_id: int,
    is_batsman: bool,
    player_team: str,
    opponent_team: str,
    ground: str,
    country: str,
    base_dir: str
) -> Optional[str]:
    """
    In 'venuewise' folder, look for {match_id}_{TEAM}_commentary_*.txt.
    We prefer files that mention `ground` in their name, else fallback to `country`.
    """
    if not os.path.isdir(base_dir):
        return None

    desired_team = player_team if is_batsman else opponent_team
    pattern_str = rf"^{match_id}_{desired_team}_commentary_.*\.txt$"
    pattern = re.compile(pattern_str, re.IGNORECASE)

    pass1_candidates = []
    pass2_candidates = []

    for fn in os.listdir(base_dir):
        if pattern.match(fn):
            fn_lower = fn.lower()
            if ground and ground.lower() in fn_lower:
                pass1_candidates.append(os.path.join(base_dir, fn))
            elif country and country.lower() in fn_lower:
                pass2_candidates.append(os.path.join(base_dir, fn))

    if pass1_candidates:
        return pass1_candidates[0]
    if pass2_candidates:
        return pass2_candidates[0]
    return None

# -------------------------------------------------------------------------
#                       SELECTING COMMENTARY FILES
# -------------------------------------------------------------------------

def select_files_against_opposition(
    records: List[dict],
    opponent_code: str,
    is_batsman: bool,
    player_team_code: str,
    base_dir: str,
    needed: int,
    exclude_match_ids: Set[int]      # <- new
) -> List[str]:
    """Select up to `needed` commentary files from 'raw' vs opponent_code, skipping duplicates."""
    chosen = []
    for rec in records:
        if rec["match_id"] in exclude_match_ids:
            continue
        t1, t2 = rec["team1_code"], rec["team2_code"]
        if opponent_code in (t1, t2):
            path = find_commentary_file_raw(
                match_id=rec["match_id"],
                is_batsman=is_batsman,
                player_team=player_team_code,
                opponent_team=opponent_code,
                base_dir=base_dir
            )
            if path:
                chosen.append(path)
                exclude_match_ids.add(rec["match_id"])
                if len(chosen) >= needed:
                    break
    debug_print(f"Found {len(chosen)} commentary files vs opponent.")
    return chosen

def select_files_venuewise(
    records: List[dict],
    is_batsman: bool,
    player_team_code: str,
    opponent_team_code: str,
    venue_dir: str,
    needed: int,
    exclude_match_ids: Set[int],
    forced_ground: str = "",
    forced_country: str = "",
) -> List[str]:
    """
    Attempt to select needed commentary files from the venuewise path,
    skipping any match already selected.
    """
    chosen: List[str] = []

    # ---------------- helper that runs one harvesting pass ---------------
    def _harvest(trg_ground: str, trg_country: str, vs_only: bool) -> None:
        nonlocal chosen, needed
        if len(chosen) >= needed:
            return
        for rec in records:
            if rec["match_id"] in exclude_match_ids:
                continue
            if vs_only and opponent_team_code not in (rec["team1_code"], rec["team2_code"]):
                continue

            gnd = trg_ground if trg_ground else (forced_ground or rec["ground"])
            ctry = trg_country if trg_country else (forced_country or rec["country"])

            path = find_commentary_file_venuewise(
                match_id=rec["match_id"],
                is_batsman=is_batsman,
                player_team=player_team_code,
                opponent_team=opponent_team_code,
                ground=gnd,
                country=ctry,
                base_dir=venue_dir,
            )
            if path:
                chosen.append(path)
                exclude_match_ids.add(rec["match_id"])
                if len(chosen) >= needed:
                    break

    # pass-1: opponent + ground
    if forced_ground:
        _harvest(forced_ground, forced_country, vs_only=True)

    # pass-2: opponent + country (ground ignored)
    if len(chosen) < needed and forced_country:
        _harvest("", forced_country, vs_only=True)

    # pass-3: previous behaviour (any opponent, ground -> country)
    if len(chosen) < needed:
        _harvest(forced_ground, forced_country, vs_only=False)

    debug_print(f"Found {len(chosen)} commentary files from venue bucket.")
    return chosen[:needed]

def select_files_any_raw(
    records: List[dict],
    is_batsman: bool,
    player_team_code: str,
    opponent_team_code: str,
    base_dir: str,
    needed: int,
    exclude_match_ids: Set[int]      # <- new
) -> List[str]:
    """Fallback: pick any from records in descending date order, skipping duplicates."""
    debug_print(f"Selecting {needed} from raw ignoring opponent/venue.")
    chosen = []
    for rec in records:
        if rec["match_id"] in exclude_match_ids:
            continue
        path = find_commentary_file_raw(
            match_id=rec["match_id"],
            is_batsman=is_batsman,
            player_team=player_team_code,
            opponent_team=opponent_team_code,
            base_dir=base_dir
        )
        if path:
            chosen.append(path)
            exclude_match_ids.add(rec["match_id"])
            if len(chosen) >= needed:
                break
    debug_print(f"Found {len(chosen)} commentary files from anyRaw.")
    return chosen

def select_final_fallback_k(
    is_batsman: bool,
    player_team_code: str,
    base_dir: str,
    k: int,
    exclude_match_ids: Set[int]      # <- new
) -> List[str]:
    """
    FINAL fallback if we couldn't get k from all prior steps.
    Grab files by match-id order, skipping already chosen ones.
    """
    debug_print(f"FINAL fallback: picking any {k} from the entire raw folder for team={player_team_code}.")
    if not os.path.isdir(base_dir):
        return []

    pattern_str = rf"^(\d+)_{player_team_code}_commentary_.*\.txt$"
    pattern = re.compile(pattern_str)

    file_match_pairs = []
    for fn in os.listdir(base_dir):
        match = pattern.match(fn)
        if match:
            match_id_str = match.group(1)
            try:
                match_id_num = int(match_id_str)
            except ValueError:
                continue
            if match_id_num in exclude_match_ids:
                continue
            full_path = os.path.join(base_dir, fn)
            file_match_pairs.append((match_id_num, full_path))

    file_match_pairs.sort(key=lambda x: x[0], reverse=True)
    final_list = []
    for mid, path in file_match_pairs:
        final_list.append(path)
        exclude_match_ids.add(mid)
        if len(final_list) >= k:
            break
    debug_print(f"FINAL fallback result: {len(final_list)} files.")
    return final_list

# -------------------------------------------------------------------------
#                           BUILD PROMPT & CALL LLM
# -------------------------------------------------------------------------

def build_single_prompt(file_paths: List[str], player_name: str) -> str:
    """Create the final user prompt with each commentary labeled."""
    header = FILTERING_PROMPT_TEMPLATE.format(PLAYER_NAME=player_name)
    body_parts = []
    for idx, fp in enumerate(file_paths, start=1):
        with open(fp, 'r', encoding='utf-8') as f:
            commentary_text = f.read()
        commentary_label = f"Commentary {idx}:"
        commentary_block = f"{commentary_label}\n{commentary_text}\n"
        body_parts.append(commentary_block)

    joined_body = "\n".join(body_parts)
    return f"{header}\n\n{joined_body}"


def call_llm(prompt: str, api_key: str = None) -> str:
    """Call LLM. If no API key, return a simulated response."""
    model = "gpt-4.1"
    if not api_key:
        print("[WARNING] OPENAI_API_KEY not found. Using a simulated response.\n")
        return (
            f"SIMULATED {model} RESPONSE:\n"
            "Commentary 1:\n"
            "Relevant lines here...\n"
            "Fraction retained: x/Total\n\n"
            "Commentary 2:\n"
            "Relevant lines here...\n"
            "Fraction retained: x/Total\n"
        )

    openai.api_key = api_key
    print(f"[INFO] Calling {model} API with the full prompt...")

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        filtered_text = response["choices"][0]["message"]["content"]

        usage = response["usage"]
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        cost_prompt = (prompt_tokens / 1_000_000) * 2.00
        cost_completion = (completion_tokens / 1_000_000) * 8.00
        total_cost = cost_prompt + cost_completion

        print("=== Token Usage ===")
        print(f"  Prompt Tokens:      {prompt_tokens}")
        print(f"  Completion Tokens:  {completion_tokens}")
        print(f"  Total Tokens:       {total_tokens}")
        print("")
        print("=== Cost Estimate (USD) ===")
        print(f"  Prompt Cost:        ${cost_prompt:.4f}")
        print(f"  Completion Cost:    ${cost_completion:.4f}")
        print(f"  Total Cost:         ${total_cost:.4f}")
        print("")
        print(f"[INFO] {model} API call complete.")
        return filtered_text
    
    except Exception as e:
        print(f"[ERROR] {model} API call failed: {e}")
        return f"[ERROR] {model} API call failed.\n"

# -------------------------------------------------------------------------
#                                MAIN
# -------------------------------------------------------------------------

def main():
    load_dotenv()

    # ---------------------------- argument parsing ------------------------
    parser = argparse.ArgumentParser(
        description="Filter commentary files with advanced selection logic, "
                    "then build a single LLM prompt."
    )
    parser.add_argument("--player_name", required=True)
    parser.add_argument("--player_team_code", required=True)
    parser.add_argument("--opponent_team_code", required=True)
    parser.add_argument("--is_batsman", choices=["True", "False"], default="True")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--against_comms", type=int, default=3)
    parser.add_argument("--venue_comms", type=int, default=2)
    parser.add_argument("--venue_country", default="")
    parser.add_argument("--venue_ground", default="")
    parser.add_argument("--end_date", default="December 31 9999")
    parser.add_argument("--base_dir",
                        default="../cricinfo_data_scrapping/CommentaryData/processed_commetaries/raw")
    parser.add_argument("--venue_dir",
                        default="../cricinfo_data_scrapping/CommentaryData/processed_commetaries/venuewise")
    parser.add_argument("--output_dir", default="./filtered_outputs")
    args = parser.parse_args()

    if args.against_comms + args.venue_comms != args.k:
        print(f"[ERROR] against_comms + venue_comms != k ({args.against_comms}+"
              f"{args.venue_comms} != {args.k})")
        sys.exit(1)

    is_batsman_bool = args.is_batsman.lower() == "true"
    api_key = os.getenv("OPENAI_API_KEY")

    # ------------------------- data prep ----------------------------------
    team_map = load_team_map()
    all_records = parse_recent100_file(args.player_name, team_map)
    if not all_records:
        print("[INFO] No Recent100 data; exiting.")
        return

    try:
        end_dt = parse_date_str(args.end_date)
    except ValueError:
        print(f"[ERROR] Could not parse end_date '{args.end_date}'.")
        sys.exit(1)

    valid_records = [r for r in all_records if date_is_on_or_before(r["date"], end_dt)]
    exclude_match_ids: Set[int] = set()

    # ---------------------- selection logic -------------------------------
    if not valid_records:
        final_files = select_final_fallback_k(
            is_batsman_bool, args.player_team_code, args.base_dir,
            args.k, exclude_match_ids
        )
    else:
        # ---- Step-1  vs-opponent -----------------------------------------
        opp_files = select_files_against_opposition(
            valid_records, args.opponent_team_code, is_batsman_bool,
            args.player_team_code, args.base_dir, args.against_comms,
            exclude_match_ids
        )

        if len(opp_files) < args.against_comms:
            # ---- Step-2  try whole-k from venue --------------------------
            venue_files = select_files_venuewise(
                valid_records, is_batsman_bool, args.player_team_code,
                args.opponent_team_code, args.venue_dir, args.k,
                exclude_match_ids, args.venue_ground, args.venue_country
            )

            # ---- NEW top-up vs-opponent before generic fallback ----------
            if len(venue_files) < args.k:
                need_left = args.k - len(venue_files)
                venue_files.extend(
                    select_files_against_opposition(
                        valid_records, args.opponent_team_code, is_batsman_bool,
                        args.player_team_code, args.base_dir, need_left,
                        exclude_match_ids
                    )
                )
            # --------------------------------------------------------------

            if len(venue_files) < args.k:
                final_files = select_files_any_raw(
                    valid_records, is_batsman_bool, args.player_team_code,
                    args.opponent_team_code, args.base_dir, args.k,
                    exclude_match_ids
                )
                if len(final_files) < args.k:
                    final_files = select_final_fallback_k(
                        is_batsman_bool, args.player_team_code,
                        args.base_dir, args.k, exclude_match_ids
                    )
            else:
                final_files = venue_files
        else:
            # ---- Step-2  venue bucket just for venue_comms ---------------
            venue_files = select_files_venuewise(
                valid_records, is_batsman_bool, args.player_team_code,
                args.opponent_team_code, args.venue_dir, args.venue_comms,
                exclude_match_ids, args.venue_ground, args.venue_country
            )

            # ---- NEW top-up vs-opponent before generic fallback ----------
            if len(venue_files) < args.venue_comms:
                need_left = args.venue_comms - len(venue_files)
                venue_files.extend(
                    select_files_against_opposition(
                        valid_records, args.opponent_team_code, is_batsman_bool,
                        args.player_team_code, args.base_dir, need_left,
                        exclude_match_ids
                    )
                )
            # --------------------------------------------------------------

            if len(venue_files) < args.venue_comms:
                final_files = select_files_any_raw(
                    valid_records, is_batsman_bool, args.player_team_code,
                    args.opponent_team_code, args.base_dir, args.k,
                    exclude_match_ids
                )
                if len(final_files) < args.k:
                    final_files = select_final_fallback_k(
                        is_batsman_bool, args.player_team_code,
                        args.base_dir, args.k, exclude_match_ids
                    )
            else:
                final_files = opp_files + venue_files

    if not final_files:
        print("[INFO] No commentary files found. Exiting.")
        return

    print("\n[INFO] Final chosen commentary files:")
    for fp in final_files:
        print("   ", fp)

    os.makedirs(args.output_dir, exist_ok=True)
    prompt_text = build_single_prompt(final_files, args.player_name)
    prompt_path = os.path.join(args.output_dir, "comm_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as fh:
        fh.write(prompt_text)

    print(f"\n[INFO] Prompt saved to {prompt_path}")
    # print("[INFO] Exiting so you can review the prompt before calling the LLM.")
    # sys.exit(0)

    filtered_result = call_llm(prompt_text, api_key)
    output_filename = f"filtered_commentaries_{args.player_name}_{args.opponent_team_code}_{args.k}.txt"
    output_path = os.path.join(args.output_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as of:
        of.write(filtered_result)
    
    print(f"\n[INFO] Filtered commentary saved to: {output_path}\n")

if __name__ == "__main__":
    main()
