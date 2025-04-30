#!/usr/bin/env python3
"""
filter_recent_comms.py   •   2025-04-rewrite (v3)

Usage:
python filter_recent_comms.py --player_name "StevenSmith" --player_team_code "AUS" --opponent_team_code "IND" --is_batsman True --k 2 --against_comms 0 --venue_comms 0 --venue_country "India" --venue_ground "Mohali" --end_date "December 23 2025" --base_dir "../cricinfo_data_scrapping/CommentaryData/processed_commetaries/raw" --venue_dir "../cricinfo_data_scrapping/CommentaryData/processed_commetaries/venuewise" --output_dir "./filtered_outputs"

Select the k most-recent commentary files from raw/ that

  • are dated strictly < --end_date
  • belong to the required TEAM   (player_team  if --is_batsman True
                                   else opponent_team)
  • prefer—but do not absolutely require—the player’s name in the text

If < k files contain the player’s name, the remainder is filled by the
most-recent files for that team.  No “other-team” fallback is used.

CLI interface is unchanged, so driver.py works untouched.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from calendar import month_abbr, month_name
from datetime import datetime
from typing import List, Tuple, Optional

import openai
from dotenv import load_dotenv


# ───────────────────────────── prompt template ────────────────────────────
PROMPT_TEMPLATE = """
You are given multiple entire cricket match commentaries. Each is labelled
as "Commentary X".

Return **only** those lines that relate to ANY of:

  1. **Venue / match context**
  2. **Pitch conditions**
  3. **Quotes about {PLAYER_NAME}**
  4. **Any dismissal of other players in the innings**
  5. **Ball-by-ball lines involving {PLAYER_NAME}**
  6. **{PLAYER_NAME}'s own dismissal**

Count the number of time the {PLAYER_NAME} occurs in each of the commentaries. That is the minimum number of lines the output should contain per filtered commentary as they reference the player in dicussion directly. Don't mention the count in your output response.
Note: Each entire cricket match commentary contains atleast 100 relevant lines which meet these conditions.

Keep the original order.  Do **NOT** paraphrase or add extra text.

Return:

Commentary 1:
<filtered lines>

…

Commentary k:
<filtered lines>
""".lstrip()


# ───────────────────── filename → date utility ────────────────────────────
MONTH_MAP: dict[str, int] = {}

for m in range(1, 13):
    MONTH_MAP[month_name[m].lower()] = m      #  "march"
    MONTH_MAP[month_abbr[m].lower()] = m      #  "mar"

FILE_RE = re.compile(
    r"""
    ^(?P<mid>\d+)              # match id
    _(?P<team>[A-Z]+)_         # team code
    (?:commentary_)?           # optional literal
    (?P<month>[A-Za-z]+)_
    (?P<day>\d{1,2})_
    (?P<year>\d{4})\.txt$
    """,
    re.VERBOSE,
)


def file_date(fname: str) -> Optional[datetime]:
    m = FILE_RE.match(fname)
    if not m:
        return None
    try:
        d   = int(m.group("day"))
        mon = MONTH_MAP.get(m.group("month").lower())
        y   = int(m.group("year"))
        return datetime(y, mon, d) if mon else None
    except Exception:
        return None


def parse_end_date(s: str) -> datetime:
    """'March 04 2025' → datetime(2025-03-04)."""
    if s.lower() in {"", "all"}:
        return datetime.max
    return datetime.strptime(s, "%B %d %Y")


# ───────────────────── helper utilities ───────────────────────────────────
def debug(msg: str) -> None:
    print(f"[DEBUG] {msg}")


def camel_to_spaced(name: str) -> str:
    """'StevenSmith' → 'Steven Smith' (noop if already spaced)."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name) if " " not in name else name


def name_variants(player_name: str) -> List[str]:
    spaced = camel_to_spaced(player_name).lower()
    no_spc = spaced.replace(" ", "")
    last   = spaced.split()[-1]
    # ensure word-boundary variants for reliable matches
    return [rf"\b{re.escape(spaced)}\b",
            rf"\b{re.escape(no_spc)}\b",
            rf"\b{re.escape(last)}\b"]


def text_contains_name(path: str, regexes: List[re.Pattern]) -> bool:
    try:
        with open(path, encoding="utf-8") as fh:
            txt = fh.read().lower()
        return any(r.search(txt) for r in regexes)
    except Exception:
        return False


def collect_candidates(
    raw_dir: str,
    team_code: str,
    end_dt: datetime
) -> List[Tuple[datetime, str]]:
    """
    Return [(date, full_path)] for files that
        • match the naming pattern
        • belong to team_code
        • have date < end_dt
    Sorted newest→oldest.
    """
    out: List[Tuple[datetime, str]] = []

    for fn in os.listdir(raw_dir):
        m = FILE_RE.match(fn)
        if not m or m.group("team") != team_code:
            continue
        dt = file_date(fn)
        if dt is None or dt >= end_dt:
            continue
        out.append((dt, os.path.join(raw_dir, fn)))

    out.sort(key=lambda t: t[0], reverse=True)
    debug(f"collect_candidates: {len(out)} files for team={team_code}")
    return out


# ───────────────────────── LLM helpers ────────────────────────────────────
def build_prompt(paths: List[str], player_name: str) -> str:
    header = PROMPT_TEMPLATE.format(PLAYER_NAME=player_name)
    body_parts = []
    for idx, p in enumerate(paths, 1):
        with open(p, encoding="utf-8") as fh:
            body_parts.append(f"Commentary {idx}:\n{fh.read().strip()}\n")
    return header + "\n\n" + "\n".join(body_parts)


def call_llm(prompt: str, api_key: Optional[str]) -> str:
    import time
    import openai.error
    if not api_key:
        print("[WARNING] OPENAI_API_KEY not set - returning stub output")
        return "SIMULATED LLM RESPONSE\n"

    openai.api_key = api_key
    max_retries = 5
    backoff = 2  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            rsp = openai.ChatCompletion.create(
                model="text-moderation-latest",
                messages=[{"role": "user", "content": prompt}],
            )
            return rsp["choices"][0]["message"]["content"]
        except openai.error.APIError as e:
            print(f"[ERROR] APIError (attempt {attempt}): {e}")
            if attempt == max_retries:
                raise
            wait_time = backoff * attempt
            print(f"[INFO] Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            raise


# ─────────────────────────── main entry ───────────────────────────────────
def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(
        description="Pick the k most-recent commentary files that are "
                    "< end_date, belong to the right team, and ideally "
                    "contain the player's name."
    )
    # essential
    ap.add_argument("--player_name", required=True)
    ap.add_argument("--player_team_code", required=True)
    ap.add_argument("--opponent_team_code", required=True)
    ap.add_argument("--is_batsman", choices=["True", "False"], default="True")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--end_date", default="December 31 9999")
    ap.add_argument("--base_dir",
        default="../cricinfo_data_scrapping/CommentaryData/processed_commetaries/raw")
    ap.add_argument("--output_dir", default="./filtered_outputs")
    # legacy (ignored) – kept for driver compatibility
    ap.add_argument("--against_comms"); ap.add_argument("--venue_comms")
    ap.add_argument("--venue_country");  ap.add_argument("--venue_ground")
    ap.add_argument("--venue_dir")
    args = ap.parse_args()

    raw_dir = args.base_dir
    if not os.path.isdir(raw_dir):
        print(f"[ERROR] raw directory not found: {raw_dir}")
        sys.exit(1)

    primary_team = (
        args.player_team_code.upper()
        if args.is_batsman.lower() == "true"
        else args.opponent_team_code.upper()
    )
    end_dt    = parse_end_date(args.end_date)
    variants  = name_variants(args.player_name)
    regexes   = [re.compile(v, re.IGNORECASE) for v in variants]

    # 1️⃣ collect by team & date
    candidates = collect_candidates(raw_dir, primary_team, end_dt)

    # 2️⃣ split into “has name” / “no name”
    with_name   = [p for dt, p in candidates if text_contains_name(p, regexes)]
    without_name = [p for dt, p in candidates if p not in with_name]

    chosen = with_name[: args.k]
    if len(chosen) < args.k:
        needed = args.k - len(chosen)
        chosen.extend(without_name[: needed])

    if not chosen:
        print("[INFO] No commentary files satisfy the criteria – exiting.")
        sys.exit(0)

    print("\n[INFO] Selected files:")
    for p in chosen:
        print("   ", p)

    # build prompt & save
    os.makedirs(args.output_dir, exist_ok=True)
    formatted_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', args.player_name)
    prompt = build_prompt(chosen, formatted_name)
    prompt_path = os.path.join(args.output_dir, "comm_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as fh:
        fh.write(prompt)
    print(f"\n[INFO] Prompt written to {prompt_path}")
    # print("[INFO] Exiting so you can review the prompt before calling the LLM.")
    # sys.exit(0)

    # ─ optional: call LLM & save output ─
    filtered = call_llm(prompt, os.getenv("OPENAI_API_KEY"))
    out_name = (
        f"filtered_commentaries_{args.player_name.replace(' ', '')}_"
        f"{primary_team}_{len(chosen)}.txt"
    )
    out_path = os.path.join(args.output_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(filtered)
    print(f"[INFO] Filtered commentary saved to {out_path}\n")


if __name__ == "__main__":
    main()
