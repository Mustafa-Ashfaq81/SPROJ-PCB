#!/usr/bin/env python3
"""
filter_comms_chunks.py

Usage
-----------
python filter_comms_chunks.py --player_name "StevenSmith" --player_team_code "AUS" --opponent_team_code "IND" --is_batsman True --k 5 --against_comms 3 --venue_comms 2 --venue_country "India" --venue_ground "Mohali" --end_date "December 23 2025" --base_dir "../cricinfo_data_scrapping/CommentaryData/processed_commetaries/raw" --venue_dir "../cricinfo_data_scrapping/CommentaryData/processed_commetaries/venuewise" --chunk_size 100 --chunk_overlap 20 --output_dir "./filtered_outputs"

High-level flow
---------------
1. **Select files** (new multi-step logic - see README above).
2. For each selected file:
      • split into overlapping chunks
      • stream chunks to GPT-4o-mini in parallel
      • clean & dedupe returned lines
3. Concatenate per-file results and write   filtered_commentaries_<player>_<opp>_<k>.txt
"""

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------
import os, re, sys, json, argparse, concurrent.futures
from typing import List, Tuple, Optional, Set, Dict
from datetime import datetime

from dotenv import load_dotenv    # load .env for OPENAI_API_KEY

# --------------------------------------------------------------------------
# =====  Section 0 ── tiny debug helper  ===================================
# --------------------------------------------------------------------------
def debug(msg: str):
    print(f"[DEBUG] {msg}")

# --------------------------------------------------------------------------
# =====  Section 1 ── PROMPT + LLM helpers (UNTOUCHED)  ====================
# --------------------------------------------------------------------------
FILTERING_PROMPT_TEMPLATE = """
You are given a **chunk** of the complete match commentary text for a cricket
match. This chunk might be a continuation from a previous portion, so it may
reference events or context from earlier lines not shown here.

Your task is to **extract and return only the exact lines** (verbatim, without
paraphrasing) from this chunk that meet the following criteria related to
**{PLAYER_NAME}**.  Specifically, search this chunk and gather lines that
include:

1. **Venue and Match Context**
   - Include lines that mention the type or significance of the match — such as
     whether it is part of a bilateral series, an ICC tournament, a group
     stage, knockout, semi-final, or final.
   - If it's a series match, include details specifying which match it is
     (e.g., 1st ODI, 2nd Test) or the current series standing
     (e.g., Series tied 1-1).
   - Any lines explicitly stating the venue or location of the match, crowd
     attendance, or references to the stadium conditions—especially if they
     refer to how it might affect batting or bowling.

2. **Pitch Conditions**
   - Any lines (including quotations) that describe the pitch behaviour, pace,
     bounce, slowness, or anything else about the surface that could be
     relevant to how batters—especially {PLAYER_NAME}—perform or get out.

3. **Comments / Quotes by or about {PLAYER_NAME}**
   - Any direct quotes where {PLAYER_NAME} discusses how they felt at the
     crease, their form, the difficulty of the pitch, or mentions of their
     batting approach and mindset.

4. **How All Batters in the Innings Got Out**
   - Extract lines describing dismissals of **any** batter in the innings.
     Include the exact commentary lines for each such dismissal.

5. **All Ball-by-Ball Entries Involving {PLAYER_NAME}**
   - Every commentary line that mentions {PLAYER_NAME}'s deliveries faced, runs
     scored, boundaries, near-dismissals, or any relevant event while
     {PLAYER_NAME} was batting.
   - If a line references {PLAYER_NAME}'s partnership, shot selection, or a
     bowler's tactics against them, include it exactly as it appears.

6. **{PLAYER_NAME}'s Mode of Dismissal**
   - The precise commentary line that states how {PLAYER_NAME} got out (caught,
     bowled, run-out, etc.).  This is **crucial**. Extract it word-for-word.

**Important Requirements**
- **Return only the exact lines from this chunk**; do not paraphrase or
  summarise.
- If a line is relevant to more than one category, include it once in full.
- Maintain the **exact order** in which these lines appear in this chunk.
- Do **not** include any commentary text that is irrelevant to the six points
  above.
"""

# -- GPT-4o call for ONE chunk ------------------------------------------------
def call_llm_filtering_api_for_chunk(
    chunk_text: str,
    player_name: str,
    api_key: Optional[str] = None,
) -> str:
    """Send one chunk to GPT-4o-mini and return its filtered output."""
    system_prompt = FILTERING_PROMPT_TEMPLATE.format(PLAYER_NAME=player_name)
    user_prompt = (
        f"{system_prompt}\n\n"
        "---\n\n"
        "Below is a **chunk** of the commentary. Return only the relevant lines "
        "in their original order, exactly as they appear:\n\n"
        f"{chunk_text}\n\n"
        "---\n\n"
        "Remember: no paraphrasing, keep the order, and include only lines that "
        "match the six categories."
    )

    if not api_key:
        print("[WARNING] OPENAI_API_KEY not set - returning simulated output.")
        return ""

    import openai
    openai.api_key = api_key
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp["choices"][0]["message"]["content"]

# --------------------------------------------------------------------------
# =====  Section 2 ── cleaning helpers (UNTOUCHED)  =========================
# --------------------------------------------------------------------------
BAD_PREFIXES = tuple(s.lower() for s in (
    "there are no lines",
    "the commentary chunk provided",
    "the chunk provided does not contain",
    "i’m sorry",
    "here are the relevant lines",
    "here's the extracted commentary",
    "none of the lines",
    "no relevant lines",
))

def _clean_llm_output(txt: str) -> str:
    out: List[str] = []
    for ln in txt.splitlines():
        if not ln.strip():
            continue
        low = ln.lower().lstrip("*•- ").strip()
        if any(low.startswith(p) for p in BAD_PREFIXES):
            continue
        out.append(ln)
    return "\n".join(out)

# --------------------------------------------------------------------------
# =====  Section 3 ── chunk-level filtering (UNTOUCHED)  ====================
# --------------------------------------------------------------------------
def filter_commentary_in_chunks(
    raw_text: str,
    player_name: str,
    api_key: Optional[str],
    chunk_size: int,
    overlap: int,
) -> str:
    """Split → parallel LLM calls → clean + dedupe."""
    lines = raw_text.splitlines()
    N = len(lines)

    # build reversed chunks (so seq 1 is earliest in final list)
    chunks: List[Tuple[int, str]] = []
    seq, end = 0, N
    while end > 0:
        seq += 1
        start = max(0, end - chunk_size)
        chunks.append((seq, "\n".join(lines[start:end])))
        if start == 0:
            break
        end = start + overlap

    print(f"[INFO] Dispatching {len(chunks)} chunks in parallel …")

    def _worker(pair: Tuple[int, str]) -> Tuple[int, str]:
        idx, txt = pair
        out = call_llm_filtering_api_for_chunk(txt, player_name, api_key)
        return idx, _clean_llm_output(out)

    max_workers = min(8, len(chunks)) or 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_worker, chunks))

    # chronological order
    results.sort(key=lambda t: t[0])
    combined = "\n".join(r for _, r in results if r)

    # de-dup keeping first occurrence
    seen: Set[str] = set()
    final: List[str] = []
    for ln in combined.splitlines():
        s = ln.strip()
        if s and s not in seen:
            final.append(ln)
            seen.add(s)
    return "\n".join(final)

# --------------------------------------------------------------------------
# =====  Section 4 ── *** NEW FILE-SELECTION LOGIC ***  =====================
# --------------------------------------------------------------------------
# Everything below is copied / adapted from your `filter_comms.py`
# (with tiny renames and zero functional changes).

# ---- date helpers ---------------------------------------------------------
def _parse_date_str(s: str) -> datetime:
    return datetime.strptime(s, "%B %d %Y")

def _date_on_or_before(d: datetime, end: datetime) -> bool:
    return d <= end

def _parse_date_from_text(txt: str) -> Optional[datetime]:
    pat = re.compile(r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b")
    m = pat.search(txt)
    if not m:
        return None
    for fm in ("%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(" ".join(m.groups()), fm)
        except ValueError:
            continue
    return None

def _parse_ground_country(header: str) -> Tuple[str, str]:
    ground, country = "", ""
    idx = header.lower().find(" at ")
    if idx >= 0:
        sub = header[idx + 4 :]
        cidx = sub.find(",")
        ground = sub[:cidx].strip() if cidx > 0 else sub.strip()
    return ground, country

# ---- team-map -------------------------------------------------------------
def _load_team_map() -> Dict[str, str]:
    here = os.path.dirname(os.path.abspath(__file__))
    fp = os.path.join(here, "team_maps.json")
    if not os.path.isfile(fp):
        debug("team_maps.json missing - returning empty map.")
        return {}
    with open(fp, encoding="utf-8") as f:
        return json.load(f)

# ---- Recent100 parser -----------------------------------------------------
def _parse_recent100(player_name: str, team_map: Dict[str, str]) -> List[dict]:
    path = os.path.join("player_matches", f"{player_name}Recent100.txt")
    if not os.path.isfile(path):
        debug(f"Missing Recent100 for {player_name}.")
        return []

    recs: List[dict] = []
    mid_pat = re.compile(r"Match ID:\s*(\d+)")
    head_pat = re.compile(r"Header:\s*(.*)")

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue

            m_mid = mid_pat.search(line)
            if not m_mid:
                continue
            mid = int(m_mid.group(1))

            header = head_pat.search(line)
            header_text = header.group(1).strip() if header else line

            date = _parse_date_from_text(header_text) or _parse_date_from_text(line)
            if not date:
                continue

            ground, country = _parse_ground_country(header_text)

            teams_str = header_text.split(",", 1)[0]
            t1 = t2 = ""
            if " vs " in teams_str:
                t1, t2 = [x.strip() for x in teams_str.split(" vs ", 1)]
            recs.append(
                {
                    "match_id": mid,
                    "date": date,
                    "team1_code": team_map.get(t1, t1.upper()),
                    "team2_code": team_map.get(t2, t2.upper()),
                    "ground": ground,
                    "country": country,
                }
            )
    recs.sort(key=lambda r: r["date"], reverse=True)
    debug(f"Recent100 parsed - {len(recs)} records.")
    return recs

# ---- finders --------------------------------------------------------------
def _find_raw(
    mid: int,
    is_bat: bool,
    player_team: str,
    opp_team: str,
    base_dir: str,
) -> Optional[str]:
    if not os.path.isdir(base_dir):
        return None
    targ_team = player_team if is_bat else opp_team
    rgx = re.compile(rf"^{mid}_{targ_team}_commentary_.*\.txt$")
    for fn in os.listdir(base_dir):
        if rgx.match(fn):
            return os.path.join(base_dir, fn)
    return None

def _find_venue(
    mid: int,
    is_bat: bool,
    player_team: str,
    opp_team: str,
    ground: str,
    country: str,
    venue_dir: str,
) -> Optional[str]:
    if not os.path.isdir(venue_dir):
        return None
    targ_team = player_team if is_bat else opp_team
    rgx = re.compile(rf"^{mid}_{targ_team}_commentary_.*\.txt$", re.IGNORECASE)
    g_pass, c_pass = [], []
    for fn in os.listdir(venue_dir):
        if rgx.match(fn):
            low = fn.lower()
            if ground and ground.lower() in low:
                g_pass.append(os.path.join(venue_dir, fn))
            elif country and country.lower() in low:
                c_pass.append(os.path.join(venue_dir, fn))
    return g_pass[0] if g_pass else c_pass[0] if c_pass else None

# ---- bucket selectors (unchanged behaviour) ------------------------------
def _sel_vs_opp(
    recs: List[dict],
    opp: str,
    is_bat: bool,
    player_team: str,
    base_dir: str,
    need: int,
    exclude: Set[int],
) -> List[str]:
    out = []
    for r in recs:
        if r["match_id"] in exclude:
            continue
        if opp not in (r["team1_code"], r["team2_code"]):
            continue
        p = _find_raw(r["match_id"], is_bat, player_team, opp, base_dir)
        if p:
            out.append(p)
            exclude.add(r["match_id"])
            if len(out) >= need:
                break
    debug(f"vs-opp bucket: {len(out)}/{need}")
    return out

def _sel_venue(
    recs: List[dict],
    is_bat: bool,
    player_team: str,
    opp_team: str,
    venue_dir: str,
    need: int,
    exclude: Set[int],
    force_ground: str,
    force_country: str,
) -> List[str]:
    """
    Priority order
    1) matches vs opp  **AND**   force_ground
    2) matches vs opp  **AND**   force_country      (ground ignored)
    3) ANY opponent    (previous behaviour - ground first, then country)
    """
    out: List[str] = []

    # ---------- helper to loop once with given ground / country ------------
    def _harvest(target_ground: str, target_country: str, vs_only: bool) -> None:
        nonlocal out, need
        for r in recs:
            if r["match_id"] in exclude:
                continue
            if vs_only and opp_team not in (r["team1_code"], r["team2_code"]):
                continue
            p = _find_venue(
                r["match_id"],
                is_bat,
                player_team,
                opp_team,
                target_ground,
                target_country,
                venue_dir,
            )
            if p:
                out.append(p)
                exclude.add(r["match_id"])
                if len(out) >= need:
                    break

    # pass‑1: opp + ground ---------------------------------------------------
    if force_ground:
        _harvest(force_ground, force_country, vs_only=True)

    # pass‑2: opp + country --------------------------------------------------
    if len(out) < need and force_country:
        _harvest("", force_country, vs_only=True)   # ground intentionally ""

    # pass‑3: old behaviour (any opponent) ----------------------------------
    if len(out) < need:
        _harvest(force_ground, force_country, vs_only=False)

    debug(f"venue bucket (new): {len(out)}/{need}")
    return out[:need]            # trim just in case

def _sel_any_raw(
    recs: List[dict],
    is_bat: bool,
    player_team: str,
    opp_team: str,
    base_dir: str,
    need: int,
    exclude: Set[int],
) -> List[str]:
    out = []
    for r in recs:
        if r["match_id"] in exclude:
            continue
        p = _find_raw(r["match_id"], is_bat, player_team, opp_team, base_dir)
        if p:
            out.append(p)
            exclude.add(r["match_id"])
            if len(out) >= need:
                break
    debug(f"any-raw bucket: {len(out)}/{need}")
    return out

def _sel_final_fallback(
    is_bat: bool,
    player_team: str,
    base_dir: str,
    k: int,
    exclude: Set[int],
) -> List[str]:
    if not os.path.isdir(base_dir):
        return []
    rgx = re.compile(rf"^(\d+)_({player_team})_commentary_.*\.txt$")
    pairs = []
    for fn in os.listdir(base_dir):
        m = rgx.match(fn)
        if m:
            mid = int(m.group(1))
            if mid in exclude:
                continue
            pairs.append((mid, os.path.join(base_dir, fn)))
    pairs.sort(key=lambda t: t[0], reverse=True)
    out = []
    for mid, path in pairs:
        out.append(path)
        exclude.add(mid)
        if len(out) >= k:
            break
    debug(f"final fallback bucket: {len(out)}/{k}")
    return out

# ---- master file-selection orchestrator -----------------------------------
def pick_commentary_files(
    player_name: str,
    player_team: str,
    opp_team: str,
    is_batsman: bool,
    k: int,
    against_comms: int,
    venue_comms: int,
    base_dir: str,
    venue_dir: str,
    venue_ground: str,
    venue_country: str,
    end_date: str,
) -> List[str]:
    if against_comms + venue_comms != k:
        raise ValueError("against_comms + venue_comms must equal k")

    team_map = _load_team_map()
    recs_all = _parse_recent100(player_name, team_map)
    if not recs_all:
        debug("No Recent100 → skip straight to final fallback")
        return _sel_final_fallback(is_batsman, player_team, base_dir, k, set())

    end_dt = _parse_date_str(end_date)
    recs = [r for r in recs_all if _date_on_or_before(r["date"], end_dt)]
    exclude: Set[int] = set()

    if not recs:
        debug("No records before end_date")
        return _sel_final_fallback(is_batsman, player_team, base_dir, k, exclude)

    # --- Step 1  vs opponent ------------------------------------------------
    opp_files = _sel_vs_opp(
        recs, opp_team, is_batsman, player_team, base_dir, against_comms, exclude
    )
    if len(opp_files) < against_comms:
        # Step 2  try fill entire k from venue
        venue_files = _sel_venue(
            recs,
            is_batsman,
            player_team,
            opp_team,
            venue_dir,
            k,
            exclude,
            venue_ground,
            venue_country,
        )
        if len(venue_files) < k:
            # Step 3 fill k from raw by recency
            need_left = k - len(venue_files)          # NEW
            # top‑up from raw vs‑opponent first       # NEW
            more_opp = _sel_vs_opp(
                recs, opp_team, is_batsman, player_team,
                base_dir, need_left, exclude
            )
            venue_files.extend(more_opp)              # NEW
            fallback_files = _sel_any_raw(
                recs,
                is_batsman,
                player_team,
                opp_team,
                base_dir,
                k,
                exclude,
            )
            if len(fallback_files) < k:
                # Step 4 final fallback
                return _sel_final_fallback(is_batsman, player_team, base_dir, k, exclude)
            return fallback_files
        return venue_files

    # If Step-1 ok, try Step-2 for the remaining quota
    venue_files = _sel_venue(
        recs,
        is_batsman,
        player_team,
        opp_team,
        venue_dir,
        venue_comms,
        exclude,
        venue_ground,
        venue_country,
    )
    if len(venue_files) < venue_comms:
        # couldn’t fill - just pick k by recency
        need_left = venue_comms - len(venue_files)    # NEW
        more_opp = _sel_vs_opp(
            recs, opp_team, is_batsman, player_team,
            base_dir, need_left, exclude
        )
        venue_files.extend(more_opp)
        fallback_files = _sel_any_raw(
            recs, is_batsman, player_team, opp_team, base_dir, k, exclude
        )
        if len(fallback_files) < k:
            return _sel_final_fallback(is_batsman, player_team, base_dir, k, exclude)
        return fallback_files

    return opp_files + venue_files

# --------------------------------------------------------------------------
# =====  Section 5 ──  MAIN  ===============================================
# --------------------------------------------------------------------------
def main() -> None:
    load_dotenv()

    p = argparse.ArgumentParser(
        description="Filter cricket commentary for a player using chunk-level "
        "LLM extraction, with advanced file-selection logic."
    )
    # --- identity / role
    p.add_argument("--player_name", required=True)
    p.add_argument("--player_team_code", required=True)
    p.add_argument("--opponent_team_code", required=True)
    p.add_argument("--is_batsman", choices=["True", "False"], default="True")
    # --- file-selection hyper-params
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--against_comms", type=int, default=3)
    p.add_argument("--venue_comms", type=int, default=2)
    p.add_argument("--venue_country", type=str, default="")
    p.add_argument("--venue_ground", type=str, default="")
    p.add_argument("--end_date", type=str, default="December 31 9999")
    p.add_argument("--base_dir", default="../CommentaryData/raw")
    p.add_argument("--venue_dir", default="../CommentaryData/venuewise")
    # --- chunking / LLM
    p.add_argument("--chunk_size", type=int, default=300)
    p.add_argument("--chunk_overlap", type=int, default=10)
    # --- output
    p.add_argument("--output_dir", default="./filtered_outputs")
    args = p.parse_args()

    # sanity
    if args.against_comms + args.venue_comms != args.k:
        print("[ERROR] against_comms + venue_comms must equal k.")
        sys.exit(1)

    is_batsman = args.is_batsman.lower() == "true"
    api_key = os.getenv("OPENAI_API_KEY")

    # ------------------------------------------------------------------ file pick
    files = pick_commentary_files(
        player_name=args.player_name,
        player_team=args.player_team_code,
        opp_team=args.opponent_team_code,
        is_batsman=is_batsman,
        k=args.k,
        against_comms=args.against_comms,
        venue_comms=args.venue_comms,
        base_dir=args.base_dir,
        venue_dir=args.venue_dir,
        venue_ground=args.venue_ground,
        venue_country=args.venue_country,
        end_date=args.end_date,
    )

    if not files:
        print("[INFO] No commentary files found - nothing to process.")
        return

    print("\n[INFO] Final commentary set:")
    for fp in files:
        print("   ", fp)

    # print(args.chunk_size, args.chunk_overlap)
    # print("[INFO] Exiting so you can verify the files before calling LLM.")
    # sys.exit(0)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ per-file processing
    combined_sections = []
    for idx, path in enumerate(files, 1):
        print(f"\n[INFO] ===== Processing file {idx}/{len(files)} =====")
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
        filtered = filter_commentary_in_chunks(
            raw,
            player_name=args.player_name,
            api_key=api_key,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
        )
        combined_sections.append(f"Commentary {idx}:\n{filtered}")

    out_name = (
        f"filtered_commentaries_{args.player_name}_"
        f"{args.opponent_team_code}_{args.k}.txt"
    )
    out_path = os.path.join(args.output_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as out_fh:
        out_fh.write("\n\n".join(combined_sections))

    print(f"\n[INFO] All done! Filtered commentary saved to:\n   {out_path}\n")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
