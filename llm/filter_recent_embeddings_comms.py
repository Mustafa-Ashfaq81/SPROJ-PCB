#!/usr/bin/env python3
"""
filter_recent_embeddings_comms.py
================================

Same purpose as before, but now:
• Embedding templates focus on batter-dismissal context
• Semantic keep-rule always retains dismissal keywords
• A ±5-line context window is captured around every kept line
• Cleaning step trims junk, standardises ball tags, and
  highlights the batter’s **last name** only (less visual clutter).

Usage:
python filter_recent_embeddings_comms.py --player_name "StevenSmith" --player_team_code "AUS" --opponent_team_code "IND" --is_batsman True --k 5 --against_comms 0 --venue_comms 0 --venue_country "India" --venue_ground "Mohali" --end_date "December 23 2025" --base_dir "../cricinfo_data_scrapping/CommentaryData/processed_commetaries/raw" --venue_dir "../cricinfo_data_scrapping/CommentaryData/processed_commetaries/venuewise" --output_dir "./filtered_outputs"
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
import os
import re
import sys
import time
import threading
import shutil
from calendar import month_abbr, month_name
from datetime import datetime
from typing import List, Optional, Tuple

from dotenv import load_dotenv
import openai


# ───────────────────────── debug helper ─────────────────────────
def debug(msg: str) -> None:
    print(f"[DEBUG] {msg}", file=sys.stderr)


# ──────────────────── filename→date utilities ───────────────────
MONTH_MAP = {month_name[m].lower(): m for m in range(1, 13)}
MONTH_MAP.update({month_abbr[m].lower(): m for m in range(1, 13)})

FILE_RE = re.compile(
    r"""
    ^(?P<mid>\d+)_                # match id
      (?P<team>[A-Z]+)_           # team code
      (?:commentary_)?            # optional literal
      (?P<month>[A-Za-z]+)_       # month
      (?P<day>\d{1,2})_           # day
      (?P<year>\d{4})\.txt$       # year
    """,
    re.VERBOSE,
)


def file_date(fname: str) -> Optional[datetime]:
    m = FILE_RE.match(fname)
    if not m:
        return None
    try:
        d = int(m.group("day"))
        mon = MONTH_MAP.get(m.group("month").lower())
        y = int(m.group("year"))
        return datetime(y, mon, d) if mon else None
    except Exception:
        return None


def parse_end_date(s: str) -> datetime:
    return (
        datetime.max if s.lower() in {"", "all"} else datetime.strptime(s, "%B %d %Y")
    )


# ─────────────────────── name helpers ──────────────────────────
def camel_to_spaced(name: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name) if " " not in name else name


def name_variants(player_name: str) -> List[str]:
    """
    Given a CamelCase name like "BabarAzam" or "StevenSmith",
    returns four regex‐strings (to be compiled CI):
      1) full name with space:   \bBabar Azam\b
      2) concatenated name:       \bBabarAzam\b
      3) last name only:          \bAzam\b
      4) first name only:         \bBabar\b
    """
    # Turn CamelCase into “First Last”
    spaced = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', player_name).strip()
    nospc  = spaced.replace(' ', '')
    parts  = spaced.split()
    first  = parts[0]
    last   = parts[-1]

    return [
        rf"\b{re.escape(spaced)}\b",  # full name
        rf"\b{re.escape(nospc)}\b",   # concatenated
        rf"\b{re.escape(last)}\b",    # last only
        rf"\b{re.escape(first)}\b",   # first only
    ]


def text_contains_name(path: str, regexes: List[re.Pattern]) -> bool:
    try:
        with open(path, encoding="utf-8") as fh:
            txt = fh.read().lower()
        return any(r.search(txt) for r in regexes)
    except Exception:
        return False


def collect_candidates(
    raw_dir: str, team_code: str, end_dt: datetime
) -> List[Tuple[datetime, str]]:
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


# ─────────────────── embeddings utilities ─────────────────────
def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


_total_tokens_used = 0
_tokens_lock = threading.Lock()


def _add_tokens(n: int) -> None:
    global _total_tokens_used
    with _tokens_lock:
        _total_tokens_used += n


def embed_texts(
    texts: List[str],
    model: str = "text-embedding-ada-002",
) -> List[List[float]]:
    clean: List[str] = [t for t in texts if isinstance(t, str) and t.strip()]
    if not clean:
        return [[] for _ in texts]

    for attempt in range(4):
        try:
            rsp = openai.Embedding.create(input=clean, model=model)
            _add_tokens(rsp.get("usage", {}).get("total_tokens", 0))
            embs = {d["index"]: d["embedding"] for d in rsp["data"]}
            return [embs.get(i, []) for i in range(len(texts))]
        except openai.error.OpenAIError as e:
            wait = 2**attempt
            debug(f"Embedding error – retrying in {wait}s: {e}")
            time.sleep(wait)
    raise RuntimeError("OpenAI Embedding failed repeatedly")


# ────────────────── CATEGORY VECTORS (UPDATED) ─────────────────
CAT_TEMPLS = (
    # batter-centric
    "{PLAYER} faces the delivery",
    "{PLAYER} scores FOUR or SIX",
    "{PLAYER} gets dismissed bowled, lbw, caught, stumped or run-out",
    "{PLAYER} edges, miscues or nicks behind",
    "{PLAYER} beaten by swing, seam, spin or bounce",
    "{PLAYER} reaches fifty or hundred",
    # contextual
    "Description of pitch or weather conditions",
    "Bowler changes field or angle",
    "Any wicket falls in the innings",
)
THRESH = 0.80          # wider net
BATCH  = 50

DISMISSAL_PAT = re.compile(
    r"\b(b |c |lbw|bowled|stumped|run ?out)\b", flags=re.IGNORECASE
)

def build_category_vecs(player: str) -> List[List[float]]:
    lines = [t.format(PLAYER=player) for t in CAT_TEMPLS]
    debug("Embedding category reference vectors …")
    return embed_texts(lines)

def keep_line(
    line: str,
    vec: List[float],
    cat_vecs: List[List[float]],
    name_regexes: List[re.Pattern],
) -> bool:
    """
    Returns True if this line MUST be kept:
      a) It matches any of the name_regexes (first/full/concat/last)
      b) Or it contains one of your key_phrases
      c) Or its embedding is semantically similar
    """
    low = line.lower()

    # a) Always keep if *any* name‐variant regex matches
    if not any(rx.search(low) for rx in name_regexes):
        return False

    # b) Key‐phrase check (unchanged)
    key_phrases = [
        "bowled", "lbw", "caught", "stumped", "run out",
        "edged", "beaten", "nick", "inside edge", "outside edge",
        "leading edge", "swing", "spin", "bounce", "footwork",
        "aggressive", "drives", "pulls", "hooks", "slogs", "reverse",
        "charges", "flick", "defends", "blocks", "top edge",
        "bottom edge", "lofted", "stepped out", "backs away",
        "rushed", "misjudged",
        # pitch/weather
        "pitch", "conditions", "surface", "weather", "dew", "grip",
        "slow", "dry", "turn", "moisture", "two-paced", "holding up",
        "bouncy", "overcast", "green top", "spin-friendly",
        "low bounce", "sticky", "outfield"
    ]
    if any(p in low for p in key_phrases):
        return True

    # c) Semantic‐similarity fallback (unchanged)
    return any(cosine(vec, cat) >= THRESH for cat in cat_vecs)

# ───────── filter_file with ±WINDOW context (UPDATED) ─────────
WINDOW = 5   # lines of context before & after every kept line

def filter_file(
    path: str,
    name_regexes: List[re.Pattern],
    cat_vecs: List[List[float]],
) -> List[str]:
    """
    Reads the raw commentary file, batches through embeddings,
    and uses keep_line() with name_regexes instead of a fixed player string.
    """
    with open(path, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh]

    keep_mask = [False] * len(lines)

    for i in range(0, len(lines), BATCH):
        chunk = lines[i : i + BATCH]
        vecs = embed_texts(chunk)
        for off, (ln, vec) in enumerate(zip(chunk, vecs)):
            if keep_line(ln, vec, cat_vecs, name_regexes):
                idx = i + off
                for j in range(max(0, idx - WINDOW), min(len(lines), idx + WINDOW + 1)):
                    keep_mask[j] = True

    return [ln for ln, keep in zip(lines, keep_mask) if keep]

def clean_filtered_commentary(raw_text: str, player_name: str) -> str:
    last = player_name.split()[-1].lower()
    last_pat = re.compile(rf"\b{re.escape(last)}\b", flags=re.IGNORECASE)

    # Tagging helpers
    DISMISSAL_RE = re.compile(r"\b(out|b |c |lbw|stumped|run[-\s]?out)\b", re.IGNORECASE)
    SHOT_RE = re.compile(r"\b(six|four|drives?|pulls?|hooks?|cuts?|flicks?|slogs?|edges?|lofts?|blocks?|defends?)\b", re.IGNORECASE)
    CONDITION_RE = re.compile(r"\b(pitch|conditions|surface|slow|turn|grip|dew|weather|moisture|bounce|holding up|two-paced|sticky|green top)\b", re.IGNORECASE)
    QUOTE_RE = re.compile(r'["“][^"”]+["”]')

    def tag_line(line: str) -> str:
        tag = ""
        l = line.lower()
        if DISMISSAL_RE.search(l):
            tag = "[DISMISSAL] "
        elif SHOT_RE.search(l):
            tag = "[SHOT] "
        elif CONDITION_RE.search(l):
            tag = "[CONDITION] "
        elif QUOTE_RE.search(line) and last in l:
            tag = "[QUOTE] "
        return tag + line

    out = []
    seen = set()
    current_commentary_block = []

    for line in raw_text.splitlines():
        line = line.strip()

        if line.startswith("Commentary"):
            if current_commentary_block:
                out.append("\n".join(current_commentary_block))
                current_commentary_block = []
            current_commentary_block.append(line)
            continue

        if not line or "[MISC]" in line:
            continue
        if not line.startswith("[Ball") and last not in line.lower():
            continue

        line = last_pat.sub(lambda m: f"**{m.group(0)}**", line)
        tagged_line = tag_line(line)

        if tagged_line not in seen:
            current_commentary_block.append(tagged_line)
            seen.add(tagged_line)

    if current_commentary_block:
        out.append("\n".join(current_commentary_block))

    return "\n\n".join(out).strip()


# ─────────────────────────────── main ─────────────────────────
def main() -> None:
    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("OPENAI_API_KEY not set (env var or .env).")
    openai.api_key = os.environ["OPENAI_API_KEY"]

    ap = argparse.ArgumentParser(description="Filter recent commentary with embeddings.")
    ap.add_argument("--player_name", required=True)
    ap.add_argument("--player_team_code", required=True)
    ap.add_argument("--opponent_team_code", required=True)
    ap.add_argument("--is_batsman", choices=["True", "False"], default="True")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--end_date", default="December 31 9999")
    ap.add_argument(
        "--base_dir",
        default="../cricinfo_data_scrapping/CommentaryData/processed_commetaries/raw",
    )
    ap.add_argument("--output_dir", default="./filtered_outputs")
    # legacy / ignored args
    ap.add_argument("--against_comms")
    ap.add_argument("--venue_comms")
    ap.add_argument("--venue_country")
    ap.add_argument("--venue_ground")
    ap.add_argument("--venue_dir")
    args = ap.parse_args()

    # Build and compile all four name‐variant patterns, CI
    raw_patterns = name_variants(args.player_name)
    name_regexes = [re.compile(pat, re.IGNORECASE) for pat in raw_patterns]

    raw_dir = args.base_dir
    if not os.path.isdir(raw_dir):
        sys.exit(f"raw directory not found: {raw_dir}")

    primary_team = (
        args.player_team_code.upper()
        if args.is_batsman.lower() == "true"
        else args.opponent_team_code.upper()
    )
    end_dt = parse_end_date(args.end_date)
    regexes = [re.compile(v, re.IGNORECASE) for v in name_variants(args.player_name)]

    candidates = collect_candidates(raw_dir, primary_team, end_dt)
    with_name = [p for _, p in candidates if text_contains_name(p, regexes)]
    without_name = [p for _, p in candidates if p not in with_name]
    chosen = with_name[: args.k]
    if len(chosen) < args.k:
        chosen.extend(without_name[: args.k - len(chosen)])

    if not chosen:
        print("[INFO] No commentary files satisfy the criteria – exiting.")
        sys.exit(0)

    print("\n[INFO] Selected files:")
    for p in chosen:
        print("   ", p)

    os.makedirs(args.output_dir, exist_ok=True)

    # ───── Enhancement: versioned filenames based on match IDs ─────
    match_ids: List[str] = []
    for path in chosen:
        fn = os.path.basename(path)
        m = FILE_RE.match(fn)
        if m:
            match_ids.append(m.group("mid"))

    base = f"filtered_commentaries_{args.player_name.replace(' ', '')}_{primary_team}_{len(chosen)}"
    versioned_name = base + "_" + "_".join(match_ids) + ".txt"
    canonical_name = base + ".txt"
    versioned_path = os.path.join(args.output_dir, versioned_name)
    canonical_path = os.path.join(args.output_dir, canonical_name)

    if os.path.exists(versioned_path):
        print(f"[INFO] Reusing existing filtered file {versioned_name}")
        shutil.copyfile(versioned_path, canonical_path)
        sys.exit(0)

    cat_vecs = build_category_vecs(args.player_name)

    def _worker(idx_path):
        """
        ThreadPool worker: filters one file using name_regexes
        """
        idx, path = idx_path
        debug(f"Filtering {path}")
        kept = filter_file(path, name_regexes, cat_vecs)
        return idx, f"Commentary {idx}:\n" + "\n".join(kept)


    max_workers = min(len(chosen), os.cpu_count() or 4)
    futures: List[cf.Future] = []
    ordered_blocks: List[str] = ["" for _ in range(len(chosen))]

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx, p in enumerate(chosen, 1):
            futures.append(ex.submit(_worker, (idx, p)))

        for fut in cf.as_completed(futures):
            idx, block = fut.result()
            ordered_blocks[idx - 1] = block  # preserve order

    # ─── CLEAN & SAVE ───
    merged_text = "\n".join(ordered_blocks)
    cleaned_text = clean_filtered_commentary(merged_text.strip(), camel_to_spaced(args.player_name))

    # write versioned file
    with open(versioned_path, "w", encoding="utf-8") as fh:
        fh.write(cleaned_text)
    print(f"\n[INFO] Filtered commentary saved to {versioned_path}\n")

    # also copy to canonical filename for downstream pipeline
    shutil.copyfile(versioned_path, canonical_path)
    print(f"[INFO] Also copied to canonical name: {canonical_name}\n")

    # ───── Cost reporting ─────
    print(f"[INFO] Total tokens used: {_total_tokens_used}")
    cost = _total_tokens_used / 1_000_000 * 0.10
    print(f"[INFO] Estimated embedding cost: ${cost:.6f} USD")

    try:
        with open("embedding_token_usage.txt", "w") as f:
            f.write(str(_total_tokens_used))
    except Exception as e:
        print(f"[WARNING] Could not write embedding token usage: {e}")


if __name__ == "__main__":
    main()
