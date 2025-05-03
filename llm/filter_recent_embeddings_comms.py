#!/usr/bin/env python3
"""
filter_recent_embeddings_comms.py
=================================

A drop-in replacement for *filter_recent_comms.py* that now:
• Uses `text-embedding-ada-002` to semantically filter commentary lines, **and**
• Processes the chosen `k` commentary files in **parallel** while keeping their
  final ordering intact.

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
    spaced = camel_to_spaced(player_name).lower()
    nospc = spaced.replace(" ", "")
    last = spaced.split()[-1]
    return [
        rf"\b{re.escape(spaced)}\b",
        rf"\b{re.escape(nospc)}\b",
        rf"\b{re.escape(last)}\b",
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


# global token tracker (thread-safe)
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
    """Call OpenAI Embedding and return vectors; thread-safe token count."""
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


CAT_TEMPLS = (
    "Venue or match context mentioning stadium or city",
    "Description of pitch conditions such as pace, bounce, spin or dryness",
    "Commentator quote or observation about {PLAYER}'s form or batting style",
    "Dismissal of any batter in the innings",
    "Ball-by-ball description involving {PLAYER}: deliveries faced, runs scored, edges, boundaries",
    "{PLAYER}'s own dismissal description",
)
THRESH = 0.83
BATCH = 100


def build_category_vecs(player: str) -> List[List[float]]:
    lines = [t.format(PLAYER=player) for t in CAT_TEMPLS]
    debug("Embedding category reference vectors …")
    return embed_texts(lines)


def keep_line(
    line: str,
    vec: List[float],
    cat_vecs: List[List[float]],
    player: str,
) -> bool:
    if player.lower() in line.lower():
        return True
    return any(cosine(vec, cat) >= THRESH for cat in cat_vecs)


def filter_file(path: str, player: str, cat_vecs: List[List[float]]) -> List[str]:
    with open(path, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh]

    kept: List[str] = []
    for i in range(0, len(lines), BATCH):
        chunk = lines[i : i + BATCH]
        vecs = embed_texts(chunk)
        for ln, vec in zip(chunk, vecs):
            if keep_line(ln, vec, cat_vecs, player):
                kept.append(ln)
    return kept


# ─────────────────────────────── main ─────────────────────────
def main() -> None:
    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("OPENAI_API_KEY not set (env var or .env).")
    openai.api_key = os.environ["OPENAI_API_KEY"]

    ap = argparse.ArgumentParser(
        description="Filter recent commentary with embeddings."
    )
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
    cat_vecs = build_category_vecs(args.player_name)

    # ───── PROCESS EACH FILE IN PARALLEL ─────
    def _worker(idx_path):
        idx, path = idx_path
        debug(f"Filtering {path}")
        kept = filter_file(path, args.player_name, cat_vecs)
        return idx, f"Commentary {idx}:\n" + "\n".join(kept) + "\n"

    max_workers = min(len(chosen), os.cpu_count() or 4)
    futures: List[cf.Future] = []
    ordered_blocks: List[str] = ["" for _ in range(len(chosen))]

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx, p in enumerate(chosen, 1):
            futures.append(ex.submit(_worker, (idx, p)))

        for fut in cf.as_completed(futures):
            idx, block = fut.result()
            ordered_blocks[idx - 1] = block  # preserve order

    out_text = "\n".join(ordered_blocks)
    out_name = (
        f"filtered_commentaries_{args.player_name.replace(' ', '')}_"
        f"{primary_team}_{len(chosen)}.txt"
    )
    out_path = os.path.join(args.output_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(out_text)

    print(f"\n[INFO] Filtered commentary saved to {out_path}\n")

    # ───── Cost reporting ─────
    print(f"\n[INFO] Total tokens used: {_total_tokens_used}")
    cost = _total_tokens_used / 1_000_000 * 0.10
    print(f"[INFO] Estimated embedding cost: ${cost:.6f} USD")

    try:
        with open("embedding_token_usage.txt", "w") as f:
            f.write(str(_total_tokens_used))
    except Exception as e:
        print(f"[WARNING] Could not write embedding token usage: {e}")


if __name__ == "__main__":
    main()
