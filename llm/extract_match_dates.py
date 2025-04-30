#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-scan date extractor for ESPNcricinfo score-card JSONs.

• 1st pass  : match_details["date"]
• 2nd pass  : match_details["match_days"] / ["header"]
• 3rd pass  : walk every value in the JSON; pick the first
              string that parses cleanly into a DD Month YYYY
              using the season context for missing years.

Result is guaranteed reproducible: the traversal order is
file order  ➜  dict key order  ➜  list index.
"""

import os
import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

# ----------------------------------------------------------------------
# 1.  Regexes & low-level utilities  (identical to the previous reply)
# ----------------------------------------------------------------------

MONTHS_FULL = ("January|February|March|April|May|June|July|August|"
               "September|October|November|December")
MONTHS_ABBR = ("Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec")
MONTH_RE    = re.compile(rf"\b(?:{MONTHS_FULL}|{MONTHS_ABBR})\b", re.I)
ORDINAL_RE  = re.compile(r"(\d+)(st|nd|rd|th)", re.I)
DASH_RE     = re.compile(r"\s*[-–—]\s*")

_PATTERNS = ["%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y"]

def _expand_two_digit_year(start: str, two: str) -> int:
    y = int(start[:2] + two.zfill(2))
    return y + 100 if y < int(start) else y

def _season_year(raw: str, season: str) -> Optional[int]:
    if not season:
        return None
    m = re.match(r"(\d{4})\D+(\d{2,4})", season)
    if m:
        s, e = m.groups()
        e = int(e) if len(e) == 4 else _expand_two_digit_year(s, e)
        mon = MONTH_RE.search(raw)
        if mon and mon.group(0)[:3].lower() in {"jan", "feb", "mar"}:
            return e
        return int(s)
    m2 = re.match(r"(\d{4})", season)
    return int(m2.group(1)) if m2 else None

def _clean(txt: str) -> str:
    if not txt:
        return ""
    txt = txt.replace("\u2013", "-").replace("\u2014", "-").strip()
    if "-" in txt:                      # keep only the LH date of a range
        for chunk in DASH_RE.split(txt):
            if MONTH_RE.search(chunk):
                txt = chunk.strip()
                break
    return ORDINAL_RE.sub(r"\1", txt)

def _try_parse(txt: str) -> Optional[datetime]:
    for fmt in _PATTERNS:
        try:
            return datetime.strptime(txt, fmt)
        except ValueError:
            continue
    return None

def parse_date(raw: str, season: str = "") -> Optional[str]:
    raw = _clean(raw)
    if not raw:
        return None
    # already includes a year?
    stamp = _try_parse(raw)
    if stamp:
        return stamp.strftime("%d %B %Y")
    # append inferred year
    yr = _season_year(raw, season)
    if yr:
        stamp = _try_parse(f"{raw} {yr}")
        if stamp:
            return stamp.strftime("%d %B %Y")
    return None

def extract_date_fragment(text: str) -> Optional[str]:
    m = re.search(rf"(?:{MONTHS_FULL}|{MONTHS_ABBR})\s+\d{{1,2}}", text, re.I)
    return m.group(0) if m else None

# ----------------------------------------------------------------------
# 2.  NEW: recursive search for dates anywhere in the JSON
# ----------------------------------------------------------------------

def _collect_strings(node: Any) -> List[str]:
    """Depth-first yield of every string value in *node* (keeps order)."""
    out: List[str] = []
    if isinstance(node, dict):
        for v in node.values():
            out.extend(_collect_strings(v))
    elif isinstance(node, list):
        for v in node:
            out.extend(_collect_strings(v))
    elif isinstance(node, str):
        out.append(node)
    return out

def _deep_search_for_date(data: dict, season: str) -> Optional[str]:
    for txt in _collect_strings(data):
        frag = extract_date_fragment(txt)
        if frag:
            parsed = parse_date(frag, season)
            if parsed:
                return parsed
    return None

# ----------------------------------------------------------------------
# 3.  High-level driver
# ----------------------------------------------------------------------

def extract_match_dates_from_directory(path: str) -> Dict[str, str]:
    result: Dict[str, str] = {}

    for fname in os.listdir(path):
        if not (fname.startswith("match_") and fname.endswith(".json")):
            continue

        match_id = fname.split("_")[1].split(".")[0]
        fpath = os.path.join(path, fname)

        try:
            with open(fpath, encoding="utf-8") as fh:
                data = json.load(fh)

            md      = data.get("match_details", {})
            season  = str(md.get("season", "")).strip()
            raw     = str(md.get("date", "")).strip()

            # 1) direct
            final = parse_date(raw, season)

            # 2) fall-back inside header / match_days
            if not final:
                for key in ("match_days", "header"):
                    final = parse_date(str(md.get(key, "")), season)
                    if final:
                        break

            # 3) deep scan (NEW)
            if not final:
                final = _deep_search_for_date(data, season)

            result[match_id] = final or "UNKNOWN"

        except Exception as exc:
            result[match_id] = f"UNKNOWN ({exc.__class__.__name__})"

    return result

def save_mapping(mapping: Dict[str, str],
                 outfile: str = "match_date_mapping.json") -> None:
    with open(outfile, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=4, ensure_ascii=False)

# ----------------------------------------------------------------------

if __name__ == "__main__":
    directory = "../cricinfo_data_scrapping/match_scorecards_json/"
    mapping   = extract_match_dates_from_directory(directory)
    save_mapping(mapping)

    unknown = sum(v == "UNKNOWN" for v in mapping.values())
    print(f"✅  Finished.  Parsed {len(mapping)-unknown} dates, {unknown} still missing.")
