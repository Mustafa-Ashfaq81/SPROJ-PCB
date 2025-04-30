#!/usr/bin/env python3
"""
rename_commentaries.py
──────────────────────

Rename every commentary file in the raw/ folder so that its name becomes

    <matchID>_<battingTeam>_<Month>_<DD>_<YYYY>.txt

The script now also copes with files that currently end in
“_UNKNOWN_DATE.txt”.  For those, it looks up the real date in
**match_date_mapping.json**:

* If the match-ID **is present** in the JSON → file is renamed.
* If the match-ID **is missing**            → file is skipped with
  the reason **“nomap”**.

Summary printed at the end:

    Done: <renamed> renamed, <skipped_pattern> skipped (pattern),
          <skipped_nomap> skipped (no mapping).

Edit the two paths below (`MAPPING_JSON` and `COMMENTARY_DIR`) if your
folder structure differs.
"""

from __future__ import annotations

from pathlib import Path
import json
import re
import sys
from typing import Tuple

# ───────────────────────── paths ──────────────────────────────────────────
MAPPING_JSON   = Path("match_date_mapping.json")   # JSON: { "1234567":"04 March 2018", ... }
COMMENTARY_DIR = Path(
    "../cricinfo_data_scrapping/CommentaryData/processed_commetaries/raw"
)

# ──────────────────────── regex for filenames ─────────────────────────────
# Matches “real-date” files *and* “…_UNKNOWN_DATE.txt”
FILE_RE = re.compile(
    r"""
    ^(?P<match>\d+)_              # match ID
    (?P<team>[^_]+?)_             # batting team (greedy, non-underscore)
    (?:commentary_)?              # optional literal 'commentary_'
    (?:
        (?P<month>[A-Za-z]+)_(?P<day>\d{1,2})_(?P<year>\d{4})   # real date
      | UNKNOWN_DATE                                            # placeholder
    )
    (?P<ext>\.txt)$
    """,
    re.VERBOSE,
)

# ─────────────────────── helpers ──────────────────────────────────────────
def map_date(match_id: str) -> Tuple[str, str, str]:
    """Return (Month, DD, YYYY) for *match_id* from the JSON mapping."""
    raw = mapping[match_id]                      # raises KeyError if absent
    dd, month, yyyy = raw.split()
    return month, dd.zfill(2), yyyy


def resolved_date(groups: dict) -> Tuple[str, str, str] | None:
    """
    Decide the final date for a file.

    • If the filename already contains a real date → use that.
    • Otherwise (UNKNOWN_DATE) try the JSON mapping.
      - If found → return mapped date.
      - If not   → return None  (caller will skip with reason 'nomap').
    """
    if groups["month"]:                          # real date present
        return groups["month"], groups["day"].zfill(2), groups["year"]
    try:
        return map_date(groups["match"])
    except KeyError:
        return None

# ─────────────────── load the mapping once ────────────────────────────────
try:
    mapping: dict[str, str] = json.loads(MAPPING_JSON.read_text(encoding="utf-8"))
except FileNotFoundError as exc:
    sys.exit(f"ERROR: mapping file not found: {exc.filename}")
except json.JSONDecodeError as exc:
    sys.exit(f"ERROR: mapping file is not valid JSON: {exc}")

# ───────────────────────── renaming loop ──────────────────────────────────
renamed = skipped_pattern = skipped_nomap = 0

for path in COMMENTARY_DIR.iterdir():
    if not path.is_file():
        continue

    match = FILE_RE.match(path.name)
    if not match:
        skipped_pattern += 1
        print("SKIP - pattern :", path.name)
        continue

    groups = match.groupdict()
    date_parts = resolved_date(groups)
    if date_parts is None:                       # still unknown → no mapping
        skipped_nomap += 1
        print("SKIP - nomap   :", path.name)
        continue

    month, day, year = date_parts
    new_name = f"{groups['match']}_{groups['team']}_{month}_{day}_{year}{groups['ext']}"
    new_path = path.with_name(new_name)

    # ---- rename on disk --------------------------------------------------
    try:
        path.rename(new_path)
        renamed += 1
    except OSError as exc:
        print(f"ERROR renaming {path.name} → {new_name}: {exc}")

# ─────────────────────────── summary ──────────────────────────────────────
print(
    f"Done: {renamed} renamed, "
    f"{skipped_pattern} skipped (pattern), "
    f"{skipped_nomap} skipped (no mapping)."
)
