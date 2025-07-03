import csv
import re
import sys
import io

# Regex setup
MODE_REGEX = r"(Bowled|Caught by a fielder|Caught by the wicketkeeper|Stumped|LBW|Hit[- ]Wicket|Caught and bowled|Caught & bowled|Not[- ]Out)"
CANON = [
    "Bowled", "Caught-Fielder", "Caught-Wicketkeeper", "Stumped",
    "LBW", "Hit-Wicket", "Caught-and-Bowled", "Not-Out"
]
SYNONYMS = {
    "bowled": "Bowled",
    "caught by a fielder": "Caught-Fielder",
    "caught by the wicketkeeper": "Caught-Wicketkeeper",
    "stumped": "Stumped",
    "lbw": "LBW",
    "hit wicket": "Hit-Wicket",
    "caught and bowled": "Caught-and-Bowled",
    "caught & bowled": "Caught-and-Bowled",
    "not out": "Not-Out",
}

prob_pattern = re.compile(
    rf"{MODE_REGEX}\s*[:：–—]\s*(\d+(?:\.\d+)?)\s*(?:%|％)",
    re.IGNORECASE
)
pred_pattern = lambda name: re.compile(
    rf"most\s+likely\s+mode\s+of\s+dismissal\s+for\s+{re.escape(name)}\s+is\s+({MODE_REGEX})",
    re.IGNORECASE
)

def is_all_zero(s: str):
    clean = s.strip().strip('"').strip("'")
    parts = [p.strip() for p in clean.split(",")]
    return len(parts) == 8 and all(p in ("0", "0.0") for p in parts)

def parse_after_step6(text: str, player: str):
    m = re.search(r"step\s*6\s*[:：–—]", text, re.IGNORECASE)
    after = text[m.end():] if m else ""
    tmp = {}
    for mode, pct in prob_pattern.findall(after):
        key = SYNONYMS.get(mode.lower())
        if not key:
            canon = mode.title().replace(" & ", "-and-").replace(" ", "-")
            if canon in CANON:
                key = canon
        if key:
            tmp[key] = float(pct)
    full = {c: tmp.get(c, 0.0) for c in CANON}
    narrative = ""
    pm = pred_pattern(player).search(after)
    if pm:
        raw = pm.group(1)
        narrative = SYNONYMS.get(raw.lower(), raw.replace(" ", "-").title())
    return full, narrative, bool(after.strip())

def rewrite_line(raw_line, header, new_row):
    """
    raw_line: original CSV line as text (including newline)
    header: list of column names
    new_row: dict mapping column->new value
    """
    # parse raw_line into fields
    reader = csv.reader([raw_line])
    fields = next(reader)
    # update only the three columns
    for col in ("mode_of_dismissal_predicted_probability",
                "mode_of_dismissal_predicted",
                "most_likely_mode"):
        idx = header.index(col)
        fields[idx] = new_row[col]
    # re-serialize
    output = io.StringIO()
    writer = csv.writer(output, lineterminator=raw_line[len(raw_line.rstrip('\r\n')):])
    writer.writerow(fields)
    return output.getvalue()

def interactive_fix(path):
    # read raw lines
    with open(path, 'r', encoding='utf-8', newline='') as f:
        raw_lines = f.readlines()

    # parse CSV rows
    with open(path, newline='', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
        header = reader[0].keys()

    modified = 0

    # iterate rows with line numbers
    for idx, row in enumerate(reader, start=2):
        orig = row["mode_of_dismissal_predicted_probability"]
        if is_all_zero(orig):
            player = row["player_name"]
            match  = row["prediction_match"]
            full_map, narrative, has_step6 = parse_after_step6(row["reasoning"], player)

            print("\n" + "-"*60)
            print(f"Line #{idx}: {player} — {match}")
            print(f"Orig Prob: {orig}")

            if not has_step6:
                print("⚠️  No Step 6 breakdown found in reasoning.")
                print(f"Current Predicted Labels: {row['mode_of_dismissal_predicted']}")
                choice = input("Enter=skip, 'r'+Enter=reset to N/A, 'q'+Enter=quit: ")
                if choice.lower() == 'q':
                    print("Aborting early.")
                    break
                if choice.lower() == 'r':
                    # reset
                    na_str = ", ".join(["N/A"]*8)
                    row["mode_of_dismissal_predicted"] = na_str
                    row["most_likely_mode"] = "N/A"
                    # rewrite only this line
                    raw_lines[idx-1] = rewrite_line(raw_lines[idx-1], list(header), row)
                    with open(path, 'w', encoding='utf-8', newline='') as f:
                        f.writelines(raw_lines)
                    print(f"[RESET] Line #{idx} labels set to N/A.")
                    modified += 1
            else:
                print(f"Parsed: {full_map}")
                sorted_items = sorted(full_map.items(), key=lambda kv: kv[1], reverse=True)
                prob_str = ", ".join(f"{v:.2f}".rstrip("0").rstrip(".") for _, v in sorted_items)
                label_str = ", ".join(lbl for lbl, _ in sorted_items)
                top = sorted_items[0][0]
                print(f"NewProb: {prob_str}")
                print(f"NewLbl:  {label_str}")
                print(f"NewTop:  {top}")
                choice = input("Enter=apply, 's'+Enter=skip, 'q'+Enter=quit: ")
                if choice.lower() == 'q':
                    print("Aborting early.")
                    break
                if choice.lower() != 's':
                    row["mode_of_dismissal_predicted_probability"] = prob_str
                    row["mode_of_dismissal_predicted"] = label_str
                    row["most_likely_mode"] = top
                    # rewrite only this line
                    raw_lines[idx-1] = rewrite_line(raw_lines[idx-1], list(header), row)
                    with open(path, 'w', encoding='utf-8', newline='') as f:
                        f.writelines(raw_lines)
                    print(f"[FLUSHED] Line #{idx} updated.")
                    modified += 1

    print(f"\n✅ Done. Total changes: {modified}. File saved: {path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python interactive_fix.py <input.csv>")
        sys.exit(1)
    interactive_fix(sys.argv[1])
