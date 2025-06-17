import os
import json
import re
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_corrected_series_name_llm(series_name, date, venue):
    """
    Send a prompt to OpenAI LLM to suggest a corrected 'series_name' 
    based on the old name, plus match date and venue context.
    """
    prompt = f"""
We have a cricket series name that may reference multiple countries with slashes or parentheses, for example:
    "Zimbabwe Triangular Series (Australia, South Africa, Zimbabwe), 2014" 
    or 
    "in Australia/South Africa",
but we want to produce a corrected or standardized series name.

You are given:
- The current 'series_name': {series_name}
- The 'match date': {date}
- The 'venue': {venue}

From this info, please respond with just a single line string: the best corrected version of the series name. 
Try to:
1) Preserve essential context, e.g., "Tri-Nation" or "Quadrangular".
2) If it references multiple countries, transform it into a standardized "In [primary host]" style, e.g. "Tri-Nation Series In Ireland, 2017".
3) Retain the year from the original series name if present.
4) Omit any slash lists or parentheses listing multiple countries. 
5) If the venue or date clarifies the host, incorporate that. 
6) Return only the corrected series name, no additional explanations.
"""

    # Using ChatCompletion for GPT-3.5 or GPT-4, for example:
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for cleaning up cricket series names."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=100,
    )

    # Extract the LLM's answer text
    answer = response["choices"][0]["message"]["content"].strip()
    return answer


def main():
    # Directory containing JSON files
    directory = "match_scorecards_json"

    # Regex pattern to match suspicious series_name with slash/country references or comma references
    # This is broad enough to catch multi-country patterns with parentheses or slash usage.
    pattern = re.compile(
        r"in \b[A-Za-z\s]+/[A-Za-z\s]+(/[A-Za-z\s]+)*\b"  # e.g. in CountryA/CountryB...
        r"|.*\([A-Za-z\s]+(,\s?[A-Za-z\s]+)+\),\s*\d{4}.*"  # e.g. (Australia, South Africa, Zimbabwe), 2014
    )

    matching_files = []

    # --- Part 1: Identify suspicious JSONs ---
    for filename in os.listdir(directory):
        if filename.endswith(".json"):  # Only process JSON files
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    # Check if "series_name" exists and matches the regex pattern
                    if "series_name" in data and pattern.search(data["series_name"]):
                        matching_files.append(filename)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {filename}")

    print("Matching files:", matching_files)
    print(f"Number of matches to correct: {len(matching_files)}")

    # --- Part 2: Automatic correction with LLM ---
    for filename in matching_files:
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {filename}")
                continue

        # Extract relevant fields
        series_name = data.get("series_name", "N/A")
        date = data.get("match_details", {}).get("date", "N/A")
        venue = data.get("match_details", {}).get("venue", "N/A")

        # Use LLM to get the corrected name
        corrected_name = get_corrected_series_name_llm(series_name, date, venue)

        # If we get a non-empty response, update the JSON
        if corrected_name and corrected_name != series_name:
            data["series_name"] = corrected_name
            # Save changes
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

            print(f"✅ Updated '{filename}' from:\n   {series_name}\n   to:\n   {corrected_name}")
        else:
            print(f"No changes made for '{filename}' (LLM returned empty or same).")

    print("\n🎉 All files processed!")

if __name__ == "__main__":
    main()
