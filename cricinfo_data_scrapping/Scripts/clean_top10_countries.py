import csv

# Define the Top 10 teams
top_teams = {
    "Australia",
    "England",
    "Pakistan",
    "India",
    "New Zealand",
    "South Africa",
    "West Indies",
    "Afghanistan",
    "Sri Lanka",
    "Bangladesh",
}

input_file = "series_match_commentary_urls.csv"
output_file = "series_filteredbyTopCountries.csv"


# Helper function to extract the team name (first two words, if possible)
def get_team_name(side):
    words = side.strip().split()
    return " ".join(words[:2]) if len(words) >= 2 else words[0]


with open(input_file, "r", encoding="utf-8") as infile, open(
    output_file, "w", newline="", encoding="utf-8"
) as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()

    for row in reader:
        match_title = row["Match Title"]

        # Split on " Vs "
        sides = match_title.split(" Vs ")

        if len(sides) == 2:
            team1 = get_team_name(sides[0])
            team2 = get_team_name(sides[1])

            # Check if either team is in top_teams
            if team1 in top_teams or team2 in top_teams:
                writer.writerow(row)
        else:
            # If match title doesn't follow the format, include the row by default
            writer.writerow(row)
