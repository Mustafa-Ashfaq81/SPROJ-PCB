# Cricket Data Science Analysis

Welcome to the Cricket Data Science Analysis repository! This project delves into the world of cricket using data science techniques to analyze player performance, focusing on dismissals and runs scored against different categories of bowlers.

## Introduction

The current notebook, `Player_Dismissals.ipynb`, provides an in-depth analysis of player dismissals and runs scored against various types of bowlers. This dynamic notebook allows users to analyze data for players from Pakistan and India. By adjusting the `team` and `player_name` variables, users can easily switch between different teams and players to perform targeted analysis.

# Running Flow (run_script.py : produced file/dir):

1. pip install -r requirements.txt
2. get_odi_series_urls.py : odi_series_urls.csv
3. get_scorecard_commentary_urls.py : series_match_commentary_urls.csv, series_match_scorecard_urls.csv
4. get_scorecard.py : (dir)match_scorecards_json
5. add_btype_scorecards.py
6. add_batting_type_scorecards.py
7. add_playingxi.py
8. find_slash.py (Scorecards which have "/" in venues)
9. correct_slash.py
10. get_player_stats.py : dir(aggregated_stats)
11. get_player_stats_venuewise.py : dir(aggregated_stats_venue_wise)

## Features

- **Dynamic Analysis**: The notebook allows for flexible analysis based on the team and player selected.
- **Supported Teams**: Pakistan and India.
- **Supported Players**:
  - **Pakistan**:
    - Azhar Ali
    - Fakhar Zaman
    - Babar Azam
    - Shoaib Malik
    - Mohammad Hafeez
    - Imad Wasim
    - Sarfaraz Ahmed
    - Mohammad Amir
    - Shadab Khan
    - Ahmed Shahzad
    - Faheem Ashraf
    - Haris Sohail
  - **India**:
    - Rohit Sharma
    - Shikhar Dhawan
    - Virat Kohli
    - Yuvraj Singh
    - MS Dhoni
    - Kedar Jadhav
    - Hardik Pandya
    - Ravindra Jadeja
    - Dinesh Karthik
    - Ajinkya Rahane

## Authors

- **Mustafa Ashfaq** - [GitHub Profile](https://github.com/Mustafa-Ashfaq81)
- **Essa Shahid Arshad** - [GitHub Profile](https://github.com/essashahid)

## Feedback and Ideas

We welcome any feedback, suggestions, or ideas for improving this project. If you have any thoughts or would like to contribute to enhancing the analysis or adding new features, please reach out to us via our GitHub profiles linked above.

Feel free to open an issue or submit a pull request with your contributions. Your input is invaluable in making this project better!