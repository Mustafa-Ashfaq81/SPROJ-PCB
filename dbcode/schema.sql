CREATE TABLE matches (
    match_id INT PRIMARY KEY,
    series_name VARCHAR(255),
    match_title VARCHAR(255),
    match_url TEXT,
    venue VARCHAR(255),
    match_date TEXT,
    toss_winner VARCHAR(50),
    elected_to VARCHAR(250),
    player_of_match VARCHAR(100),
    match_result TEXT,
    team_1 VARCHAR(50),
    team_2 VARCHAR(50),
    year INT,
    playing_xi JSONB  -- Stores an array of player IDs playing in this match
);

CREATE TABLE players (
    player_id SERIAL PRIMARY KEY,
    player_name VARCHAR(100) UNIQUE,
    teams JSONB  -- Stores a list of teams the player has played for
);

CREATE TABLE player_performances (
    performance_id SERIAL PRIMARY KEY,
    match_id INT REFERENCES matches(match_id),
    player_id INT REFERENCES players(player_id),
    team VARCHAR(50),  -- The team player was playing for in that match
    -- create two flags one captain to tell if a player is captain or not
    -- second a wk flag to tell if a player is wk or not
    -- create a sub flag to tell if a player is a substitute

    -- Batting Performance
    runs INT DEFAULT 0,
    balls INT DEFAULT 0,
    strike_rate DECIMAL(5,2),
    fours INT DEFAULT 0,
    sixes INT DEFAULT 0,
    dismissal TEXT,

    -- Bowling Performance
    overs DECIMAL(4,1) DEFAULT 0,
    maidens INT DEFAULT 0,
    runs_conceded INT DEFAULT 0,
    wickets INT DEFAULT 0,
    economy DECIMAL(4,2),

    -- Fielding Performance
    catches INT DEFAULT 0,
    stumpings INT DEFAULT 0,
    run_outs INT DEFAULT 0
);

ALTER TABLE player_performances ADD CONSTRAINT unique_match_player UNIQUE (match_id, player_id);

