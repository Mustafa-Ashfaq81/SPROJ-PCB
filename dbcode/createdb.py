import psycopg2

# Database Connection Configurations
DB_CONFIG = {
    "dbname": "cricket_db",  # Replace with your database name
    "user": "postgres",
    "password": "Csmajor@lums25",  # Replace with your actual password
    "host": "localhost",
    "port": "5432",
}

# SQL Query to Create Tables
CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS matches (
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

CREATE TABLE IF NOT EXISTS players (
    player_id SERIAL PRIMARY KEY,
    player_name VARCHAR(100) UNIQUE,
    teams JSONB  -- Stores a list of teams the player has played for
);

CREATE TABLE IF NOT EXISTS player_performances (
    performance_id SERIAL PRIMARY KEY,
    match_id INT REFERENCES matches(match_id) ON DELETE CASCADE,
    player_id INT REFERENCES players(player_id) ON DELETE CASCADE,
    team VARCHAR(50),  -- The team player was playing for in that match

    -- Captain, Wicketkeeper & Substitute Flags
    is_captain BOOLEAN DEFAULT FALSE,  -- True if player was captain
    is_wk BOOLEAN DEFAULT FALSE,       -- True if player was wicketkeeper
    is_sub BOOLEAN DEFAULT FALSE,      -- True if player was a substitute

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
    run_outs INT DEFAULT 0,

    CONSTRAINT unique_match_player UNIQUE (match_id, player_id)
);
"""

def create_tables():
    """Connects to PostgreSQL and creates the required tables."""
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        print("📌 Creating tables...")
        cursor.execute(CREATE_TABLES_SQL)

        # Commit changes
        conn.commit()
        print("✅ Tables created successfully!")

    except psycopg2.Error as e:
        print(f"❌ Error: {e}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        print("🔒 Connection closed.")

# Run the function
if __name__ == "__main__":
    create_tables()
