import psycopg2

# Database Connection Configurations
DB_CONFIG = {
    "dbname": "cricket_db",  # Replace with your database name
    "user": "postgres",
    "password": "Csmajor@lums25",  # Replace with your actual password
    "host": "localhost",
    "port": "5432",
}

# SQL Query to Drop All Tables
DROP_ALL_TABLES_SQL = """
DO $$ 
DECLARE 
    r RECORD;
BEGIN 
    FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') 
    LOOP 
        EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
    END LOOP; 
END $$;
"""

def drop_all_tables():
    """Connects to PostgreSQL and drops all tables in the public schema."""
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        print("🛑 Dropping all tables...")
        cursor.execute(DROP_ALL_TABLES_SQL)

        # Commit changes
        conn.commit()
        print("✅ All tables dropped successfully!")

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
    drop_all_tables()
