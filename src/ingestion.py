import pandas as pd
import sqlite3
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOVIE_CSV_PATH = os.path.join(ROOT_DIR, 'data', 'wiki_movie_plots.csv')
DB_PATH = os.path.join(ROOT_DIR, 'data', 'movie_features.db')
PARQUET_PATH = os.path.join(ROOT_DIR, 'data', 'movie_features.parquet')

def check_file_exists(file_path):
    if not os.path.isfile(file_path):
        print(f"Error: The file {file_path} does not exist.")
        sys.exit(1)

def check_directory_writable(directory):
    if not os.access(directory, os.W_OK):
        print(f"Error: The directory {directory} is not writable.")
        sys.exit(1)

def ingest_data(csv_path: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PARQUET_PATH), exist_ok=True)
    check_file_exists(csv_path)
    
    try:
        df = pd.read_csv(csv_path)
        
        if df.empty:
            print(f"Error: The file {csv_path} is empty.")
            sys.exit(1)

        conn = sqlite3.connect(DB_PATH)
        conn.execute('''
        CREATE TABLE IF NOT EXISTS movies (
            movie_id TEXT PRIMARY KEY,
            Title TEXT,
            Plot TEXT,
            Year INTEGER,
            event_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        df.to_sql('movies', conn, if_exists='replace', index=False)
        df.to_parquet(PARQUET_PATH, index=False)
        
        print(f"Data successfully ingested into SQLite database: {DB_PATH}")
        print(f"Data successfully saved as Parquet: {PARQUET_PATH}")
        
        return df

    except pd.errors.EmptyDataError:
        print(f"Error: The file {csv_path} is empty.")
        sys.exit(1)
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during data ingestion: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

def query_database(query: str) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(DB_PATH)
        result = pd.read_sql_query(query, conn)
        return result
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    df = ingest_data(MOVIE_CSV_PATH)
    print(f"Successfully processed {len(df)} movie records")
