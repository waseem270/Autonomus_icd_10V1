import sqlite3
import tqdm
import argparse
import logging
import re
import os
import sys
from datetime import datetime

# Set working directory to project root if script is run from seeds folder
if os.path.basename(os.getcwd()) == 'seeds':
    os.chdir('../..')

# Configure paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

# Configure paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

DB_PATH = os.path.join(PROJECT_ROOT, "database", "medical_icd.db")
DATA_PATH = os.path.join(PROJECT_ROOT, "database", "seeds", "data", "icd10cm-codes-April-2025.txt")
LOG_DIR = os.path.join(PROJECT_ROOT, "database", "seeds")
ERROR_LOG = os.path.join(LOG_DIR, "icd_load_errors.log")

# Ensure directories exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=ERROR_LOG,
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ICD10Loader:
    def __init__(self, db_path, data_path):
        # Enforce absolute paths
        self.db_path = os.path.abspath(db_path)
        self.data_path = os.path.abspath(data_path)
        self.batch_size = 5000
        self.commit_interval = 25000
        # Regex for ICD10-CM Code: Letter, Digit, then up to 5 letters/digits (April 2025 format)
        self.code_regex = re.compile(r'^[A-Z][0-9][0-9A-Z]{1,5}$')

    def get_connection(self):
        """Establish SQLite connection."""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            sys.exit(1)

    def setup_database(self, cursor, reset=False):
        """Create table and indexes."""
        if reset:
            print(f"Resetting table: icd10_codes...")
            cursor.execute("DROP TABLE IF EXISTS icd10_codes")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS icd10_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Performance Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_code ON icd10_codes(code)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_description ON icd10_codes(description)")

    def parse_line(self, line):
        """Extract code and description from line."""
        line = line.strip()
        if not line:
            return None
        
        # Split by first whitespace block
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            return None
        
        code = parts[0].strip()
        description = parts[1].strip()
        
        if not self.code_regex.match(code):
            logging.error(f"Validation failed for code: '{code}' in line: '{line}'")
            return None
            
        return (code, description)

    def load(self, reset=False, limit=None):
        """Process file and load into SQLite."""
        if not os.path.exists(self.data_path):
            print(f"Error: Data file not found at {self.data_path}")
            print("Please ensure the file exists before running the loader.")
            return

        conn = self.get_connection()
        cursor = conn.cursor()
        
        self.setup_database(cursor, reset)
        conn.commit()

        # Count total lines for progress bar
        print(f"Reading {self.data_path}...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        load_limit = limit if limit else total_lines
        
        records_to_insert = []
        count_loaded = 0
        count_errors = 0

        print(f"Starting load into {self.db_path}...")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            # Use tqdm for progress tracking
            with tqdm.tqdm(total=load_limit, desc="Loading ICD-10", unit="codes") as pbar:
                for line in f:
                    if limit and count_loaded >= limit:
                        break
                        
                    data = self.parse_line(line)
                    if data:
                        records_to_insert.append(data)
                        count_loaded += 1
                        pbar.update(1)
                    else:
                        if line.strip():
                            count_errors += 1
                    
                    # Batch processing
                    if len(records_to_insert) >= self.batch_size:
                        self._execute_batch(cursor, records_to_insert)
                        records_to_insert = []
                        
                        # Transaction commitment
                        if count_loaded % self.commit_interval == 0:
                            conn.commit()

                # Insert remaining
                if records_to_insert:
                    self._execute_batch(cursor, records_to_insert)
                    conn.commit()

        print(f"\n--- Load Summary ---")
        print(f"Status: Success")
        print(f"Total Loaded: {count_loaded}")
        print(f"Total Errors: {count_errors} (Details in {ERROR_LOG})")
        
        self.verify_counts(cursor)
        conn.close()

    def _execute_batch(self, cursor, records):
        """Perform batch insert."""
        try:
            cursor.executemany(
                "INSERT OR REPLACE INTO icd10_codes (code, description) VALUES (?, ?)", 
                records
            )
        except sqlite3.Error as e:
            logging.error(f"Batch insert error: {e}")

    def verify_counts(self, cursor):
        """Quick verification of data."""
        cursor.execute("SELECT COUNT(*) FROM icd10_codes")
        total = cursor.fetchone()[0]
        print(f"Database Record Count: {total}")
        
        print("\nSample (First 5):")
        cursor.execute("SELECT code, description FROM icd10_codes ORDER BY id ASC LIMIT 5")
        for row in cursor.fetchall():
            print(f"  [{row[0]}] {row[1]}")
            
        print("\nSample (Last 5):")
        cursor.execute("SELECT code, description FROM icd10_codes ORDER BY id DESC LIMIT 5")
        rows = cursor.fetchall()
        for row in reversed(rows):
            print(f"  [{row[0]}] {row[1]}")

def main():
    parser = argparse.ArgumentParser(description="Medical ICD-10 SQLite Loader")
    parser.add_argument("--reset", action="store_true", help="Recreate table before loading")
    parser.add_argument("--verify-only", action="store_true", help="Just check existing counts")
    parser.add_argument("--limit", type=int, help="Stop after loading N records")
    parser.add_argument("--db", default=DB_PATH, help="SQLite DB path")
    parser.add_argument("--data", default=DATA_PATH, help="Source text file path")
    
    args = parser.parse_args()
    
    loader = ICD10Loader(args.db, args.data)
    
    if args.verify_only:
        conn = loader.get_connection()
        loader.verify_counts(conn.cursor())
        conn.close()
    else:
        loader.load(reset=args.reset, limit=args.limit)

if __name__ == "__main__":
    main()
