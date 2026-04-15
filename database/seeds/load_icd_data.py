import sqlite3
import tqdm
import argparse
import logging
import os
import sys
from datetime import datetime

try:
    import openpyxl
    _OPENPYXL_AVAILABLE = True
except ImportError:
    _OPENPYXL_AVAILABLE = False

# Set working directory to project root if script is run from seeds folder
if os.path.basename(os.getcwd()) == 'seeds':
    os.chdir('../..')

# Configure paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

DB_PATH = os.path.join(PROJECT_ROOT, "database", "medical_icd.db")
DATA_PATH = os.path.join(PROJECT_ROOT, "database", "Final_ICD_10_CM.xlsx")
TXT_DATA_PATH = os.path.join(SCRIPT_DIR, "data", "icd10cm-codes-April-2025.txt")
LOG_DIR = os.path.join(PROJECT_ROOT, "database", "seeds")
ERROR_LOG = os.path.join(LOG_DIR, "icd_load_errors.log")

# Ensure directories exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
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
                hcc_status TEXT DEFAULT 'Non-HCC',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add hcc_status column if upgrading from old schema
        try:
            cursor.execute("ALTER TABLE icd10_codes ADD COLUMN hcc_status TEXT DEFAULT 'Non-HCC'")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Performance Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_code ON icd10_codes(code)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_description ON icd10_codes(description)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hcc_status ON icd10_codes(hcc_status)")

    def parse_txt(self):
        """Read ICD-10 records from the plain-text fallback file.
        
        Format: each line is <CODE><whitespace><Description>
        e.g.: A000    Cholera due to Vibrio cholerae 01, biovar cholerae
        """
        print(f"Opening txt fallback: {TXT_DATA_PATH}...")
        records = []
        errors = 0
        with open(TXT_DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                # Split on first whitespace sequence: CODE + rest = description
                parts = line.split(None, 1)
                if len(parts) < 2:
                    errors += 1
                    continue
                code = parts[0].strip()
                description = parts[1].strip()
                if not code or not description:
                    errors += 1
                    continue
                records.append((code, description, "Non-HCC"))
        print(f"Parsed {len(records)} valid records from txt, {errors} skipped.")
        return records

    def parse_excel(self):
        """Read ICD-10 records from the Excel file."""
        if not _OPENPYXL_AVAILABLE:
            print("openpyxl not installed — cannot parse xlsx. Using txt fallback.")
            return self.parse_txt()
        print(f"Opening {self.data_path}...")
        wb = openpyxl.load_workbook(self.data_path, read_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        wb.close()

        # Detect header row
        headers = [str(h).strip() if h else "" for h in rows[0]]
        print(f"Detected columns: {headers}")

        # Find column indexes
        code_col = next((i for i, h in enumerate(headers) if "decimal" in h.lower()), None)
        if code_col is None:
            code_col = next((i for i, h in enumerate(headers) if h.upper() == "CODE"), 0)
        desc_col = next((i for i, h in enumerate(headers) if "description" in h.lower()), 2)
        hcc_col = next((i for i, h in enumerate(headers) if "hcc status" in h.lower()), None)

        records = []
        errors = 0
        for row in rows[1:]:
            try:
                code = str(row[code_col]).strip() if row[code_col] else None
                description = str(row[desc_col]).strip() if row[desc_col] else None
                hcc_status = str(row[hcc_col]).strip() if (hcc_col is not None and row[hcc_col]) else "Non-HCC"

                if not code or not description or code == "None":
                    errors += 1
                    continue

                records.append((code, description, hcc_status))
            except Exception as e:
                logging.error(f"Row parse error: {e} | row: {row}")
                errors += 1

        print(f"Parsed {len(records)} valid records, {errors} skipped rows.")
        return records

    def _execute_batch(self, cursor, records):
        """Perform batch insert."""
        try:
            cursor.executemany(
                "INSERT OR REPLACE INTO icd10_codes (code, description, hcc_status) VALUES (?, ?, ?)",
                records
            )
        except sqlite3.Error as e:
            logging.error(f"Batch insert error: {e}")

    def load(self, reset=False, limit=None):
        """Process ICD data file and load into SQLite.
        
        Uses Final_ICD_10_CM.xlsx if available, otherwise falls back to
        the plain-text file (database/seeds/data/icd10cm-codes-April-2025.txt).
        """
        xlsx_available = os.path.exists(self.data_path) and _OPENPYXL_AVAILABLE
        txt_available = os.path.exists(TXT_DATA_PATH)

        if not xlsx_available and not txt_available:
            print(f"Error: No ICD data file found.")
            print(f"  Tried xlsx: {self.data_path}")
            print(f"  Tried txt:  {TXT_DATA_PATH}")
            sys.exit(1)

        if not xlsx_available:
            print(f"xlsx not found — using txt fallback: {TXT_DATA_PATH}")

        conn = self.get_connection()
        cursor = conn.cursor()

        self.setup_database(cursor, reset)
        conn.commit()

        all_records = self.parse_excel() if xlsx_available else self.parse_txt()
        if limit:
            all_records = all_records[:limit]

        total = len(all_records)
        count_loaded = 0

        print(f"Starting load into {self.db_path}...")
        with tqdm.tqdm(total=total, desc="Loading ICD-10", unit="codes") as pbar:
            batch = []
            for record in all_records:
                batch.append(record)
                count_loaded += 1
                pbar.update(1)

                if len(batch) >= self.batch_size:
                    self._execute_batch(cursor, batch)
                    batch = []

                if count_loaded % self.commit_interval == 0:
                    conn.commit()

            if batch:
                self._execute_batch(cursor, batch)
                conn.commit()

        print(f"\n--- Load Summary ---")
        print(f"Status: Success")
        print(f"Total Loaded: {count_loaded}")
        print(f"Source: Final_ICD_10_CM.xlsx (FY2026)")

        self.verify_counts(cursor)
        conn.close()

    def verify_counts(self, cursor):
        """Quick verification of data."""
        cursor.execute("SELECT COUNT(*) FROM icd10_codes")
        total = cursor.fetchone()[0]
        print(f"Database Record Count: {total}")

        print("\nSample (First 5):")
        cursor.execute("SELECT code, description, hcc_status FROM icd10_codes ORDER BY id ASC LIMIT 5")
        for row in cursor.fetchall():
            print(f"  [{row[0]}] {row[1]} | HCC: {row[2]}")

        print("\nSample (Last 5):")
        cursor.execute("SELECT code, description, hcc_status FROM icd10_codes ORDER BY id DESC LIMIT 5")
        rows = cursor.fetchall()
        for row in reversed(rows):
            print(f"  [{row[0]}] {row[1]} | HCC: {row[2]}")

def main():
    parser = argparse.ArgumentParser(description="Medical ICD-10 SQLite Loader")
    parser.add_argument("--reset", action="store_true", help="Recreate table before loading")
    parser.add_argument("--verify-only", action="store_true", help="Just check existing counts")
    parser.add_argument("--limit", type=int, help="Stop after loading N records")
    parser.add_argument("--db", default=DB_PATH, help="SQLite DB path")
    parser.add_argument("--data", default=DATA_PATH, help="Source Excel file path")

    args = parser.parse_args()

    loader = ICD10Loader(args.db, args.data)

    if args.verify_only:
        conn = loader.get_connection()
        loader.verify_counts(conn.cursor())
        conn.close()
    else:
        loader.load(reset=args.reset, limit=args.limit)

if "__main__" == __name__:
    main()
