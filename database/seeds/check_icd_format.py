import os
import re

FILE_PATH = "database/seeds/data/icd10cm-codes-April-2025.txt"

def analyze_icd_file(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return

    records = []
    malformed = []
    total_lines = 0
    empty_lines = 0

    # ICD-10-CM Regex: Letter + Number + 1-5 Alphanumeric (with optional decimal)
    icd_pattern = re.compile(r"^[A-Z][0-9][A-Z0-9\.]{1,5}$")

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()
            
            if not line:
                empty_lines += 1
                continue

            # Split on first whitespace block
            match = re.match(r"^(\S+)\s+(.*)$", line)
            if not match:
                malformed.append((i, line, "Missing description or code"))
                continue

            code, description = match.groups()
            
            if not icd_pattern.match(code):
                malformed.append((i, line, "Invalid ICD-10 code format"))
                continue

            records.append({"code": code, "desc": description})

    # Stats
    decimal_codes = [r for r in records if "." in r["code"]]
    non_decimal_codes = [r for r in records if "." not in r["code"]]
    
    print("=" * 40)
    print("ICD-10 FILE ANALYSIS REPORT")
    print("=" * 40)
    print(f"File: {path}")
    print(f"File size: {os.path.getsize(path) / 1024:.2f} KB")
    
    print(f"\nPARSING RESULTS:")
    print(f"- Total lines: {total_lines}")
    print(f"- Valid records: {len(records)}")
    print(f"- Empty lines skipped: {empty_lines}")
    print(f"- Malformed lines: {len(malformed)}")

    print(f"\nCODE FORMAT BREAKDOWN:")
    print(f"- Codes without decimal: {len(non_decimal_codes)}")
    print(f"- Codes with decimal: {len(decimal_codes)}")

    print(f"\nSAMPLE RECORDS (first 10):")
    for r in records[:10]:
        print(f"Code: {r['code']:<7} | Description: {r['desc']}")

    if malformed:
        print(f"\nISSUES FOUND (first 5):")
        for line_no, content, reason in malformed[:5]:
            # Truncate content for cleaner output
            content_display = (content[:50] + '..') if len(content) > 50 else content
            print(f"Line {line_no:5}: '{content_display}' -> {reason}")
    
    print(f"\nREADY TO LOAD: {'Yes' if len(records) > 0 and not malformed else 'No'}")

if __name__ == "__main__":
    analyze_icd_file(FILE_PATH)
