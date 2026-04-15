"""
Deterministic Validator — rule-based ICD-10 code validation.

Validates ICD codes against:
  1. Format regex (letter + 2 digits + optional dot + up to 4 alphanum)
  2. Known ICD-10-CM code dictionary in the database
  3. Cross-validation: disease name must plausibly match the ICD description

This layer prevents LLM-hallucinated or malformed ICD codes from reaching output.
"""

import logging
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from fuzzywuzzy import fuzz

from ..core.config import settings

logger = logging.getLogger(__name__)

# ICD-10-CM code format: A00–Z99 with optional dot and subcategory (up to 7 chars total + dot)
_ICD_FORMAT_RE = re.compile(
    r'^[A-Z]\d{2}(?:\.[A-Z0-9]{1,5})?$',
    re.IGNORECASE,
)


class DeterministicValidator:
    """Validate ICD codes deterministically against format rules and the code dictionary."""

    def __init__(self):
        self._db_path = self._find_icd_db()
        self._code_cache: Optional[Dict[str, str]] = None

    @staticmethod
    def _find_icd_db() -> Optional[str]:
        """Locate the ICD SQLite database file."""
        base = Path(settings.DATABASE_URL.replace("sqlite:///", "")).parent
        # Try common locations
        for candidate in [
            base / "medical_icd.db",
            base / "icd_codes.db",
            Path(settings.UPLOAD_FOLDER).parent / "database" / "medical_icd.db",
        ]:
            if candidate.exists():
                return str(candidate)
        # Fall back to the configured DB URL itself
        db_file = settings.DATABASE_URL.replace("sqlite:///", "")
        if Path(db_file).exists():
            return db_file
        return None

    def _load_code_cache(self) -> Dict[str, str]:
        """Load ICD codes into memory cache from the database."""
        if self._code_cache is not None:
            return self._code_cache

        self._code_cache = {}
        if not self._db_path:
            logger.warning("No ICD database found for deterministic validation.")
            return self._code_cache

        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            # Try to read from the ICD codes table
            for table in ("icd_codes", "icd10_codes", "icd_10_codes"):
                try:
                    cursor.execute(f"SELECT code, description FROM {table}")  # noqa: S608
                    for code, desc in cursor.fetchall():
                        self._code_cache[code.upper().strip()] = (desc or "").strip()
                    if self._code_cache:
                        break
                except sqlite3.OperationalError:
                    continue
            conn.close()
            logger.info(f"Loaded {len(self._code_cache)} ICD codes for validation.")
        except Exception as e:
            logger.error(f"Failed to load ICD code cache: {e}")
        return self._code_cache

    def validate_icd_code(self, code: str) -> Dict:
        """
        Validate a single ICD-10 code.

        Returns:
            {"valid": bool, "reason": str, "normalized_code": str}
        """
        if not code or code in ("—", "-", ""):
            return {"valid": False, "reason": "Empty or placeholder code", "normalized_code": ""}

        normalized = code.upper().strip()

        # Step 1: Format validation
        if not _ICD_FORMAT_RE.match(normalized):
            return {"valid": False, "reason": f"Invalid format: '{code}'", "normalized_code": normalized}

        # Step 2: Dictionary lookup (if cache is available)
        cache = self._load_code_cache()
        if cache:
            # Try exact match
            if normalized in cache:
                return {"valid": True, "reason": "Exact match in ICD dictionary", "normalized_code": normalized}
            # Try without dot
            no_dot = normalized.replace(".", "")
            for cached_code in cache:
                if cached_code.replace(".", "") == no_dot:
                    return {"valid": True, "reason": "Match after dot normalization", "normalized_code": cached_code}
            # Try 3-char category code
            category = normalized[:3]
            if any(c.startswith(category) for c in cache):
                return {"valid": True, "reason": f"Valid category {category}", "normalized_code": normalized}
            return {"valid": False, "reason": f"Code {normalized} not in ICD dictionary", "normalized_code": normalized}

        # No dictionary available — format-only validation (permissive)
        return {"valid": True, "reason": "Format valid (no dictionary check)", "normalized_code": normalized}

    def cross_validate_disease_icd(self, disease_name: str, icd_code: str) -> Dict:
        """
        Check if the ICD code description plausibly matches the disease name.
        Uses fuzzy matching to detect gross mismatches (hallucinated codes).
        """
        cache = self._load_code_cache()
        if not cache:
            return {"match": True, "score": 0.0, "reason": "No dictionary for cross-validation"}

        normalized = icd_code.upper().strip()
        description = cache.get(normalized, "")
        if not description:
            # Try without dot
            no_dot = normalized.replace(".", "")
            for c, d in cache.items():
                if c.replace(".", "") == no_dot:
                    description = d
                    break

        if not description:
            return {"match": False, "score": 0.0, "reason": "Code not in dictionary"}

        # Fuzzy match disease name against ICD description
        score = fuzz.token_set_ratio(disease_name.lower(), description.lower())
        threshold = 40  # Lenient — ICD descriptions often differ from clinical names
        matched = score >= threshold
        return {
            "match": matched,
            "score": score / 100.0,
            "reason": f"Fuzzy match {score}% ({'pass' if matched else 'fail'})",
            "icd_description": description,
        }


# Singleton
deterministic_validator = DeterministicValidator()
