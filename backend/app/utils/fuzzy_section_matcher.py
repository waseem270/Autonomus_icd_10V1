"""
Fuzzy Section Matcher
======================
Provides semantically-resilient matching for clinical section headers.

Instead of rigid exact-string comparisons, this module uses multiple
similarity signals to tolerate:
  - Typos ("Immunatization" → immunizations)
  - Plurals / suffixes ("Medication" vs "Medications")
  - Extra words ("Assessment Note" → assessment)
  - Combined headers ("Assessment and Plan" → assessment_and_plan)
  - Minor formatting ("Assessment:" vs "[Assessment]" vs "ASSESSMENT")

All matching uses Python stdlib only (difflib.SequenceMatcher) — no
external NLP dependencies are required.

Design:
    1.  Normalize: strip formatting, lowercase, remove stop-words
    2.  Token overlap: Jaccard on significant word tokens
    3.  Edit similarity: SequenceMatcher ratio on normalized strings
    4.  Root matching: crude suffix stripping for plural/gerund tolerance
    5.  Combined weighted score with configurable threshold
"""

import re
import logging
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Canonical section knowledge base
# ──────────────────────────────────────────────────────────────────────

# Maps canonical_name → set of known variants (lowercased, normalized).
# This is the SINGLE source of truth for section vocabulary.
CANONICAL_VARIANTS: Dict[str, List[str]] = {
    "chief_complaint": [
        "chief complaint", "cc", "complaint", "presenting problem",
        "presenting complaint", "reason for visit", "reason for encounter",
        "exam reason", "chief complaints",
    ],
    "history_present_illness": [
        "history of present illness", "hpi", "present illness",
        "history present illness", "subjective", "hx present illness",
        "hx of present illness", "interval history",
    ],
    "past_medical_history": [
        "past medical history", "pmh", "medical history",
        "active problems", "problem list", "active problem list",
        "past med hx", "past medical hx", "significant history",
        "significant past history",
    ],
    "past_surgical_history": [
        "past surgical history", "psh", "surgical history",
        "past surg hx", "prior surgeries", "previous surgeries",
    ],
    "medications": [
        "medications", "medication", "meds", "current medications",
        "active medications", "medication list", "rx", "current meds",
        "home medications", "outpatient medications", "med list",
        "medication reconciliation",
    ],
    "allergies": [
        "allergies", "allergy", "allergy list", "drug allergies",
        "allergic reactions", "medication allergies", "nkda",
        "known allergies",
    ],
    "social_history": [
        "social history", "sh", "social hx", "social",
    ],
    "family_history": [
        "family history", "fh", "fam hx", "family hx",
        "family medical history",
    ],
    "review_of_systems": [
        "review of systems", "ros", "systems review",
        "review of system", "general adult ros", "system review",
    ],
    "vitals": [
        "vitals", "vital signs", "vs", "vital sign",
    ],
    "physical_exam": [
        "physical exam", "physical examination", "pe", "exam",
        "examination", "objective exam", "phys exam",
    ],
    "objective": [
        "objective",
    ],
    "immunizations": [
        "immunizations", "immunization", "immunisation", "immunisations",
        "vaccines", "vaccination", "vaccinations", "immunization history",
        "vaccine history", "immunization record",
    ],
    "lab_results": [
        "lab results", "labs", "laboratory", "laboratory results",
        "test results", "lab values", "lab data", "diagnostic results",
        "lab result",
    ],
    "imaging": [
        "imaging", "imaging results", "radiology", "x-ray", "xray",
        "diagnostic imaging", "radiological findings",
    ],
    "assessment": [
        "assessment", "impression", "dx", "diagnoses", "diagnosis",
        "clinical impression", "assessment note",
    ],
    "plan": [
        "plan", "treatment plan", "tx", "treatment", "recommendations",
        "plan of care", "care plan", "management plan",
    ],
    "assessment_and_plan": [
        "assessment and plan", "assessment & plan", "a/p", "a&p",
        "assessment/plan", "assessment plan", "a and p",
        "impression and plan", "impression & plan",
        "impression and recommendations",
    ],
    "follow_up": [
        "follow up", "f/u", "followup", "follow-up", "return visit",
        "follow up plan", "disposition",
    ],
}

# Flat lookup: normalized_variant → canonical_name (built once at import)
_VARIANT_TO_CANONICAL: Dict[str, str] = {}
for _canonical, _variants in CANONICAL_VARIANTS.items():
    for _v in _variants:
        _VARIANT_TO_CANONICAL[_v.lower().strip()] = _canonical

# All canonical names for iteration
ALL_CANONICAL_NAMES = list(CANONICAL_VARIANTS.keys())

# Table column labels and non-section text that should NEVER be matched
# to a canonical section name (even though they may fuzzy-match variants
# like "diagnosis" → "assessment").
_NOT_SECTION_LABELS = frozenset({
    "diagnosis date", "date", "description",
    "status", "code", "comments", "onset date", "resolved date",
    "priority", "severity", "provider", "location", "type",
    "note", "value", "result", "range", "units",
    "reference range", "flag", "category", "name", "dose",
    "frequency", "route", "start date", "end date", "sig",
    "quantity", "refills", "pharmacy", "prescriber",
    "diagnosis code", "icd code", "icd 10", "cpt code",
    "procedure date",
    # Common non-section content that pattern matchers may pick up
    "patient educational handouts", "medical equipment",
    "advance directives", "functional status",
    "none recorded", "no information available",
    "unknown", "reminders", "orders", "dme orders",
})


# ──────────────────────────────────────────────────────────────────────
# Text normalization helpers
# ──────────────────────────────────────────────────────────────────────

# Words that carry no section-discriminating information
_STOP_WORDS = frozenset({
    "the", "a", "an", "of", "and", "&", "or", "in", "for", "to",
    "is", "are", "was", "were", "on", "at", "by", "with", "from",
    "note", "notes", "section", "report", "current", "active",
    "patient", "clinical", "general",
})

# Common medical suffixes for crude stemming
_SUFFIXES = [
    "ations", "ation", "tions", "tion", "sions", "sion",
    "ments", "ment", "ings", "ing", "ness",
    "ous", "ive", "al", "ly",
]

# Separate plural/ending rules (applied after main suffixes fail)
_PLURAL_RULES = [
    # "allergies" → "allerg" (y→ies plurals)
    ("ies", 3, "y"),   # remove "ies", require 3+ stem chars, treat as if ended in "y"
    ("es", 3, None),
    ("s", 3, None),
    ("ed", 3, None),
    ("y", 4, None),   # "allergy" → "allerg" (only if stem ≥4 chars)
]


def normalize_header(text: str) -> str:
    """
    Normalize a header string for comparison:
    - lowercase
    - strip brackets, colons, asterisks, dashes, underscores
    - collapse whitespace
    """
    t = text.lower().strip()
    # Remove brackets, colons, asterisks, equals, underscores, dashes (decorators)
    t = re.sub(r"[\[\]:*=_\-–—]+", " ", t)
    # Remove trailing/leading parenthetical abbreviations like "(HPI)"
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t)
    t = re.sub(r"^\s*\([^)]*\)\s*", "", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize(text: str) -> List[str]:
    """Split normalized text into meaningful tokens (no stop-words)."""
    words = normalize_header(text).split()
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def crude_stem(word: str) -> str:
    """
    Crude suffix stripping for English medical terms.
    NOT a real stemmer — just enough to equate plurals / gerunds.
    """
    w = word.lower()

    # Main suffixes (longest first)
    for suffix in _SUFFIXES:
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            return w[: -len(suffix)]

    # Plural / ending rules
    for ending, min_stem, replacement in _PLURAL_RULES:
        if w.endswith(ending) and len(w) - len(ending) >= min_stem:
            stem = w[: -len(ending)]
            return stem

    return w


def stemmed_tokens(text: str) -> List[str]:
    """Tokenize, then stem each token."""
    return [crude_stem(t) for t in tokenize(text)]


# ──────────────────────────────────────────────────────────────────────
# Similarity functions
# ──────────────────────────────────────────────────────────────────────

def edit_similarity(a: str, b: str) -> float:
    """
    Character-level edit similarity using SequenceMatcher.
    Returns 0.0–1.0 (1.0 = identical).
    """
    na = normalize_header(a)
    nb = normalize_header(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def token_overlap(a: str, b: str) -> float:
    """
    Jaccard similarity on stemmed token sets.
    Returns 0.0–1.0 (1.0 = identical token sets).
    """
    ta = set(stemmed_tokens(a))
    tb = set(stemmed_tokens(b))
    if not ta or not tb:
        return 0.0
    intersection = ta & tb
    union = ta | tb
    return len(intersection) / len(union)


def combined_similarity(a: str, b: str) -> float:
    """
    Multi-strategy similarity combining edit distance and token overlap.
    Returns the BEST score across strategies, ensuring that single-word
    typos (high edit_sim, low token_overlap) score well.
    """
    edit = edit_similarity(a, b)
    tokens = token_overlap(a, b)

    na = normalize_header(a)
    nb = normalize_header(b)

    # For very short strings (abbreviations ≤3 chars), require near-exact match
    if len(na) <= 3 or len(nb) <= 3:
        return edit

    # Weight: edit similarity matters more for short strings (typos),
    # token overlap matters more for longer strings (extra words)
    avg_len = (len(na.split()) + len(nb.split())) / 2
    if avg_len <= 2:
        # Short headers: edit distance dominates
        weighted = 0.65 * edit + 0.35 * tokens
    else:
        # Longer headers: token overlap dominates
        weighted = 0.45 * edit + 0.55 * tokens

    # KEY: return the best of pure edit_sim and the weighted score.
    # This ensures single-word typos (where token_overlap=0 because
    # stems don't match) still get the full edit_similarity score.
    return max(edit, weighted)


# ──────────────────────────────────────────────────────────────────────
# Canonical matching (header text → canonical section name)
# ──────────────────────────────────────────────────────────────────────

def match_canonical(
    header: str,
    threshold: float = 0.62,
) -> Tuple[Optional[str], float]:
    """
    Match a raw header string to a canonical section name.

    Matching strategy (fast → slow, stops at first hit):
      1. Exact lookup in variant table
      2. Normalized exact lookup
      3. Fuzzy match against all known variants

    Returns
    -------
    (canonical_name, confidence)
        canonical_name is None if no match above threshold.
    """
    if not header or not header.strip():
        return None, 0.0

    norm = normalize_header(header)

    # Short inputs that aren't known abbreviations: reject immediately.
    # This prevents random 2–3 char strings from fuzzy-matching.
    if len(norm) <= 3 and norm not in _VARIANT_TO_CANONICAL:
        return None, 0.0

    # ── 0. Reject known non-section labels (column headers, etc.) ─
    # These must be checked BEFORE exact lookup because "diagnosis"
    # is a valid variant of "assessment" but "diagnosis date" is not.
    if norm in _NOT_SECTION_LABELS:
        return None, 0.0

    # ── 1. Exact lookup ──────────────────────────────────────────
    exact = _VARIANT_TO_CANONICAL.get(norm)
    if exact:
        return exact, 1.0

    # ── 2. Normalized-token exact lookup ─────────────────────────
    # Try stripping stop-words and re-matching
    stripped = " ".join(tokenize(header))
    exact2 = _VARIANT_TO_CANONICAL.get(stripped)
    if exact2:
        return exact2, 0.98

    # ── 3. Fuzzy match against all variants ──────────────────────
    best_canonical: Optional[str] = None
    best_score: float = 0.0

    for canonical_name, variants in CANONICAL_VARIANTS.items():
        for variant in variants:
            score = combined_similarity(norm, variant)
            if score > best_score:
                best_score = score
                best_canonical = canonical_name

    if best_score >= threshold and best_canonical:
        logger.debug(
            f"Fuzzy canonical match: '{header}' → {best_canonical} "
            f"(score={best_score:.3f})"
        )
        return best_canonical, best_score

    # ── 4. Last-ditch: check if any canonical variant is a SUBSTRING
    #       of the header or vice versa (handles "Assessment Note" etc.)
    #       Require minimum 6 chars to avoid false positives like
    #       2-char strings matching inside longer words.
    #       Also require the variant to cover at least 75% of the header
    #       length to prevent "diagnosis" matching "diagnosis date".
    for canonical_name, variants in CANONICAL_VARIANTS.items():
        for variant in variants:
            # Variant is substring of header — require high coverage
            if len(variant) >= 6 and variant in norm:
                coverage = len(variant) / len(norm) if norm else 0
                if coverage >= 0.75:
                    return canonical_name, 0.85
            # Header is substring of variant
            if len(norm) >= 6 and norm in variant:
                coverage = len(norm) / len(variant) if variant else 0
                if coverage >= 0.75:
                    return canonical_name, 0.80

    return None, best_score


def match_canonical_batch(
    headers: List[str],
    threshold: float = 0.62,
) -> Dict[str, str]:
    """
    Map a list of raw header strings to canonical names.
    Returns {raw_header_lower: canonical_name} for matches above threshold.
    """
    result: Dict[str, str] = {}
    for header in headers:
        canonical, score = match_canonical(header, threshold)
        if canonical:
            key = header.lower().strip().strip("[]").rstrip(":").strip()
            result[key] = canonical
    return result


# ──────────────────────────────────────────────────────────────────────
# Fuzzy text search (find header position in document)
# ──────────────────────────────────────────────────────────────────────

def _extract_candidate_lines(
    text: str,
    start: int = 0,
    max_lines: int = 500,
) -> List[Tuple[int, str]]:
    """
    Extract lines from the text that COULD be section headers.
    Returns list of (char_position, line_text) tuples.

    A line is considered a potential header if it:
    - Is relatively short (< 80 chars after stripping)
    - Contains at least one letter
    - Is not purely numeric / date / whitespace
    """
    results: List[Tuple[int, str]] = []
    pos = start
    count = 0

    for line in text[start:].split("\n"):
        stripped = line.strip()
        if (
            stripped
            and len(stripped) < 80
            and any(c.isalpha() for c in stripped)
            and not re.fullmatch(r"[\d/\-., :;()]+", stripped)
        ):
            results.append((pos, stripped))
            count += 1
            if count >= max_lines:
                break
        pos += len(line) + 1  # +1 for the \n

    return results


# Common medical abbreviations used in section headers.
# Used to expand abbreviations before fuzzy text comparison.
_MEDICAL_ABBREVS: Dict[str, str] = {
    "hx": "history",
    "dx": "diagnosis",
    "tx": "treatment",
    "rx": "prescription",
    "sx": "symptoms",
    "fx": "fracture",
    "cx": "culture",
    "bx": "biopsy",
    "pt": "patient",
    "htn": "hypertension",
    "eval": "evaluation",
    "exam": "examination",
    "med": "medical",
    "surg": "surgical",
    "fam": "family",
    "prev": "previous",
    "fhx": "family history",
}


def _expand_abbreviations(text: str) -> str:
    """Expand common medical abbreviations in header text for comparison."""
    words = text.lower().split()
    expanded = []
    for w in words:
        expanded.append(_MEDICAL_ABBREVS.get(w, w))
    return " ".join(expanded)


def fuzzy_find_in_text(
    header: str,
    text: str,
    start: int = 0,
    threshold: float = 0.65,
) -> int:
    """
    Find the position of a header in the document text using fuzzy matching.

    This is a FALLBACK — callers should attempt exact/case-insensitive
    matching first (which is fast). This function is invoked only when
    rigid matching fails.

    Strategy:
      1. Extract all short lines (potential headers) from the text.
      2. Score each against the target header using combined_similarity.
      3. Also try abbreviation-expanded form for better coverage.
      4. Return the position of the best match above threshold.

    Returns -1 if no match found.
    """
    target_norm = normalize_header(header)
    if not target_norm:
        return -1

    # Also prepare abbreviation-expanded form
    target_expanded = _expand_abbreviations(target_norm)

    candidate_lines = _extract_candidate_lines(text, start)
    if not candidate_lines:
        return -1

    best_pos = -1
    best_score = 0.0

    for line_pos, line_text in candidate_lines:
        line_norm = normalize_header(line_text)
        if not line_norm:
            continue

        # Also expand abbreviations in the line
        line_expanded = _expand_abbreviations(line_norm)

        # For very short targets (abbreviations like "CC", "PE"),
        # check if the abbreviation appears as a word in the line
        if len(target_norm) <= 3:
            line_words = set(line_norm.split())
            if target_norm in line_words:
                return line_pos
            continue  # Don't fuzzy match abbreviations against long lines

        # Score: take best of direct comparison and abbreviation-expanded
        score = combined_similarity(target_norm, line_norm)

        # Abbreviation-expanded comparison (e.g., "Social History" vs "Social Hx")
        if target_expanded != target_norm or line_expanded != line_norm:
            exp_score = combined_similarity(target_expanded, line_expanded)
            score = max(score, exp_score)

        # Bonus: if the target header appears as a prefix of the line,
        # boost score (handles "Assessment" matching "Assessment Note:")
        if line_norm.startswith(target_norm) or target_norm.startswith(line_norm):
            score = max(score, 0.88)

        if score > best_score:
            best_score = score
            best_pos = line_pos

    if best_score >= threshold:
        logger.debug(
            f"Fuzzy text match: '{header}' found at pos {best_pos} "
            f"(score={best_score:.3f})"
        )
        return best_pos

    return -1


def fuzzy_find_in_text_with_header(
    header: str,
    text: str,
    start: int = 0,
    threshold: float = 0.65,
) -> Tuple[int, Optional[str]]:
    """
    Like fuzzy_find_in_text but also returns the actual matched line text.
    Useful for getting the real header text from the document (with typos).
    
    Returns (position, matched_line_text) or (-1, None).
    """
    target_norm = normalize_header(header)
    if not target_norm:
        return -1, None

    target_expanded = _expand_abbreviations(target_norm)

    candidate_lines = _extract_candidate_lines(text, start)
    if not candidate_lines:
        return -1, None

    best_pos = -1
    best_score = 0.0
    best_line: Optional[str] = None

    for line_pos, line_text in candidate_lines:
        line_norm = normalize_header(line_text)
        if not line_norm:
            continue

        line_expanded = _expand_abbreviations(line_norm)

        if len(target_norm) <= 3:
            line_words = set(line_norm.split())
            if target_norm in line_words:
                return line_pos, line_text
            continue

        score = combined_similarity(target_norm, line_norm)

        if target_expanded != target_norm or line_expanded != line_norm:
            exp_score = combined_similarity(target_expanded, line_expanded)
            score = max(score, exp_score)

        if line_norm.startswith(target_norm) or target_norm.startswith(line_norm):
            score = max(score, 0.88)

        if score > best_score:
            best_score = score
            best_pos = line_pos
            best_line = line_text

    if best_score >= threshold:
        logger.debug(
            f"Fuzzy text match: '{header}' → '{best_line}' at pos {best_pos} "
            f"(score={best_score:.3f})"
        )
        return best_pos, best_line

    return -1, None
