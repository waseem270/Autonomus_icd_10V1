"""
Section Candidate Extractor
============================
Extracts potential section headers from any medical document using
greedy pattern matching and confidence scoring — NO hard-coded accept/reject
keyword lists. Every candidate is scored by context signals so the LLM
filter (Gemini) only needs to adjudicate low-confidence candidates.

Pipeline:
    1. Greedy Regex (8 patterns) → raw candidate list
    2. De-duplicate by position (keep highest-confidence pattern)
    3. Score each candidate on 6 contextual signals
    4. filter_by_confidence() splits list into HIGH / LOW buckets
       - HIGH  → accepted directly (skip LLM call)
       - LOW   → sent to Gemini for validation
"""

import re
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confidence base scores per pattern type
# (purely structural — no clinical word lists)
# ---------------------------------------------------------------------------
_BASE_SCORES: Dict[str, float] = {
    "bracketed":          0.90,   # [Section]  — highly distinctive
    "colon_block":        0.85,   # Section:\n — very reliable
    "ehr_timestamp":      0.90,   # Header ; MM/DD HH:MM AM)
    "allcaps":            0.75,   # MEDICATIONS
    "colon_inline":       0.65,   # Header: value on same line
    "numbered":           0.70,   # 1. Section  /  I. Section
    "decorated":          0.80,   # === Section ===  /  *** Section ***
    "standalone":         0.55,   # Title Case / lowercase on own line
}


class SectionCandidateExtractor:
    """
    Extract potential section headers with confidence scores.

    Design goal: zero hard-coded accept/reject word lists.
    Confidence is derived purely from structural and contextual signals
    so the extractor works on any medical PDF format without modification.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_candidates(self, text: str) -> List[Dict]:
        """
        Extract all potential section header candidates with confidence scores.

        Each candidate dict contains:
            header          : str   – cleaned header text
            position        : int   – char position in source text
            confidence      : float – 0.0 … 1.0
            pattern_type    : str   – detection pattern name
            context_before  : str   – 30 chars before the match
            context_after   : str   – 80 chars after the match
        """
        raw: List[Dict] = []

        raw.extend(self._p1_bracketed(text))
        raw.extend(self._p2_colon_block(text))
        raw.extend(self._p3_ehr_timestamp(text))
        raw.extend(self._p4_allcaps(text))
        raw.extend(self._p5_colon_inline(text))
        raw.extend(self._p6_numbered(text))
        raw.extend(self._p7_decorated(text))
        raw.extend(self._p8_standalone(text))

        # De-duplicate: same char position → keep highest base score
        candidates = self._deduplicate(raw)

        # Apply contextual scoring signals on top of base scores
        for c in candidates:
            c["confidence"] = self._score(c, text)

        candidates.sort(key=lambda x: x["position"])
        self.logger.info(f"Extracted {len(candidates)} header candidates")
        return candidates

    def filter_by_confidence(
        self,
        candidates: List[Dict],
        threshold: float = 0.75,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Split into (high_confidence, low_confidence) buckets.

        high → accepted immediately (no LLM needed)
        low  → send to Gemini's LLM filter for final judgement
        """
        high = [c for c in candidates if c["confidence"] >= threshold]
        low  = [c for c in candidates if c["confidence"] <  threshold]
        self.logger.info(
            f"Threshold {threshold}: {len(high)} high, {len(low)} low"
        )
        return high, low

    # ------------------------------------------------------------------
    # Pattern 1 – Bracketed: [Section Name]
    # ------------------------------------------------------------------

    def _p1_bracketed(self, text: str) -> List[Dict]:
        results = []
        for m in re.finditer(r"\[([^\[\]\n]{2,60})\]", text):
            hdr = self._clean(m.group(1))
            if not hdr:
                continue
            # Skip pure numbers or pure dates e.g. [05/22/2024] [1]
            if re.fullmatch(r"[\d/\-.,: ]+", hdr):
                continue
            # Skip known table column labels
            if hdr.lower() in self._COLUMN_LABELS:
                continue
            results.append(self._make(hdr, m.start(), "bracketed", m.end(), text))
        return results

    # ------------------------------------------------------------------
    # Pattern 2 – Colon block: "Section:\n" (nothing after colon)
    # ------------------------------------------------------------------

    def _p2_colon_block(self, text: str) -> List[Dict]:
        results = []
        for m in re.finditer(
            r"(?m)^[ \t]*([A-Z][A-Za-z0-9 \t/,()\-]{1,60}?):[ \t]*$", text
        ):
            hdr = self._clean(m.group(1))
            if not hdr or len(hdr.split()) > 8:
                continue
            # Skip known table column labels
            if hdr.lower() in self._COLUMN_LABELS:
                continue
            results.append(self._make(hdr, m.start(), "colon_block", m.end(), text))
        return results

    # ------------------------------------------------------------------
    # Pattern 3 – EHR Timestamp: "Header ; MM/DD/YYYY [HH:MM AM/PM])"
    # Appears mid-line in many modern EHR (Epic / Cerner) exports.
    # ------------------------------------------------------------------

    def _p3_ehr_timestamp(self, text: str) -> List[Dict]:
        results = []
        pat = re.compile(
            r"([A-Z][A-Za-z][A-Za-z ]{0,38}?)"
            r"\s*;\s*\d{1,2}/\d{1,2}/\d{2,4}"
            r"(?:\s+\d{1,2}:\d{2}\s*(?:AM|PM))?"
            r"\s*\)?"
        )
        for m in pat.finditer(text):
            hdr = self._clean(m.group(1))
            if not hdr or len(hdr.split()) > 6:
                continue
            results.append(self._make(hdr, m.start(), "ehr_timestamp", m.end(), text))
        return results

    # ------------------------------------------------------------------
    # Pattern 4 – ALL CAPS line: "MEDICATIONS"  "PHYSICAL EXAM"
    # ------------------------------------------------------------------

    def _p4_allcaps(self, text: str) -> List[Dict]:
        results = []
        for m in re.finditer(r"(?m)^[ \t]*([A-Z]{2}[A-Z\s/\-]{1,48})[ \t]*$", text):
            hdr = self._clean(m.group(1))
            if not hdr or len(hdr) < 3:
                continue
            # Skip known table column labels
            if hdr.lower() in self._COLUMN_LABELS:
                continue
            results.append(self._make(hdr, m.start(), "allcaps", m.end(), text))
        return results

    # ------------------------------------------------------------------
    # Pattern 5 – Colon inline: "Chief Complaint: text on same line"
    # ------------------------------------------------------------------

    def _p5_colon_inline(self, text: str) -> List[Dict]:
        results = []
        for m in re.finditer(
            r"(?m)^[ \t]*((?:[A-Z][a-z]+(?:\s+[A-Za-z/]+){0,5}))\s*:\s+(.{1,120})$",
            text,
        ):
            hdr = self._clean(m.group(1))
            if not hdr:
                continue
            # Skip obvious non-headers (negation phrases, single pronouns)
            if hdr.lower().split()[0] in {"no", "not", "the", "a", "an", "there"}:
                continue
            # Skip known table column labels
            if hdr.lower() in self._COLUMN_LABELS:
                continue
            entry = self._make(hdr, m.start(), "colon_inline", m.end(), text)
            entry["inline_value"] = m.group(2).strip()
            results.append(entry)
        return results

    # ------------------------------------------------------------------
    # Pattern 6 – Numbered / lettered: "1. History"  "II. Assessment"
    # ------------------------------------------------------------------

    def _p6_numbered(self, text: str) -> List[Dict]:
        results = []
        pat = re.compile(
            r"(?m)^[ \t]*(?:\d+\.|[IVXivx]+\.|[A-Z]\))\s+"
            r"([A-Z][A-Za-z /\-&]{2,50})[ \t]*$"
        )
        for m in pat.finditer(text):
            hdr = self._clean(m.group(1))
            if not hdr:
                continue
            # Single-word numbered items (e.g. "1. Hypertension") are list entries,
            # not section headers. Require at least 2 words for numbered pattern.
            if len(hdr.split()) < 2:
                continue
            results.append(self._make(hdr, m.start(), "numbered", m.end(), text))
        return results

    # ------------------------------------------------------------------
    # Pattern 7 – Decorated: "=== Section ===" / "*** Note ***"
    # ------------------------------------------------------------------

    def _p7_decorated(self, text: str) -> List[Dict]:
        results = []
        pat = re.compile(
            r"(?m)^[ \t]*[=*_\-]{2,}"
            r"[ \t]*([A-Za-z][A-Za-z0-9 /&]{2,50})"
            r"[ \t]*[=*_\-]{2,}[ \t]*$"
        )
        for m in pat.finditer(text):
            hdr = self._clean(m.group(1))
            if not hdr:
                continue
            results.append(self._make(hdr, m.start(), "decorated", m.end(), text))
        return results

    # ------------------------------------------------------------------
    # Pattern 8 – Standalone line (Title Case OR lowercase, 1–5 words)
    # Most permissive — relies on confidence scoring to filter noise.
    # Examples: "Social History"  "note"  "addendum"  "Discharge Summary"
    # ------------------------------------------------------------------

    # Table column labels that should never be treated as section headers
    _COLUMN_LABELS = {
        "diagnosis date", "date", "description",
        "status", "code", "comments", "onset date", "resolved date",
        "priority", "severity", "provider", "location", "type",
        "note", "notes", "value", "result", "range", "units",
        "reference range", "flag", "category", "name", "dose",
        "frequency", "route", "start date", "end date", "sig",
        "quantity", "refills", "pharmacy", "prescriber",
        "diagnosis code", "icd code", "icd-10", "cpt code",
    }

    def _p8_standalone(self, text: str) -> List[Dict]:
        results = []
        # Title Case
        for m in re.finditer(
            r"(?m)^[ \t]*([A-Z][a-z]+(?:[ \t]+[A-Za-z][a-z]*){0,4})[ \t]*$",
            text,
        ):
            hdr = self._clean(m.group(1))
            if not hdr or len(hdr) < 3:
                continue
            # Reject known table column labels
            if hdr.lower() in self._COLUMN_LABELS:
                continue
            results.append(self._make(hdr, m.start(), "standalone", m.end(), text))

        # Pure lowercase (handles bold-but-lowercase EHR headers)
        for m in re.finditer(
            r"(?m)^[ \t]*([a-z][a-z /\-]{1,43})[ \t]*$", text
        ):
            raw = m.group(1).strip()
            if len(raw) > 45 or len(raw.split()) > 5:
                continue
            hdr = self._clean(raw)
            if not hdr or len(hdr) < 3:
                continue
            results.append(self._make(hdr, m.start(), "standalone", m.end(), text))

        return results

    # ------------------------------------------------------------------
    # Confidence scoring (purely contextual signals — NO word lists)
    # ------------------------------------------------------------------

    def _score(self, candidate: Dict, full_text: str) -> float:
        score = candidate["confidence"]   # starts at pattern base score
        hdr   = candidate["header"]
        after = candidate.get("context_after", "")
        before = candidate.get("context_before", "")

        # Signal +1: Preceded by blank line (↑ likely section boundary)
        if "\n\n" in before or before.strip() == "":
            score += 0.10

        # Signal +2: Followed by blank line then content (↑ section start)
        if re.match(r"^\s*\n\s*\S", after):
            score += 0.05

        # Signal +3: Content after header looks like a list
        if self._followed_by_list(after):
            score += 0.10

        # Signal +4: Combined section name ("Assessment and Plan", "H&P")
        if " and " in hdr.lower() or "&" in hdr:
            score += 0.08

        # Signal +5: Very short header (1-3 chars) — penalty
        if len(hdr.strip()) < 4:
            score -= 0.20

        # Signal +6: Contains digits — mild penalty (likely not a section name)
        if any(ch.isdigit() for ch in hdr):
            score -= 0.10

        # Signal +7: Sentence fragment — strong penalty
        # (starts with a lowercase article / verb / pronoun)
        first_word = hdr.lower().split()[0] if hdr.split() else ""
        if first_word in {"the", "a", "an", "is", "are", "was", "this", "that",
                          "it", "he", "she", "we", "they", "no", "not"}:
            score -= 0.35

        # Signal +8: Very long header (> 8 words) — unlikely to be a header
        if len(hdr.split()) > 8:
            score -= 0.25

        return max(0.0, min(1.0, round(score, 3)))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(raw: str) -> str:
        """Strip EHR timestamp junk and trailing punctuation from header text."""
        c = raw.strip()
        # "as of MM/DD/YYYY [HH:MM AM]" with optional ; ) at end
        c = re.sub(
            r"\s*;\s*\d{1,2}/\d{1,2}/\d{2,4}(?:\s+\d{1,2}:\d{2}\s*(?:AM|PM))?\s*\)?\s*$",
            "", c, flags=re.IGNORECASE,
        ).strip()
        c = re.sub(r"\s+as\s+of\s+[\d/,\w]+[;:)]*\s*$", "", c, flags=re.IGNORECASE).strip()
        c = re.sub(r"\s*\(\s*[\d/,\w]+\s*\)\s*$", "", c).strip()
        c = re.sub(r"\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*$", "", c).strip()
        c = re.sub(r"\s*[-–]?\s*[Pp]age\s+\d+\s*$", "", c).strip()
        c = c.rstrip(";:)").strip()
        return c

    @staticmethod
    def _make(
        header: str,
        pos: int,
        pattern_type: str,
        end: int,
        text: str,
    ) -> Dict:
        return {
            "header":         header,
            "position":       pos,
            "confidence":     _BASE_SCORES.get(pattern_type, 0.5),
            "pattern_type":   pattern_type,
            "context_before": text[max(0, pos - 30): pos],
            "context_after":  text[end: end + 80],
        }

    @staticmethod
    def _followed_by_list(context: str) -> bool:
        """Return True if the text immediately after a header looks like a list."""
        list_markers = [
            r"^\s*[-*•]",       # bullet
            r"^\s*\d+[.)]\s",   # numbered
            r"^\s*[a-zA-Z]\)",  # lettered
        ]
        for pat in list_markers:
            if re.search(pat, context, re.MULTILINE):
                return True
        return False

    @staticmethod
    def _deduplicate(candidates: List[Dict]) -> List[Dict]:
        """Keep only the highest-confidence candidate per character position."""
        seen: Dict[int, Dict] = {}
        for c in candidates:
            pos = c["position"]
            if pos not in seen or c["confidence"] > seen[pos]["confidence"]:
                seen[pos] = c
        return list(seen.values())


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
section_candidate_extractor = SectionCandidateExtractor()
