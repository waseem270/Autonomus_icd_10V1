"""
Regex-based disease candidate extractor.

Replaces scispaCy NER with deterministic regex patterns that identify
disease candidates from clinical text.  The LLM (Gemini) is the PRIMARY
disease detector; this module provides a fast BACKUP / seed list so
that diseases still surface even when the LLM is unavailable or
rate-limited.

Strategy
--------
1. **Structural patterns** — numbered lists, bullet lists, PMH entries
2. **Clinical prefix/suffix patterns** — "history of …", "diagnosis: …",
   "… disease", "… disorder", "… syndrome", etc.
3. **Keyword context patterns** — terms near diagnostic language
4. **PMH structured entry parsing** — "• Disease Name Active MM/DD/YYYY"
"""

import logging
import re
from typing import Dict, List, Optional, Any, Set

logger = logging.getLogger(__name__)

# ── Sections where entries should NOT be treated as diseases ──────────
_NON_DISEASE_SECTIONS: Set[str] = {
    "medications", "allergies", "immunizations", "vitals",
    "social_history", "family_history", "imaging",
    "lab_results", "past_surgical_history",
}

# ── Known medication names — reject if an extracted term IS a drug ────
_MEDICATION_NAMES: Set[str] = {
    "metformin", "insulin", "glipizide", "sitagliptin", "jardiance",
    "ozempic", "lisinopril", "amlodipine", "losartan", "metoprolol",
    "atenolol", "hydrochlorothiazide", "atorvastatin", "rosuvastatin",
    "simvastatin", "pravastatin", "albuterol", "fluticasone",
    "montelukast", "salmeterol", "budesonide", "sertraline",
    "escitalopram", "fluoxetine", "bupropion", "venlafaxine",
    "buspirone", "lorazepam", "alprazolam", "citalopram",
    "levothyroxine", "synthroid", "omeprazole", "pantoprazole",
    "esomeprazole", "ranitidine", "famotidine", "tiotropium",
    "ipratropium", "furosemide", "carvedilol", "digoxin",
    "spironolactone", "warfarin", "aspirin", "acetaminophen",
    "ibuprofen", "naproxen", "prednisone", "prednisolone",
    "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline",
    "gabapentin", "pregabalin", "tramadol", "oxycodone",
    "hydrocodone", "morphine", "fentanyl", "duloxetine",
    "trazodone", "quetiapine", "olanzapine", "risperidone",
    "aripiprazole", "lithium", "valproic", "carbamazepine",
    "phenytoin", "levetiracetam", "lamotrigine", "topiramate",
    "clopidogrel", "rivaroxaban", "apixaban", "enoxaparin",
    "heparin", "anastrozole", "tamoxifen", "letrozole",
    "alendronate", "risedronate", "cholecalciferol",
    "ergocalciferol", "multivitamin", "tylenol", "seroquel",
    "norvasc", "lipitor",
}

# ── Garbage / admin words that should never be a disease ─────────────
_SINGLE_WORD_REJECT: Set[str] = {
    "patient", "visit", "follow", "followup", "stable", "normal",
    "negative", "positive", "history", "review", "current", "prior",
    "date", "none", "unknown", "monitor", "refilled", "continue",
    "discussed", "counseling", "examination", "screening", "vitamin",
    "blood", "pressure", "level", "value", "lab", "test",
    "medication", "prescription", "diet", "exercise", "weight",
    "height", "pulse", "temperature", "alert", "oriented",
    "assessment", "plan", "impression", "return", "appointment",
    "referral", "consult", "encounter", "note",
}

# ── Medical suffixes that indicate a disease/condition name ──────────
_DISEASE_SUFFIXES = re.compile(
    r'(?:disease|disorder|syndrome|deficiency|failure|insufficiency|'
    r'infection|inflammation|neoplasm|carcinoma|tumor|malignancy|'
    r'stenosis|occlusion|obstruction|fibrosis|sclerosis|neuropathy|'
    r'myopathy|arthritis|itis|emia|osis|opathy|penia|algia|uria|'
    r'plegia|paresis|ectasia|trophy|philia|cytosis|edema|dysplasia)\b',
    re.IGNORECASE,
)

# ── Medical prefixes/contexts that precede a disease mention ─────────
# "history of DISEASE", "diagnosed with DISEASE", "assessment: DISEASE"
_PREFIX_PATTERNS = [
    # "history of X", "h/o X", "known X", "diagnosis of X"
    re.compile(
        r'(?:history\s+of|h/o|diagnosed\s+with|diagnosis\s+of|'
        r'known|assessment\s*(?:of)?|impression\s*:?)\s+'
        r'([A-Z][A-Za-z0-9,/\s\-\(\)]{2,60})',
        re.IGNORECASE,
    ),
]

# ── Numbered-list pattern (Assessment & Plan, Problem List) ──────────
# Captures: "1. Essential hypertension: BP stable..." → "Essential hypertension"
_NUMBERED_ITEM = re.compile(
    r'(?:^|\n)\s*(\d{1,2})\.\s*([A-Z][^:\n]{2,80}?)(?:\s*:|$)',
    re.MULTILINE,
)

# ── Bullet-list pattern (PMH, Problem List) ──────────────────────────
# Captures: "• Anxiety", "- Depression", "* Osteoporosis"
_BULLET_ITEM = re.compile(
    r'(?:^|\n)\s*[•\-\*►▪▸]\s*([A-Z][A-Za-z0-9,/\s\-\(\)]{2,80}?)(?:\s*$|\s*(?:Active|Inactive|Resolved|History|\d{1,2}/\d{1,2}/\d{2,4}))',
    re.MULTILINE,
)

# ── Reject patterns — clearly NOT diseases ───────────────────────────
_REJECT_PATTERNS = [
    re.compile(r'^\d+$'),                                # Pure numbers
    re.compile(r'\d+\s*(mg|mcg|ml|units?|tabs?|caps?)'),  # Dosage
    re.compile(r'^\d{1,2}/\d{1,2}/\d{2,4}$'),           # Dates
    re.compile(r'^(dr|mr|mrs|ms|md|do|np|pa|rn)\.?\s', re.I),
    re.compile(r'ectomy\b', re.I),                        # Procedure names
    re.compile(r'plasty\b', re.I),
    re.compile(r'otomy\b', re.I),
    re.compile(r'ostomy\b', re.I),
    re.compile(r'\bpost[\s-]?op\b', re.I),
    re.compile(r'^(Continue|Monitor|Maintain|Refer|Counsel|Discuss|Order|Schedule|Return)\b', re.I),
    re.compile(r'\bvaccine\b', re.I),
    re.compile(r'\bimmunization\b', re.I),
]


# =====================================================================
# Public API
# =====================================================================

def extract_disease_candidates(
    text: str,
    section: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Extract disease/condition candidates from *one section* of clinical
    text using deterministic regex patterns.

    Parameters
    ----------
    text : str
        The raw text of a single clinical section.
    section : str | None
        Canonical section name (e.g. ``"assessment"``, ``"past_medical_history"``).

    Returns
    -------
    list[dict]
        Each dict has: disease_name, normalized_name, confidence_score,
        negated, section, section_sources, entity_type, sentence_number,
        start_char, end_char.
    """
    if not text or not text.strip():
        return []

    # Skip non-disease sections entirely
    if section and section.lower() in _NON_DISEASE_SECTIONS:
        return []

    candidates: List[Dict[str, Any]] = []
    seen_norms: Set[str] = set()

    section_lower = (section or "unknown").lower()

    # ── 1. Numbered-list items (Assessment/Plan, Problem List) ────────
    for m in _NUMBERED_ITEM.finditer(text):
        raw_item = m.group(2).strip()
        # Split comma-separated diseases in a single numbered entry
        # e.g. "Left leg pain, osteoporosis" → two candidates
        parts = re.split(r',\s+', raw_item)
        for part in parts:
            part = part.strip()
            # Also split slash-combined entries: "Major depression/moderate/recurrent"
            # Keep the full slash-combined form (the LLM will expand it)
            _add_candidate(candidates, seen_norms, part, section_lower, confidence=0.90)

    # ── 2. Bullet-list items (PMH, active problem list) ──────────────
    for m in _BULLET_ITEM.finditer(text):
        raw_item = m.group(1).strip()
        _add_candidate(candidates, seen_norms, raw_item, section_lower, confidence=0.85)

    # ── 3. "Diagnosis Date • Disease" PMH pattern ────────────────────
    # Many EMRs list PMH as: "Diagnosis Date • Anxiety • Depression ..."
    pmh_bullet = re.compile(r'[•]\s*([A-Z][A-Za-z\s\-,/\(\)]{2,60}?)(?=\s*[•]|\s*$)', re.MULTILINE)
    for m in pmh_bullet.finditer(text):
        raw_item = m.group(1).strip()
        # Skip dates and metadata
        if re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}', raw_item):
            continue
        _add_candidate(candidates, seen_norms, raw_item, section_lower, confidence=0.80)

    # ── 4. Clinical-prefix patterns ──────────────────────────────────
    for pat in _PREFIX_PATTERNS:
        for m in pat.finditer(text):
            raw = m.group(1).strip()
            # Trim at common stop words
            raw = re.split(r'\b(?:and|with|on|for|since|but|that|which|who|was|will|she|he|is)\b', raw, maxsplit=1, flags=re.I)[0].strip()
            # Split comma-separated lists into individual diseases
            parts = re.split(r',\s+', raw)
            for part in parts:
                part = part.strip()
                if part and len(part) >= 3:
                    _add_candidate(candidates, seen_norms, part, section_lower, confidence=0.70)

    # ── 5. Suffix-heuristic scan (terms ending in "-itis", "-emia", etc.) ─
    for m in _DISEASE_SUFFIXES.finditer(text):
        # Expand to capture the full term: go back up to 5 words
        start = max(0, m.start() - 60)
        snippet = text[start:m.end()]
        # Find the start of the disease phrase (cap letter or start of snippet)
        phrase_match = re.search(r'([A-Z][A-Za-z\s\-]{0,50}?' + re.escape(m.group(0)) + r')', snippet)
        if phrase_match:
            raw = phrase_match.group(1).strip()
            _add_candidate(candidates, seen_norms, raw, section_lower, confidence=0.65)

    return candidates


def extract_candidates_from_sections(
    sections: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Run regex extraction over all clinical sections.

    Parameters
    ----------
    sections : dict
        Canonical sections dict (same as used by the structuring pipeline).

    Returns
    -------
    list[dict]
        Merged, deduplicated candidate list across all sections.
    """
    all_candidates: List[Dict[str, Any]] = []
    seen_norms: Set[str] = set()

    for section_name, section_data in sections.items():
        text = section_data.get("text", "")
        if not text:
            continue
        candidates = extract_disease_candidates(text, section=section_name)
        for c in candidates:
            norm = c["normalized_name"]
            if norm not in seen_norms:
                all_candidates.append(c)
                seen_norms.add(norm)
            else:
                # Merge section_sources
                for existing in all_candidates:
                    if existing["normalized_name"] == norm:
                        sources = existing.get("section_sources", [])
                        new_source = c.get("section", "unknown")
                        if new_source not in sources:
                            sources.append(new_source)
                            existing["section_sources"] = sources
                        break

    return all_candidates


# =====================================================================
# Internal helpers
# =====================================================================

def _add_candidate(
    candidates: List[Dict],
    seen_norms: Set[str],
    raw_name: str,
    section: str,
    confidence: float,
) -> None:
    """Validate, clean, and append a disease candidate if it passes filters."""
    name = _clean_name(raw_name)
    if not name or len(name) < 3:
        return

    norm = name.lower().strip()

    # Reject single-word garbage
    if " " not in norm and norm in _SINGLE_WORD_REJECT:
        return

    # Reject medications
    first_word = norm.split()[0]
    if first_word in _MEDICATION_NAMES or norm in _MEDICATION_NAMES:
        return

    # Reject regex-matched patterns
    if any(p.search(name) for p in _REJECT_PATTERNS):
        return

    # Dedup
    if norm in seen_norms:
        return
    seen_norms.add(norm)

    candidates.append({
        "disease_name": name,
        "normalized_name": norm,
        "confidence_score": confidence,
        "negated": False,
        "start_char": 0,
        "end_char": 0,
        "section": section,
        "section_sources": [section],
        "entity_type": "CONDITION",
        "sentence_number": None,
    })


def _clean_name(raw: str) -> str:
    """Strip list prefixes, trailing punctuation, and whitespace artefacts."""
    # Remove numbered/bullet prefixes
    cleaned = re.sub(r'^\s*(?:\d+[.\)\s]|[a-zA-Z][.\)]|[•\-\*►▪▸])\s*', '', raw)
    # Remove trailing punctuation and whitespace
    cleaned = cleaned.strip('.,;:\t\n ')
    # Remove trailing metadata: "Active 01/01/2024", "History Type: ..."
    cleaned = re.sub(r'\s*Active\s+\d{1,2}/\d{1,2}/\d{2,4}.*$', '', cleaned, flags=re.I)
    cleaned = re.sub(r'\s*History\s+Type.*$', '', cleaned, flags=re.I)
    # Collapse whitespace
    cleaned = ' '.join(cleaned.split())
    # Remove content after double newlines
    cleaned = re.sub(r'\n\n.*$', '', cleaned).strip()
    return cleaned
