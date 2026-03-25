"""
Content-Based Section Inferrer
===============================
Detects clinical document sections by analyzing content patterns
when explicit section headers are missing or poorly formatted.

Complements the regex-based SectionCandidateExtractor by providing
a fallback for headerless documents or documents with non-standard
formatting.

Pipeline:
    1. Scan each line against content-specific regex patterns
    2. Group matching lines into proximity blocks
    3. Blocks exceeding the minimum-line threshold → inferred section
    4. Return position + canonical name for marker injection

Content patterns detected:
  - Medications: drug names with dosages/units
  - Vital Signs: BP, HR, Temp, SpO2, etc.
  - Assessment/Diagnoses: numbered diagnosis lists
  - Physical Exam: system-by-system exam findings
  - Allergies: NKDA, drug-reaction pairs
  - Review of Systems: denies/endorses patterns
  - Lab Results: lab values with numbers
  - Social History: smoking/alcohol/occupation mentions
  - Family History: family member + disease patterns
"""

import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class ContentSectionInferrer:
    """Infer section boundaries from content patterns when headers are absent."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Content pattern definitions
    # ------------------------------------------------------------------

    # Medication line: drug name + dose + unit
    # e.g. "• acetaminophen (TYLENOL) 500 MG tablet"
    #      "lisinopril 10 mg daily"
    #      "insulin glargine 20 units at bedtime"
    _MED_PATTERN = re.compile(
        r"(?i)"
        r"[a-z]+(?:\s+[a-z]+){0,3}"       # drug name 1-4 words
        r"\s*(?:\([^)]{1,30}\))?"          # optional brand in parens
        r"\s+\d+(?:\.\d+)?"               # dose number
        r"\s*(?:mg|mcg|µg|ml|mL|units?|tabs?|tablets?|caps?|capsules?|"
        r"cream|ointment|solution|injection|patch|spray|drops?|"
        r"suppository|inhaler|pen|vial|gm?|%)\b"
    )

    # Vital sign: known vital abbreviation + numeric value
    _VITAL_PATTERN = re.compile(
        r"(?i)(?:"
        r"\b(?:BP|Blood\s*Pressure)\s*:?\s*\d{2,3}\s*/\s*\d{2,3}"
        r"|\b(?:HR|Heart\s*Rate|Pulse)\s*:?\s*\d{2,3}\b"
        r"|\b(?:Temp|Temperature)\s*:?\s*\d{2,3}(?:\.\d{1,2})?\s*(?:°?\s*[FCfc])?"
        r"|\b(?:SpO2|O2\s*Sat|Oxygen\s*Sat)\s*:?\s*\d{2,3}\s*%?"
        r"|\b(?:RR|Resp(?:iratory)?\s*Rate)\s*:?\s*\d{1,2}\b"
        r"|\b(?:Wt|Weight)\s*:?\s*\d{2,4}(?:\.\d+)?\s*(?:kg|lbs?|pounds?)?"
        r"|\b(?:Ht|Height)\s*:?\s*\d[''′]?\s*\d{1,2}[\"″]?"
        r"|\bBMI\s*:?\s*\d{2}(?:\.\d+)?"
        r")"
    )

    # Assessment: numbered diagnosis line ("1. Diabetes mellitus type 2")
    _ASSESSMENT_PATTERN = re.compile(
        r"(?m)^\s*\d{1,2}\.\s+[A-Z][a-zA-Z\s,\-()]{3,}"
    )

    # Physical exam sub-headers ("HEENT:", "Lungs: clear", "Heart: RRR")
    _PE_PATTERN = re.compile(
        r"(?i)\b(?:"
        r"HEENT|Head|Eyes?|Ears?|Nose|Throat|Neck|Lungs?|Chest|"
        r"Heart|Cardiac|Cardiovascular|Abdomen|Abdominal|"
        r"Extremities|Neuro(?:logical)?|Skin|Integumentary|"
        r"Musculoskeletal|MSK|Psych(?:iatric)?|Back|Spine|"
        r"Lymph(?:atic)?|Genitourinary"
        r")\s*:"
    )

    # Allergy patterns ("NKDA", drug-reaction pairs)
    _ALLERGY_PATTERN = re.compile(
        r"(?i)(?:"
        r"\bNKDA\b|\bNKA\b"
        r"|\bNo\s+Known\s+(?:Drug\s+)?Allergies\b"
        r"|\b[A-Za-z]+\s*[-–—:]\s*(?:rash|hives|anaphylax|swelling|itching|"
        r"nausea|vomiting|angioedema|urticaria|dyspnea|throat\s+swell)"
        r")"
    )

    # ROS: "denies", "endorses", system-based review
    _ROS_PATTERN = re.compile(
        r"(?i)(?:"
        r"\b(?:denies?|positive\s+for|negative\s+for|endorses?|"
        r"no\s+complaints?\s+of)\b"
        r"|\b(?:Constitutional|Cardiovascular|Respiratory|Gastrointestinal|"
        r"Genitourinary|Musculoskeletal|Neurologic(?:al)?|Psychiatric|"
        r"Endocrine|Hematologic|Integumentary|Immunologic)\s*:"
        r")"
    )

    # Lab result: lab name + numeric value
    _LAB_PATTERN = re.compile(
        r"(?i)\b(?:"
        r"WBC|RBC|Hgb|Hemoglobin|Hct|Hematocrit|PLT|Platelets|"
        r"BUN|Cr(?:eatinine)?|Na|K|Cl|CO2|Bicarb|"
        r"Glucose|A1[cC]|HbA1[cC]|TSH|T[34]|"
        r"ALT|AST|ALP|Bilirubin|Albumin|"
        r"GFR|eGFR|Cholesterol|LDL|HDL|Triglycerides|"
        r"INR|PT|PTT|aPTT|ESR|CRP|BNP|Troponin"
        r")\s*:?\s*[<>]?\s*[\d.]+"
    )

    # Social history: smoking, alcohol, occupation, etc.
    _SOCIAL_PATTERN = re.compile(
        r"(?i)\b(?:"
        r"smok(?:es?|ing|er)|tobacco|cigarettes?|pack[- ]?year|"
        r"alcohol|drinks?|etoh|marijuana|cannabis|"
        r"illicit\s+drugs?|substance\s+(?:use|abuse)|"
        r"occupation|employed|retired|lives\s+(?:alone|with)|"
        r"married|divorced|widowed"
        r")\b"
    )

    # Family history: family member + disease context
    _FAMILY_PATTERN = re.compile(
        r"(?i)\b(?:"
        r"mother|father|brother|sister|son|daughter|aunt|uncle|"
        r"maternal|paternal|grandmother|grandfather|sibling|parent"
        r")\b"
        r".*\b(?:"
        r"diabetes|hypertension|cancer|heart|stroke|asthma|"
        r"copd|depression|alzheimer|dementia|arthritis|"
        r"kidney|liver|thyroid|epilepsy|diagnosed|died|deceased"
        r")\b"
    )

    # ------------------------------------------------------------------
    # Tuning: min matching lines to infer section, max gap between lines
    # ------------------------------------------------------------------
    _MIN_LINES: Dict[str, int] = {
        "medications": 2,
        "vitals": 1,
        "assessment": 1,
        "physical_exam": 2,
        "allergies": 1,
        "review_of_systems": 2,
        "lab_results": 2,
        "social_history": 1,
        "family_history": 1,
    }

    _MAX_GAP: Dict[str, int] = {
        "medications": 3,
        "vitals": 3,
        "assessment": 4,
        "physical_exam": 3,
        "allergies": 2,
        "review_of_systems": 3,
        "lab_results": 3,
        "social_history": 5,
        "family_history": 5,
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer_sections(self, text: str) -> List[Dict]:
        """Analyze document content to infer section boundaries.

        Returns list of dicts:
            position   : int   — char offset in text
            canonical  : str   — canonical section name
            confidence : float — 0.0–1.0
            source     : str   — always "content_inference"
        """
        lines = text.split("\n")

        # Build char offset for each line start
        line_offsets: List[int] = []
        offset = 0
        for line in lines:
            line_offsets.append(offset)
            offset += len(line) + 1  # +1 for newline

        # Map pattern name → compiled regex
        patterns: Dict[str, re.Pattern] = {
            "medications": self._MED_PATTERN,
            "vitals": self._VITAL_PATTERN,
            "assessment": self._ASSESSMENT_PATTERN,
            "physical_exam": self._PE_PATTERN,
            "allergies": self._ALLERGY_PATTERN,
            "review_of_systems": self._ROS_PATTERN,
            "lab_results": self._LAB_PATTERN,
            "social_history": self._SOCIAL_PATTERN,
            "family_history": self._FAMILY_PATTERN,
        }

        inferred: List[Dict] = []

        for section_name, pattern in patterns.items():
            matching_lines = [i for i, line in enumerate(lines) if pattern.search(line)]
            if not matching_lines:
                continue

            max_gap = self._MAX_GAP.get(section_name, 3)
            min_lines = self._MIN_LINES.get(section_name, 2)

            blocks = self._group_into_blocks(matching_lines, max_gap)

            for block in blocks:
                if len(block) < min_lines:
                    continue
                start_line = block[0]
                # Confidence: denser blocks → higher confidence
                block_span = block[-1] - block[0] + 1
                density = len(block) / max(block_span, 1)
                confidence = min(0.50 + density * 0.30 + len(block) * 0.03, 0.88)

                inferred.append({
                    "position": line_offsets[start_line] if start_line < len(line_offsets) else 0,
                    "canonical": section_name,
                    "confidence": round(confidence, 3),
                    "line_count": len(block),
                    "source": "content_inference",
                })

        # De-duplicate: keep strongest block per section type
        best_per_section: Dict[str, Dict] = {}
        for item in sorted(inferred, key=lambda x: -x["line_count"]):
            if item["canonical"] not in best_per_section:
                best_per_section[item["canonical"]] = item

        result = sorted(best_per_section.values(), key=lambda x: x["position"])

        if result:
            self.logger.info(
                f"Content inference: {len(result)} sections — "
                + ", ".join(f"{s['canonical']}({s['line_count']}L)" for s in result)
            )
        else:
            self.logger.info("Content inference: no sections detected from content patterns.")

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_into_blocks(line_indices: List[int], max_gap: int) -> List[List[int]]:
        """Group line indices into proximity blocks (gap ≤ max_gap)."""
        if not line_indices:
            return []

        blocks: List[List[int]] = [[line_indices[0]]]
        for idx in line_indices[1:]:
            if idx - blocks[-1][-1] <= max_gap:
                blocks[-1].append(idx)
            else:
                blocks.append([idx])

        return blocks
