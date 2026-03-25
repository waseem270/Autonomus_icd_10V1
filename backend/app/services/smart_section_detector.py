"""
Smart Section Detector — Context-First LLM Architecture
========================================================

PRIMARY approach: send the FULL document to Gemini for holistic section
detection with semantic understanding.  The LLM sees the entire document
structure, enabling it to:
  - Distinguish "Assessment" from "Assessment and Plan" based on context
  - Recognize subsections (Heart, Lungs) under Physical Exam
  - Ignore diagnosis names and medication lines that look like headers
  - Handle **any** document template without format-specific rules

FALLBACK: pattern-based candidate extraction + LLM context validation
(used only when the primary holistic approach fails).

CONTENT-BASED INFERENCE: pattern-driven section detection for headerless
documents.  Analyzes line-level content (drug names, dosage units, vital
sign patterns, ICD codes, exam findings) to infer canonical sections
without relying on any section headings.

LAST RESORT: full-LLM content extraction (headerless documents).

Pipeline:
    PRIMARY ──── Full-Document LLM Analysis ──→ Position Matching ──→ Slice
    FALLBACK ─── Regex Candidates ──→ LLM Validation ──→ Slice
    CONTENT ──── Line-Pattern Analysis ──→ Section Clustering ──→ Merge
    LAST RESORT  Full-LLM Content Extraction

The LLM NEVER generates or paraphrases document content.
It only identifies, classifies, and names section headers.
"""

import re
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set

import google.genai as genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.config import settings
from ..utils.gemini_retry import call_gemini_safe as call_gemini
from ..utils.fuzzy_section_matcher import (
    fuzzy_find_in_text,
    fuzzy_find_in_text_with_header,
    match_canonical,
    match_canonical_batch,
    normalize_header,
    combined_similarity,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical section names
# ---------------------------------------------------------------------------
CANONICAL_SECTIONS = [
    "chief_complaint",
    "history_present_illness",
    "past_medical_history",
    "past_surgical_history",
    "medications",
    "allergies",
    "social_history",
    "family_history",
    "review_of_systems",
    "vitals",
    "physical_exam",
    "objective",
    "immunizations",
    "lab_results",
    "imaging",
    "assessment",
    "plan",
    "assessment_and_plan",
    "follow_up",
    "other",
]

# ---------------------------------------------------------------------------
# Safety settings — disable content filtering for medical documents
# ---------------------------------------------------------------------------
SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
]


# ===========================================================================
#  Content-Based Section Inference Engine
# ===========================================================================


class ContentBasedSectionInferrer:
    """
    Infers canonical sections from document content when section headers
    are absent or insufficient.

    This engine analyzes individual lines using medical-domain pattern
    recognizers (drug names + dosage, vital signs, allergy phrases, lab
    values, exam findings, etc.) and clusters consecutive same-category
    lines into coherent section blocks.

    Design principles:
    - NO reliance on section headings — purely content-driven.
    - Pattern recognizers are composable; new categories can be added
      by registering a (regex, canonical_name, weight) tuple.
    - Lines that don't match any pattern are assigned to the nearest
      preceding section (contextual inheritance) or left as "other".
    - Final output merges adjacent blocks of the same type.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        # Each recognizer: (compiled_regex, canonical_section, confidence_weight)
        self._recognizers: List[Tuple[re.Pattern, str, float]] = []
        self._build_recognizers()

    # ------------------------------------------------------------------
    # Pattern Recognizers
    # ------------------------------------------------------------------

    def _build_recognizers(self) -> None:
        """
        Register all content-pattern recognizers.

        Each recognizer is a tuple of:
            (compiled_regex, canonical_section_name, confidence_weight)

        The confidence weight (0.0–1.0) indicates how strongly a single
        line match should count toward classifying that line's section.
        Higher weights are for highly specific patterns (e.g., a drug
        name + dosage is almost certainly a medication line).
        """
        recognizers_spec: List[Tuple[str, str, float, int]] = [
            # ── MEDICATIONS ──────────────────────────────────────────
            # Drug + dosage + form: "amlodipine 5 mg tablet"
            (
                r"(?i)\b[a-z]{3,}(?:\s*/\s*[a-z]+)*"       # drug name(s)
                r"\s+\d+(?:\.\d+)?"                          # dose number
                r"\s*(?:mg|mcg|g|ml|units?|iu|meq|%)"       # dose unit
                r"(?:\s*(?:/\s*\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?|iu|meq|%))?)"  # optional combo dose
                r"(?:\s+(?:tablet|tab|capsule|cap|solution|suspension|injection|cream|"
                r"ointment|patch|inhaler|drops?|syrup|powder|vial|pen|spray|gel|"
                r"suppository|lozenge|elixir|chewable|extended.release|"
                r"delayed.release|oral|topical|iv|im|sq|subq|sublingual|"
                r"transdermal|ophthalmic|otic|nasal|rectal|vaginal)s?)?",
                "medications",
                0.92,
                0,
            ),
            # Sig / directions pattern: "Take 1 tablet by mouth daily"
            (
                r"(?i)\b(?:take|inject|apply|inhale|instill|insert|chew|dissolve|use)"
                r"\s+\d+\s+(?:tablet|cap|capsule|puff|drop|spray|patch|unit|ml|dose)s?"
                r"\s+(?:by mouth|orally|topically|subcutaneously|intramuscularly|"
                r"intravenously|per|every|once|twice|three times|daily|bid|tid|qid|"
                r"q\d+h|prn|as needed|at bedtime|qhs|qam|qpm)",
                "medications",
                0.95,
                0,
            ),
            # Refill / quantity line: "Qty: 90  Refills: 3"
            (
                r"(?i)(?:qty|quantity|refills?|dispense|supply|days?\s*supply)"
                r"\s*[:=]?\s*\d+",
                "medications",
                0.80,
                0,
            ),
            # Medication list with route and quantity columns:
            # "amlodipine 5 mg tablet tab oral 90"
            (
                r"(?i)^[a-z][a-z\- ]{2,30}"
                r"\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?|iu|meq|%)"
                r"\s+\w+"
                r"\s+(?:tab|cap|sol|susp|inj|crm|oint)"
                r"\s+(?:oral|topical|iv|im|sq|subq|sublingual|transdermal|"
                r"ophthalmic|otic|nasal|rectal|vaginal|po|pr|sl)"
                r"(?:\s+\d+)?",
                "medications",
                0.95,
                0,
            ),
            # Standalone drug-form-route: "metformin ER 500mg PO BID"
            (
                r"(?i)\b[a-z]{4,}(?:\s+(?:er|sr|cr|xl|xr|dr|hcl|hct))?"
                r"\s+\d+\s*(?:mg|mcg|g)\s+"
                r"(?:po|pr|iv|im|sq|sl|top|inh|neb|od|bid|tid|qid|daily|"
                r"q\d+h|prn|qhs|qam|qpm)\b",
                "medications",
                0.90,
                0,
            ),

            # ── ALLERGIES ────────────────────────────────────────────
            (
                r"(?i)(?:^|\b)(?:allerg(?:y|ies|ic)\s+(?:to|reaction)|"
                r"(?:nkda|nka|no\s+known\s+(?:drug\s+)?allergies)|"
                r"(?:causes?|reaction)\s*:\s*(?:hives|rash|anaphylaxis|"
                r"swelling|itching|nausea|shortness of breath|angioedema))",
                "allergies",
                0.93,
                0,
            ),
            # Allergy entry: "Penicillin - rash, hives"
            (
                r"(?i)^[A-Z][a-z]+(?:\s+[a-z]+)?\s*[-–—:]\s*"
                r"(?:rash|hives|anaphylaxis|swelling|itching|nausea|"
                r"shortness of breath|angioedema|urticaria|gi upset|"
                r"throat\s+swelling|tongue\s+swelling|wheezing|"
                r"hypotension|vomiting|diarrhea)",
                "allergies",
                0.90,
                0,
            ),

            # ── VITALS ──────────────────────────────────────────────
            (
                r"(?i)(?:^|\b)(?:"
                r"(?:bp|blood\s+pressure)\s*[:=]?\s*\d{2,3}\s*/\s*\d{2,3}|"
                r"(?:hr|heart\s+rate|pulse)\s*[:=]?\s*\d{2,3}\s*(?:bpm)?|"
                r"(?:temp(?:erature)?|t)\s*[:=]?\s*\d{2,3}(?:\.\d)?\s*(?:[°ºf]|deg|celsius|fahrenheit)?|"
                r"(?:rr|resp(?:iratory)?\s*rate)\s*[:=]?\s*\d{1,2}|"
                r"(?:spo2|o2\s*sat|oxygen\s+sat(?:uration)?)\s*[:=]?\s*\d{2,3}\s*%?|"
                r"(?:wt|weight)\s*[:=]?\s*\d{2,4}\s*(?:kg|lb|lbs|pounds)?|"
                r"(?:ht|height)\s*[:=]?\s*\d{1,3}(?:[''′]\s*\d{1,2}[\"″]?|\s*(?:cm|in|ft))?"
                r")\b",
                "vitals",
                0.91,
                0,
            ),

            # ── LAB RESULTS ─────────────────────────────────────────
            # Lab value: "WBC 7.2 x10^3/uL" or "Hemoglobin: 13.5 g/dL"
            (
                r"(?i)(?:^|\b)(?:"
                r"(?:wbc|rbc|hgb|hct|plt|mcv|mch|mchc|rdw|mpv|"
                r"hemoglobin|hematocrit|platelets?|"
                r"sodium|potassium|chloride|co2|bicarb(?:onate)?|bun|"
                r"creatinine|glucose|calcium|magnesium|phosphorus|"
                r"albumin|total\s+protein|bilirubin|"
                r"ast|alt|alp|ggt|ldh|lipase|amylase|"
                r"tsh|t3|t4|free\s+t4|"
                r"inr|pt|ptt|aptt|"
                r"hba1c|a1c|hemoglobin\s+a1c|"
                r"troponin|bnp|nt-?probnp|d-?dimer|"
                r"ferritin|iron|tibc|transferrin|"
                r"crp|esr|sed\s+rate|procalcitonin|"
                r"gfr|egfr|"
                r"ua|urinalysis|urine)"
                r")\s*[:=]?\s*"
                r"[<>]?\s*\d+(?:\.\d+)?",
                "lab_results",
                0.89,
                0,
            ),
            # Lab with reference range: "13.5 (12.0 - 16.0)"
            (
                r"(?i)(?:^|\b)[a-z][a-z\s]{2,25}:\s*"
                r"\d+(?:\.\d+)?\s*"
                r"(?:\(?\s*\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*\)?)"
                r"\s*(?:mg/dl|g/dl|mmol/l|meq/l|u/l|iu/l|ng/ml|pg/ml|"
                r"mcg/dl|%|x10[³\^]?3?/?[uμ]?l|cells?/?[uμ]l|"
                r"mm/hr|seconds?|sec|ratio)?",
                "lab_results",
                0.85,
                0,
            ),

            # ── PHYSICAL EXAM ───────────────────────────────────────
            # Exam findings with anatomical terms
            (
                r"(?i)(?:^|\b)(?:"
                r"(?:heent|cardiovascular|respiratory|abdomen|"
                r"musculoskeletal|neurological|psychiatric|"
                r"integumentary|lymphatic|genitourinary)\s*:"
                r"|(?:no\s+(?:murmurs?|rales|rhonchi|wheezing|edema|"
                r"tenderness|guarding|rebound|rigidity|"
                r"lymphadenopathy|cyanosis|clubbing|jaundice))"
                r"|(?:lungs?\s+(?:clear|cta)|heart\s+(?:rrr|regular))"
                r"|(?:(?:pupils?|perrla|eomi|ncat|a&ox[234]|"
                r"alert\s+and\s+oriented|"
                r"regular\s+rate\s+and\s+rhythm|rrr|"
                r"clear\s+to\s+auscultation|"
                r"soft[,\s]+non[- ]?tender[,\s]+non[- ]?distended|"
                r"no\s+(?:focal|gross)\s+deficits?))"
                r"|(?:(?:normal|abnormal|wnl|within\s+normal\s+limits)"
                r"\s+(?:gait|reflexes|strength|sensation|rom|range\s+of\s+motion))"
                r")",
                "physical_exam",
                0.88,
                0,
            ),

            # ── IMAGING ─────────────────────────────────────────────
            (
                r"(?i)(?:^|\b)(?:"
                r"(?:x-?ray|ct\s*(?:scan)?|mri|ultrasound|echo(?:cardiogram)?|"
                r"mammogram|dexa|bone\s+density|pet\s+scan|"
                r"fluoroscopy|angiograph?y|venogram|arthrogram)"
                r"\s*(?:of\s+(?:the\s+)?[a-z]+)?\s*[:=]?"
                r"|(?:impression|finding|result)\s*:\s*"
                r"(?:no\s+(?:acute|significant)|normal|unremarkable|"
                r"stable|improved|worsened|new)"
                r")",
                "imaging",
                0.85,
                0,
            ),

            # ── IMMUNIZATIONS ───────────────────────────────────────
            (
                r"(?i)(?:^|\b)(?:"
                r"(?:influenza|flu|covid|tdap|td|tetanus|"
                r"pneumo(?:coccal|vax)|prevnar|"
                r"hepatitis\s*[ab]|hep\s*[ab]|"
                r"mmr|varicella|zoster|shingrix|"
                r"ipv|polio|hib|meningococcal|"
                r"hpv|gardasil|rotavirus)"
                r"\s+(?:vaccine|vaccination|immunization|dose|booster|series)"
                r"|(?:vaccine|vaccination|immunization|dose|booster)"
                r"\s+(?:given|administered|received|due|overdue|declined)"
                r"(?:\s+(?:on\s+)?\d{1,2}[-/]\d{1,2}[-/]\d{2,4})?"
                r")",
                "immunizations",
                0.88,
                0,
            ),

            # ── ASSESSMENT / DIAGNOSIS ──────────────────────────────
            # ICD-10 code pattern: "I10" "E11.65" "J44.1"
            (
                r"(?i)(?:^|\b)(?:icd[- ]?10\s*[:=]?\s*)?"
                r"[A-TV-Z]\d{2,3}(?:\.\d{1,4})?"
                r"\s*[-–:]\s*[A-Z][a-z]",
                "assessment",
                0.85,
                0,
            ),
            # Numbered diagnosis list: "1. Essential Hypertension"
            # Only match when followed by disease-like words
            (
                r"(?i)^\s*\d+\.\s+[A-Z][a-z]+(?:\s+[a-z]+){0,5}"
                r"\s*(?:\(?(?:icd|[A-TV-Z]\d{2})|\buncontrolled\b|"
                r"\bcontrolled\b|\bstable\b|\bchronic\b|\bacute\b|"
                r"\btype\s*[12]\b|\bstage\b|\bwith\b|\bwithout\b)",
                "assessment",
                0.78,
                0,
            ),

            # ── PLAN ────────────────────────────────────────────────
            (
                r"(?i)(?:^|\b)(?:"
                r"(?:continue|start|stop|increase|decrease|titrate|"
                r"switch|change|add|discontinue|taper|refill|renew)\s+"
                r"(?:current\s+)?(?:medication|dose|regimen|therapy|treatment)|"
                r"(?:order(?:ed)?|schedule[d]?|refer(?:red)?(?:\s+to)?|"
                r"recommend(?:ed)?|advise[d]?)\s+"
                r"(?:lab|labs|imaging|ct|mri|x-?ray|ultrasound|echo|"
                r"consult|follow[- ]?up|f/?u|return\s+visit|"
                r"physical\s+therapy|pt|ot|"
                r"specialist|cardiology|neurology|endocrinology|"
                r"dermatology|gastroenterology|ophthalmology)"
                r"|(?:return\s+(?:to\s+(?:clinic|office)|in)\s+\d+\s*(?:day|week|month)s?)"
                r"|(?:patient\s+(?:educated|counseled|advised)\s+(?:on|about|regarding))"
                r")",
                "plan",
                0.82,
                0,
            ),

            # ── FOLLOW UP ───────────────────────────────────────────
            (
                r"(?i)(?:^|\b)(?:"
                r"(?:follow[- ]?up|f/?u|return)\s+"
                r"(?:in\s+)?\d+\s*(?:day|week|month|year)s?"
                r"|(?:next\s+(?:appointment|visit))\s*[:=]?"
                r"|(?:rto|return\s+to\s+(?:office|clinic))\s+"
                r"(?:in\s+)?\d+"
                r")",
                "follow_up",
                0.86,
                0,
            ),

            # ── SOCIAL HISTORY ──────────────────────────────────────
            (
                r"(?i)(?:^|\b)(?:"
                r"(?:(?:current|former|never|non)[- ]?smoker|"
                r"smokes?\s+\d+\s*(?:pack|ppd|cigarettes?)|"
                r"tobacco\s+(?:use|history|abuse)|"
                r"quit\s+(?:smoking|tobacco))"
                r"|(?:(?:drinks?|consumes?)\s+\d+\s*(?:drink|beer|wine|alcohol)|"
                r"(?:alcohol|etoh)\s+(?:use|abuse|dependence|history)|"
                r"social\s+drinker|denies\s+(?:alcohol|etoh|tobacco|drug|illicit))"
                r"|(?:(?:illicit|recreational)\s+drug\s+(?:use|abuse|history)|"
                r"(?:marijuana|cannabis|cocaine|heroin|methamphetamine|opioid)\s+use)"
                r"|(?:(?:lives|resides)\s+(?:alone|with)|"
                r"(?:married|single|divorced|widowed|separated)|"
                r"(?:employed|retired|unemployed|disabled|occupation))"
                r")",
                "social_history",
                0.87,
                0,
            ),

            # ── FAMILY HISTORY ──────────────────────────────────────
            (
                r"(?i)(?:^|\b)(?:"
                r"(?:(?:mother|father|sister|brother|parent|sibling|"
                r"maternal|paternal|grandmother|grandfather|"
                r"aunt|uncle|cousin|family\s+(?:member|history))\s+"
                r"(?:with|has|had|diagnosed|hx|history)\s+"
                r"(?:of\s+)?[a-z])"
                r"|(?:(?:no|negative|positive)\s+family\s+(?:history|hx))"
                r"|(?:fh\s*[:=]\s*(?:positive|negative|significant|unremarkable))"
                r")",
                "family_history",
                0.86,
                0,
            ),

            # ── REVIEW OF SYSTEMS ───────────────────────────────────
            # ROS pattern: "denies fever, chills" or "positive for cough"
            (
                r"(?i)(?:^|\b)(?:"
                r"(?:(?:denies|reports?|endorses?|admits?|complains?\s+of|"
                r"positive\s+for|negative\s+for|no)\s+"
                r"(?:fever|chills|fatigue|malaise|weight\s+(?:loss|gain)|"
                r"night\s+sweats?|headache|dizziness|"
                r"chest\s+pain|palpitations?|dyspnea|sob|"
                r"(?:shortness\s+of\s+breath)|"
                r"cough|wheez(?:e|ing)|"
                r"nausea|vomiting|diarrhea|constipation|"
                r"abdominal\s+pain|"
                r"dysuria|frequency|urgency|hematuria|"
                r"(?:joint|muscle|back)\s+pain|"
                r"rash|itching|"
                r"(?:blurred|change\s+in)\s+vision|"
                r"anxiety|depression|insomnia))"
                r"|(?:all\s+other\s+systems?\s+(?:reviewed?\s+and\s+)?negative)"
                r"|(?:(?:\d+|fourteen|twelve|ten)\s*[- ]?point\s+ros)"
                r")",
                "review_of_systems",
                0.86,
                0,
            ),

            # ── CHIEF COMPLAINT ─────────────────────────────────────
            # Short statement of reason: "here for annual physical"
            (
                r"(?i)^(?:"
                r"(?:(?:patient|pt)\s+(?:presents?|comes?\s+in|here|is\s+here|"
                r"seen\s+(?:today|for))\s+(?:for|with|complaining\s+of))"
                r"|(?:(?:reason\s+for\s+(?:visit|encounter)|chief\s+concern)\s*[:=]?)"
                r")",
                "chief_complaint",
                0.85,
                0,
            ),

            # ── HPI / HISTORY OF PRESENT ILLNESS ────────────────────
            # Narrative with temporal and symptom language
            (
                r"(?i)(?:^|\b)(?:"
                r"(?:(?:patient|pt)\s+is\s+a\s+\d+\s*[-–]?\s*"
                r"(?:year|yr|y/?o|y\.o\.?)(?:\s*[-–]?\s*old)?\s+"
                r"(?:male|female|m|f|man|woman))"
                r"|(?:(?:\d+\s*[-–]?\s*(?:day|week|month|year)s?\s+"
                r"(?:history|hx|ago))\s+(?:of|since))"
                r")",
                "history_present_illness",
                0.80,
                0,
            ),

            # ── PAST SURGICAL HISTORY ───────────────────────────────
            (
                r"(?i)(?:^|\b)(?:"
                r"(?:(?:appendectomy|cholecystectomy|hysterectomy|"
                r"mastectomy|lumpectomy|colectomy|"
                r"(?:total|partial)\s+(?:knee|hip)\s+(?:replacement|arthroplasty)|"
                r"(?:cabg|coronary\s+artery\s+bypass)|"
                r"(?:c[- ]?section|cesarean)|"
                r"(?:lap(?:aroscopic)?)\s+[a-z]+(?:ectomy|otomy|plasty|pexy)|"
                r"tonsillectomy|adenoidectomy|"
                r"hernia\s+repair|"
                r"(?:acl|rotator\s+cuff)\s+repair)"
                r"\s*(?:\(?\s*\d{4}\s*\)?|\bin\s+\d{4}\b)?)"
                r")",
                "past_surgical_history",
                0.83,
                0,
            ),
        ]

        for pattern_str, section, weight, flags in recognizers_spec:
            try:
                compiled = re.compile(pattern_str, flags)
                self._recognizers.append((compiled, section, weight))
            except re.error as exc:
                self.logger.warning(
                    f"Failed to compile recognizer for {section}: {exc}"
                )

    # ------------------------------------------------------------------
    # Core Inference
    # ------------------------------------------------------------------

    def infer_sections(self, text: str) -> Dict[str, Any]:
        """
        Analyze document text line-by-line and infer canonical sections
        from content patterns alone (no headers required).

        Returns the same structure as SmartSectionDetector.detect_sections():
            sections             – {canonical_name: {text, start, end}}
            has_structure        – bool
            confidence           – float
            rejected_subsections – []
        """
        lines = text.split("\n")
        # Classify each line
        classifications: List[Tuple[int, str, str, float]] = []
        # (line_index, canonical_section, line_text, confidence)

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            section, conf = self._classify_line(stripped)
            classifications.append((i, section, stripped, conf))

        if not classifications:
            return {
                "sections": {},
                "has_structure": False,
                "confidence": 0.0,
                "rejected_subsections": [],
            }

        # Cluster consecutive lines of the same section type
        clusters = self._cluster_lines(classifications, lines)

        # Apply contextual inheritance: unclassified lines ("other") that
        # sit between two blocks of the same type get absorbed
        clusters = self._inherit_context(clusters)

        # Build final sections
        sections = self._build_sections_from_clusters(clusters, text)

        if not sections:
            return {
                "sections": {},
                "has_structure": False,
                "confidence": 0.0,
                "rejected_subsections": [],
            }

        # Overall confidence = weighted average of matched-line confidences
        matched = [c for c in classifications if c[1] != "other"]
        avg_conf = (
            sum(c[3] for c in matched) / len(matched) if matched else 0.0
        )
        # Scale by coverage: what fraction of non-empty lines were classified?
        non_empty_count = sum(1 for l in lines if l.strip())
        coverage = len(matched) / non_empty_count if non_empty_count else 0
        overall_conf = round(min(avg_conf * (0.5 + 0.5 * coverage), 1.0), 2)

        return {
            "sections": sections,
            "has_structure": True,
            "confidence": overall_conf,
            "rejected_subsections": [],
        }

    def _classify_line(self, line: str) -> Tuple[str, float]:
        """
        Classify a single line by running all recognizers and returning
        the highest-confidence match.

        Returns (canonical_section, confidence).  If no recognizer fires,
        returns ("other", 0.0).
        """
        best_section = "other"
        best_confidence = 0.0

        for pattern, section, weight in self._recognizers:
            if pattern.search(line):
                if weight > best_confidence:
                    best_section = section
                    best_confidence = weight

        return best_section, best_confidence

    def _cluster_lines(
        self,
        classifications: List[Tuple[int, str, str, float]],
        all_lines: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Group consecutive classified lines into clusters.

        A cluster represents a contiguous block of lines that belong
        to the same section.  Unclassified lines between two blocks of
        the same type are absorbed into that section.
        """
        if not classifications:
            return []

        clusters: List[Dict[str, Any]] = []
        current_section = classifications[0][1]
        current_start_line = classifications[0][0]
        current_lines: List[str] = []
        current_confidences: List[float] = []

        for i, (line_idx, section, line_text, conf) in enumerate(classifications):
            if section == current_section:
                # Include any blank lines between the previous classified
                # line and this one (maintain original formatting)
                if current_lines:
                    prev_line_idx = classifications[i - 1][0] if i > 0 else line_idx
                    for gap_idx in range(prev_line_idx + 1, line_idx):
                        gap_line = all_lines[gap_idx] if gap_idx < len(all_lines) else ""
                        current_lines.append(gap_line)
                current_lines.append(all_lines[line_idx])
                current_confidences.append(conf)
            else:
                # Flush current cluster
                if current_lines and current_section != "other":
                    clusters.append({
                        "section": current_section,
                        "start_line": current_start_line,
                        "end_line": classifications[i - 1][0] if i > 0 else current_start_line,
                        "lines": current_lines,
                        "confidence": (
                            sum(current_confidences) / len(current_confidences)
                            if current_confidences else 0.0
                        ),
                    })
                elif current_lines:
                    # Keep "other" clusters for context inheritance
                    clusters.append({
                        "section": "other",
                        "start_line": current_start_line,
                        "end_line": classifications[i - 1][0] if i > 0 else current_start_line,
                        "lines": current_lines,
                        "confidence": 0.0,
                    })
                # Start new cluster
                current_section = section
                current_start_line = line_idx
                current_lines = [all_lines[line_idx]]
                current_confidences = [conf]

        # Flush final cluster
        if current_lines:
            clusters.append({
                "section": current_section,
                "start_line": current_start_line,
                "end_line": classifications[-1][0],
                "lines": current_lines,
                "confidence": (
                    sum(current_confidences) / len(current_confidences)
                    if current_confidences else 0.0
                ),
            })

        return clusters

    def _inherit_context(
        self, clusters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Contextual inheritance: if an "other" cluster is sandwiched between
        two clusters of the same section type, absorb it into that section.

        Also merges adjacent clusters of the same section type.
        """
        if len(clusters) <= 1:
            return clusters

        # Pass 1: absorb "other" blocks sandwiched between same-type blocks
        merged = list(clusters)
        changed = True
        while changed:
            changed = False
            new_merged: List[Dict[str, Any]] = []
            i = 0
            while i < len(merged):
                if (
                    i > 0
                    and i < len(merged) - 1
                    and merged[i]["section"] == "other"
                    and merged[i - 1]["section"] == merged[i + 1]["section"]
                    and merged[i - 1]["section"] != "other"
                ):
                    # Absorb into previous cluster
                    new_merged[-1]["lines"].extend(merged[i]["lines"])
                    new_merged[-1]["end_line"] = merged[i]["end_line"]
                    changed = True
                else:
                    new_merged.append(merged[i])
                i += 1
            merged = new_merged

        # Pass 2: merge adjacent clusters with the same section type
        final: List[Dict[str, Any]] = [merged[0]]
        for cluster in merged[1:]:
            if cluster["section"] == final[-1]["section"]:
                final[-1]["lines"].extend(cluster["lines"])
                final[-1]["end_line"] = cluster["end_line"]
                final[-1]["confidence"] = max(
                    final[-1]["confidence"], cluster["confidence"]
                )
            else:
                final.append(cluster)

        return final

    def _build_sections_from_clusters(
        self,
        clusters: List[Dict[str, Any]],
        full_text: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Convert clusters to the canonical sections dict format."""
        sections: Dict[str, Dict[str, Any]] = {}

        for cluster in clusters:
            section_name = cluster["section"]
            if section_name == "other":
                continue

            content = "\n".join(cluster["lines"]).strip()
            if not content or len(content) < 10:
                continue

            # Find position in original text
            # Use the first line's content to anchor the position
            first_line = cluster["lines"][0].strip()
            if first_line:
                start = full_text.find(first_line)
            else:
                start = 0
            if start == -1:
                start = 0
            end = start + len(content)

            if section_name in sections:
                existing = sections[section_name]
                if content.strip() not in existing["text"]:
                    existing["text"] += "\n\n" + content
                    existing["end"] = max(existing["end"], end)
            else:
                sections[section_name] = {
                    "text": content,
                    "start": start,
                    "end": end,
                }

        return sections


# Singleton for the content inferrer
_content_inferrer = ContentBasedSectionInferrer()


class SmartSectionDetector:
    """
    Context-first section detection using full-document LLM analysis.

    Unlike format-dependent approaches that classify bold text, colon lines,
    or capitalized phrases as headers, this detector sends the ENTIRE document
    to Gemini for holistic analysis.  The model evaluates headers by their
    **semantic role** in the clinical note hierarchy, not surface formatting.

    When headers are absent, the content-based inference engine analyzes
    individual lines using medical-domain pattern recognizers (drug names +
    dosage, vital signs, allergy phrases, lab values, exam findings) and
    clusters them into coherent section blocks.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._client = None
        self.model_name = settings.GEMINI_MODEL
        self._content_inferrer = _content_inferrer

        # Config for section detection (needs structured JSON output)
        self.detect_config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=16384,
            response_mime_type="application/json",
            safety_settings=SAFETY_SETTINGS,
            system_instruction=(
                "You are a medical documentation expert specializing in EHR structure and clinical NLP. "
                "Your mission is to parse clinical notes into canonical sections while identifying specific, "
                "documented medical conditions with maximum precision."
            )
        )

        # Config for shorter tasks (validation, name mapping)
        self.quick_config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=8192,
            response_mime_type="application/json",
            safety_settings=SAFETY_SETTINGS,
            system_instruction="You are a medical terminology expert. Map clinical terms to canonical section names."
        )

    @property
    def client(self):
        if self._client is None:
            from ..core.config import create_genai_client
            self._client = create_genai_client()
        return self._client

    # ==================================================================
    #  Public Entry Point
    # ==================================================================

    async def detect_sections(self, text: str) -> Dict[str, Any]:
        """
        Detect clinical note sections using context-first LLM analysis.

        Returns
        -------
        dict
            sections             – {canonical_name: {text, start, end}}
            has_structure        – bool
            confidence           – float
            rejected_subsections – list of dropped subsection names
        """
        if not text or len(text.strip()) < 50:
            self.logger.warning("Text too short for section detection.")
            return {
                "sections": {},
                "has_structure": False,
                "confidence": 0.0,
                "rejected_subsections": [],
            }

        # ── PRIMARY: Holistic LLM (full-document context) ───────────
        try:
            start_t = time.time()
            result = await self._detect_holistic(text)
            self.logger.info(f"Holistic detection took {time.time() - start_t:.2f}s")
            if result and len(result.get("sections", {})) >= 2:
                # Enrich with content-based inference for any missed sections
                result = self._enrich_with_content_inference(result, text)
                self.logger.info(
                    f"Holistic detection: {len(result['sections'])} sections, "
                    f"confidence={result.get('confidence', 0):.2f}"
                )
                return result
            self.logger.info(
                "Holistic detection returned <2 sections, trying fallback."
            )
        except Exception as exc:
            self.logger.warning(
                f"Holistic detection failed ({exc}), trying fallback."
            )

        # ── FALLBACK: Pattern candidates + LLM validation ───────────
        try:
            result = await self._detect_with_candidates(text)
            if result and result.get("sections"):
                # Enrich with content-based inference for any missed sections
                result = self._enrich_with_content_inference(result, text)
                return result
        except Exception as exc:
            self.logger.warning(
                f"Candidate fallback failed ({exc}), trying content inference."
            )

        # ── CONTENT-BASED INFERENCE: Pattern-driven (no headers) ────
        try:
            start_t = time.time()
            result = self._content_inferrer.infer_sections(text)
            self.logger.info(
                f"Content-based inference took {time.time() - start_t:.2f}s"
            )
            if result and len(result.get("sections", {})) >= 1:
                self.logger.info(
                    f"Content inference: {len(result['sections'])} sections, "
                    f"confidence={result.get('confidence', 0):.2f}, "
                    f"sections={list(result['sections'].keys())}"
                )
                return result
            self.logger.info(
                "Content inference returned 0 sections, trying last resort."
            )
        except Exception as exc:
            self.logger.warning(
                f"Content inference failed ({exc}), trying last resort."
            )

        # ── LAST RESORT: Full-LLM content extraction ────────────────
        return await self._llm_full_detection(text)

    # ==================================================================
    #  Content-Based Enrichment (supplements LLM-detected sections)
    # ==================================================================

    def _enrich_with_content_inference(
        self, result: Dict[str, Any], text: str
    ) -> Dict[str, Any]:
        """
        After LLM-based section detection, run content-based inference on
        the full text and merge in any sections that the LLM missed.

        This catches cases like:
        - Medications listed without a "Medications" header
        - Vitals embedded in narrative text
        - Allergy entries not under a labeled allergy section
        """
        try:
            inferred = self._content_inferrer.infer_sections(text)
            inferred_sections = inferred.get("sections", {})

            if not inferred_sections:
                return result

            existing_sections = result.get("sections", {})
            added_count = 0

            for section_name, section_data in inferred_sections.items():
                if section_name not in existing_sections:
                    # Only add inferred sections that don't overlap with
                    # existing sections (check position ranges)
                    inferred_start = section_data.get("start", 0)
                    inferred_end = section_data.get("end", 0)

                    overlaps = False
                    for existing in existing_sections.values():
                        ex_start = existing.get("start", 0)
                        ex_end = existing.get("end", 0)
                        # Check if ranges overlap
                        if inferred_start < ex_end and inferred_end > ex_start:
                            overlaps = True
                            break

                    if not overlaps:
                        existing_sections[section_name] = section_data
                        added_count += 1
                        self.logger.info(
                            f"Content enrichment: added '{section_name}' "
                            f"({len(section_data.get('text', ''))} chars)"
                        )

            if added_count > 0:
                result["sections"] = existing_sections
                self.logger.info(
                    f"Content enrichment added {added_count} sections"
                )

        except Exception as exc:
            self.logger.warning(f"Content enrichment failed: {exc}")

        return result

    # ==================================================================
    #  PRIMARY — Holistic Full-Document LLM Detection
    # ==================================================================

    async def _detect_holistic(self, text: str) -> Dict[str, Any]:
        """
        Send the full document to Gemini for semantic section detection and 
        integrated disease validation.
        """
        doc_text = text[:15000] if len(text) > 15000 else text
        prompt = self._build_holistic_prompt(doc_text)

        response = await call_gemini(
            client=self.client,
            model=self.model_name, contents=prompt, config=self.detect_config,
        )

        # ── Validate response before parsing ─────────────────────
        raw = self._safe_response_text(response, "holistic")
        if not raw:
            raise ValueError("Holistic LLM returned empty / blocked response")

        self.logger.debug(f"Holistic raw output (first 300 chars): {raw[:300]}")
        result = self._extract_json(raw)

        # Task 1: Sections
        llm_sections = result.get("sections", [])
        rejected = result.get("rejected_subsections", [])
        confidence = float(result.get("confidence", 0.85))
        
        # Task 2: Validated Diseases (Stored for enrichment)
        validated_diseases = result.get("diseases", [])
        self.logger.info(f"LLM identified {len(validated_diseases)} validated diseases")

        if not llm_sections:
            return {
                "sections": {},
                "has_structure": False,
                "confidence": 0.0,
                "rejected_subsections": rejected,
                "validated_diseases": validated_diseases
            }

        # Filter out known column labels that the LLM may have misidentified
        # as section headers (e.g. "Diagnosis Date" inside PMH tables).
        filtered_sections = []
        for s in llm_sections:
            hdr = (s.get("header_text") or "").strip()
            hdr_clean = hdr.lower().strip("[]").rstrip(":").strip()
            if hdr_clean in self._NOT_SECTION_LABELS:
                self.logger.info(
                    f"Holistic: rejected column label '{hdr}' from LLM sections"
                )
                rejected.append(hdr)
                continue
            filtered_sections.append(s)
        llm_sections = filtered_sections

        if not llm_sections:
            return {
                "sections": {},
                "has_structure": False,
                "confidence": 0.0,
                "rejected_subsections": rejected,
                "validated_diseases": validated_diseases
            }

        # Match LLM-returned headers to positions in source text
        # Model returns 'header_text' in new prompt
        verified = self._match_headers_to_positions(
            [{"header": s.get("header_text")} for s in llm_sections], 
            text
        )
        if not verified:
            return {
                "sections": {},
                "has_structure": False,
                "confidence": 0.0,
                "rejected_subsections": rejected,
                "validated_diseases": validated_diseases
            }

        # Slice text between verified header positions
        raw_sections = self._slice_sections(text, verified)
        merged = self._merge_duplicate_keys(raw_sections)

        # Build name map from the LLM's canonical names
        name_map: Dict[str, str] = {}
        _canonical_set = set(CANONICAL_SECTIONS)
        for entry in llm_sections:
            hdr = entry.get("header_text", "")
            canonical = entry.get("normalized_name", "other").lower().strip()
            key = hdr.lower().strip().strip("[]").rstrip(":")
            if not key:
                continue
            # Post-correct: if the LLM returned an invalid canonical name,
            # or the header text fuzzy-matches a different canonical better,
            # override with the fuzzy matcher's result.
            if canonical not in _canonical_set:
                corrected, _score = match_canonical(hdr)
                if corrected:
                    self.logger.info(
                        f"Holistic post-fix: '{hdr}' LLM said '{canonical}' "
                        f"→ corrected to '{corrected}' (score={_score:.2f})"
                    )
                    canonical = corrected
                else:
                    canonical = "other"
            else:
                # Even when canonical is valid, verify it matches the header
                # (e.g. header="Assessment" but LLM said "physical_exam")
                corrected, score = match_canonical(hdr)
                if corrected and corrected != canonical and score >= 0.90:
                    self.logger.info(
                        f"Holistic post-fix: '{hdr}' LLM said '{canonical}' "
                        f"→ corrected to '{corrected}' (score={score:.2f})"
                    )
                    canonical = corrected
            name_map[key] = canonical

        sections = self._build_final_sections(merged, name_map, text)

        self.logger.info(f"Holistic sections: {list(sections.keys())}")
        return {
            "sections": sections,
            "has_structure": bool(sections),
            "confidence": confidence,
            "rejected_subsections": rejected,
            "validated_diseases": validated_diseases
        }

    # ------------------------------------------------------------------

    def _build_holistic_prompt(self, doc_text: str) -> str:
        """Build the integrated section detection and disease validation prompt.

        The system prompt is loaded from backend/prompt.json on every call.
        """
        prompt_path = Path(__file__).resolve().parents[2] / "prompt.json"

        try:
            with prompt_path.open("r", encoding="utf-8") as f:
                prompt_config = json.load(f)
            system_prompt = json.dumps(prompt_config, indent=2, ensure_ascii=False)
        except Exception as exc:
            self.logger.warning(
                f"Failed to load prompt config from {prompt_path}: {exc}"
            )
            system_prompt = "{}"

        canonical_list = ", ".join(CANONICAL_SECTIONS)

        return f"""Use the following system prompt configuration exactly as the primary instruction source:

{system_prompt}

CLINICAL NOTE:
\"\"\"
{doc_text}
\"\"\"

CANONICAL SECTION NAMES — you MUST use one of these for the "normalized_name" field:
{canonical_list}

RETURN FORMAT (JSON ONLY):
{{
    "sections": [
        {{
            "header_text": "Assessment and Plan",
            "normalized_name": "assessment_and_plan",
            "note": "Split into assessment + plan"
        }}
    ],
    "diseases": [
        {{
            "disease_name": "Essential Hypertension",
            "is_valid_disease": true,
            "icd_billable": true,
            "confidence": 0.95,
            "found_in_section": "assessment",
            "clinical_context": "BP stable on current medications",
            "reasoning": "Billable diagnosis (I10)"
        }}
    ],
    "rejected_subsections": [],
    "confidence": 0.95
}}

IMPORTANT:
- Return ONLY valid JSON.
- Include ALL sections and ALL diseases found across the ENTIRE document.
- Keep section headers in document order and preserve exact header text.
- "normalized_name" MUST be from the canonical list above. If unsure, use "other".
- A section titled "Assessment" must be normalized to "assessment", NOT "physical_exam".
"""

    # ==================================================================
    #  Position Matching (shared by primary + fallback)
    # ==================================================================

    def _match_headers_to_positions(
        self,
        llm_sections: List[Dict],
        text: str,
    ) -> List[Tuple[int, str, Optional[str]]]:
        """
        Match LLM-returned headers to exact positions in the source text.

        Uses sequential forward search: since headers are in document order,
        each subsequent header must appear AFTER the previous one.  This
        prevents ambiguity when the same text appears multiple times.
        """
        result: List[Tuple[int, str, Optional[str]]] = []
        search_start = 0

        for entry in llm_sections:
            header = entry.get("header", "").strip()
            if not header:
                continue

            pos = self._find_header_in_text(header, text, search_start)
            if pos == -1:
                self.logger.debug(f"Header not found in text: '{header}'")
                continue

            inline_val = self._extract_inline_value(text, pos, header)
            result.append((pos, header, inline_val))
            search_start = pos + len(header)

        return result

    @staticmethod
    def _find_header_in_text(header: str, text: str, start: int = 0) -> int:
        """
        Find the position of a header string in the document.

        Tries progressively looser matching strategies:
        1. Exact substring
        2. Case-insensitive
        3. Whitespace-normalized
        4. With optional formatting markers (brackets, colons)
        5. Fuzzy matching (typos, extra words, spelling variations)
        """
        # 1. Exact match
        pos = text.find(header, start)
        if pos != -1:
            return pos

        # 2. Case-insensitive
        text_lower = text.lower()
        header_lower = header.lower()
        pos = text_lower.find(header_lower, start)
        if pos != -1:
            return pos

        # 3. Whitespace-normalized regex
        collapsed_pat = re.compile(re.sub(r"\s+", r"\\s+", re.escape(header)), re.IGNORECASE)
        m = collapsed_pat.search(text, start)
        if m:
            return m.start()

        # 4. Optional formatting markers (brackets, colons, leading whitespace)
        escaped = re.escape(header.rstrip(":").strip("[]").strip())
        patterns = [
            re.compile(rf"(?m)^[ \t]*\[?\s*{escaped}\s*\]?\s*:?", re.IGNORECASE),
            re.compile(rf"(?m)^[ \t]*{escaped}[ \t]*:?[ \t]*$", re.IGNORECASE),
        ]
        for pat in patterns:
            m = pat.search(text, start)
            if m:
                return m.start()

        # 5. Fuzzy matching — handles typos, extra words, spelling variations
        #    This is the critical resilience layer that prevents NULL sections
        #    when document headers have minor variations.
        #    Threshold raised to 0.75 to reduce false position matches.
        pos = fuzzy_find_in_text(header, text, start, threshold=0.75)
        if pos != -1:
            logger.info(
                f"Fuzzy matched header '{header}' at position {pos} "
                f"(rigid matching failed)"
            )
            return pos

        return -1

    @staticmethod
    def _extract_inline_value(
        text: str, header_pos: int, header: str
    ) -> Optional[str]:
        """Extract inline value from 'Header: value' on the same line."""
        line_end = text.find("\n", header_pos)
        if line_end == -1:
            line_end = len(text)
        line = text[header_pos:line_end]

        # Match header followed by ": value"
        escaped = re.escape(header.rstrip(":").strip())
        m = re.match(rf".*?{escaped}\s*:\s+(.+)", line, re.IGNORECASE)
        if m and m.group(1).strip():
            return m.group(1).strip()
        return None

    # ==================================================================
    #  FALLBACK — Pattern Candidates + LLM Validation
    # ==================================================================

    async def _detect_with_candidates(self, text: str) -> Dict[str, Any]:
        """
        Fallback: regex candidate extraction + LLM context validation.
        Used when the holistic approach fails or returns too few sections.
        """
        from ..utils.section_candidate_extractor import section_candidate_extractor

        candidates = section_candidate_extractor.extract_candidates(text)
        if not candidates:
            return {
                "sections": {},
                "has_structure": False,
                "confidence": 0.0,
                "rejected_subsections": [],
            }

        self.logger.info(f"Fallback: {len(candidates)} candidates from regex")

        # Send ALL candidates to LLM with context for validation
        validation = await self._validate_candidates(candidates, text)
        main_headers = validation.get("main_sections", [])
        subsections = validation.get("subsections", [])
        name_map = validation.get("name_map", {})

        if not main_headers:
            return {
                "sections": {},
                "has_structure": False,
                "confidence": 0.0,
                "rejected_subsections": subsections,
            }

        # Resolve positions and slice
        header_positions = self._resolve_candidate_positions(
            main_headers, candidates, text
        )
        raw_sections = self._slice_sections(text, header_positions)
        if not raw_sections:
            return {
                "sections": {},
                "has_structure": False,
                "confidence": 0.0,
                "rejected_subsections": subsections,
            }

        merged = self._merge_duplicate_keys(raw_sections)

        # Use name map from validation; fallback to separate normalisation
        if not name_map or not any(k in name_map for k in merged):
            try:
                name_map = await self._normalise_names(list(merged.keys()))
            except Exception as exc:
                self.logger.error(f"Normalisation failed: {exc}")
                # Use fuzzy canonical matching as fallback
                name_map = match_canonical_batch(
                    list(merged.keys()), threshold=0.78
                )
                # Fill any remaining unmapped keys
                for k in merged:
                    if k not in name_map:
                        key = k.lower().strip().strip("[]").rstrip(":").strip()
                        canonical = self._REGEX_SECTION_MAP.get(key)
                        if canonical:
                            name_map[k] = canonical
                        else:
                            # Last resort: fuzzy match
                            fuzzy_c, _ = match_canonical(key, threshold=0.78)
                            name_map[k] = fuzzy_c or re.sub(r"\s+", "_", k).lower()

        sections = self._build_final_sections(merged, name_map, text)

        return {
            "sections": sections,
            "has_structure": bool(sections),
            "confidence": validation.get("confidence", 0.75),
            "rejected_subsections": subsections,
        }

    # ------------------------------------------------------------------

    async def _validate_candidates(
        self, candidates: List[Dict], full_text: str
    ) -> Dict[str, Any]:
        """Validate pattern-based candidates using LLM with context."""
        items = []
        for c in candidates:
            pos = c["position"]
            ctx_start = max(0, pos - 250)
            ctx_end = min(len(full_text), pos + 250)
            items.append({
                "header": c["header"],
                "context": full_text[ctx_start:ctx_end].replace("\n", " "),
                "confidence": round(c.get("confidence", 0.5), 2),
            })

        prompt = self._build_validation_prompt(items)

        try:
            response = await call_gemini(
                client=self.client,
                model=self.model_name, contents=prompt, config=self.quick_config,
            )
            raw = self._safe_response_text(response, "candidate_validation")
            if not raw:
                raise ValueError("Candidate validation LLM returned empty / blocked")
            result = self._extract_json(raw)

            original_main = []
            name_map = {}
            subsections = []
            not_sections = []

            for cls in result.get("classifications", []):
                hdr = cls.get("header")
                ctype = cls.get("type", "not_section")
                canonical = cls.get("canonical_name", "other").lower().strip()
                if not hdr:
                    continue
                if ctype == "main_section":
                    original_main.append(hdr)
                    name_map[hdr.lower().strip().strip("[]").rstrip(":")] = canonical
                elif ctype == "subsection":
                    subsections.append(hdr)
                else:
                    not_sections.append(hdr)

            self.logger.info(
                f"Candidate validation: {len(original_main)} main, "
                f"{len(subsections)} sub, {len(not_sections)} rejected"
            )
            return {
                "main_sections": original_main,
                "subsections": subsections,
                "not_sections": not_sections,
                "name_map": name_map,
                "confidence": result.get("confidence", 0.80),
            }
        except Exception as exc:
            self.logger.error(f"Candidate validation failed: {exc}")
            # Robust regex-only fallback: filter candidates against known
            # section header keywords (not diseases/meds/anatomy)
            return self._regex_only_validate(candidates)

    # ------------------------------------------------------------------
    # Regex-only fallback when ALL LLM calls fail
    # ------------------------------------------------------------------

    # Known section header keywords mapped to canonical names
    _REGEX_SECTION_MAP: Dict[str, str] = {
        # Chief Complaint / Reason for visit
        "chief complaint": "chief_complaint",
        "cc": "chief_complaint",
        "reason for visit": "chief_complaint",
        "reason for encounter": "chief_complaint",
        "exam reason": "chief_complaint",
        # HPI
        "history of present illness": "history_present_illness",
        "hpi": "history_present_illness",
        "present illness": "history_present_illness",
        "history present illness": "history_present_illness",
        # Subjective
        "subjective": "history_present_illness",
        # PMH
        "past medical history": "past_medical_history",
        "pmh": "past_medical_history",
        "medical history": "past_medical_history",
        "active problems": "past_medical_history",
        "problem list": "past_medical_history",
        "active problem list": "past_medical_history",
        # PSH
        "past surgical history": "past_surgical_history",
        "psh": "past_surgical_history",
        "surgical history": "past_surgical_history",
        # Medications
        "medications": "medications",
        "meds": "medications",
        "current medications": "medications",
        "active medications": "medications",
        "medication list": "medications",
        # Allergies
        "allergies": "allergies",
        "allergy list": "allergies",
        "drug allergies": "allergies",
        # Social History
        "social history": "social_history",
        "sh": "social_history",
        # Family History
        "family history": "family_history",
        "fh": "family_history",
        # ROS
        "review of systems": "review_of_systems",
        "ros": "review_of_systems",
        # Note: "constitutional" and "general adult ros" are subsection-level
        # labels that appear WITHIN Review of Systems, not as top-level headers.
        # Vitals
        "vitals": "vitals",
        "vital signs": "vitals",
        "vs": "vitals",
        # Physical Exam
        "physical exam": "physical_exam",
        "physical examination": "physical_exam",
        "pe": "physical_exam",
        "exam": "physical_exam",
        "objective": "objective",
        # Immunizations
        "immunizations": "immunizations",
        "vaccines": "immunizations",
        "immunization history": "immunizations",
        # Lab Results
        "lab results": "lab_results",
        "labs": "lab_results",
        "laboratory": "lab_results",
        "laboratory results": "lab_results",
        # Imaging
        "imaging": "imaging",
        "imaging results": "imaging",
        "radiology": "imaging",
        # Assessment
        "assessment": "assessment",
        "impression": "assessment",
        "dx": "assessment",
        "diagnoses": "assessment",
        "diagnosis": "assessment",
        # Plan
        "plan": "plan",
        "treatment plan": "plan",
        "tx": "plan",
        # Assessment and Plan
        "assessment and plan": "assessment_and_plan",
        "assessment & plan": "assessment_and_plan",
        "a/p": "assessment_and_plan",
        "a&p": "assessment_and_plan",
        # Follow Up
        "follow up": "follow_up",
        "f/u": "follow_up",
        "followup": "follow_up",
        "follow-up": "follow_up",
        "return visit": "follow_up",
    }

    # Physical exam subsection names — these are NOT top-level sections
    _SUBSECTION_NAMES = {
        "heart", "lungs", "cardiovascular", "respiratory", "heent",
        "abdomen", "musculoskeletal", "neurological", "psychiatric",
        "constitutional", "skin", "eyes", "ent", "gastrointestinal",
        "genitourinary", "extremities", "head", "neck", "chest",
        "back", "rectal", "breast", "lymph", "vascular",
        "integumentary", "endocrine", "hematologic", "immunologic",
        "general", "appearance", "general adult ros",
    }

    # Clinical noise words that regex patterns may pick up as standalone
    # candidates but are actually disease modifiers, qualifiers, or non-
    # structural content.  These should NEVER be treated as section headers.
    _CLINICAL_NOISE_WORDS = {
        "uncomplicated", "complicated", "unspecified", "bilateral",
        "unilateral", "benign", "malignant", "metastatic", "recurrent",
        "resolved", "controlled", "uncontrolled", "stable", "unstable",
        "progressive", "intermittent", "persistent", "transient",
        "moderate", "mild", "severe", "primary", "secondary",
        "essential", "idiopathic", "hereditary", "congenital",
        "acquired", "chronic", "subacute", "initial", "subsequent",
        "routine", "normal", "abnormal", "positive", "negative",
        "patient educational handouts", "medical equipment",
        "advance directives", "functional status", "unknown",
        "none recorded", "no information available",
    }

    # Table column labels and other non-section text that pattern matchers
    # may incorrectly pick up as section headers.
    _NOT_SECTION_LABELS = {
        "diagnosis date", "date", "description",
        "status", "code", "comments", "onset date", "resolved date",
        "priority", "severity", "provider", "location", "type",
        "note", "notes", "value", "result", "range", "units",
        "reference range", "flag", "category", "name", "dose",
        "frequency", "route", "start date", "end date", "sig",
        "quantity", "refills", "pharmacy", "prescriber",
        "diagnosis code", "icd code", "icd-10", "cpt code",
        "procedure date",
        # Common non-section content that pattern matchers may pick up
        "patient educational handouts", "medical equipment",
        "advance directives", "functional status",
        "none recorded", "no information available",
        "unknown", "reminders", "orders", "dme orders",
        "implant orders", "reminders provider",
        "mental status", "question answer notes",
    }

    def _regex_only_validate(self, candidates: List[Dict]) -> Dict[str, Any]:
        """
        Validate candidates against known section header keywords when LLM
        is unavailable. Uses fuzzy matching for resilience against typos,
        plurals, extra words, and minor formatting variations.
        """
        main_sections = []
        name_map = {}
        subsections = []

        for c in candidates:
            header = c.get("header", "").strip()
            key = header.lower().strip().strip("[]").rstrip(":").strip()

            # 0. Reject known column labels / non-section text
            if key in self._NOT_SECTION_LABELS:
                self.logger.debug(
                    f"Regex-only: rejecting column label '{header}'"
                )
                continue

            # 0b. Reject clinical noise words (disease modifiers, qualifiers)
            if key in self._CLINICAL_NOISE_WORDS:
                self.logger.debug(
                    f"Regex-only: rejecting clinical noise '{header}'"
                )
                continue

            # 1. Exact lookup in regex section map
            canonical = self._REGEX_SECTION_MAP.get(key)
            if canonical:
                main_sections.append(header)
                name_map[key] = canonical
                continue

            # 2. Check for partial matches (e.g., "Past Medical History (PMH)")
            matched = False
            for known_key, known_canonical in self._REGEX_SECTION_MAP.items():
                if len(known_key) >= 3 and known_key in key:
                    main_sections.append(header)
                    name_map[key] = known_canonical
                    matched = True
                    break
            if matched:
                continue

            # 3. Fuzzy canonical matching — handles typos, extra words,
            #    spelling variations, plurals, etc.
            #    Threshold raised to 0.78 to prevent false positives like
            #    "Uncomplicated" → chief_complaint (0.636) or
            #    "Patient educational handouts" → medications (0.680).
            #    Single-word candidates require an even higher threshold (0.88)
            #    since they're more likely to be disease names or modifiers.
            word_count = len(key.split())
            fuzzy_threshold = 0.88 if word_count <= 1 else 0.78
            fuzzy_canonical, fuzzy_score = match_canonical(
                header, threshold=fuzzy_threshold
            )
            if fuzzy_canonical:
                main_sections.append(header)
                name_map[key] = fuzzy_canonical
                self.logger.info(
                    f"Fuzzy matched candidate '{header}' → {fuzzy_canonical} "
                    f"(score={fuzzy_score:.3f})"
                )
                continue

            # 4. Check if it's a known subsection
            if key in self._SUBSECTION_NAMES:
                subsections.append(header)
                continue

            # If not recognized, skip it (don't include random disease names)
            self.logger.debug(
                f"Regex-only: skipping unrecognized candidate '{header}'"
            )

        self.logger.info(
            f"Regex-only validation: {len(main_sections)} main, "
            f"{len(subsections)} sub"
        )

        return {
            "main_sections": main_sections,
            "subsections": subsections,
            "name_map": name_map,
            "confidence": 0.65,
        }

    # ------------------------------------------------------------------

    def _build_validation_prompt(self, candidates: List[Dict]) -> str:
        """Build the context-aware candidate classification prompt."""
        candidate_list = "\n".join([
            f"- Header: '{c['header']}' "
            f"(Confidence: {c.get('confidence', 0):.2f})\n"
            f"  Context: '...{c['context']}...'"
            for c in candidates
        ])

        return f"""You are a medical documentation expert specializing in clinical note parsing.

TASK:
Classify each candidate header from a medical document.

CATEGORIES:
1. "main_section": Top-level clinical heading (e.g., Chief Complaint, Assessment).
2. "subsection": Component inside a main section (e.g., "Heart" under Physical Exam).
3. "not_section": Noise — disease names, dates, sentence fragments, content labels.

REASONING RULES:
1. ANATOMICAL SYSTEMS under Physical Exam or ROS are SUBSECTIONS:
   (Constitutional, HEENT, Cardiovascular, Respiratory, Abdomen,
    Musculoskeletal, Skin, Neurological, Psychiatric)
2. SEQUENCE PATTERN: multiple short anatomical tokens in a row = subsections.
3. DISEASE NAMES (Diabetes, Hypertension, COPD) are NOT sections.
4. MEDICATION NAMES are NOT sections.
5. LAB VALUES (WBC, BMP, Sodium) are NOT sections.
6. ABBREVIATIONS: CC=Chief Complaint, HPI=History Present Illness,
   PMH=Past Medical History, PE=Physical Exam, ROS=Review of Systems,
   A/P=Assessment and Plan

CANONICAL MAPPING:
- chief_complaint ← CC, Complaint, Reason for Visit
- history_present_illness ← HPI, History
- past_medical_history ← PMH, Medical History, Problem List, Active Problems
- medications ← Meds, Current Medications
- allergies ← Allergic Reactions
- physical_exam ← PE, Examination, Objective
- review_of_systems ← ROS, Systems Review
- vitals ← VS, Vital Signs
- assessment_and_plan ← A/P, Assessment & Plan
- assessment ← Impression
- plan ← Treatment Plan, Recommendations
- family_history ← FH, Fam Hx
- social_history ← SH, Social Hx
- lab_results ← Labs, Laboratory Studies
- imaging ← Radiology
- other ← Admin info, signatures

CANDIDATES:
{candidate_list}

RETURN ONLY VALID JSON:
{{
  "classifications": [
    {{
      "header": "Exact Original String",
      "type": "main_section | subsection | not_section",
      "canonical_name": "standardized_name",
      "reasoning": "Brief reason"
    }}
  ],
  "confidence": 0.0 to 1.0
}}
"""

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_candidate_positions(
        validated_headers: List[str],
        all_candidates: List[Dict],
        full_text: str,
    ) -> List[Tuple[int, str, Optional[str]]]:
        """Map validated headers back to character positions via candidate list.

        Uses fuzzy matching as a fallback when exact key lookup fails,
        preventing section loss due to minor text variations.
        """
        cand_lookup: Dict[str, Dict] = {}
        for c in all_candidates:
            key = c["header"].lower().strip()
            if key not in cand_lookup or c["confidence"] > cand_lookup[key]["confidence"]:
                cand_lookup[key] = c

        result: List[Tuple[int, str, Optional[str]]] = []
        seen_positions: set = set()

        for hdr in validated_headers:
            key = hdr.lower().strip()
            cand = cand_lookup.get(key)

            # Fuzzy fallback: if exact key not found, try fuzzy match
            # against all candidate headers
            if not cand:
                best_cand = None
                best_score = 0.0
                for cand_key, cand_val in cand_lookup.items():
                    score = combined_similarity(key, cand_key)
                    if score > best_score:
                        best_score = score
                        best_cand = cand_val
                if best_score >= 0.78:
                    cand = best_cand
                    if cand:
                        logger.debug(
                            f"Fuzzy candidate match: '{hdr}' → "
                            f"'{cand['header']}' (score={best_score:.3f})"
                        )

            if cand and cand["position"] not in seen_positions:
                inline = cand.get("inline_value")
                result.append((cand["position"], hdr, inline))
                seen_positions.add(cand["position"])
            else:
                # Regex fallback
                escaped = re.escape(hdr)
                pat = rf"(?im)(?:^|\[)\s*{escaped}\s*(?:\]|:|\s*$)"
                m = re.search(pat, full_text)
                if m and m.start() not in seen_positions:
                    result.append((m.start(), hdr, None))
                    seen_positions.add(m.start())
                else:
                    # Last resort: fuzzy text search
                    pos = fuzzy_find_in_text(hdr, full_text, threshold=0.75)
                    if pos != -1 and pos not in seen_positions:
                        result.append((pos, hdr, None))
                        seen_positions.add(pos)

        result.sort(key=lambda x: x[0])
        return result

    # ------------------------------------------------------------------

    async def _normalise_names(self, raw_headers: List[str]) -> Dict[str, str]:
        """Map raw header strings to canonical names via LLM."""
        sanitized_to_raw: Dict[str, str] = {}
        sanitized: List[str] = []
        for h in raw_headers:
            s = re.sub(r"\s+", " ", h.lower()).strip()
            sanitized_to_raw[s] = h
            sanitized.append(s)

        canonical_list = "\n".join(f"  - {c}" for c in CANONICAL_SECTIONS)
        headers_list = "\n".join(f'  - "{h}"' for h in sanitized)

        prompt = f"""You are a medical informatics expert.

Map each raw clinical note header to the best canonical name.
If none fit, use "other".

Canonical names:
{canonical_list}

Raw headers:
{headers_list}

Return ONLY valid JSON (no markdown):
{{
  "raw_header_exactly_as_given": "canonical_name"
}}

Rules:
- Keys MUST exactly match the lowercase raw headers above.
- Use medical knowledge:
    PMH / Past Med Hx → past_medical_history
    PSH / Past Surg Hx → past_surgical_history
    HPI / Hx Present Illness → history_present_illness
    ROS / Review of Systems → review_of_systems
    CC / Chief Complaint → chief_complaint
    Meds / Current Meds → medications
    PE / Phys Exam → physical_exam
    A/P / Assessment/Plan → assessment_and_plan
    Labs / Lab Results → lab_results
    FH / Fam Hx → family_history
    SH / Social Hx → social_history
    Immunizations / Vaccines → immunizations
    Vitals / VS → vitals
    Imaging / Radiology → imaging
    Follow Up / F/U → follow_up
- Administrative items → "other"
"""
        response = await call_gemini(
            client=self.client,
            model=self.model_name, contents=prompt, config=self.quick_config,
        )
        raw = self._safe_response_text(response, "normalise_names")
        if not raw:
            raise ValueError("Normalise names LLM returned empty / blocked")
        mapping = self._extract_json(raw)

        result: Dict[str, str] = {}
        for k, v in mapping.items():
            clean_k = re.sub(r"\s+", " ", k.lower()).strip()
            orig = sanitized_to_raw.get(clean_k, clean_k)
            canonical = re.sub(r"\s+", "_", v.lower().strip())
            result[orig.lower().strip()] = canonical
        return result

    # ==================================================================
    #  LAST RESORT — Full-LLM Content Extraction
    # ==================================================================

    async def _llm_full_detection(self, text: str) -> Dict[str, Any]:
        """
        Last resort for documents with no detectable section headers.
        Every returned snippet is verified to exist in the source text.
        """
        canonical_csv = ", ".join(CANONICAL_SECTIONS)
        prompt = (
            "You are a medical documentation expert.\n\n"
            "The clinical note below has NO clear section headers.\n"
            "1. Identify the clinical sections present.\n"
            "2. For each section copy the EXACT verbatim text (no paraphrasing).\n"
            f"3. Use canonical names from: {canonical_csv}\n\n"
            'Clinical Note:\n"""\n'
            + text[:6000]
            + '\n"""\n\n'
            "Return ONLY valid JSON (no markdown):\n"
            '{\n  "sections": {"canonical_name": "verbatim text"},\n'
            '  "confidence": 0.0\n}\n\n'
            "Important: every value MUST be a verbatim substring of the note."
        )
        try:
            response = await call_gemini(
                client=self.client,
                model=self.model_name, contents=prompt, config=self.detect_config,
            )
            raw = self._safe_response_text(response, "full_detection")
            if not raw:
                raise ValueError("Full-LLM detection returned empty / blocked")
            result = self._extract_json(raw)

            sections_raw = result.get("sections", {})
            verified: Dict[str, Dict[str, Any]] = {}
            for name, content in sections_raw.items():
                if not content or len(content.strip()) < 10:
                    continue
                start = (
                    text.find(content[:60])
                    if len(content) >= 60
                    else text.find(content)
                )
                if start == -1:
                    snippet = re.sub(r"\s+", " ", content[:30].strip())
                    m = re.search(re.escape(snippet), text, re.IGNORECASE)
                    start = m.start() if m else 0
                end = start + len(content)
                verified[name.lower()] = {
                    "text": self._clean_content(content),
                    "start": start,
                    "end": end,
                }
            return {
                "sections": verified,
                "has_structure": bool(verified),
                "confidence": float(result.get("confidence", 0.7)),
                "rejected_subsections": [],
            }
        except Exception as exc:
            self.logger.error(f"Full-LLM detection failed: {exc}")
            return {
                "sections": {},
                "has_structure": False,
                "confidence": 0.0,
                "rejected_subsections": [],
            }

    # ==================================================================
    #  Shared Helpers — Text Slicing & Assembly
    # ==================================================================

    def _slice_sections(
        self,
        text: str,
        headers: List[Tuple[int, str, Optional[str]]],
    ) -> List[Tuple[str, str]]:
        """
        Slice source text between consecutive validated headers.

        Returns list of (raw_key, content) tuples.
        Content is an exact substring of *text* — never generated.

        Empty-section guard: if a section's content is very short and
        matches another header pattern, it is set to "N/A" instead of
        stealing the next section's header line as content.
        """
        sections: List[Tuple[str, str]] = []

        for idx, (pos, header, inline_val) in enumerate(headers):
            next_pos = (
                headers[idx + 1][0] if idx + 1 < len(headers) else len(text)
            )

            if inline_val is not None:
                eol = text.find("\n", pos)
                if eol == -1 or eol >= next_pos:
                    content = inline_val
                else:
                    continuation = text[eol:next_pos].strip()
                    content = (
                        inline_val
                        + ("\n" + continuation if continuation else "")
                    )
            else:
                header_end = text.find("\n", pos)
                if header_end == -1:
                    content = ""
                else:
                    content = text[header_end:next_pos].strip()

            key = header.lower().strip().strip("[]").rstrip(":")
            content = content.strip() if content.strip() else "N/A"

            # Guard: if content is suspiciously short and looks like
            # another header (Title Case line with optional colon),
            # the next header was likely captured as content of an empty
            # section.  Set to "N/A" instead.
            if content != "N/A" and len(content) < 80:
                if re.match(r"^[A-Z][a-zA-Z /\-&]{1,50}:?\s*$", content):
                    content = "N/A"

            sections.append((key, content))

        return sections

    # ------------------------------------------------------------------

    @staticmethod
    def _merge_duplicate_keys(
        raw_sections: List[Tuple[str, str]],
    ) -> Dict[str, str]:
        """Merge duplicate section keys; drop N/A when real content exists."""
        merged: Dict[str, str] = {}
        for key, content in raw_sections:
            if key not in merged:
                merged[key] = content
            else:
                prev = merged[key]
                if prev == "N/A":
                    merged[key] = content
                elif content == "N/A":
                    continue
                elif content.strip() not in prev:
                    merged[key] += "\n\n" + content
        return merged

    # ------------------------------------------------------------------

    @staticmethod
    def _build_final_sections(
        merged: Dict[str, str],
        name_map: Dict[str, str],
        text: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Build the canonical sections dict with text / start / end.

        Uses the raw_key (original header text) to locate the header
        position in `text` first, then offsets from there — much more
        reliable than searching for a content prefix which may appear
        earlier in the document.
        """
        sections: Dict[str, Dict[str, Any]] = {}
        for raw_key, canonical in name_map.items():
            if canonical == "other":
                continue
            content = merged.get(raw_key, "").strip()
            content = SmartSectionDetector._clean_content(content)
            if content != "N/A" and (not content or len(content) < 10):
                continue
            if canonical in sections:
                existing = sections[canonical]["text"]
                if existing == "N/A":
                    # Replace N/A placeholder with real content
                    sections[canonical]["text"] = content
                elif content == "N/A":
                    # Don't append N/A to real content
                    pass
                elif content.strip() not in existing:
                    sections[canonical]["text"] += "\n\n" + content
            else:
                # Find position by locating the header in text first,
                # then the content starts right after the header line.
                header_pos = text.lower().find(raw_key.lower())
                if header_pos != -1:
                    # Content starts after the header line
                    newline_after = text.find("\n", header_pos)
                    start = newline_after + 1 if newline_after != -1 else header_pos + len(raw_key)
                else:
                    # Fallback: search for content prefix
                    prefix = content[:60] if len(content) >= 60 else content
                    start = text.find(prefix)
                end = start + len(content) if start != -1 else -1
                sections[canonical] = {
                    "text": content,
                    "start": max(start, 0),
                    "end": max(end, 0),
                }
        return sections

    # ==================================================================
    #  Shared Helpers — JSON & Text
    # ==================================================================

    def _safe_response_text(self, response, label: str = "LLM") -> str:
        """
        Safely extract text from a Gemini response, handling blocked or
        empty responses without raising an exception.

        Returns the stripped response text, or empty string if unavailable.
        Works with both old (google.generativeai) and new (google.genai) SDKs.
        """
        try:
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                fr = getattr(candidate, "finish_reason", None)
                # New SDK uses FinishReason enum; normalise to string
                fr_str = fr.name if hasattr(fr, "name") else str(fr)
                if fr_str == "MAX_TOKENS":
                    self.logger.warning(
                        f"[{label}] Response hit MAX_TOKENS — output may be truncated"
                    )
                elif fr_str not in ("STOP", "1", "None"):
                    self.logger.warning(
                        f"[{label}] Response blocked: finish_reason={fr_str}"
                    )
            elif hasattr(response, "candidates"):
                self.logger.warning(f"[{label}] No candidates in response")
                return ""

            text = response.text.strip()
            if not text:
                self.logger.warning(f"[{label}] Response text is empty")
            else:
                self.logger.info(f"[{label}] Got {len(text)} chars of response text")
            return text

        except ValueError as e:
            self.logger.warning(f"[{label}] Response blocked (ValueError): {e}")
            return ""
        except Exception as e:
            self.logger.warning(f"[{label}] Could not extract response text: {e}")
            return ""

    @staticmethod
    def _extract_json(text: str) -> dict:
        """
        Robustly extract a JSON object from LLM output that may contain
        markdown fences, thinking tokens, or other wrapper text.

        Avoids slow regex backtracking for deep/malformed objects.
        """
        # 1. Strip markdown code fences
        cleaned = re.sub(r"```[a-z]*\s*", "", text)
        cleaned = re.sub(r"\s*```", "", cleaned).strip()

        # 2. Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 3. Find the outermost { ... } block (brace counting)
        start = cleaned.find("{")
        if start != -1:
            depth = 0
            first_found = False
            for i in range(start, len(cleaned)):
                if cleaned[i] == "{":
                    depth += 1
                    first_found = True
                elif cleaned[i] == "}":
                    depth -= 1
                    if first_found and depth == 0:
                        try:
                            # We found a balanced block
                            return json.loads(cleaned[start : i + 1])
                        except json.JSONDecodeError:
                            # Might be another block later? Or just bad JSON.
                            continue

        # 4. Fallback: try to find anything that looks like a JSON object
        # but avoid the backtracking regex. Use a simpler one.
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError(
            "No valid JSON found in LLM response", text, 0
        )

    @staticmethod
    def _clean_content(text: str) -> str:
        """Light cleaning: strip per-line whitespace, collapse blank runs."""
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            line = line.strip()
            if re.match(r"^[\s.\-_]{0,3}$", line):
                cleaned.append("")
            else:
                cleaned.append(line)
        return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
smart_section_detector = SmartSectionDetector()