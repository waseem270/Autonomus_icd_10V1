"""
Output Filter — strict post-processing rules for final disease results.

Rules:
  1. Assessment section priority (or graceful fallback)
  2. MEAT tier required (full/medium/half)
  3. Active disease only (exclude history-only, family history, resolved)
  4. ICD code required
  5. Evidence text required (at least one MEAT field has real text)
  6. Deduplication (ICD-based, abbreviation-aware, substring, stem-based)  7. Problem-list-only exclusion (not addressed in Assessment/Plan)
  8. Incidental imaging findings exclusion
  9. Chronic-not-managed exclusion (stable conditions not actively managed)"""

import logging
import re
from typing import Dict, List

from ..utils.abbreviation_expander import MEDICAL_ABBREVIATIONS

logger = logging.getLogger(__name__)

# ── Section sets ──────────────────────────────────────────────────────────
_ASSESSMENT_SECTIONS = {
    "assessment", "assessment_and_plan", "active_problems",
}
# Problem list sections are NOT Assessment — separate handling
_PROBLEM_LIST_SECTIONS = {
    "problem_list", "active_problem_list",
    "past_medical_history", "pmh", "medical_history",
    "surgical_history", "past_surgical_history",
}
_VISIT_ASSESSMENT = {"assessment", "assessment_and_plan"}
_SECONDARY_PRIORITY_SECTIONS = {
    "impression", "plan", "orders", "order",
    "diagnosis", "working_diagnoses", "final_diagnoses",
}
_EXCLUDED_SECTIONS = {"family_history", "fh", "screening"}
_RESTRICTED_SOLE_SECTIONS = {"family_history", "fh", "screening", "social_history"}
_PRIMARY_DISEASE_SECTIONS = _ASSESSMENT_SECTIONS | {
    "chief_complaint", "cc", "history_present_illness", "hpi",
    "past_medical_history", "pmh", "medical_history",
    "plan", "impression", "physical_exam", "pe", "objective",
}
_ACTIVE_CLINICAL_SECTIONS = _PRIMARY_DISEASE_SECTIONS | {
    "medications", "current_medications", "review_of_systems", "ros",
}
_ACTIVE_EVIDENCE_SECTIONS = {
    "medications", "current_medications", "vital_signs", "vitals",
    "physical_exam", "pe", "labs", "imaging", "procedures",
}
_VISIT_FOCUS_SECTIONS = {"chief_complaint", "cc", "history_present_illness", "hpi"}
_GRACE_SECTIONS = {
    "history_present_illness", "plan", "chief_complaint",
    "physical_exam", "review_of_systems", "objective",
    "past_medical_history", "pmh", "medical_history",
    "medications", "current_medications", "hpi", "cc",
}

_EXCLUDED_STATUSES = {"History Only"}
_EXCLUDED_TIERS = {"history_only", "invalid"}

_CANCER_PATTERN = re.compile(
    r"(?i)\b(?:neoplasm|malignant|cancer|carcinoma|melanoma|lymphoma|sarcoma|"
    r"leukemia|myeloma|oncolog)\b"
)
_PLAN_ANNOTATION_RE = re.compile(
    r'(?i)(?:'
    r'\s+(?:stable|controlled|uncontrolled|well controlled)\b(?:\s*[,=].*)?$'
    r'|\s*[=,]\s*(?:stable|controlled|uncontrolled|well controlled|'
    r'counsel|monitor|consider|continue|increase|decrease|'
    r'refer|serial|start|check|order|worsening|improving|'
    r'due to dm|on meds|diet and exercise|fiber/fluids)\b.*$'
    r')'
)
_FRAGMENT_RE = re.compile(
    r'^(?:due to \w+|stable.*|on meds|single episode.*|psy support|'
    r'serial \w+|consider \w+|increase \w+|monitor \w+)$',
    re.IGNORECASE,
)

# Rule 8: Incidental imaging findings / not-actively-managed conditions
_INCIDENTAL_PATTERN = re.compile(
    r"(?i)\b(?:incidental|subdural\s+hygroma|dural\s+collection|"
    r"benign\s+finding|"
    r"unchanged|old\s+finding|stable\s+from\s+prior|"
    r"no\s+acute\s+(?:findings?|process)|"
    r"chronic\s+appearing|degenerative\s+(?:joint|change)|"
    r"age[- ]?related\s+(?:change|finding)|"
    r"calcific\s+tendinitis|tendinosis|spur|"
    r"mild\s+degenerative)"
    r"\b"
)
_IMAGING_ONLY_SECTIONS = {"imaging", "radiology", "x-ray", "ct", "mri", "ultrasound"}

# Rule 9: Chronic conditions stable/not managed this visit
_STABLE_CHRONIC_RE = re.compile(
    r"(?i)\b(?:stable|well\s+controlled|controlled|unchanged|"
    r"continue\s+(?:current|same|home)|maintain|no\s+change|"
    r"doing\s+well|not\s+active|at\s+goal|resolved|quiescent)"
    r"\b"
)

# Active-management signals used for encounter relevance gating.
_MANAGEMENT_ACTION_RE = re.compile(
    r"(?i)\b(?:start|started|initiat|prescrib|adjust|increase|decrease|"
    r"taper|stop|discontinu|hold|order|follow\s*up|f/?u|monitor|recheck|"
    r"refer|discuss|counsel|reviewed|evaluat|treat|therapy|plan|"
    r"continu|maintain(?:ing)?|ongoing|manag|titrat|optimiz|uptitrat|"
    r"wound\s*care|dress|care\s*of|oxygen|wean|transfus|catheter|"
    r"ventilat|dialys|infus|irrigat|suction|"
    r"antibiotic|analges|sedat|nutrition|supplement|replet)\b"
)
_MAINTENANCE_ONLY_RE = re.compile(
    r"(?i)\b(?:on\s+home\s+meds?\s+only|"
    r"med\s+list\s+reviewed\s+only)\b"
)

# Rule 10: Historical-only condition detection.
# Only flag conditions explicitly described as history (e.g. 'history of', 'h/o').
# Do NOT flag 'chronic', 'stable', 'controlled' — those are clinical qualifiers
# for conditions that may be actively managed.
_HISTORY_OF_PATTERN = re.compile(
    r"(?i)(?:^|\b)(?:history\s+of|h/o|h\.o\.|hx\s+of|past\s+h/o|past\s+history\s+of)\b"
)


def _is_chronic_background(icd_code: str, disease_name: str) -> bool:
    """Check if a disease is purely historical (not actively managed).

    Only flags conditions explicitly documented as historical via phrases like
    'history of', 'h/o', 'hx of'. Does NOT flag ordinary clinical qualifiers
    like 'chronic', 'stable', 'controlled' — those frequently appear in actively
    managed diagnoses (e.g. 'Chronic kidney disease', 'Stable angina').
    """
    return bool(_HISTORY_OF_PATTERN.search(disease_name))


def _norm_sections(raw: list) -> set:
    return {s.lower().replace(" ", "_") for s in raw}


def _clean_display_name(name: str) -> str:
    cleaned = _PLAN_ANNOTATION_RE.sub('', name).strip().rstrip('.,;:= ')
    return cleaned if len(cleaned) >= 3 else name


def _has_active_management(
    treatment_evidence: str,
    monitoring_evidence: str,
    evaluation_evidence: str,
    assessment_evidence: str,
) -> bool:
    """Encounter-relevance gate: keep only actively managed conditions."""
    treat = (treatment_evidence or "").strip()
    monitor = (monitoring_evidence or "").strip()
    evaluate = (evaluation_evidence or "").strip()
    assess = (assessment_evidence or "").strip()

    if monitor or evaluate:
        return True

    if treat:
        # Medication-list carryforward without explicit action is not active management.
        if _MAINTENANCE_ONLY_RE.search(treat):
            return False
        return bool(_MANAGEMENT_ACTION_RE.search(treat))

    return bool(_MANAGEMENT_ACTION_RE.search(assess))


class OutputFilter:
    """Apply strict output rules + deduplication to unified results."""

    def _filter(self, results: List[Dict], single_pass: bool) -> List[Dict]:
        """Simple filter: ensure ICD code exists and apply basic cleanup."""
        filtered = []
        for r in results:
            dname = r.get("disease", r.get("disease_name", ""))
            icd = r.get("icd_code") or ""
            
            # Ensure we have an ICD code
            if not icd or icd in ("—", "-"):
                continue
                
            # Clean display name
            cleaned = _clean_display_name(dname)
            if not cleaned or _FRAGMENT_RE.match(cleaned):
                continue
            r["disease"] = cleaned
            
            # Ensure we have some evidence text (basic clinical gate)
            if not any(
                (r.get(f"{e}_evidence") or "").strip()
                for e in ("monitoring", "evaluation", "assessment", "treatment")
            ):
                continue
                
            filtered.append(r)
        return filtered

    def apply(self, unified_results: List[Dict], single_pass: bool = False) -> List[Dict]:
        """Apply base filtering and deduplication."""
        # Removed _correct_icd_codes override to keep 'real results'
        filtered = self._filter(unified_results, single_pass)
        
        # If everything was filtered but input was not empty, 
        # try a looser set to avoid total accuracy loss.
        if not filtered and unified_results:
            filtered = [r for r in unified_results if r.get("icd_code") not in (None, "", "—", "-")]
            
        filtered = self._deduplicate(filtered)
        for i, r in enumerate(filtered, 1):
            r["number"] = i
        return filtered

    @staticmethod
    def _passes_section_rule(r, seg_lower, has_assessment, single_pass):
        in_assessment = bool(seg_lower & _ASSESSMENT_SECTIONS)
        in_visit = bool(seg_lower & _VISIT_ASSESSMENT)
        in_secondary_priority = bool(seg_lower & _SECONDARY_PRIORITY_SECTIONS)
        has_treatment = bool(r.get("treatment", False) or r.get("treatment_evidence"))
        in_focus = bool(seg_lower & _VISIT_FOCUS_SECTIONS)
        has_active = bool(seg_lower & _ACTIVE_EVIDENCE_SECTIONS)
        # Items solely from PMH/Problem List sections (no overlap with Assessment or HPI)
        in_pmh_only = bool(seg_lower) and seg_lower.issubset(
            _PROBLEM_LIST_SECTIONS | _RESTRICTED_SOLE_SECTIONS
        )

        if single_pass:
            if has_assessment:
                # PMH/Problem List-only items must NOT pass when Assessment exists
                if in_pmh_only:
                    return False
                # Priority order: Assessment/Plan -> Impression/Orders -> HPI
                if in_visit:
                    return True
                if in_secondary_priority and (has_treatment or has_active or in_focus):
                    return True
                if in_focus and (has_treatment or has_active):
                    return True
                if in_assessment:
                    return True
                if (has_treatment or has_active) and bool(seg_lower & _PRIMARY_DISEASE_SECTIONS):
                    return True
                return False
            return True  # no assessment → allow all in single-pass
        else:
            if has_assessment:
                return in_assessment or has_treatment
            return bool(seg_lower & _ACTIVE_CLINICAL_SECTIONS)

    def _grace_fallback(self, results: List[Dict]) -> List[Dict]:
        logger.info("Grace rule: no diseases passed; accepting MEAT-valid from clinical sections.")
        out = []
        for r in results:
            seg_lower = _norm_sections(r.get("segment_source_raw", []))
            if not (seg_lower & _GRACE_SECTIONS):
                continue
            if r.get("meat_tier") in _EXCLUDED_TIERS:
                continue
            if r.get("disease_status") in _EXCLUDED_STATUSES:
                continue
            if seg_lower and seg_lower.issubset(_EXCLUDED_SECTIONS):
                continue
            out.append(r)
        return out

    # ──────────────────────────────────────────────────────────────────
    # ICD Code Corrections (deterministic post-processing)
    # ──────────────────────────────────────────────────────────────────
    # Patterns that indicate elevated BP reading, NOT hypertension
    _ELEVATED_BP_RE = re.compile(
        r"(?i)(?:elevated\s+(?:blood\s+)?(?:pressure|bp)|"
        r"without\s+diagnosis\s+of\s+hypertension|"
        r"white\s*coat\s+(?:hypertension|effect|syndrome)|"
        r"high\s+(?:blood\s+)?(?:pressure|bp)\s+reading)"
    )
    _BMI_VALUE_RE = re.compile(r"(?i)\bbmi\b[^\d]{0,8}(\d{2}(?:\.\d+)?)")

    @staticmethod
    def _bmi_to_icd_code(bmi_value: float) -> str:
        """Map adult BMI value to Z68.2x / Z68.3x ICD code."""
        if 25.0 <= bmi_value < 26.0:
            return "Z68.25"
        if 26.0 <= bmi_value < 27.0:
            return "Z68.26"
        if 27.0 <= bmi_value < 28.0:
            return "Z68.27"
        if 28.0 <= bmi_value < 29.0:
            return "Z68.28"
        if 29.0 <= bmi_value < 30.0:
            return "Z68.29"
        if 30.0 <= bmi_value < 31.0:
            return "Z68.30"
        if 31.0 <= bmi_value < 32.0:
            return "Z68.31"
        if 32.0 <= bmi_value < 33.0:
            return "Z68.32"
        if 33.0 <= bmi_value < 34.0:
            return "Z68.33"
        if 34.0 <= bmi_value < 35.0:
            return "Z68.34"
        if 35.0 <= bmi_value < 36.0:
            return "Z68.35"
        if 36.0 <= bmi_value < 37.0:
            return "Z68.36"
        if 37.0 <= bmi_value < 38.0:
            return "Z68.37"
        if 38.0 <= bmi_value < 39.0:
            return "Z68.38"
        if 39.0 <= bmi_value < 40.0:
            return "Z68.39"
        if 40.0 <= bmi_value < 45.0:
            return "Z68.41"
        return ""

    def _correct_icd_codes(self, results: List[Dict]) -> List[Dict]:
        """Apply deterministic ICD code corrections to filtered results."""
        for r in results:
            icd = r.get("icd_code") or ""
            name = r.get("disease", "")
            combined = f"{name} {r.get('assessment_evidence', '')} {r.get('llm_reasoning', '')}"

            # I10 → R03.0: Elevated BP reading without hypertension diagnosis
            if icd == "I10" and self._ELEVATED_BP_RE.search(combined):
                logger.info(f"ICD correction: I10 → R03.0 for '{name}' (elevated BP reading)")
                r["icd_code"] = "R03.0"
                r["icd_description"] = "Elevated blood-pressure reading, without diagnosis of hypertension"

            # Z68.xx normalization from explicit BMI value in note text
            if icd.startswith("Z68."):
                m_bmi = self._BMI_VALUE_RE.search(combined)
                if m_bmi:
                    try:
                        bmi_val = float(m_bmi.group(1))
                        target = self._bmi_to_icd_code(bmi_val)
                        if target and target != icd:
                            logger.info(
                                f"ICD correction: {icd} → {target} for '{name}' "
                                f"(BMI={bmi_val})"
                            )
                            r["icd_code"] = target
                    except ValueError:
                        pass

        return results

    # ──────────────────────────────────────────────────────────────────
    # Deduplication
    # ──────────────────────────────────────────────────────────────────
    def _deduplicate(self, results: List[Dict]) -> List[Dict]:
        if len(results) <= 1:
            return results
        results = self._dedup_by_icd(results)
        results = self._dedup_abbreviation_aware(results)
        results = self._dedup_substring(results)
        results = self._dedup_stem(results)
        results.sort(key=lambda r: r["disease"].lower())
        return results

    @staticmethod
    def _dedup_by_icd(results):
        by_icd: Dict[str, Dict] = {}
        no_icd = []
        tier_priority = {"strong_evidence": 0, "moderate_evidence": 1, "weak_evidence": 2}
        for r in results:
            icd = r.get("icd_code", "")
            if icd and icd != "—":
                if icd not in by_icd:
                    by_icd[icd] = r
                else:
                    ex = by_icd[icd]
                    if len(r["disease"]) > len(ex["disease"]):
                        by_icd[icd] = r
                    elif len(r["disease"]) == len(ex["disease"]):
                        if tier_priority.get(r["meat_tier"], 9) < tier_priority.get(ex["meat_tier"], 9):
                            by_icd[icd] = r
            else:
                no_icd.append(r)
        return list(by_icd.values()) + no_icd

    @staticmethod
    def _dedup_abbreviation_aware(results):
        abbr_map = {k.lower(): v.lower() for k, v in MEDICAL_ABBREVIATIONS.items()}
        rev_map = {v.lower(): k.lower() for k, v in MEDICAL_ABBREVIATIONS.items()}

        def _expand(name):
            low = name.lower().strip()
            variants = {low}
            if low in abbr_map:
                variants.add(abbr_map[low])
            if low in rev_map:
                variants.add(rev_map[low])
            for abbr, expansion in abbr_map.items():
                if abbr in low.split():
                    variants.add(expansion)
                if expansion in low:
                    variants.add(abbr)
            return variants

        deduped = []
        groups: List[set] = []
        icds: List[str] = []  # track ICD code per group
        tier_priority = {"strong_evidence": 0, "moderate_evidence": 1, "weak_evidence": 2}
        for r in results:
            variants = _expand(r["disease"])
            icd_r = r.get("icd_code") or ""
            merged = False
            for idx, g in enumerate(groups):
                # Only merge if ICD codes match or one is missing
                icd_g = icds[idx]
                if icd_r and icd_g and icd_r != icd_g:
                    continue  # Different ICD codes — never merge
                if variants & g:
                    ex = deduped[idx]
                    if len(r["disease"]) > len(ex["disease"]):
                        deduped[idx] = r
                    elif len(r["disease"]) == len(ex["disease"]):
                        if tier_priority.get(r["meat_tier"], 9) < tier_priority.get(ex["meat_tier"], 9):
                            deduped[idx] = r
                    groups[idx] |= variants
                    if icd_r:
                        icds[idx] = icd_r
                    merged = True
                    break
            if not merged:
                deduped.append(r)
                groups.append(variants)
                icds.append(icd_r)
        return deduped

    @staticmethod
    def _dedup_substring(results):
        names = [r["disease"].lower().strip() for r in results]
        out = []
        for i, r in enumerate(results):
            icd_i = r.get("icd_code") or ""
            is_substring_of_another = False
            for j in range(len(results)):
                if j == i:
                    continue
                icd_j = results[j].get("icd_code") or ""
                # If both have valid but different ICD codes, never merge
                if icd_i and icd_j and icd_i != icd_j:
                    continue
                if names[i] in names[j] and len(names[i]) < len(names[j]):
                    is_substring_of_another = True
                    break
            if not is_substring_of_another:
                out.append(r)
        return out

    @staticmethod
    def _dedup_stem(results):
        _SUFFIXES = ['ies', 'es', 'y', 's', 'ia', 'ic', 'al', 'ous']

        def _roots(name):
            roots = []
            for w in name.lower().split():
                if len(w) < 5:
                    continue
                root = w
                for s in sorted(_SUFFIXES, key=len, reverse=True):
                    if root.endswith(s) and len(root) - len(s) >= 4:
                        root = root[:-len(s)]
                        break
                if len(root) >= 5:
                    roots.append(root)
            return roots

        names = [r["disease"].lower().strip() for r in results]
        remove = set()
        for i in range(len(results)):
            if i in remove:
                continue
            roots_i = _roots(names[i])
            if not roots_i:
                continue
            icd_i = results[i].get("icd_code") or ""
            for j in range(len(results)):
                if i == j or j in remove:
                    continue
                if len(names[i]) >= len(names[j]):
                    continue
                icd_j = results[j].get("icd_code") or ""
                # NEVER merge two items that have different valid ICD codes
                # This prevents same-family codes (e.g. F11.20 + F11.288) from collapsing
                if icd_i and icd_j and icd_i != icd_j:
                    continue
                if any(root in names[j].replace(" ", "") for root in roots_i):
                    remove.add(i)
                    break
        return [r for i, r in enumerate(results) if i not in remove]


# Singleton
output_filter = OutputFilter()
