"""
Medical Coding Rules Engine — deterministic ICD-10 coding compliance.

Enforces real-world medical coding guidelines (CMS / AMA / AHA) that the
LLM-based pipeline cannot guarantee on its own.  Every rule is pure Python
logic — no LLM calls.

Rules implemented:
 1. Documented-Only (no medication/lab inference)
 2. Diagnosis-Over-Symptoms (suppress integral symptoms)
 3. Symptoms-Only-If-No-Diagnosis
 4. Rule-Out / Suspected Filtering (OPD: do not code uncertain dx)
 5. Chronic Condition Retention
 6. Abbreviation Deduplication
 7. Specificity Preference (most specific ICD wins)
 8. Medication ≠ Diagnosis guard
 9. Lab Result ≠ Diagnosis guard
10. Negation Detection
11. Combination Code Handling
12. Etiology + Manifestation sequencing
13. Laterality Detection
14. Excludes1 Conflict Removal
15. Integral-Symptom Removal
16. Primary vs Secondary Ranking
17. Narrative Pharyngitis Backfill (J02.9)
18. CAD Semantic Deduplication
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from ..utils.abbreviation_expander import MEDICAL_ABBREVIATIONS

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Static knowledge tables (deterministic — no LLM)
# ═══════════════════════════════════════════════════════════════════════════

# Rule 4 — phrases that indicate uncertain / suspected diagnosis (OPD logic)
_SUSPECTED_RE = re.compile(
    r"(?i)\b(?:rule\s*out|r/o|ruled\s*out|possible|probable|suspected|"
    r"likely|differential|questionable|uncertain|cannot\s+exclude|"
    r"to\s+be\s+determined|pending\s+workup|presumed)\b"
)

# Rule 8 — Medication names that should NOT be promoted to diagnoses.
# Key = med-like substring (lowercase).  Presence in disease name → reject.
_MEDICATION_MARKERS = {
    "metformin", "lisinopril", "atorvastatin", "omeprazole", "amlodipine",
    "losartan", "metoprolol", "gabapentin", "levothyroxine", "acetaminophen",
    "ibuprofen", "aspirin", "warfarin", "insulin", "albuterol", "prednisone",
    "amoxicillin", "azithromycin", "furosemide", "hydrochlorothiazide",
    "pantoprazole", "sertraline", "fluoxetine", "citalopram", "duloxetine",
    "tramadol", "oxycodone", "hydrocodone", "morphine", "fentanyl",
    "rosuvastatin", "simvastatin", "pravastatin", "carvedilol", "digoxin",
    "spironolactone", "clopidogrel", "apixaban", "rivaroxaban", "enoxaparin",
}

# Rule 9 — Lab-result strings that are NOT diagnoses on their own.
_LAB_INDICATOR_RE = re.compile(
    r"(?i)\b(?:elevated\s|low\s|high\s|abnormal\s|positive\s|negative\s)?"
    r"(?:a1c|hba1c|hemoglobin\s*a1c|creatinine|bun|gfr|egfr|wbc|rbc|"
    r"platelet|troponin|bnp|inr|ptt|pt\b|alt|ast|alkaline\s*phosphatase|"
    r"bilirubin|albumin|sodium|potassium|chloride|calcium|tsh|t3|t4|"
    r"ldl|hdl|triglyceride|glucose|fasting\s*glucose|psa|esr|crp|"
    r"hemoglobin|hematocrit|ferritin|iron|vitamin\s*d|b12)\b"
    r"(?:\s*(?:level|value|result|of|at|is|was|=|:)\s*[\d.]+)?"
)

# Rule 10 — Negation window  (expanded from meat_gate)
_NEGATION_RE = re.compile(
    r"(?i)\b(?:no\s+(?:history\s+of|evidence\s+of|signs?\s+of|diagnosis\s+of)|"
    r"denies|without|negative\s+for|ruled\s+out|not\s+(?:found|identified|present|seen)|"
    r"absence\s+of|free\s+of)\b"
)

# Rule 2 / 15 —  Diagnosis → integral / subsumable symptoms.
# Key = diagnosis regex pattern  →  set of symptom patterns to remove.
_DIAGNOSIS_SUBSUMES_SYMPTOMS: List[Tuple[re.Pattern, Set[str]]] = [
    (re.compile(r"(?i)\bpneumonia\b"),
     {"cough", "fever", "dyspnea", "shortness of breath", "chest pain",
      "tachypnea", "pleuritic pain"}),
    (re.compile(r"(?i)\b(?:urinary\s+tract\s+infection|uti)\b"),
     {"dysuria", "frequency", "urgency", "hematuria", "suprapubic pain", "fever"}),
    (re.compile(r"(?i)\b(?:myocardial\s+infarction|mi|heart\s+attack)\b"),
     {"chest pain", "diaphoresis", "nausea", "jaw pain", "arm pain"}),
    (re.compile(r"(?i)\b(?:congestive\s+heart\s+failure|chf|heart\s+failure)\b"),
     {"dyspnea", "shortness of breath", "edema", "peripheral edema",
      "orthopnea", "paroxysmal nocturnal dyspnea"}),
    (re.compile(r"(?i)\b(?:migraine)\b"),
     {"headache", "nausea", "photophobia", "phonophobia", "aura"}),
    (re.compile(r"(?i)\b(?:appendicitis)\b"),
     {"abdominal pain", "nausea", "vomiting", "fever", "rlq pain"}),
    (re.compile(r"(?i)\b(?:cholecystitis)\b"),
     {"abdominal pain", "ruq pain", "nausea", "vomiting", "fever"}),
    (re.compile(r"(?i)\b(?:acute\s+bronchitis|bronchitis)\b"),
     {"cough", "sore throat", "chest congestion"}),
    (re.compile(r"(?i)\b(?:influenza|flu)\b"),
     {"fever", "cough", "myalgia", "fatigue", "headache", "sore throat"}),
    (re.compile(r"(?i)\b(?:strep\s+pharyngitis|pharyngitis|strep\s+throat)\b"),
     {"sore throat", "odynophagia", "fever"}),
    (re.compile(r"(?i)\b(?:gastroenteritis)\b"),
     {"nausea", "vomiting", "diarrhea", "abdominal pain", "abdominal cramps"}),
    (re.compile(r"(?i)\b(?:copd|chronic\s+obstructive\s+pulmonary\s+disease)\b"),
     {"dyspnea", "shortness of breath", "wheezing", "cough"}),
    (re.compile(r"(?i)\b(?:asthma)\b"),
     {"wheezing", "cough", "dyspnea", "shortness of breath", "chest tightness"}),
    (re.compile(r"(?i)\b(?:dvt|deep\s+vein\s+thrombosis)\b"),
     {"leg swelling", "leg pain", "calf pain", "edema"}),
    (re.compile(r"(?i)\b(?:pulmonary\s+embolism|pe)\b"),
     {"dyspnea", "shortness of breath", "chest pain", "tachycardia", "hemoptysis"}),
    (re.compile(r"(?i)\b(?:cellulitis)\b"),
     {"erythema", "swelling", "warmth", "pain", "fever"}),
    (re.compile(r"(?i)\b(?:pancreatitis)\b"),
     {"abdominal pain", "epigastric pain", "nausea", "vomiting"}),
    (re.compile(r"(?i)\b(?:meningitis)\b"),
     {"headache", "fever", "neck stiffness", "photophobia", "nausea"}),
    (re.compile(r"(?i)\b(?:diverticulitis)\b"),
     {"abdominal pain", "llq pain", "fever", "nausea"}),
]

# Rule 5 — Chronic conditions (ICD categories that are always retained).
_CHRONIC_ICD_PREFIXES = {
    "E08", "E09", "E10", "E11", "E13",   # Diabetes
    "I10", "I11", "I12", "I13",           # Hypertension
    "I25",                                 # Chronic ischemic heart disease
    "I50",                                 # Heart failure
    "J44",                                 # COPD
    "N18",                                 # CKD
    "K21",                                 # GERD
    "M05", "M06",                          # RA
    "G35",                                 # Multiple sclerosis
    "E03",                                 # Hypothyroidism
    "E05",                                 # Hyperthyroidism
    "E78",                                 # Hyperlipidemia
    "G47",                                 # Sleep disorders / OSA
    "J45",                                 # Asthma
    "F32", "F33",                          # Depression
    "F41",                                 # Anxiety
    "N40",                                 # BPH
    "G43",                                 # Migraine
    "M15", "M16", "M17", "M18", "M19",    # Osteoarthritis
}

# Rule 11 — Common combination codes  (disease_a + disease_b → combo ICD).
# Checked AFTER individual mapping so both entries can be collapsed.
_COMBINATION_CODES: List[Dict] = [
    {"terms": [r"(?i)diabetes", r"(?i)neuropathy"],
     "icd": "E11.40", "desc": "Type 2 diabetes mellitus with diabetic neuropathy, unspecified"},
    {"terms": [r"(?i)diabetes", r"(?i)nephropathy|kidney\s+disease"],
     "icd": "E11.22", "desc": "Type 2 diabetes mellitus with diabetic chronic kidney disease"},
    {"terms": [r"(?i)diabetes", r"(?i)retinopathy"],
     "icd": "E11.319", "desc": "Type 2 diabetes mellitus with unspecified diabetic retinopathy without macular edema"},
    {"terms": [r"(?i)diabetes", r"(?i)peripheral\s+(?:vascular|angiopathy)"],
     "icd": "E11.51", "desc": "Type 2 diabetes mellitus with diabetic peripheral angiopathy without gangrene"},
    {"terms": [r"(?i)hypertension|htn", r"(?i)(?:chronic\s+)?kidney\s+disease|ckd"],
     "icd": "I12.9", "desc": "Hypertensive chronic kidney disease with stage 1-4 or unspecified CKD"},
    {"terms": [r"(?i)hypertension|htn", r"(?i)heart\s+(?:failure|disease)|chf"],
     "icd": "I11.0", "desc": "Hypertensive heart disease with heart failure"},
]

# Rule 13 — Laterality keywords
_LATERALITY_RE = re.compile(
    r"(?i)\b(right|left|bilateral|unilateral|rt|lt|r/l|l/r)\b"
)

# Rule 14 — Excludes1 conflict pairs (code A excludes code B).
# More pairs can be added as the ICD dictionary is enriched.
_EXCLUDES1_PAIRS: List[Tuple[str, str]] = [
    ("E11", "E10"),   # T2DM excludes T1DM
    ("I10", "I15"),   # Essential HTN excludes Secondary HTN
    ("J45", "J44"),   # Asthma (certain categories) and COPD overlap
    ("F32", "F33"),   # Single episode depression vs recurrent
]

# ── Symptom ICD ranges (R00–R99) ─────────────────────────────────────────
_SYMPTOM_ICD_RE = re.compile(r"^R\d{2}", re.IGNORECASE)

# ═══════════════════════════════════════════════════════════════════════════
# Rule Engine
# ═══════════════════════════════════════════════════════════════════════════

class MedicalCodingRules:
    """
    Deterministic medical coding rule engine.

    Call ``apply(results)`` with the unified disease/ICD list *after*
    MEAT validation and ICD mapping.  Returns a cleaned list plus a
    structured audit log of every rule action.
    """

    def __init__(self):
        self._abbr_map = {k.lower(): v.lower() for k, v in MEDICAL_ABBREVIATIONS.items()}

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────
    def apply(self, results: List[Dict], raw_text: str = "") -> Dict:
        """
        Run all 16 rules on the unified result list.

        Returns:
            {
                "results": [filtered list],
                "audit_log": [{"rule": ..., "action": ..., "disease": ...}, ...],
                "stats": {"total_input": N, "total_output": M, "removed": K, ...},
            }
        """
        audit: List[Dict] = []
        original_count = len(results)

        # Phase 1 — per-item rules (Bypassed to keep 'real results')
        # results = self._rule_01_documented_only(results, raw_text, audit)
        # results = self._rule_04_suspected_filtering(results, audit)
        # results = self._rule_08_medication_not_diagnosis(results, audit)
        # results = self._rule_09_lab_not_diagnosis(results, audit)
        # results = self._rule_10_negation_detection(results, audit)
        
        # Keep basic abbreviation deduplication
        results = self._rule_06_abbreviation_dedup(results, audit)

        # Phase 2 — cross-item rules (Bypassed to keep 'real results')
        # results = self._rule_02_15_diagnosis_over_symptoms(results, audit)
        # results = self._rule_03_symptoms_only_if_no_diagnosis(results, audit)
        # results = self._rule_05_chronic_retention(results, audit)
        # results = self._rule_14_excludes1(results, audit)

        # Phase 3 — enrichment / upgrade rules (Keep for better organization)
        results = self._rule_07_specificity(results, audit)
        results = self._rule_11_combination_codes(results, audit)
        results = self._rule_12_etiology_manifestation(results, audit)
        results = self._rule_13_laterality(results, raw_text, audit)
        results = self._rule_16_primary_secondary_ranking(results, audit)


        stats = {
            "total_input": original_count,
            "total_output": len(results),
            "removed": original_count - len(results),
            "rules_applied": len(set(a["rule"] for a in audit)),
            "actions": len(audit),
        }
        logger.info(
            "Medical coding rules applied",
            extra={"stage": "medical_coding_rules", **stats},
        )
        return {"results": results, "audit_log": audit, "stats": stats}

    # ──────────────────────────────────────────────────────────────────
    # Individual Rules
    # ──────────────────────────────────────────────────────────────────

    # R1  Documented-Only — reject if disease name looks inferred
    def _rule_01_documented_only(
        self, results: List[Dict], raw_text: str, audit: List[Dict]
    ) -> List[Dict]:
        if not raw_text:
            return results
        raw_lower = raw_text.lower()
        kept = []
        for r in results:
            name = r.get("disease", "").lower().strip()

            # Assessment/Plan items with any MEAT evidence are trusted —
            # the LLM may use clinical nomenclature (e.g. "pharyngitis")
            # that doesn't literally appear in the source text.
            seg = r.get("segment", "").lower().replace(" ", "_")
            seg_raw = {s.lower().replace(" ", "_") for s in r.get("segment_source_raw", [])}
            in_assessment = seg in {"assessment", "assessment_and_plan"} or bool(
                seg_raw & {"assessment", "assessment_and_plan", "assessment_plan"}
            )
            has_evidence = any(
                (r.get(f"{e}_evidence") or "").strip()
                for e in ("monitoring", "evaluation", "assessment", "treatment")
            )
            if in_assessment and has_evidence:
                kept.append(r)
                continue

            # Check if the disease (or a close abbreviation) actually appears in text
            found = name in raw_lower
            if not found:
                # Try abbreviation → expansion  (e.g. "htn" → "hypertension")
                expanded = self._abbr_map.get(name)
                if expanded:
                    found = expanded in raw_lower
                # Try reverse: expansion → abbreviation  (e.g. "urinary tract infection" → "uti")
                if not found:
                    rev_map = {v: k for k, v in self._abbr_map.items()}
                    abbr = rev_map.get(name)
                    if abbr:
                        found = abbr in raw_lower
                # Try each word (≥4 chars) of multi-word disease names
                if not found and len(name.split()) > 1:
                    long_words = [w for w in name.split() if len(w) >= 4]
                    if long_words:
                        found = sum(1 for w in long_words if w in raw_lower) / len(long_words) >= 0.6
                # Try ICD code description match
                if not found:
                    icd_desc = r.get("icd_description", "").lower()
                    if icd_desc and len(icd_desc) >= 6:
                        icd_words = [w for w in icd_desc.split() if len(w) >= 4]
                        if icd_words:
                            found = sum(1 for w in icd_words if w in raw_lower) / len(icd_words) >= 0.5
            if found:
                kept.append(r)
            else:
                audit.append({
                    "rule": "R01_documented_only",
                    "action": "removed",
                    "disease": r.get("disease", ""),
                    "reason": "Disease term not found in source document text",
                })
                logger.info(
                    "R01: removed_by_rule",
                    extra={"rule": "documented_only", "disease": r.get("disease", "")},
                )
        return kept

    # R4  Rule-Out / Suspected Filtering (OPD)
    def _rule_04_suspected_filtering(
        self, results: List[Dict], audit: List[Dict]
    ) -> List[Dict]:
        kept = []
        for r in results:
            name = r.get("disease", "")
            evidence = " ".join(
                r.get(f"{e}_evidence", "") or ""
                for e in ("monitoring", "evaluation", "assessment", "treatment")
            )
            text_to_check = f"{name} {evidence}"
            if _SUSPECTED_RE.search(text_to_check):
                audit.append({
                    "rule": "R04_suspected_filtering",
                    "action": "removed",
                    "disease": name,
                    "reason": f"Suspected/rule-out language detected: "
                              f"'{_SUSPECTED_RE.search(text_to_check).group()}'",
                })
                logger.info(
                    "R04: suspected_removed",
                    extra={"rule": "suspected_filtering", "disease": name},
                )
            else:
                kept.append(r)
        return kept

    # R8  Medication ≠ Diagnosis
    def _rule_08_medication_not_diagnosis(
        self, results: List[Dict], audit: List[Dict]
    ) -> List[Dict]:
        kept = []
        for r in results:
            name_lower = r.get("disease", "").lower().strip()
            # Reject if the disease name IS a medication name
            rejected = False
            for med in _MEDICATION_MARKERS:
                if name_lower == med or (
                    name_lower.startswith(med) and len(name_lower) - len(med) <= 4
                ):
                    audit.append({
                        "rule": "R08_medication_not_diagnosis",
                        "action": "removed",
                        "disease": r.get("disease", ""),
                        "reason": f"Disease name matches medication '{med}'",
                    })
                    logger.info(
                        "R08: medication_rejected",
                        extra={"rule": "medication_not_diagnosis",
                               "disease": r.get("disease", "")},
                    )
                    rejected = True
                    break
            if not rejected:
                kept.append(r)
        return kept

    # R9  Lab Result ≠ Diagnosis
    def _rule_09_lab_not_diagnosis(
        self, results: List[Dict], audit: List[Dict]
    ) -> List[Dict]:
        kept = []
        for r in results:
            name = r.get("disease", "").strip()
            # If the entire disease name is essentially a lab indicator, reject
            # (but allow if a real diagnosis is appended, e.g. "Elevated A1C — Diabetes")
            if _LAB_INDICATOR_RE.fullmatch(name.strip()):
                audit.append({
                    "rule": "R09_lab_not_diagnosis",
                    "action": "removed",
                    "disease": name,
                    "reason": "Disease name is a standalone lab result",
                })
                logger.info(
                    "R09: lab_rejected",
                    extra={"rule": "lab_not_diagnosis", "disease": name},
                )
            else:
                kept.append(r)
        return kept

    # R10  Negation Detection
    def _rule_10_negation_detection(
        self, results: List[Dict], audit: List[Dict]
    ) -> List[Dict]:
        kept = []
        for r in results:
            # Check assessment evidence for leading negation
            assessment_ev = (r.get("assessment_evidence") or "").lower()
            name_lower = r.get("disease", "").lower()
            combined = f"{assessment_ev} {name_lower}"
            m = _NEGATION_RE.search(combined)
            if m:
                # Verify negation is close to the disease name (within 60 chars)
                pos = m.end()
                remaining = combined[pos:pos + 60]
                if any(w in remaining for w in name_lower.split() if len(w) >= 4):
                    audit.append({
                        "rule": "R10_negation_detection",
                        "action": "removed",
                        "disease": r.get("disease", ""),
                        "reason": f"Negation phrase '{m.group()}' precedes disease name",
                    })
                    logger.info(
                        "R10: negation_removed",
                        extra={"rule": "negation_detection",
                               "disease": r.get("disease", "")},
                    )
                    continue
            kept.append(r)
        return kept

    # R6  Abbreviation Dedup
    def _rule_06_abbreviation_dedup(
        self, results: List[Dict], audit: List[Dict]
    ) -> List[Dict]:
        # Build groups of abbreviation ↔ expansion
        groups: List[Tuple[int, Set[str]]] = []
        deduped: List[Dict] = []
        for r in results:
            name = r.get("disease", "").lower().strip()
            variants = {name}
            if name in self._abbr_map:
                variants.add(self._abbr_map[name])
            rev = {v: k for k, v in self._abbr_map.items()}
            if name in rev:
                variants.add(rev[name])

            merged = False
            for idx, (_, group_variants) in enumerate(groups):
                if variants & group_variants:
                    existing = deduped[idx]
                    # Keep the longer (more descriptive) name
                    if len(r.get("disease", "")) > len(existing.get("disease", "")):
                        audit.append({
                            "rule": "R06_abbreviation_dedup",
                            "action": "merged",
                            "disease": existing.get("disease", ""),
                            "kept": r.get("disease", ""),
                            "reason": "Abbreviation duplicate — kept longer name",
                        })
                        deduped[idx] = r
                    else:
                        audit.append({
                            "rule": "R06_abbreviation_dedup",
                            "action": "merged",
                            "disease": r.get("disease", ""),
                            "kept": existing.get("disease", ""),
                            "reason": "Abbreviation duplicate — kept longer name",
                        })
                    groups[idx] = (idx, group_variants | variants)
                    merged = True
                    break
            if not merged:
                groups.append((len(deduped), variants))
                deduped.append(r)
        return deduped

    # R2 + R15  Diagnosis Over Symptoms + Integral Symptom Removal
    def _rule_02_15_diagnosis_over_symptoms(
        self, results: List[Dict], audit: List[Dict]
    ) -> List[Dict]:
        # Identify which diagnoses are present
        diagnoses_present: List[re.Pattern] = []
        subsumable_symptoms: Set[str] = set()
        for r in results:
            name = r.get("disease", "")
            for dx_pattern, symptoms in _DIAGNOSIS_SUBSUMES_SYMPTOMS:
                if dx_pattern.search(name):
                    diagnoses_present.append(dx_pattern)
                    subsumable_symptoms |= symptoms

        if not subsumable_symptoms:
            return results

        kept = []
        for r in results:
            name_lower = r.get("disease", "").lower().strip()
            if name_lower in subsumable_symptoms:
                audit.append({
                    "rule": "R02_diagnosis_over_symptoms",
                    "action": "removed",
                    "disease": r.get("disease", ""),
                    "reason": "Symptom subsumed by a confirmed diagnosis",
                })
                logger.info(
                    "R02: symptom_removed",
                    extra={"rule": "diagnosis_over_symptoms",
                           "disease": r.get("disease", ""),
                           "subsumed_by": "confirmed_diagnosis"},
                )
            else:
                kept.append(r)
        return kept

    # R3  Symptoms Only If No Diagnosis
    def _rule_03_symptoms_only_if_no_diagnosis(
        self, results: List[Dict], audit: List[Dict]
    ) -> List[Dict]:
        has_real_diagnosis = any(
            not _SYMPTOM_ICD_RE.match(r.get("icd_code", "") or "")
            for r in results
            if r.get("icd_code") and r["icd_code"] not in ("—", "-")
        )
        if not has_real_diagnosis:
            # No diagnoses — keep symptoms
            for r in results:
                icd = r.get("icd_code", "") or ""
                if _SYMPTOM_ICD_RE.match(icd):
                    audit.append({
                        "rule": "R03_symptoms_only_if_no_diagnosis",
                        "action": "retained",
                        "disease": r.get("disease", ""),
                        "reason": "No confirmed diagnosis — symptom coding allowed",
                    })
            return results
        # Has diagnoses — symptoms already handled by R02/R15
        return results

    # R5  Chronic Condition Retention
    def _rule_05_chronic_retention(
        self, results: List[Dict], audit: List[Dict]
    ) -> List[Dict]:
        # This is a safeguard — ensure chronic conditions are never removed
        # by earlier rules.  We mark them with a flag.
        for r in results:
            icd = (r.get("icd_code") or "").upper().replace(".", "")
            for prefix in _CHRONIC_ICD_PREFIXES:
                if icd.startswith(prefix):
                    r["_chronic"] = True
                    audit.append({
                        "rule": "R05_chronic_retention",
                        "action": "flagged_chronic",
                        "disease": r.get("disease", ""),
                        "icd_code": r.get("icd_code", ""),
                        "reason": "Chronic condition — protected from removal",
                    })
                    break
        return results

    # R7  Specificity — prefer most specific ICD (longer code)
    def _rule_07_specificity(
        self, results: List[Dict], audit: List[Dict]
    ) -> List[Dict]:
        # Within the same 3-char category, keep the most specific code
        by_category: Dict[str, List[Dict]] = {}
        for r in results:
            icd = (r.get("icd_code") or "").upper().strip()
            if not icd or icd in ("—", "-") or len(icd) < 3:
                continue
            cat = icd[:3]
            by_category.setdefault(cat, []).append(r)

        remove_set = set()
        for cat, group in by_category.items():
            if len(group) <= 1:
                continue
            # Sort by code length desc — longest = most specific
            group.sort(key=lambda r: len(r.get("icd_code", "")), reverse=True)
            most_specific = group[0]
            for lesser in group[1:]:
                lesser_icd = lesser.get("icd_code", "")
                specific_icd = most_specific.get("icd_code", "")
                # Only remove if the shorter code is a prefix of the longer
                if specific_icd.startswith(lesser_icd.replace(".", "")):
                    remove_set.add(id(lesser))
                    audit.append({
                        "rule": "R07_specificity",
                        "action": "removed_less_specific",
                        "disease": lesser.get("disease", ""),
                        "icd_removed": lesser_icd,
                        "icd_kept": specific_icd,
                        "reason": f"Less specific {lesser_icd} replaced by {specific_icd}",
                    })
        return [r for r in results if id(r) not in remove_set]

    # R11  Combination Code Handling
    def _rule_11_combination_codes(
        self, results: List[Dict], audit: List[Dict]
    ) -> List[Dict]:
        names = [r.get("disease", "") for r in results]
        names_str = " | ".join(names).lower()
        combo_applied: List[Dict] = []

        for combo in _COMBINATION_CODES:
            patterns = combo["terms"]
            if all(re.search(p, names_str) for p in patterns):
                combo_applied.append(combo)
                audit.append({
                    "rule": "R11_combination_code",
                    "action": "suggested",
                    "combination_icd": combo["icd"],
                    "description": combo["desc"],
                    "reason": f"Combination code available for co-occurring conditions",
                })
        # Enrich — add suggestion flag to results (don't replace existing codes)
        if combo_applied:
            for r in results:
                r.setdefault("combination_code_suggestions", [])
                for c in combo_applied:
                    r["combination_code_suggestions"].append(
                        {"icd": c["icd"], "description": c["desc"]}
                    )
                break  # Add suggestion to first result only
        return results

    # R12  Etiology + Manifestation
    def _rule_12_etiology_manifestation(
        self, results: List[Dict], audit: List[Dict]
    ) -> List[Dict]:
        # Flag etiology/manifestation pairs for correct sequencing
        # The ICD standard requires etiology (cause) code first, then manifestation
        # This is informational — we tag but don't reorder to avoid breaking display
        for i, r in enumerate(results):
            icd = r.get("icd_code", "") or ""
            # Manifestation codes often have dagger/asterisk conventions
            # In ICD-10-CM, codes in certain ranges are manifestations
            # (e.g., G63.* diabetic polyneuropathy manifestation)
            if re.match(r"^[GHN]\d{2}", icd):
                # Check if there's a likely etiology code in the same set
                for j, other in enumerate(results):
                    if i == j:
                        continue
                    other_icd = other.get("icd_code", "") or ""
                    if re.match(r"^E1[0-3]", other_icd):  # Diabetes as etiology
                        r["_etiology_pair"] = other_icd
                        audit.append({
                            "rule": "R12_etiology_manifestation",
                            "action": "paired",
                            "manifestation": icd,
                            "etiology": other_icd,
                            "reason": "Etiology/manifestation pair detected",
                        })
                        break
        return results

    # R13  Laterality Detection
    def _rule_13_laterality(
        self, results: List[Dict], raw_text: str, audit: List[Dict]
    ) -> List[Dict]:
        if not raw_text:
            return results
        for r in results:
            name = r.get("disease", "")
            # Check disease name + evidence for laterality
            evidence = " ".join(
                r.get(f"{e}_evidence", "") or ""
                for e in ("monitoring", "evaluation", "assessment", "treatment")
            )
            combined = f"{name} {evidence}"
            m = _LATERALITY_RE.search(combined)
            if m:
                laterality = m.group(1).lower()
                _MAP = {"rt": "right", "lt": "left", "r/l": "right/left", "l/r": "left/right"}
                normalized = _MAP.get(laterality, laterality)
                r["laterality"] = normalized
                audit.append({
                    "rule": "R13_laterality",
                    "action": "detected",
                    "disease": name,
                    "laterality": normalized,
                    "reason": f"Laterality '{normalized}' detected",
                })
        return results

    # R14  Excludes1 Conflict Removal
    def _rule_14_excludes1(
        self, results: List[Dict], audit: List[Dict]
    ) -> List[Dict]:
        icd_set = {}
        for r in results:
            icd = (r.get("icd_code") or "").upper().replace(".", "")
            if icd and len(icd) >= 3:
                icd_set[icd[:3]] = r

        remove_ids = set()
        for cat_a, cat_b in _EXCLUDES1_PAIRS:
            if cat_a in icd_set and cat_b in icd_set:
                # Keep the one with higher confidence
                r_a = icd_set[cat_a]
                r_b = icd_set[cat_b]
                conf_a = r_a.get("confidence", 0) or r_a.get("icd_confidence", 0)
                conf_b = r_b.get("confidence", 0) or r_b.get("icd_confidence", 0)
                loser = r_b if conf_a >= conf_b else r_a
                winner = r_a if conf_a >= conf_b else r_b
                remove_ids.add(id(loser))
                audit.append({
                    "rule": "R14_excludes1",
                    "action": "removed_conflict",
                    "removed": loser.get("disease", ""),
                    "kept": winner.get("disease", ""),
                    "reason": f"Excludes1: {cat_a} and {cat_b} cannot coexist",
                })
        return [r for r in results if id(r) not in remove_ids]

    # R16  Primary vs Secondary Ranking
    def _rule_16_primary_secondary_ranking(
        self, results: List[Dict], audit: List[Dict]
    ) -> List[Dict]:
        _SECTION_RANK = {
            "assessment_and_plan": 1, "assessment": 1,
            "chief_complaint": 2, "cc": 2,
            "history_present_illness": 3, "hpi": 3,
            "active_problems": 1, "active_problem_list": 1, "problem_list": 1,
            "plan": 4, "impression": 4,
            "past_medical_history": 6, "pmh": 6, "medical_history": 6,
        }
        for r in results:
            sections = r.get("segment_source_raw", [])
            section_score = min(
                (_SECTION_RANK.get(s.lower(), 8) for s in sections),
                default=8,
            )
            meat_bonus = 0
            if r.get("meat_tier") == "strong_evidence":
                meat_bonus = 2
            elif r.get("meat_tier") == "moderate_evidence":
                meat_bonus = 1
            conf = r.get("confidence", 0) or r.get("icd_confidence", 0)
            chronic_bonus = 1 if r.get("_chronic") else 0
            # Lower = higher priority (like golf score)
            rank_score = section_score - meat_bonus - chronic_bonus - (conf * 2)
            r["ranking_score"] = round(rank_score, 2)

        results.sort(key=lambda r: r.get("ranking_score", 99))

        for i, r in enumerate(results):
            r["coding_rank"] = i + 1
            rank_label = "primary" if i == 0 else "secondary"
            r["coding_priority"] = rank_label
            audit.append({
                "rule": "R16_ranking",
                "action": "ranked",
                "disease": r.get("disease", ""),
                "rank": i + 1,
                "score": r.get("ranking_score"),
                "priority": rank_label,
            })
        return results



# Singleton
medical_coding_rules = MedicalCodingRules()
