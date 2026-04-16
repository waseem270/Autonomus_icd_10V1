"""
Ground Truth Validator — reads the Excel coding results file and
compares pipeline output against expected ICD-10 codes.

Provides:
  - Per-PDF precision, recall, F1, accuracy
  - Error classification (overcoding / undercoding / wrong-code)
  - Section-source analysis for extra codes
  - Overall summary statistics
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import openpyxl

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Excel parser
# ═══════════════════════════════════════════════════════════════════════════

def parse_ground_truth_excel(excel_path: str) -> Dict[str, Set[str]]:
    """
    Parse 'Set 11 Coding Results.xlsx' → { mrn: {ICD codes} }.

    Expects columns:  A=MRN, B=DOS, C–N = DX1–DX12
    """
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    ws = wb.active

    ground_truth: Dict[str, Set[str]] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        raw_mrn = row[0]
        if raw_mrn is None:
            continue
        mrn = str(int(raw_mrn)) if isinstance(raw_mrn, (int, float)) else str(raw_mrn).strip()
        codes: Set[str] = set()
        for cell in row[2:]:  # columns C..N  (DX1–DX12)
            if cell and str(cell).strip():
                codes.add(str(cell).strip().upper())
        if codes:
            ground_truth[mrn] = codes
    wb.close()
    logger.info(f"Parsed {len(ground_truth)} PDFs from ground truth: {excel_path}")
    return ground_truth


# ═══════════════════════════════════════════════════════════════════════════
# Comparison helpers
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_code(code: str) -> str:
    """Uppercase, strip spaces, ensure dot notation."""
    c = code.strip().upper()
    if "." not in c and len(c) > 3:
        c = c[:3] + "." + c[3:]
    return c


def compare_codes(
    predicted: Set[str],
    expected: Set[str],
) -> Dict:
    """
    Compare predicted ICD codes vs expected.

    Returns dict with: matched, extra, missing, precision, recall, f1, accuracy.
    """
    pred_norm = {_normalize_code(c) for c in predicted}
    exp_norm = {_normalize_code(c) for c in expected}

    matched = pred_norm & exp_norm
    extra = pred_norm - exp_norm
    missing = exp_norm - pred_norm

    precision = len(matched) / len(pred_norm) if pred_norm else 0.0
    recall = len(matched) / len(exp_norm) if exp_norm else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = len(matched) / len(exp_norm) if exp_norm else 0.0

    return {
        "matched": sorted(matched),
        "extra": sorted(extra),
        "missing": sorted(missing),
        "total_predicted": len(pred_norm),
        "total_expected": len(exp_norm),
        "total_matched": len(matched),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Error classification
# ═══════════════════════════════════════════════════════════════════════════

_PROBLEM_LIST_SECTIONS = {
    "problem_list", "active_problems", "active_problem_list",
    "past_medical_history", "pmh", "medical_history",
    "surgical_history", "past_surgical_history",
}

_INCIDENTAL_PATTERNS = re.compile(
    r"(?i)\b(?:incidental|subdural\s+hygroma|benign\s+finding|"
    r"unchanged|old\s+finding|stable\s+from\s+prior|"
    r"no\s+acute\s+(?:findings?|process)|"
    r"chronic\s+appearing|age[- ]?related|degenerative)\b"
)

_HISTORICAL_PATTERNS = re.compile(
    r"(?i)\b(?:history\s+of|h/o|hx\s+of|prior|previous|"
    r"remote|past\s+(?:medical\s+)?history|resolved|"
    r"childhood|years?\s+ago)\b"
)


def classify_extra_code(
    icd_code: str,
    disease_info: Optional[Dict] = None,
) -> str:
    """
    Classify WHY an extra ICD code was produced.

    Returns one of:
      'overcoding_problem_list'
      'overcoding_incidental'
      'overcoding_historical'
      'overcoding_inactive_chronic'
      'overcoding_unknown'
    """
    if disease_info is None:
        return "overcoding_unknown"

    sections = disease_info.get("segment_source_raw", [])
    sec_lower = {s.lower().replace(" ", "_") for s in sections} if sections else set()

    disease_name = disease_info.get("disease", "").lower()
    evidence = " ".join([
        disease_info.get("monitoring_evidence", ""),
        disease_info.get("evaluation_evidence", ""),
        disease_info.get("assessment_evidence", ""),
        disease_info.get("treatment_evidence", ""),
    ]).lower()

    # Problem list coding
    if sec_lower and sec_lower.issubset(_PROBLEM_LIST_SECTIONS):
        return "overcoding_problem_list"

    # Incidental imaging findings
    if _INCIDENTAL_PATTERNS.search(disease_name) or _INCIDENTAL_PATTERNS.search(evidence):
        return "overcoding_incidental"

    # Historical / resolved conditions
    if _HISTORICAL_PATTERNS.search(disease_name) or _HISTORICAL_PATTERNS.search(evidence):
        return "overcoding_historical"

    # Chronic not addressed in visit (weak/no MEAT from non-assessment sections)
    meat_tier = disease_info.get("meat_tier", "")
    in_assessment = bool(sec_lower & {"assessment", "assessment_and_plan"})
    if meat_tier in ("weak_evidence", "no_meat") and not in_assessment:
        return "overcoding_inactive_chronic"

    return "overcoding_unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Full PDF Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_pdf(
    mrn: str,
    predicted_results: List[Dict],
    expected_codes: Set[str],
) -> Dict:
    """
    Full evaluation for a single PDF.

    Args:
        mrn: Patient MRN / PDF identifier
        predicted_results: unified_results list from pipeline
        expected_codes: set of expected ICD codes from ground truth

    Returns:
        Detailed evaluation dict with metrics and error analysis.
    """
    # Extract predicted codes
    predicted_codes = set()
    code_to_result = {}
    for r in predicted_results:
        code = r.get("icd_code", "").strip()
        if code and code != "—":
            norm = _normalize_code(code)
            predicted_codes.add(norm)
            code_to_result[norm] = r

    comparison = compare_codes(predicted_codes, expected_codes)

    # Classify extra codes
    extra_analysis = []
    for extra_code in comparison["extra"]:
        disease_info = code_to_result.get(extra_code)
        error_type = classify_extra_code(extra_code, disease_info)
        extra_analysis.append({
            "code": extra_code,
            "disease": disease_info.get("disease", "?") if disease_info else "?",
            "section": disease_info.get("segment", "?") if disease_info else "?",
            "meat_tier": disease_info.get("meat_tier", "?") if disease_info else "?",
            "error_type": error_type,
        })

    return {
        "mrn": mrn,
        **comparison,
        "extra_analysis": extra_analysis,
        "all_predicted": [
            {
                "code": _normalize_code(r.get("icd_code", "")),
                "disease": r.get("disease", ""),
                "section": r.get("segment", ""),
                "meat_tier": r.get("meat_tier", ""),
                "in_expected": _normalize_code(r.get("icd_code", "")) in {_normalize_code(c) for c in expected_codes},
            }
            for r in predicted_results
            if r.get("icd_code", "").strip() and r.get("icd_code") != "—"
        ],
    }


def aggregate_report(evaluations: List[Dict]) -> Dict:
    """
    Aggregate per-PDF evaluations into an overall report.
    """
    total_expected = sum(e["total_expected"] for e in evaluations)
    total_predicted = sum(e["total_predicted"] for e in evaluations)
    total_matched = sum(e["total_matched"] for e in evaluations)
    total_extra = sum(len(e["extra"]) for e in evaluations)
    total_missing = sum(len(e["missing"]) for e in evaluations)

    overall_precision = total_matched / total_predicted if total_predicted else 0
    overall_recall = total_matched / total_expected if total_expected else 0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0 else 0
    )

    # Error type breakdown
    error_breakdown = {}
    for e in evaluations:
        for ea in e.get("extra_analysis", []):
            et = ea["error_type"]
            error_breakdown[et] = error_breakdown.get(et, 0) + 1

    return {
        "total_pdfs": len(evaluations),
        "total_expected": total_expected,
        "total_predicted": total_predicted,
        "total_matched": total_matched,
        "total_extra": total_extra,
        "total_missing": total_missing,
        "overall_precision": round(overall_precision, 4),
        "overall_recall": round(overall_recall, 4),
        "overall_f1": round(overall_f1, 4),
        "error_breakdown": error_breakdown,
        "per_pdf": [
            {
                "mrn": e["mrn"],
                "expected": e["total_expected"],
                "predicted": e["total_predicted"],
                "matched": e["total_matched"],
                "extra": e["extra"],
                "missing": e["missing"],
                "precision": e["precision"],
                "recall": e["recall"],
                "f1": e["f1"],
            }
            for e in evaluations
        ],
    }
