"""
Set 11 Evaluation Runner — runs all 10 PDFs from 'test pdf/' through the
full ClinicalDocumentAnalyzer pipeline and compares against Excel ground truth.

Usage:
    cd medical-icd-mapper
    python -m evaluation.set11_evaluator

Generates: evaluation/set11_report.json
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.services.clinical_document_analyzer import ClinicalDocumentAnalyzer
from backend.app.services.text_extraction import TextExtractionService
from backend.app.services.output_filter import OutputFilter
from backend.app.services.medical_coding_rules import MedicalCodingRules
from backend.app.services.confidence_scorer import ConfidenceScorer
from evaluation.ground_truth_validator import (
    parse_ground_truth_excel,
    evaluate_pdf,
    aggregate_report,
)

# ─── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXCEL_PATH = PROJECT_ROOT.parent / "test pdf" / "Set 11 Coding Results.xlsx"
PDF_DIR = PROJECT_ROOT.parent / "test pdf"
REPORT_PATH = PROJECT_ROOT / "evaluation" / "set11_report.json"

output_filter = OutputFilter()
medical_coding_rules = MedicalCodingRules()
confidence_scorer = ConfidenceScorer()


def find_pdf(mrn: str) -> Path:
    """Find the PDF in 'test pdf/' whose name starts with the MRN."""
    for f in PDF_DIR.iterdir():
        if f.name.startswith(mrn) and f.suffix.lower() == ".pdf":
            return f
    raise FileNotFoundError(f"No PDF found for MRN {mrn} in {PDF_DIR}")


async def run_single_pdf(
    mrn: str,
    expected_codes: set,
    analyzer: ClinicalDocumentAnalyzer,
    extractor: TextExtractionService,
) -> dict:
    """Run one PDF through the full pipeline and evaluate."""
    pdf_path = find_pdf(mrn)
    print(f"\n{'─'*60}")
    print(f"  MRN: {mrn}  |  PDF: {pdf_path.name}")
    print(f"  Expected: {sorted(expected_codes)}")

    start = time.time()

    # 1. Extract text
    extraction = await extractor.extract_text(str(pdf_path))
    raw_text = extraction["raw_text"]

    # 2. Analyze with LLM (single-pass)
    result = await analyzer.analyze_document(raw_text)
    if not result:
        print(f"  ❌ LLM returned no result")
        return evaluate_pdf(mrn, [], expected_codes)

    # 3. Convert to unified results
    unified = analyzer.convert_to_unified_results(result)

    # 4. Apply output filter
    filtered = output_filter.apply(unified, single_pass=True)

    # 5. Apply medical coding rules
    rules_result = medical_coding_rules.apply(filtered, raw_text=raw_text)
    scored = confidence_scorer.score_batch(rules_result["results"])

    elapsed = time.time() - start

    # 6. Evaluate against ground truth
    evaluation = evaluate_pdf(mrn, scored, expected_codes)
    evaluation["processing_time"] = round(elapsed, 1)
    evaluation["rules_audit"] = rules_result.get("audit_log", [])

    # Print summary
    status = "✅" if not evaluation["extra"] and not evaluation["missing"] else "❌"
    print(f"  Predicted: {sorted(set(r['code'] for r in evaluation['all_predicted']))}")
    print(f"  Matched:   {evaluation['matched']}")
    print(f"  Extra:     {evaluation['extra']}")
    print(f"  Missing:   {evaluation['missing']}")
    print(f"  Precision: {evaluation['precision']:.0%}  Recall: {evaluation['recall']:.0%}  F1: {evaluation['f1']:.0%}")
    print(f"  {status} Time: {elapsed:.1f}s")

    if evaluation["extra_analysis"]:
        print(f"  Extra code analysis:")
        for ea in evaluation["extra_analysis"]:
            print(f"    {ea['code']:10s} | {ea['disease'][:40]:40s} | [{ea['section']}] | {ea['error_type']}")

    return evaluation


async def main():
    print("=" * 60)
    print("  SET 11 GROUND TRUTH EVALUATION")
    print("=" * 60)

    # Parse ground truth
    if not EXCEL_PATH.exists():
        print(f"❌ Ground truth file not found: {EXCEL_PATH}")
        sys.exit(1)
    ground_truth = parse_ground_truth_excel(str(EXCEL_PATH))
    print(f"Loaded {len(ground_truth)} PDFs from ground truth")
    print(f"Total expected ICD codes: {sum(len(v) for v in ground_truth.values())}")

    # Init services
    analyzer = ClinicalDocumentAnalyzer()
    extractor = TextExtractionService()

    evaluations = []
    overall_start = time.time()

    for mrn in sorted(ground_truth.keys()):
        try:
            result = await run_single_pdf(mrn, ground_truth[mrn], analyzer, extractor)
            evaluations.append(result)
        except Exception as e:
            print(f"  ❌ ERROR for {mrn}: {e}")
            evaluations.append({
                "mrn": mrn,
                "total_expected": len(ground_truth[mrn]),
                "total_predicted": 0,
                "total_matched": 0,
                "matched": [],
                "extra": [],
                "missing": sorted(ground_truth[mrn]),
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
                "extra_analysis": [],
                "all_predicted": [],
                "error": str(e),
            })

    # Aggregate
    report = aggregate_report(evaluations)
    report["total_time"] = round(time.time() - overall_start, 1)

    # ── Print overall summary ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  OVERALL RESULTS")
    print(f"{'='*60}")
    print(f"  PDFs:       {report['total_pdfs']}")
    print(f"  Expected:   {report['total_expected']} ICD codes")
    print(f"  Predicted:  {report['total_predicted']} ICD codes")
    print(f"  Matched:    {report['total_matched']}")
    print(f"  Extra:      {report['total_extra']}")
    print(f"  Missing:    {report['total_missing']}")
    print(f"  Precision:  {report['overall_precision']:.1%}")
    print(f"  Recall:     {report['overall_recall']:.1%}")
    print(f"  F1:         {report['overall_f1']:.1%}")
    print(f"  Time:       {report['total_time']:.0f}s")

    if report["error_breakdown"]:
        print(f"\n  Error Breakdown:")
        for et, count in sorted(report["error_breakdown"].items()):
            print(f"    {et}: {count}")

    print(f"\n  Per-PDF:")
    for p in report["per_pdf"]:
        status = "✅" if not p["extra"] and not p["missing"] else "❌"
        miss = ", ".join(p["missing"][:5]) if p["missing"] else "-"
        extra = ", ".join(p["extra"][:5]) if p["extra"] else "-"
        print(f"    {status} {p['mrn']:10s} P={p['precision']:.0%} R={p['recall']:.0%} F1={p['f1']:.0%}  extra=[{extra}]  miss=[{miss}]")

    # Save report
    REPORT_PATH.parent.mkdir(exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
