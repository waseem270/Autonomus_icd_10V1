"""
Pipeline Evaluation Script
==========================
Measures extraction accuracy against a ground truth annotation file.

Metrics computed:
  - Disease Detection: Precision, Recall, F1 (fuzzy name matching)
  - ICD Mapping:       Exact code match accuracy, Top-3 accuracy
  - MEAT Validation:   True positive rate, False positive rate

Usage:
  python -m evaluation.evaluate --ground-truth evaluation/ground_truth.json --results evaluation/pipeline_output.json
  
Or run programmatically:
  from evaluation.evaluate import evaluate_pipeline
  report = evaluate_pipeline(ground_truth_path, results_path)
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from fuzzywuzzy import fuzz


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(name: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return " ".join(name.lower().strip().split())


def fuzzy_match_name(predicted: str, expected: str, threshold: int = 80) -> bool:
    """Check if two disease names are close enough to count as a match."""
    return fuzz.token_set_ratio(normalize(predicted), normalize(expected)) >= threshold


# ---------------------------------------------------------------------------
# Disease Detection Metrics
# ---------------------------------------------------------------------------

def evaluate_disease_detection(
    predicted_diseases: List[str],
    expected_diseases: List[Dict],
) -> Dict[str, Any]:
    """
    Compute precision, recall, F1 for disease detection.
    Uses fuzzy name matching (Token Set Ratio >= 80).
    """
    expected_names = [d["name"] for d in expected_diseases if not d.get("negated", False)]
    
    tp = 0
    fp = 0
    matched_expected = set()
    
    for pred in predicted_diseases:
        found = False
        for i, exp in enumerate(expected_names):
            if i in matched_expected:
                continue
            if fuzzy_match_name(pred, exp):
                tp += 1
                matched_expected.add(i)
                found = True
                break
        if not found:
            fp += 1
    
    fn = len(expected_names) - len(matched_expected)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "expected_count": len(expected_names),
        "predicted_count": len(predicted_diseases),
    }


# ---------------------------------------------------------------------------
# ICD Mapping Metrics
# ---------------------------------------------------------------------------

def evaluate_icd_mapping(
    predicted_mappings: List[Dict],
    expected_diseases: List[Dict],
) -> Dict[str, Any]:
    """
    Evaluate ICD code assignment accuracy.
    - exact_match: predicted code == expected code
    - top3_match:  expected code appears in top-3 candidates
    """
    exact_correct = 0
    top3_correct = 0
    total_expected = 0
    
    for exp in expected_diseases:
        if exp.get("negated", False):
            continue
        total_expected += 1
        exp_code = exp["icd_code"].strip().upper()
        exp_name = exp["name"]
        
        # Find matching predicted mapping
        matched_pred = None
        for pred in predicted_mappings:
            if fuzzy_match_name(pred.get("disease", ""), exp_name):
                matched_pred = pred
                break
        
        if not matched_pred:
            continue
        
        pred_code = (matched_pred.get("icd_code") or "").strip().upper()
        
        if pred_code == exp_code:
            exact_correct += 1
            top3_correct += 1
        else:
            # Check top-3 candidates
            candidates = matched_pred.get("candidates", [])[:3]
            if any(c.get("icd_code", "").strip().upper() == exp_code for c in candidates):
                top3_correct += 1
    
    return {
        "exact_match_accuracy": round(exact_correct / total_expected, 4) if total_expected > 0 else 0.0,
        "top3_accuracy": round(top3_correct / total_expected, 4) if total_expected > 0 else 0.0,
        "exact_correct": exact_correct,
        "top3_correct": top3_correct,
        "total_expected": total_expected,
    }


# ---------------------------------------------------------------------------
# MEAT Validation Metrics
# ---------------------------------------------------------------------------

def evaluate_meat_validation(
    predicted_meat: List[Dict],
    expected_diseases: List[Dict],
) -> Dict[str, Any]:
    """
    Evaluate MEAT validation accuracy.
    """
    tp = 0  # Correctly identified as MEAT-valid
    fp = 0  # Incorrectly marked as MEAT-valid
    fn = 0  # Expected MEAT-valid but marked invalid
    tn = 0  # Correctly identified as MEAT-invalid
    
    for exp in expected_diseases:
        if exp.get("negated", False):
            continue
        
        exp_meat = exp.get("meat_valid", False)
        exp_name = exp["name"]
        
        # Find matching predicted MEAT result
        matched_pred = None
        for pred in predicted_meat:
            if fuzzy_match_name(pred.get("disease", ""), exp_name):
                matched_pred = pred
                break
        
        if not matched_pred:
            if exp_meat:
                fn += 1
            continue
        
        pred_meat = matched_pred.get("meat_valid", False)
        
        if exp_meat and pred_meat:
            tp += 1
        elif not exp_meat and not pred_meat:
            tn += 1
        elif pred_meat and not exp_meat:
            fp += 1
        else:
            fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------

def evaluate_pipeline(
    ground_truth_path: str,
    results_path: str,
) -> Dict[str, Any]:
    """
    Run full evaluation comparing pipeline output to ground truth.
    
    ground_truth_path: Path to annotated ground truth JSON (see template)
    results_path:      Path to pipeline output JSON (exported from the system)
    """
    gt = load_json(ground_truth_path)
    results = load_json(results_path)
    
    all_reports = []
    
    for gt_doc in gt.get("documents", []):
        doc_id = gt_doc["document_id"]
        expected = gt_doc.get("diseases", [])
        
        # Find matching result document
        result_doc = None
        for r in results.get("documents", []):
            if r.get("document_id") == doc_id:
                result_doc = r
                break
        
        if not result_doc:
            print(f"WARNING: No results found for document '{doc_id}'")
            continue
        
        predicted_diseases = [d.get("disease_name", d.get("name", "")) 
                             for d in result_doc.get("detected_diseases", [])]
        predicted_mappings = result_doc.get("icd_mappings", [])
        predicted_meat = result_doc.get("meat_results", [])
        
        detection = evaluate_disease_detection(predicted_diseases, expected)
        mapping = evaluate_icd_mapping(predicted_mappings, expected)
        meat = evaluate_meat_validation(predicted_meat, expected)
        
        doc_report = {
            "document_id": doc_id,
            "disease_detection": detection,
            "icd_mapping": mapping,
            "meat_validation": meat,
        }
        all_reports.append(doc_report)
        
        print(f"\n{'='*60}")
        print(f"Document: {doc_id}")
        print(f"{'='*60}")
        print(f"  Disease Detection:  P={detection['precision']:.2f}  R={detection['recall']:.2f}  F1={detection['f1']:.2f}")
        print(f"    ({detection['predicted_count']} predicted, {detection['expected_count']} expected)")
        print(f"  ICD Mapping:        Exact={mapping['exact_match_accuracy']:.2f}  Top3={mapping['top3_accuracy']:.2f}")
        print(f"  MEAT Validation:    P={meat['precision']:.2f}  R={meat['recall']:.2f}  F1={meat['f1']:.2f}")
    
    # Aggregate across documents
    if all_reports:
        avg = lambda key, subkey: sum(
            r[key][subkey] for r in all_reports
        ) / len(all_reports)
        
        print(f"\n{'='*60}")
        print(f"AGGREGATE ({len(all_reports)} documents)")
        print(f"{'='*60}")
        print(f"  Avg Disease Detection F1: {avg('disease_detection', 'f1'):.4f}")
        print(f"  Avg ICD Exact Accuracy:   {avg('icd_mapping', 'exact_match_accuracy'):.4f}")
        print(f"  Avg ICD Top-3 Accuracy:   {avg('icd_mapping', 'top3_accuracy'):.4f}")
        print(f"  Avg MEAT Validation F1:   {avg('meat_validation', 'f1'):.4f}")
    
    return {
        "per_document": all_reports,
        "aggregate": {
            "num_documents": len(all_reports),
            "avg_detection_f1": round(sum(r["disease_detection"]["f1"] for r in all_reports) / max(len(all_reports), 1), 4),
            "avg_icd_exact": round(sum(r["icd_mapping"]["exact_match_accuracy"] for r in all_reports) / max(len(all_reports), 1), 4),
            "avg_icd_top3": round(sum(r["icd_mapping"]["top3_accuracy"] for r in all_reports) / max(len(all_reports), 1), 4),
            "avg_meat_f1": round(sum(r["meat_validation"]["f1"] for r in all_reports) / max(len(all_reports), 1), 4),
        } if all_reports else {}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pipeline accuracy")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth JSON")
    parser.add_argument("--results", required=True, help="Path to pipeline output JSON")
    parser.add_argument("--output", help="Path to save evaluation report JSON")
    args = parser.parse_args()
    
    report = evaluate_pipeline(args.ground_truth, args.results)
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.output}")
