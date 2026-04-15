import logging
import re
from typing import Dict, List, Any
from fuzzywuzzy import fuzz
from ..core.config import settings

logger = logging.getLogger(__name__)


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace (including newlines) to single spaces, lowercase."""
    return " ".join(text.lower().split())


class MEATValidationGate:
    """
    Deterministic gate that enforces strict clinical and logical rules on top of 
    LLM-generated MEAT results. This layer ensures auditability and 
    prevents LLM hallucinations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_meat_result(self, meat_result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply deterministic validation rules on a single MEAT result.
        """
        disease_name = meat_result.get("disease", "Unknown")
        
        validation = {
            "meat_valid": False,
            "validation_rule": "",
            "confidence_tier": "low",
            "requires_manual_review": False,
            "issues_found": [],
            "final_meat": meat_result.copy()
        }
        
        M = meat_result.get("monitoring", False)
        E = meat_result.get("evaluation", False)
        A = meat_result.get("assessment", False)
        T = meat_result.get("treatment", False)
        
        # 1. Core Clinical Logic Rule:
        #    Primary: A AND (M OR E OR T) — standard MEAT
        #    Secondary: T AND (M OR E) — disease actively treated without explicit assessment
        #    This allows PMH/Problem List diseases with active medications to pass
        meat_count = sum([bool(M), bool(E), bool(A), bool(T)])
        if A and (M or E or T):
            validation["meat_valid"] = True
            validation["validation_rule"] = "A AND (M OR E OR T) [PASS]"
        elif T and (M or E):
            validation["meat_valid"] = True
            validation["validation_rule"] = "T AND (M OR E) [PASS — active treatment without explicit assessment]"
        elif meat_count >= 2:
            validation["meat_valid"] = True
            validation["validation_rule"] = f"MEAT count >= 2 ({meat_count}/4) [PASS]"
        else:
            validation["meat_valid"] = False
            validation["validation_rule"] = "Insufficient MEAT evidence [FAIL]"
            if not A:
                validation["issues_found"].append("Missing Assessment (A): Diagnosis not explicitly stated as active.")
            if not (M or E or T):
                validation["issues_found"].append("Lacks Clinical Management (M/E/T): No evidence of active management.")

        # 2. Hallucination Check: Verify evidence quotes exist in original context
        #    Uses FUZZY matching to handle minor whitespace/formatting differences.
        #    ALSO: reject MEAT elements that have no evidence text at all.
        all_context_text = _normalize_ws(self._get_all_context_text(context))
        
        for element in ["monitoring", "evaluation", "assessment", "treatment"]:
            if not meat_result.get(element, False):
                continue
                
            evidence_key = f"{element}_evidence"
            evidence = meat_result.get(evidence_key, "").strip()
            
            # Strict: evidence text is REQUIRED for every claimed MEAT element
            if not evidence or len(evidence) < 5:
                validation["issues_found"].append(
                    f"{element.upper()}: No evidence text provided — element rejected."
                )
                validation["final_meat"][element] = False
                validation["final_meat"][f"{element}_confidence"] = 0.0
                continue

            evidence_norm = _normalize_ws(evidence)
            
            # Try exact normalized substring first
            found = evidence_norm in all_context_text
            
            if not found:
                # Fuzzy fallback: check if the evidence is a close partial match
                # using fuzz.partial_ratio (handles substring matching well)
                fuzzy_score = fuzz.partial_ratio(evidence_norm, all_context_text)
                found = fuzzy_score >= 75  # 75% partial match threshold
                
                if not found:
                    # Last resort: check if >=70% of evidence words appear in context
                    evidence_words = set(evidence_norm.split())
                    if evidence_words:
                        context_words = set(all_context_text.split())
                        overlap = len(evidence_words & context_words) / len(evidence_words)
                        found = overlap >= 0.70

            if not found:
                validation["issues_found"].append(
                    f"{element.upper()}: Evidence quote not found in source text (Potential Hallucination)."
                )
                validation["requires_manual_review"] = True
                conf_key = f"{element}_confidence"
                validation["final_meat"][conf_key] = max(0.0, meat_result.get(conf_key, 0.0) - 0.4)
                if len(evidence) > 15:
                    validation["final_meat"][element] = False

        # 3. Confidence Tiering
        overall_conf = validation["final_meat"].get("overall_confidence", 0.0)
        
        if overall_conf >= settings.MEAT_CONFIDENCE_HIGH:
            validation["confidence_tier"] = "high"
        elif overall_conf >= settings.MEAT_CONFIDENCE_MEDIUM:
            validation["confidence_tier"] = "medium"
            validation["requires_manual_review"] = True
        else:
            validation["confidence_tier"] = "low"
            validation["requires_manual_review"] = True
            validation["issues_found"].append("Low composite confidence score.")

        # 4. Temporal Validation (Exclude History of / Resolved conditions)
        #    BUT: skip this check when the disease is confirmed in an active
        #    section (Assessment, A&P, HPI, CC).  A disease explicitly listed
        #    in Assessment/Plan IS an active diagnosis — the presence of
        #    "history of" in the evidence quote is incidental context, not a
        #    signal that the disease is inactive.
        context_section_sources_temporal = context.get("section_sources", [])
        ctx_lower_temporal = {(s or "").lower() for s in context_section_sources_temporal}
        _ACTIVE_SECTIONS_TEMPORAL = {
            "assessment_and_plan", "assessment", "chief_complaint", "history_present_illness",
            "active_problems", "active_problem_list", "problem_list",
            "medications", "current_medications", "plan",
        }
        in_active_section = bool(ctx_lower_temporal & _ACTIVE_SECTIONS_TEMPORAL)

        if not in_active_section:
            assessment_evidence = (meat_result.get("assessment_evidence") or "").lower()
            temporal_markers = ["history of", "past history", "previous", "prior to", "resolved",
                                "remission", "pmh:", "pmh includes", "past medical history",
                                "known history", "hx of", "h/o", "h/x of"]
            
            if any(marker in assessment_evidence for marker in temporal_markers):
                validation["issues_found"].append("Assessment evidence contains temporal markers suggesting a past condition.")
                validation["final_meat"]["assessment"] = False
                validation["final_meat"]["assessment_confidence"] = 0.0
                validation["meat_valid"] = False

        # 4b. PMH-Only Gate: If context reports section_sources as PMH-only,
        #     force override Assessment=False and re-check validity.
        context_section_sources = context.get("section_sources", [])
        ctx_lower = {(s or "").lower() for s in context_section_sources}
        _PMH_NAMES = {"past_medical_history", "medical_history", "pmh"}
        _ACTIVE_NAMES = {"assessment_and_plan", "assessment", "chief_complaint", "history_present_illness"}
        is_pmh_only = bool(ctx_lower & _PMH_NAMES) and not bool(ctx_lower & _ACTIVE_NAMES)
        
        if is_pmh_only:
            # PMH-only: Set Assessment=False since not in active assessment section.
            # But preserve Treatment if there's real medication evidence — a PMH disease
            # with active medication is still being actively managed.
            if validation["final_meat"].get("assessment", False):
                validation["final_meat"]["assessment"] = False
                validation["final_meat"]["assessment_confidence"] = 0.0
                validation["issues_found"].append(
                    "PMH-Only disease: Assessment forced to False — disease found exclusively in Past Medical History."
                )
            # Re-check: if Treatment (or Monitoring/Evaluation) still holds, disease can be valid
            pmh_T = validation["final_meat"].get("treatment", False)
            pmh_M = validation["final_meat"].get("monitoring", False)
            pmh_E = validation["final_meat"].get("evaluation", False)
            if pmh_T and (pmh_M or pmh_E):
                validation["meat_valid"] = True
                validation["validation_rule"] = "PMH with active treatment + monitoring/evaluation [PASS]"
            elif sum([bool(pmh_T), bool(pmh_M), bool(pmh_E)]) >= 2:
                validation["meat_valid"] = True
                validation["validation_rule"] = "PMH with 2+ management criteria [PASS]"
            else:
                validation["meat_valid"] = False

        # 5. Semantic Negation Check (Safety layer)
        negation_words = ["no evidence", "denies", "without", "negative for", "ruled out", "not identified"]
        
        for element in ["monitoring", "evaluation", "assessment", "treatment"]:
            if not validation["final_meat"].get(element, False):
                continue
                
            evidence = (validation["final_meat"].get(f"{element}_evidence") or "").lower()
            if any(neg in evidence for neg in negation_words):
                validation["issues_found"].append(f"{element.upper()}: Evidence quote contains semantic negation.")
                validation["final_meat"][element] = False
                validation["final_meat"][f"{element}_confidence"] = 0.0

        # 6. Re-evaluating Final Validity after deterministic adjustments
        final_M = validation["final_meat"].get("monitoring", False)
        final_E = validation["final_meat"].get("evaluation", False)
        final_A = validation["final_meat"].get("assessment", False)
        final_T = validation["final_meat"].get("treatment", False)
        final_count = sum([bool(final_M), bool(final_E), bool(final_A), bool(final_T)])
        
        # Valid if: A+(M|E|T), or T+(M|E), or 2+ criteria met
        if final_A and (final_M or final_E or final_T):
            pass  # keep valid
        elif final_T and (final_M or final_E):
            pass  # keep valid — active treatment with monitoring/evaluation
        elif final_count >= 2:
            pass  # keep valid — sufficient evidence
        else:
            validation["meat_valid"] = False
            if validation["validation_rule"].endswith("[PASS]"):
                validation["validation_rule"] = "REVERSED: Failed deterministic clinical rules"

        # 7. Audit Compliance Check
        if settings.MEAT_REQUIRE_EVIDENCE and validation["meat_valid"]:
            has_any_verifiable_evidence = False
            for element in ["monitoring", "evaluation", "assessment", "treatment"]:
                if validation["final_meat"].get(element, False):
                    ev = validation["final_meat"].get(f"{element}_evidence", "")
                    if len(ev.strip()) < 8:
                        validation["issues_found"].append(f"{element.upper()}: Insufficient evidence for audit compliance.")
                        validation["requires_manual_review"] = True
                    else:
                        has_any_verifiable_evidence = True
            # If no element has verifiable evidence, MEAT is invalid
            if not has_any_verifiable_evidence:
                validation["meat_valid"] = False
                validation["issues_found"].append("No MEAT element has verifiable evidence text — overall MEAT rejected.")

        self.logger.info(
            f"MEAT Gate [{disease_name}]: Valid={validation['meat_valid']}, "
            f"Tier={validation['confidence_tier']}, Issues={len(validation['issues_found'])}"
        )
        
        return validation

    def _get_all_context_text(self, context: Dict[str, Any]) -> str:
        """Helper to flatten context for verbatim matching."""
        return " ".join([
            context.get("primary_mention", ""),
            " ".join(context.get("context_sentences", [])),
            context.get("section_context", ""),
            " ".join(context.get("related_sentences", []))
        ])

    def batch_validate(
        self,
        meat_results: Dict[str, Dict[str, Any]],
        contexts: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate a batch of results against deterministic rules.
        """
        validations = {}
        for disease_name, meat_result in meat_results.items():
            context = contexts.get(disease_name, {})
            validations[disease_name] = self.validate_meat_result(meat_result, context)
            
        valid_count = sum(1 for v in validations.values() if v["meat_valid"])
        self.logger.info(f"Batch validation complete: {valid_count}/{len(validations)} entities passed the gate.")
        return validations

# Singleton instance
meat_gate = MEATValidationGate()
