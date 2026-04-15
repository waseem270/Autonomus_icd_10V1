import logging
from typing import Dict, List, Any, Optional

from .context_builder import context_builder
from .meat_validator import meat_validator
from .meat_gate import meat_gate

logger = logging.getLogger(__name__)

class MEATProcessor:
    """
    Orchestrates the complete MEAT (Monitoring, Evaluation, Assessment, Treatment) 
    validation pipeline for clinical entities.
    
    Pipeline Steps:
    1. Filter: Remove negated or low-confidence raw detections.
    2. Context: Build rich clinical windows around each entity.
    3. Reasoning: Use Gemini to identify MEAT evidence.
    4. Gating: Enforce deterministic rules and prevent hallucinations.
    5. Categorization: Finalize status (Valid, Review, Invalid, History Only).
    6. Disease Status Classification: Active Chronic, Active Acute, Chronic.
    """

    # Section-based status classification
    _CHRONIC_KEYWORDS = {
        "chronic", "long-standing", "longstanding", "ongoing", "maintained",
        "well-controlled", "poorly controlled", "stable", "uncontrolled",
        "recurrent", "persistent", "established", "known history",
        "history of", "hx of", "moderate", "severe",
    }
    _ACUTE_KEYWORDS = {
        "acute", "new onset", "new-onset", "sudden", "recent", "worsening",
        "exacerbation", "flare", "presenting", "presents with", "today",
        "current episode", "active",
    }
    _PMH_SECTION_NAMES = {
        "past_medical_history", "medical_history", "pmh",
    }
    _ACTIVE_SECTION_NAMES = {
        "assessment_and_plan", "assessment", "chief_complaint",
        "history_present_illness", "active_problems", "active_problem_list",
        "problem_list", "medications", "current_medications", "plan",
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def process_meat_validation(
        self,
        detected_diseases: List[Dict[str, Any]],
        full_text: str,
        sentences: List[Dict[str, Any]],
        sections: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run the full validation pipeline on a set of detected clinical entities.
        
        Args:
            detected_diseases: List of entities from the NER phase.
            full_text: The complete cleaned document text.
            sentences: Segmented sentences with metadata.
            sections: Detected clinical sections.
            
        Returns:
            A results object containing individual validations and summary stats.
        """
        self.logger.info("🚀 Initializing MEAT validation pipeline...")

        # 1. Filter: We only validate conditions that were actually detected as present
        non_negated_diseases = [
            d for d in detected_diseases 
            if not d.get("negated", False)
        ]
        
        self.logger.info(
            f"Pipeline input: {len(non_negated_diseases)} active conditions "
            f"(filtered from {len(detected_diseases)} total detections)"
        )

        if not non_negated_diseases:
            return self._empty_response("No non-negated diseases found for validation.")

        # 2. Context Building: Gather evidence from surrounding text
        self.logger.info("Step 1/3: Building clinical context windows...")
        disease_contexts = context_builder.build_contexts_for_all_diseases(
            detected_diseases=non_negated_diseases,
            full_text=full_text,
            sentences=sentences,
            sections=sections
        )

        # 3. LLM Reasoning: Ask Gemini to find M/E/A/T markers
        self.logger.info(f"Step 2/3: Performing LLM-based MEAT reasoning for {len(disease_contexts)} entities...")
        meat_results = await meat_validator.validate_multiple_diseases(
            disease_contexts
        )

        # 4. Gating: Run deterministic rules (Firewall)
        self.logger.info("Step 3/3: Applying deterministic validation gate (Hallucination check)...")
        validations = meat_gate.batch_validate(
            meat_results=meat_results,
            contexts=disease_contexts
        )

        # 5. Final Assembly & Categorization
        combined_results = []
        valid_count = 0
        review_count = 0
        invalid_count = 0
        history_only_count = 0

        for disease_name, meat_data in meat_results.items():
            validation = validations.get(disease_name)
            if not validation:
                continue

            # ── MEAT Tier Classification ──────────────────────────────────
            # MUST use gate-adjusted values (final_meat), not raw LLM output
            final = validation.get("final_meat", meat_data)
            M = bool(final.get("monitoring", False))
            E = bool(final.get("evaluation", False))
            A = bool(final.get("assessment", False))
            T = bool(final.get("treatment", False))
            meat_count = M + E + A + T

            # Find the original disease entry to get section_sources
            original_disease = next(
                (d for d in non_negated_diseases if d["disease_name"] == disease_name),
                None
            )
            segment_sources = []
            if original_disease:
                segment_sources = original_disease.get("section_sources", [original_disease.get("section", "unknown")])

            # Get disease context for status classification
            disease_context = disease_contexts.get(disease_name, {})

            # Check if this disease is ONLY from Past Medical History
            sections_lower = {s.lower() for s in segment_sources}
            is_pmh_only = bool(sections_lower & self._PMH_SECTION_NAMES) and not bool(sections_lower & self._ACTIVE_SECTION_NAMES)

            # Tier Rules (evidence-based):
            #   Strong Evidence → 3+ criteria with documented evidence
            #   Moderate Evidence → 2 criteria with documented evidence
            #   Weak Evidence → 1 criterion with documented evidence
            #   No MEAT → No documented evidence found
            if validation["meat_valid"] and meat_count >= 3:
                meat_tier = "strong_evidence"
            elif validation["meat_valid"] and meat_count >= 2:
                meat_tier = "moderate_evidence"
            elif meat_count >= 1:
                meat_tier = "weak_evidence"
            elif is_pmh_only:
                meat_tier = "no_meat"
            else:
                meat_tier = "no_meat"

            # Determine final status for the caller
            if meat_tier == "no_meat" and is_pmh_only:
                final_status = "history_only"
                history_only_count += 1
            elif validation["meat_valid"]:
                if validation["requires_manual_review"]:
                    final_status = "review"
                    review_count += 1
                else:
                    final_status = "valid"
                    valid_count += 1
            else:
                final_status = "invalid"
                invalid_count += 1

            # Classify disease status (Active Chronic, Active Acute, Chronic, History Only)
            disease_status = self._classify_disease_status(
                disease_name, segment_sources, disease_context
            )

            # Build record for this disease
            combined_results.append({
                "disease": disease_name,
                "final_status": final_status,
                "meat_tier": meat_tier,
                "disease_status": disease_status,
                "meat_criteria_count": meat_count,
                "meat_data": meat_data,
                "validation": validation,
                "segment_source": segment_sources,
                "context_summary": {
                    "window_size": disease_contexts[disease_name].get("window_size", 0),
                    "has_medications": len(disease_contexts[disease_name].get("medication_mentions", [])) > 0,
                    "has_labs": len(disease_contexts[disease_name].get("lab_mentions", [])) > 0
                }
            })

        summary = {
            "status": "completed",
            "total_processed": len(combined_results),
            "valid": valid_count,
            "review_required": review_count,
            "invalid": invalid_count,
            "history_only": history_only_count,
            "success_rate": round((valid_count + review_count) / len(combined_results), 2) if combined_results else 0
        }

        self.logger.info(
            f"✅ Pipeline Complete: {valid_count} Valid, {review_count} Review, "
            f"{history_only_count} History Only, {invalid_count} Invalid."
        )

        return {
            "total_diseases": len(combined_results),
            "valid_meat_count": valid_count,
            "requires_review_count": review_count,
            "meat_results": combined_results,
            "processing_summary": summary
        }

    def _empty_response(self, message: str) -> Dict[str, Any]:
        """Return a structured empty response."""
        return {
            "total_diseases": 0,
            "valid_meat_count": 0,
            "requires_review_count": 0,
            "meat_results": [],
            "processing_summary": {
                "status": "no_data",
                "message": message,
                "total_processed": 0
            }
        }

    def _classify_disease_status(
        self,
        disease_name: str,
        section_sources: List[str],
        context: Dict[str, Any],
    ) -> str:
        """
        Classify a disease as: Active Chronic, Active Acute, Chronic, or History Only.
        
        Logic:
        - History Only: Found ONLY in Past Medical History, no mention in active sections
        - Active Chronic: Mentioned in active sections (Assessment, HPI) + chronic keywords
        - Active Acute: Mentioned in active sections + acute keywords (or Chief Complaint)
        - Chronic: Mentioned in PMH + at least one active section, OR has chronic markers
        """
        sections_lower = {s.lower() for s in section_sources}
        
        in_pmh = bool(sections_lower & self._PMH_SECTION_NAMES)
        in_active = bool(sections_lower & self._ACTIVE_SECTION_NAMES)
        
        # Gather all text context for keyword search
        context_text = " ".join([
            context.get("primary_mention", ""),
            " ".join(context.get("context_sentences", [])),
            context.get("section_context", ""),
        ]).lower()
        
        has_chronic_markers = any(kw in context_text for kw in self._CHRONIC_KEYWORDS)
        has_acute_markers = any(kw in context_text for kw in self._ACUTE_KEYWORDS)
        
        # Priority classification
        # PMH diseases with medication evidence are actively treated, not just history
        has_meds = bool(context.get("medication_mentions"))
        if in_pmh and not in_active and not has_meds and not has_chronic_markers:
            return "History Only"
        
        if in_active and has_acute_markers and not has_chronic_markers:
            return "Active Acute"
        
        if in_active and has_chronic_markers:
            return "Active Chronic"
        
        if in_active:
            # Default active diseases to Active Chronic if no clear marker
            return "Active Chronic"
        
        # Fallback: in some section but not clearly active or PMH
        if has_chronic_markers:
            return "Chronic"
        
        return "Chronic"

# Singleton instance
meat_processor = MEATProcessor()
