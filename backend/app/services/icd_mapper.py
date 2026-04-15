from typing import Dict, List, Optional, Any
import logging

from .icd_lookup import icd_lookup
from .icd_ranker import icd_ranker
from .deterministic_validator import deterministic_validator
from ..core.config import settings

logger = logging.getLogger(__name__)

class ICDMapper:
    """
    Complete ICD-10 mapping pipeline.
    Orchestrates deterministic exact lookup, multi-candidate fuzzy retrieval, 
    and LLM-based clinical ranking against MEAT evidence.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def map_disease_to_icd(
        self,
        disease_name: str,
        normalized_name: str,
        meat_validation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Map a disease to its most appropriate ICD-10 code.
        
        Args:
            disease_name: Original clinical term
            normalized_name: NLP-standardized term
            meat_validation: Structured output from the MEAT gate containing evidence.
            
        Returns:
            A finalized mapping record with status (auto_assigned/manual_review/not_found).
        """
        result = {
            "disease": disease_name,
            "icd_code": None,
            "icd_description": None,
            "mapping_method": "none",
            "confidence": 0.0,
            "status": "not_found",
            "candidates": [],
            "llm_reasoning": ""
        }
        
        self.logger.info(f"🔍 Initiating mapping for clinical entity: '{disease_name}'")
        
        # Step 1: Sequential Exact Match (Normalized -> Original)
        search_result = icd_lookup.search_combined(disease_name, normalized_name)
        
        if search_result.get("exact_match"):
            exact = search_result["exact_match"]
            result.update({
                "icd_code": exact["icd_code"],
                "icd_description": exact["description"],
                "mapping_method": "exact_match",
                "confidence": 1.0,
                "status": "auto_assigned",
                "candidates": [exact]
            })
            self.logger.info(f"✅ Exact match found: {exact['icd_code']}")
            return result
        
        # Step 2: Broad Fuzzy Search
        fuzzy_candidates = search_result.get("fuzzy_matches", [])
        
        if not fuzzy_candidates:
            self.logger.warning(f"⚠️ No matching ICD codes found for '{disease_name}'")
            return result
        
        self.logger.info(f"Retrieved {len(fuzzy_candidates)} candidates. Performing AI clinical ranking...")
        
        # Step 3: LLM Clinical Priority Ranking
        # Skip LLM ranker if top fuzzy match is already high-confidence
        top_conf = fuzzy_candidates[0].get("confidence", 0) if fuzzy_candidates else 0
        if top_conf >= 0.95:
            self.logger.info(f"Top fuzzy match confidence {top_conf:.2f} >= 0.95, skipping LLM ranker")
            ranked_candidates = fuzzy_candidates
        elif meat_validation and meat_validation.get("meat_valid"):
            meat_evidence = {
                "assessment_evidence": meat_validation.get("assessment_evidence", ""),
                "treatment_evidence": meat_validation.get("treatment_evidence", ""),
                "evaluation_evidence": meat_validation.get("evaluation_evidence", ""),
                "monitoring_evidence": meat_validation.get("monitoring_evidence", "")
            }
            
            ranked_candidates = await icd_ranker.rank_candidates(
                disease_name=disease_name,
                candidates=fuzzy_candidates,
                meat_evidence=meat_evidence
            )
        else:
            # Fallback: No clinical audit evidence provided, use string similarity ranks
            ranked_candidates = fuzzy_candidates
            
        result["candidates"] = ranked_candidates
        
        if not ranked_candidates:
            return result
            
        # Step 4: Final Selection
        best_candidate = ranked_candidates[0]
        # Normalize match_type to DB enum values
        _METHOD_MAP = {"exact": "exact_match", "fuzzy": "fuzzy_match",
                       "exact_match": "exact_match", "fuzzy_match": "fuzzy_match",
                       "llm_ranked": "llm_ranked"}
        raw_method = best_candidate.get("match_type", "fuzzy_match")
        result.update({
            "icd_code": best_candidate["icd_code"],
            "icd_description": best_candidate["description"],
            "confidence": best_candidate["confidence"],
            "mapping_method": _METHOD_MAP.get(raw_method, "fuzzy_match"),
            "llm_reasoning": best_candidate.get("llm_reasoning", "")
        })
        
        # Step 5: Deterministic Validation — reject hallucinated codes
        if result["icd_code"]:
            validation = deterministic_validator.validate_icd_code(result["icd_code"])
            if not validation["valid"]:
                self.logger.warning(
                    f"❌ ICD code '{result['icd_code']}' failed validation: {validation['reason']}"
                )
                result.update({
                    "icd_code": None, "icd_description": None,
                    "confidence": 0.0, "status": "not_found",
                    "mapping_method": "rejected_invalid",
                    "llm_reasoning": f"Code rejected: {validation['reason']}"
                })
                return result

            # Cross-validate disease name against ICD description
            xval = deterministic_validator.cross_validate_disease_icd(
                disease_name, result["icd_code"]
            )
            if not xval["match"]:
                self.logger.warning(
                    f"❌ Cross-validation failed for '{disease_name}' → {result['icd_code']}: {xval['reason']}"
                )
                result["status"] = "manual_review"
                result["confidence"] = min(result["confidence"], 0.5)
                result["llm_reasoning"] += f" | Cross-validation warning: {xval['reason']}"

        # Step 6: Confidence-Based Control (Gating)
        # Auto-assign high-confidence matches; lower confidence → manual review
        auto_threshold = getattr(settings, "ICD_AUTO_ASSIGN_THRESHOLD", 0.90)
        review_threshold = getattr(settings, "ICD_MANUAL_REVIEW_THRESHOLD", 0.70)

        if result["confidence"] >= auto_threshold:
            result["status"] = "auto_assigned"
        elif result["confidence"] >= review_threshold:
            result["status"] = "manual_review"
        else:
            result["status"] = "manual_review"
                
        self.logger.info(
            f"🎯 Final ICD Assignment: {result['icd_code']} | "
            f"Confidence: {result['confidence']:.2f} | "
            f"Status: {result['status']}"
        )
        
        return result
    
    async def map_multiple_diseases(
        self,
        diseases_with_meat: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Orchestrate the mapping of all detected diseases in parallel.
        """
        import asyncio
        semaphore = asyncio.Semaphore(5)

        async def _map_one(item):
            async with semaphore:
                return await self.map_disease_to_icd(
                    disease_name=item["disease_name"],
                    normalized_name=item["normalized_name"],
                    meat_validation=item.get("meat_validation", {})
                )

        self.logger.info(f"Mapping {len(diseases_with_meat)} diseases to ICD codes in parallel...")
        results = await asyncio.gather(
            *[_map_one(item) for item in diseases_with_meat],
            return_exceptions=True
        )

        mappings = []
        stats = {"auto_assigned": 0, "manual_review": 0, "not_found": 0}
        for r in results:
            if isinstance(r, Exception):
                self.logger.error(f"ICD mapping task failed: {r}")
                continue
            mappings.append(r)
            status = r["status"]
            if status in stats:
                stats[status] += 1

        return {
            "total_diseases": len(mappings),
            "auto_assigned": stats["auto_assigned"],
            "manual_review": stats["manual_review"],
            "not_found": stats["not_found"],
            "mappings": mappings
        }

# Create singleton instance for application use
icd_mapper = ICDMapper()
