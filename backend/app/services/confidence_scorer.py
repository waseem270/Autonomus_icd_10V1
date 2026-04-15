"""
Confidence Scorer — multi-factor confidence scoring for disease results.

Computes a composite score from three weighted components:
  1. MEAT completeness (how many of M/E/A/T are satisfied)
  2. ICD mapping confidence (from the mapping pipeline)
  3. Disease extraction confidence (how reliably the disease was detected)

Weights are configurable via settings.CONFIDENCE_WEIGHT_*.
"""

import logging
from typing import Dict, List

from ..core.config import settings

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Compute composite confidence scores for disease results."""

    def score(self, result: Dict) -> float:
        """Score a single disease result and return the composite confidence."""
        w_meat = settings.CONFIDENCE_WEIGHT_MEAT
        w_icd = settings.CONFIDENCE_WEIGHT_ICD
        w_disease = settings.CONFIDENCE_WEIGHT_DISEASE

        # MEAT completeness (0.0–1.0)
        meat_count = sum([
            bool(result.get("monitoring")),
            bool(result.get("evaluation")),
            bool(result.get("assessment")),
            bool(result.get("treatment")),
        ])
        meat_score = meat_count / 4.0

        # Bonus for having real evidence text (not just booleans)
        evidence_fields = [
            "monitoring_evidence", "evaluation_evidence",
            "assessment_evidence", "treatment_evidence",
        ]
        evidence_count = sum(
            1 for f in evidence_fields if (result.get(f) or "").strip()
        )
        evidence_bonus = min(evidence_count * 0.05, 0.15)  # up to +0.15
        meat_score = min(meat_score + evidence_bonus, 1.0)

        # ICD confidence (0.0–1.0)
        icd_conf = result.get("icd_confidence", 0.0)
        if isinstance(icd_conf, str):
            try:
                icd_conf = float(icd_conf)
            except ValueError:
                icd_conf = 0.0

        # Disease extraction confidence — from tier mapping
        tier = result.get("meat_tier", "no_meat")
        tier_scores = {
            "strong_evidence": 1.0,
            "moderate_evidence": 0.75,
            "agent2_recall": 0.55,
            "weak_evidence": 0.4,
            "no_meat": 0.0,
        }
        disease_conf = tier_scores.get(tier, 0.0)

        composite = round(
            w_meat * meat_score + w_icd * icd_conf + w_disease * disease_conf,
            3,
        )
        return max(0.0, min(composite, 1.0))

    def score_batch(self, results: List[Dict]) -> List[Dict]:
        """Score a batch and inject ``confidence`` into each result dict."""
        for r in results:
            r["confidence"] = self.score(r)
        return results


# Singleton
confidence_scorer = ConfidenceScorer()
