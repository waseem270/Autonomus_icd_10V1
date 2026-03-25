from google import genai
from google.genai import types
from typing import List, Dict, Any, Optional
import json
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.config import settings
from ..utils.gemini_retry import call_gemini_safe as call_gemini

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safety settings — disable content filtering for medical documents
# ---------------------------------------------------------------------------
SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
]


class ICDRanker:
    """
    Rank ICD-10 candidates using Gemini based on clinical context 
    and MEAT (Monitoring, Evaluation, Assessment, Treatment) evidence.
    
    Uses the new google.genai SDK with async support and ThinkingConfig
    for better clinical reasoning about code specificity.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        try:
            self._client = None
            self.model_name = settings.GEMINI_MODEL
            
            # Config for ICD ranking — structured JSON, deterministic, 
            # with extended thinking for better clinical reasoning
            self.config = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=4096,
                response_mime_type="application/json",
                safety_settings=SAFETY_SETTINGS,
                system_instruction=(
                    "You are a clinical coding expert. Your mission is to rank "
                    "ICD-10-CM codes based on clinical documentation specificity "
                    "and the official coding guidelines. Always prefer the most "
                    "detailed code supported by documented evidence (laterality, chronicity, severity)."
                )
            )
            self.logger.info(f"ICD Ranker initialized with model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini for ICD Ranker: {e}")
            self._client = None

    @property
    def client(self):
        if self._client is None:
            from ..core.config import create_genai_client
            self._client = create_genai_client()
        return self._client
    
    async def rank_candidates(
        self,
        disease_name: str,
        candidates: List[Dict[str, Any]],
        meat_evidence: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Rank ICD candidates by clinical appropriateness using LLM reasoning.
        Uses the new google.genai async API.
        """
        if not candidates:
            return []
            
        if not self.client:
            self.logger.error("Gemini client not initialized. Using fallback ranking.")
            return self._fallback_ranking(candidates)
        
        self.logger.info(f"Ranking {len(candidates)} candidates for diagnosis: '{disease_name}'")
        
        prompt = self._build_ranking_prompt(disease_name, candidates, meat_evidence)
        
        try:
            response = await call_gemini(
                client=self.client,
                model=self.model_name,
                contents=prompt,
                config=self.config,
            )
            
            # Extract text from response parts (skip thinking parts)
            response_text = ""
            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if not getattr(part, "thought", False) and part.text:
                        response_text += part.text
            
            response_text = response_text.strip()
            
            # Extract JSON from markdown if necessary
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].strip()
            
            ranking_result = json.loads(response_text)
            
            ranked_candidates = []
            for rank_data in ranking_result.get("rankings", []):
                icd_code = rank_data.get("icd_code")
                
                # Cross-reference with our actual candidates to preserve description and metadata
                original = next(
                    (c for c in candidates if c["icd_code"] == icd_code),
                    None
                )
                
                if original:
                    llm_score = rank_data.get("score", 0.5)
                    original_conf = original.get("confidence", 0.5)
                    
                    # Weighted logic: 70% AI clinical reasoning, 30% fuzzy string match
                    final_confidence = round((llm_score * 0.7) + (original_conf * 0.3), 3)
                    
                    ranked_candidates.append({
                        "icd_code": icd_code,
                        "description": original["description"],
                        "confidence": final_confidence,
                        "match_type": "llm_ranked",
                        "llm_rank": rank_data.get("rank", 999),
                        "llm_score": llm_score,
                        "llm_reasoning": rank_data.get("reasoning", "")
                    })
            
            # Sort by rank (primary) then confidence (secondary)
            ranked_candidates.sort(key=lambda x: (x["llm_rank"], -x["confidence"]))
            
            return ranked_candidates
            
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"Ranking failed for '{disease_name}': {e}")
            return self._fallback_ranking(candidates)
    
    def _build_ranking_prompt(
        self,
        disease_name: str,
        candidates: List[Dict[str, Any]],
        meat_evidence: Dict[str, Any]
    ) -> str:
        """Construct a high-precision clinical ranking prompt."""
        
        # Format top 10 candidates for the LLM
        candidates_text = ""
        for i, candidate in enumerate(candidates[:10], 1):
            candidates_text += f"{i}. {candidate['icd_code']}: {candidate['description']}\n"
        
        # Pull MEAT evidence details
        assessment = meat_evidence.get("assessment_evidence", "No assessment documented.")
        treatment = meat_evidence.get("treatment_evidence", "No specific treatment mentioned.")
        evaluation = meat_evidence.get("evaluation_evidence", "No evaluation markers found.")
        monitoring = meat_evidence.get("monitoring_evidence", "No active monitoring found.")
        
        prompt = f"""As a Senior Medical Coder (HIM Expert), rank the following ICD-10 code candidates for a detected diagnosis.

DIAGNOSIS DETECTED: "{disease_name}"

CLINICAL EVIDENCE (MEAT):
- ASSESSMENT: {assessment}
- TREATMENT: {treatment}
- EVALUATION: {evaluation}
- MONITORING: {monitoring}

ICD-10 CANDIDATES (FROM DATABASE):
{candidates_text}

RANKING RULES — READ CAREFULLY:
1. PRIMARY CONDITION MATCH (CRITICAL): The ICD-10 code must represent "{disease_name}" as the MAIN/PRIMARY diagnosis — NOT as a secondary complication or manifestation of another condition.
   - CORRECT: "Anxiety disorder, unspecified" (F41.9) for "anxiety" — anxiety IS the primary diagnosis
   - WRONG: "Vascular dementia ... with anxiety" (F01.50) for "anxiety" — here anxiety is a secondary feature of dementia
   - CORRECT: "Major depressive disorder, recurrent, moderate" (F33.1) for "Major Depression"
   - WRONG: "Depressive episode due to vascular disease" for "Major Depression"
2. SPECIFICITY MATCHING: Choose codes that match the SPECIFIC qualifiers documented:
   - If "recurrent" is documented → prefer recurrent codes (F33.x) over single episode (F32.x)
   - If "moderate" is documented → prefer moderate codes
   - If "unspecified" in the disease name → use unspecified code
3. AVOID codes where the diagnosis is only mentioned as:
   - A complication of a different major condition
   - A manifestation "due to" another disease
   - A secondary feature listed after commas
4. If documentation is generic → select the most appropriate "unspecified" code for that primary condition.
5. NO upcoding: only assign codes for what is EXPLICITLY documented.

TASK:
- Rank ALL provided candidates.
- Give score 0.0 to 1.0 (1.0 = Perfect match as PRIMARY condition).
- The #1 ranked candidate must be the one where "{disease_name}" is definitively the MAIN diagnosis.

Return ONLY valid JSON:
{{
  "rankings": [
    {{
      "rank": 1,
      "icd_code": "CODE",
      "score": 0.95,
      "reasoning": "Brief clinical rationale — why this code correctly represents the primary condition"
    }}
  ]
}}
"""
        return prompt
    
    def _fallback_ranking(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback to original fuzzy order if LLM fails."""
        ranked = []
        for i, candidate in enumerate(candidates, 1):
            ranked.append({
                "icd_code": candidate["icd_code"],
                "description": candidate["description"],
                "confidence": candidate.get("confidence", 0.5),
                "match_type": "fuzzy_match",
                "llm_rank": i,
                "llm_score": 0.5,
                "llm_reasoning": "Fallback order (AI ranking currently unavailable)"
            })
        return ranked

# Singleton instance
icd_ranker = ICDRanker()
