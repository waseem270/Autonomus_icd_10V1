import json
import logging
from typing import Dict, List, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential

from google import genai
from google.genai import types

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


class MEATValidator:
    """
    Validate MEAT (Monitoring, Evaluation, Assessment, Treatment) criteria for 
    clinical diagnoses using Gemini's reasoning capabilities.
    
    Uses the new google.genai SDK with async support and ThinkingConfig
    for better clinical reasoning.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        try:
            self._client = None
            from ..core.config import get_active_model
            self.model_name = get_active_model()
            
            # Config for MEAT validation — needs structured JSON output
            # with extended thinking for better clinical reasoning
            self.config = types.GenerateContentConfig(
                temperature=0.1,   # Slight temperature for varied evidence extraction
                max_output_tokens=4096,
                response_mime_type="application/json",
                safety_settings=SAFETY_SETTINGS,
                system_instruction=(
                    "You are a specialized Medical Risk Adjustment Auditor. "
                    "Your task is to validate MEAT (Monitoring, Evaluation, Assessment, Treatment) "
                    "criteria for clinical diagnoses using documented evidence. "
                    "Be strict: evidence must be explicit and relevant to the specific diagnosis."
                )
            )
            self.logger.info(f"MEAT Validator initialized with model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM for MEAT Validation: {e}")
            self._client = None

    @property
    def client(self):
        if self._client is None:
            from ..core.config import get_llm_provider, create_genai_client, create_openai_client
            provider = get_llm_provider()
            if provider == "openai":
                oai = create_openai_client()
                self._client = oai if oai else "__openai_sentinel__"
            else:
                self._client = create_genai_client()
        return self._client

    async def validate_meat(
        self,
        disease_name: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate MEAT for a specific disease using its clinical context.
        
        Uses the new google.genai async API (client.aio.models.generate_content)
        instead of deprecated synchronous generate_content().
        """
        if not self.client:
            self.logger.warning("Gemini client not initialized. Using fallback validation.")
            return self._fallback_meat_validation(disease_name, context)

        prompt = self._build_meat_prompt(disease_name, context)
        
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
            
            # Clean JSON formatting if Gemini returns markdown blocks
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            result = json.loads(response_text)
            
            result["disease"] = disease_name
            result["overall_confidence"] = self._calculate_overall_confidence(result)
            
            self.logger.info(
                f"MEAT validation for '{disease_name}': "
                f"M={result.get('monitoring')}, E={result.get('evaluation')}, "
                f"A={result.get('assessment')}, T={result.get('treatment')} "
                f"[confidence={result['overall_confidence']:.2f}]"
            )
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON Parsing failed for MEAT response ('{disease_name}'): {e}")
            self.logger.debug(f"Raw response: {response_text[:500]}")
            return self._fallback_meat_validation(disease_name, context)
        
        except Exception as e:
            self.logger.error(f"MEAT validation failed for '{disease_name}': {e}")
            return self._fallback_meat_validation(disease_name, context)

    def _build_meat_prompt(self, disease_name: str, context: Dict[str, Any]) -> str:
        """Construct a high-precision clinical reasoning prompt for Gemini."""
        
        # Format lists for cleaner prompt
        related = "\n".join(context.get('related_sentences', [])[:3])
        meds = "\n".join(context.get('medication_mentions', [])[:5])
        labs = "\n".join(context.get('lab_mentions', [])[:5])
        surrounding = "\n".join(context.get('context_sentences', []))

        # Build section provenance note for the LLM
        section_sources = context.get('section_sources', [context.get('section', 'unknown')])
        sections_lower = {s.lower() for s in section_sources}
        _PMH = {"past_medical_history", "medical_history", "pmh"}
        _ACTIVE = {"assessment_and_plan", "assessment", "chief_complaint", "history_present_illness"}
        is_pmh_only = bool(sections_lower & _PMH) and not bool(sections_lower & _ACTIVE)
        section_note = f"Detected in sections: {', '.join(section_sources)}"
        pmh_warning = ""
        if is_pmh_only:
            pmh_warning = """
⚠️ PMH-ONLY DISEASE: This condition was found EXCLUSIVELY in Past Medical History.
   - ASSESSMENT should be FALSE unless the note EXPLICITLY lists this as an ACTIVE problem in Assessment/Plan.
   - A listing in PMH (e.g. "PMH: hypertension") does NOT count as Assessment.
   - TREATMENT should be FALSE unless a NEW prescription or active management plan is stated for this visit.
   - "Continue current medications" or general medication lists do NOT count as Treatment for this specific disease unless the disease is explicitly named."""

        prompt = f"""You are a specialized Medical Risk Adjustment Auditor. Your task is to validate MEAT (Monitoring, Evaluation, Assessment, Treatment) criteria for ONE specific clinical diagnosis using extracted document context.

⚠️  CRITICAL: You are ONLY validating evidence for "{disease_name}" — NOT for any other condition. 
If the context contains evidence for other diseases (e.g. hypertension evidence when validating anxiety), you MUST IGNORE that evidence completely.

DIAGNOSIS TO VALIDATE: {disease_name}
SECTION SOURCE: {section_note}{pmh_warning}

CLINICAL CONTEXT (some surrounding text may mention other diseases — focus only on {disease_name}):

[Primary Mention of {disease_name}]
{context.get('primary_mention', 'N/A')}

[Surrounding Context Lines]
{surrounding}

[Filtered Section Content for {disease_name}]
{context.get('section_context', 'N/A')}

[Other Mentions of {disease_name} Elsewhere in Document]
{related or 'None found'}

[Medication Context — focus on medications for {disease_name}]
{meds or 'None found'}

[Lab/Diagnostic Context — focus on labs related to {disease_name}]
{labs or 'None found'}

INSTRUCTIONS:
Determine if there is EXPLICIT evidence in the provided context for any of the MEAT elements FOR {disease_name} SPECIFICALLY:
- MONITORING: Tracking/observing THIS condition. (e.g. if validating "anxiety": "anxiety monitored", "follow up for anxiety")
- EVALUATION: Diagnostic tests or assessments FOR THIS SPECIFIC condition
- ASSESSMENT: THIS SPECIFIC DISEASE explicitly listed as an active current diagnosis in Assessment/Plan
- TREATMENT: Active medication/therapy specifically for THIS disease in this visit

CRITICAL RULES:
1. DISEASE-SPECIFIC ONLY: Evidence must be specifically about "{disease_name}" — NOT about other conditions.
   - If context contains "hypertension: blood pressure monitored" while you are validating "anxiety" → DO NOT use that as monitoring evidence for anxiety.
   - If context contains "sertraline for depression" while validating "insomnia" → DO NOT count that as treatment for insomnia unless insomnia is explicitly mentioned with the medication.
2. EXPLICIT EVIDENCE ONLY: Do not infer or hallucinate clinical facts.
3. EXACT QUOTES: The evidence string MUST be a direct, verbatim quote relating specifically to "{disease_name}".
4. CONFIDENCE SCORING: 0.0 to 1.0. High confidence (>0.8) requires specific, unambiguous text.
5. If an element is missing for THIS specific disease, set value to false and confidence to 0.0.
6. PMH RULE: If disease is listed ONLY in Past Medical History, Assessment=false and Treatment=false unless THIS visit explicitly re-addresses it.

REQUIRED JSON FORMAT (Return ONLY valid JSON):
{{
  "monitoring": true/false,
  "monitoring_evidence": "exact quote about monitoring {disease_name} specifically",
  "monitoring_confidence": 0.0,
  "evaluation": true/false,
  "evaluation_evidence": "exact quote about evaluating {disease_name} specifically",
  "evaluation_confidence": 0.0,
  "assessment": true/false,
  "assessment_evidence": "exact quote listing {disease_name} as active diagnosis",
  "assessment_confidence": 0.0,
  "treatment": true/false,
  "treatment_evidence": "exact quote about treating {disease_name} specifically",
  "treatment_confidence": 0.0,
  "llm_reasoning": "Brief clinician rationale for your decisions about {disease_name}"
}}
"""
        return prompt

    def _calculate_overall_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate weighted average confidence for identified MEAT elements."""
        confidences = [
            result.get("monitoring_confidence", 0.0),
            result.get("evaluation_confidence", 0.0),
            result.get("assessment_confidence", 0.0),
            result.get("treatment_confidence", 0.0)
        ]
        active_credits = [c for c in confidences if c > 0.0]
        if not active_credits:
            return 0.0
        return round(sum(active_credits) / len(active_credits), 2)

    # ── Keyword lists for rule-based MEAT fallback ──────────────────────────
    _MONITORING_KW = [
        "follow up", "followup", "f/u", "monitor", "stable", "tracking",
        "watch", "recheck", "re-check", "continue to", "maintaining",
        "managed", "controlled", "checked", "surveillance", "observe",
        "recurrence", "return visit", "return to clinic", "routine",
        "interval", "reassess", "reevaluate", "re-evaluate", "annual",
        "periodic", "ongoing", "maintenance",
    ]
    _EVALUATION_KW = [
        "lab ", "labs", "test ", "tests", "tested", "result", "showed",
        "reveal", "indicate", "level", "score", "measure", "imaging",
        "x-ray", "mri", "ct scan", "biopsy", "hba1c", "a1c",
        "creatinine", "gfr", "egfr", "cholesterol", "blood pressure",
        "bp ", "ekg", "ecg", "echocardiogram", "ultrasound", "diagnosed",
        "workup", "evaluated", "physical exam", "examined",
        "urinalysis", "panel", "culture", "pathology",
        "spirometry", "dexa", "bmi", "phq", "phq9", "phq-9",
        "worsening", "improving", "unchanged", "noted", "observed",
        "finding", "monitor cr", "serial", "stable on",
    ]
    _ASSESSMENT_KW = [
        "diagnosis", "diagnosed", "assessment", "impression", "presents with",
        "problem list", "active problem", "condition", "clinical finding",
        "consistent with", "suggestive of", "confirm", "establish",
        "known history", "history of", "chronic", "acute", "uncontrolled",
        "well-controlled", "poorly controlled", "stage", "type 2", "type 1",
        "moderate", "severe", "mild",
    ]
    _TREATMENT_KW = [
        "prescrib", "taking", " mg ", " mg,", "mg/", "tablet", "refill",
        "medication", "rx ", "dose", "therapy", "surgery", "injection",
        "continue ", "initiated", "started on", "treated with", "treating",
        "capsule", "inhaler", "cream", "ointment", "procedure",
        "referred", "referral", "counsel", " diet", "exercise",
        "lifestyle", "increase", "decrease", "reduce", "add ",
        "switch", "ordered", "mcg", " daily", "consider ",
        " bid ", " tid ", " qd ", " prn", "linzess",
        "metformin", "insulin", "statin", "atorvastatin", "lisinopril",
        "amlodipine", "losartan", "sertraline", "omeprazole",
        "fiber", "fluids", "lose weight",
    ]

    def _fallback_meat_validation(self, disease_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based fallback if LLM is unavailable or fails to return valid JSON.
        Extracts actual evidence sentences from context (not just boolean flags)
        so that the MEAT gate can verify them against source text.
        """
        import re as _re

        # ── Helper: split a text block into individual atomic lines ──────────
        def _atomize(text: str) -> List[str]:
            """Split a block into individual lines/sentences."""
            # First split on double-newlines, then single newlines
            parts: List[str] = []
            for chunk in _re.split(r'\n\n+', text):
                for line in chunk.split('\n'):
                    line = line.strip()
                    if len(line) >= 8:  # Skip trivially short fragments
                        parts.append(line)
            return parts

        # Gather all context as atomic lines
        raw_texts: List[str] = []
        primary = context.get("primary_mention", "")
        if primary:
            raw_texts.extend(_atomize(primary))
        for s in context.get("context_sentences", []):
            raw_texts.extend(_atomize(s))
        section_text = context.get("section_context", "")
        if section_text:
            raw_texts.extend(_atomize(section_text))
        for s in context.get("related_sentences", []):
            raw_texts.extend(_atomize(s))
        for s in context.get("medication_mentions", []):
            raw_texts.extend(_atomize(s))
        for s in context.get("lab_mentions", []):
            raw_texts.extend(_atomize(s))

        # Deduplicate while preserving order
        seen: set = set()
        unique_sentences: List[str] = []
        for s in raw_texts:
            key = s.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique_sentences.append(s.strip())

        result: Dict[str, Any] = {
            "disease": disease_name,
            "monitoring": False, "monitoring_evidence": "", "monitoring_confidence": 0.0,
            "evaluation": False, "evaluation_evidence": "", "evaluation_confidence": 0.0,
            "assessment": False, "assessment_evidence": "", "assessment_confidence": 0.0,
            "treatment": False, "treatment_evidence": "", "treatment_confidence": 0.0,
            "llm_reasoning": "Fallback rule-based check applied.",
            "overall_confidence": 0.0,
        }

        dn_lower = disease_name.lower()

        def _find_evidence(keywords: List[str], boost_disease_mention: bool = False) -> tuple:
            """
            Return (best_sentence, confidence) for a MEAT element.
            Heavily prefers sentences that mention the disease by name,
            and favours shorter (more specific) matches over long blocks.
            """
            best_sent = ""
            best_score = 0.0
            for sent in unique_sentences:
                sl = sent.lower()
                matched = any(kw in sl for kw in keywords)
                if not matched:
                    continue
                mentions_disease = dn_lower in sl
                # Base confidence
                conf = 0.55
                if mentions_disease:
                    conf = 0.70
                if boost_disease_mention and mentions_disease:
                    conf = 0.75
                # Prefer shorter, more specific evidence (bonus for short)
                length_bonus = 0.05 if len(sent) < 120 else 0.0
                # Disease mention is a strong preference signal
                disease_bonus = 0.10 if mentions_disease else 0.0
                score = conf + length_bonus + disease_bonus
                if score > best_score:
                    best_score = score
                    best_sent = sent
                    best_conf = conf
            return (best_sent[:150].strip(), best_conf) if best_sent else ("", 0.0)

        # --- Assessment: disease name appearing in context IS assessment ---
        # First check: prefer a SHORT sentence containing the disease name
        assessment_evidence = ""
        assessment_conf = 0.0
        for sent in unique_sentences:
            if dn_lower in sent.lower():
                # Prefer shorter sentence mentioning the disease
                if not assessment_evidence or len(sent) < len(assessment_evidence):
                    assessment_evidence = sent[:150].strip()
                    assessment_conf = 0.65
        # Also try keyword-based assessment
        kw_ev, kw_conf = _find_evidence(self._ASSESSMENT_KW, boost_disease_mention=True)
        if kw_conf > assessment_conf:
            assessment_evidence = kw_ev
            assessment_conf = kw_conf
        if assessment_evidence:
            result["assessment"] = True
            result["assessment_evidence"] = assessment_evidence
            result["assessment_confidence"] = assessment_conf

        # --- Monitoring ---
        ev, conf = _find_evidence(self._MONITORING_KW)
        if ev:
            result["monitoring"] = True
            result["monitoring_evidence"] = ev
            result["monitoring_confidence"] = conf

        # --- Evaluation ---
        ev, conf = _find_evidence(self._EVALUATION_KW)
        if ev:
            result["evaluation"] = True
            result["evaluation_evidence"] = ev
            result["evaluation_confidence"] = conf

        # --- Treatment ---
        ev, conf = _find_evidence(self._TREATMENT_KW)
        if ev:
            result["treatment"] = True
            result["treatment_evidence"] = ev
            result["treatment_confidence"] = conf

        result["overall_confidence"] = self._calculate_overall_confidence(result)
        self.logger.info(
            f"MEAT fallback for '{disease_name}': "
            f"A={result['assessment']}, M={result['monitoring']}, "
            f"E={result['evaluation']}, T={result['treatment']} "
            f"[conf={result['overall_confidence']:.2f}]"
        )
        return result

    async def validate_multiple_diseases(
        self,
        disease_contexts: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Orchestrate MEAT validation for all detected clinical entities in parallel.
        Uses a semaphore to cap concurrent Gemini calls and avoid rate-limit errors.
        """
        import asyncio

        # Max 5 concurrent Gemini calls — balances speed vs API rate limits
        semaphore = asyncio.Semaphore(5)

        async def _validated_call(disease_name: str, context: Dict[str, Any]):
            async with semaphore:
                return disease_name, await self.validate_meat(disease_name, context)

        tasks = [
            _validated_call(name, ctx)
            for name, ctx in disease_contexts.items()
        ]

        self.logger.info(
            f"Running {len(tasks)} MEAT validations in parallel (max 5 concurrent)..."
        )
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        for item in completed:
            if isinstance(item, Exception):
                self.logger.error(f"Parallel MEAT task failed: {item}")
                continue
            disease_name, result = item
            results[disease_name] = result

        self.logger.info(f"Completed MEAT validation batch for {len(results)} entities.")
        return results

# Singleton instance
meat_validator = MEATValidator()
