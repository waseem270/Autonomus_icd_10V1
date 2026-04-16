import json
import logging
import re
from types import SimpleNamespace
from typing import Dict, List, Any

from ..core.config import settings
from ..utils.gemini_retry import call_gemini_safe as call_gemini

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safety settings — disable content filtering for medical documents
# ---------------------------------------------------------------------------
SAFETY_SETTINGS = []

# Sections containing primary diagnosis lists — ORDERED: assessment sections FIRST
# (using a list, not a set, to guarantee deterministic order)
_PRIMARY_SECTIONS = [
    "assessment_and_plan",          # Assessment & Plan wins as canonical
    "assessment",                   # Simple "Assessment" label
    "active_problems",              # Prescription active problem lists
    "active_problem_list",          # Alternative naming
    "problem_list",                 # Common in prescriptions
    "past_medical_history",         # Contains chronic diseases
    "pmh",                          # Abbreviation
    "medical_history",              # Alternative naming
    "chronic_conditions",           # Alternative naming
]

# Assessment section keys (subset of _PRIMARY_SECTIONS, used for priority checks)
_ASSESSMENT_KEYS = {"assessment_and_plan", "assessment", "active_problems", "active_problem_list", "problem_list"}

# Additional sections to extract diseases from (lower priority)
_SECONDARY_SECTIONS = {
    "history_present_illness", 
    "hpi",                          # NEW: Abbreviation
    "chief_complaint", 
    "cc",                           # NEW: Abbreviation
    "review_of_systems",
    "ros",                          # NEW: Abbreviation
    "medications",                  # NEW: For context/cross-reference
    "current_medications",          # NEW: Alternative naming
    "impression"                    # NEW: Common in reports
}

# Sections to cross-reference for source tracking
_CROSS_REF_SECTIONS = {
    "chief_complaint", 
    "cc",
    "history_present_illness",
    "hpi"
}


class LLMDiseaseExtractor:
    """
    Use Gemini to extract ALL diagnoses from clinical note sections.
    This is the PRIMARY disease detector — reads full clinical names
    with proper prefixes and suffixes (laterality, chronicity, severity,
    type/subtype).

    Regex-based pattern extraction serves as a lightweight backup
    for when the LLM is rate-limited or unavailable.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        try:
            self._client = None
            from ..core.config import get_active_model
            self.model_name = get_active_model()

            self.config = SimpleNamespace(
                temperature=0.0,
                max_output_tokens=8192,
                response_mime_type="application/json",
                response_schema=None,
                safety_settings=SAFETY_SETTINGS,
                system_instruction=(
                    "You are a Clinical NLP Engine. Your mission is to extract ONLY diseases "
                    "that are EXPLICITLY NAMED in the clinical document text. \n\n"
                    "STRICT RULES:\n"
                    "1. ONLY extract diseases that are directly written/named in the document. "
                    "Do NOT infer or fabricate diseases that are not explicitly stated.\n"
                    "2. SPECIFICITY: If multiple levels of specificity are mentioned, "
                    "select the most granular clinical name (Laterality, Stage, Type, Severity).\n"
                    "3. TEMPORAL: Distinguish between active pathology and resolved/historical events.\n"
                    "4. ABBREVIATION EXPANSION: Convert clinical shorthand (HTN, T2DM, CKD) "
                    "into formal medical terminology.\n"
                    "5. NEVER infer a disease from a medication alone. A medication confirms a disease "
                    "ONLY if the disease is also named elsewhere in the document.\n"
                    "6. NEVER infer a disease from lab values alone. Labs confirm diseases "
                    "that are already named in the document."
                )
            )
            self.logger.info(
                f"LLMDiseaseExtractor initialized with model: {self.model_name}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to initialize LLM for disease extraction: {e}"
            )
            self._client = None

    @property
    def client(self):
        if self._client is None:
            from ..core.config import create_openai_client
            oai = create_openai_client()
            self._client = oai if oai else "__openai_sentinel__"
        return self._client

    def _is_rate_limited(self, error: Exception) -> bool:
        """
        Check if error is a Gemini rate limit (429) error.
        
        Returns:
            True if rate limited, False otherwise
        """
        error_str = str(error).lower()
        return any(indicator in error_str for indicator in [
            "429", 
            "quota", 
            "rate limit", 
            "resource exhausted",
            "too many requests"
        ])

    async def extract_from_sections(
        self, sections: Dict[str, Dict[str, Any]]
    ) -> List[Dict]:
        """
        Extract every diagnosis/condition from clinical note sections.

        Parameters
        ----------
        sections : dict
            Canonical sections dict keyed by section name (e.g.
            ``"assessment_and_plan"``, ``"chief_complaint"``).  Each value is
            a dict that must contain at least a ``"text"`` key.

        Returns
        -------
        list[dict]
            Each dict mirrors the disease-detector output format:
            disease_name, normalized_name, confidence_score, negated,
            section, section_sources, entity_type, sentence_number.
        """
        if not self.client:
            self.logger.warning(
                "Gemini client not initialized — skipping LLM disease extraction."
            )
            return []

        # ── Collect text from ALL primary sections (ordered: assessment first) ──
        def _get_sec_text(sec: Any) -> str:
            """Safely extract text from a section value (dict or str)."""
            if isinstance(sec, dict):
                return sec.get("text", "") or ""
            return str(sec) if sec else ""

        primary_texts: Dict[str, str] = {}
        for key in _PRIMARY_SECTIONS:          # _PRIMARY_SECTIONS is a list, order preserved
            sec = sections.get(key)
            if sec and _get_sec_text(sec).strip():
                primary_texts[key] = _get_sec_text(sec).strip()

        if not primary_texts:
            self.logger.info("No primary section text found — trying secondary sections.")
            for key in _SECONDARY_SECTIONS:
                sec = sections.get(key)
                if sec and _get_sec_text(sec).strip():
                    primary_texts[key] = _get_sec_text(sec).strip()
            if not primary_texts:
                self.logger.info("No suitable section text found — nothing to extract.")
                return []

        # ── Prefer assessment section as canonical key ──────────────────
        # (primary_texts preserves insertion order from _PRIMARY_SECTIONS list,
        #  so the first assessment key found will be first in the dict)
        primary_section_key = next(
            (k for k in _PRIMARY_SECTIONS if k in primary_texts),
            list(primary_texts.keys())[0]
        )

        # Build combined primary text — assessment section FIRST so LLM sees it early
        primary_parts = []
        # Assessment sections first
        for sec_name in _PRIMARY_SECTIONS:
            if sec_name in primary_texts:
                primary_parts.append(f"[{sec_name}]\n{primary_texts[sec_name]}")
        primary_text = "\n\n".join(primary_parts)

        # Also collect secondary + cross-ref section texts for additional extraction
        all_other_texts: Dict[str, str] = {}
        for key in _SECONDARY_SECTIONS | _CROSS_REF_SECTIONS:
            if key in primary_texts:
                continue
            sec = sections.get(key)
            if sec and _get_sec_text(sec).strip():
                all_other_texts[key] = _get_sec_text(sec).strip()

        prompt = self._build_prompt(primary_text, primary_section_key, all_other_texts)

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

            parsed = json.loads(response_text)

            # OpenAI json_object mode wraps arrays in an object — unwrap
            if isinstance(parsed, dict):
                # Find the list value (could be "diseases", "results", etc.)
                raw_diseases = []
                for v in parsed.values():
                    if isinstance(v, list):
                        raw_diseases = v
                        break
                if not raw_diseases:
                    self.logger.warning(
                        f"LLM returned dict with no list values: {list(parsed.keys())}"
                    )
                    raw_diseases = []
            elif isinstance(parsed, list):
                raw_diseases = parsed
            else:
                self.logger.warning(f"Unexpected JSON type from LLM: {type(parsed)}")
                raw_diseases = []

            # Normalise: if LLM returned a list of strings instead of dicts,
            # wrap each string into the expected dict format.
            if raw_diseases and isinstance(raw_diseases[0], str):
                raw_diseases = [
                    {"disease_name": name, "section_sources": [primary_section_key],
                     "confidence": 0.90, "status": "active"}
                    for name in raw_diseases if name.strip()
                ]

            # ── Post-process: patch section sources using Assessment text ────
            # This deterministically ensures diseases in Assessment numbered
            # items are tagged with "assessment" regardless of LLM output.
            raw_diseases = self._patch_assessment_sources(raw_diseases, sections)

            # Normalise into the canonical disease-detector format
            results: List[Dict] = []
            _PMH_KEYS = {"past_medical_history", "medical_history", "pmh"}
            for item in raw_diseases:
                name = (item.get("disease_name") or "").strip()
                if not name:
                    continue
                raw_sources = item.get("section_sources", [primary_section_key])
                # Determine the canonical 'section' field:
                # If LLM says disease is ONLY from PMH, honour that.
                # If it has ANY assessment source, use assessment.
                # Otherwise use the primary section key.
                sources_lower = {s.lower() for s in raw_sources}
                if sources_lower & _ASSESSMENT_KEYS:
                    canonical_section = next(
                        (k for k in _PRIMARY_SECTIONS if k in sources_lower), primary_section_key
                    )
                elif sources_lower and sources_lower.issubset(_PMH_KEYS):
                    canonical_section = list(sources_lower)[0]
                else:
                    canonical_section = primary_section_key
                results.append({
                    "disease_name": name,
                    "normalized_name": self._normalize_name(name),
                    "confidence_score": float(item.get("confidence", 0.95)),
                    "negated": False,
                    "section": canonical_section,
                    "section_sources": raw_sources,
                    "status": item.get("status", "active"),
                    "entity_type": "CONDITION",
                    "sentence_number": 0,
                })

            self.logger.info(
                f"LLM disease extraction found {len(results)} diagnoses "
                f"from '{primary_section_key}'."
            )
            return results

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed for LLM extraction response: {e}")
            self.logger.debug(f"Raw response: {response_text[:500]}")
            return []
        except Exception as e:
            # After retries exhausted by call_gemini, propagate so caller knows
            self.logger.error(
                f"❌ Gemini disease extraction failed after retries: {e}"
            )
            raise

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        primary_text: str,
        primary_section_key: str,
        cross_ref_texts: Dict[str, str],
    ) -> str:
        cross_ref_block = ""
        if cross_ref_texts:
            parts = []
            for sec_name, text in cross_ref_texts.items():
                parts.append(f"[{sec_name}]\n{text}")
            cross_ref_block = "\n\n".join(parts)

        prompt = f"""You are a clinical NLP assistant and expert medical coder. Your task is to extract EVERY diagnosis, disease, or medical condition mentioned in ALL sections of a clinical note, with FULL CLINICAL SPECIFICITY exactly as documented.

PRIMARY SECTION ({primary_section_key}):
\"\"\"
{primary_text}
\"\"\"
"""
        if cross_ref_block:
            prompt += f"""
OTHER CLINICAL SECTIONS (extract diseases from these too):
\"\"\"
{cross_ref_block}
\"\"\"
"""

        prompt += """
EXTRACTION RULES (STRICT):
1. EXPLICIT NAMES ONLY: 
   - Extract ONLY diseases, diagnoses, and conditions that are EXPLICITLY NAMED in the text.
   - If a disease name does not appear written in the document, do NOT include it.
   - Do NOT infer diseases from medications alone. A medication confirms a disease ONLY if the disease is also named elsewhere.
   - Do NOT infer diseases from lab values alone. Labs confirm documented diseases only.
2. SPECIFICITY: 
   - If Section A says "Diabetes" and Section B says "Type 2 Diabetes with kidney disease," extract ONLY the MOST SPECIFIC name.
   - Include ALL qualifiers: Laterality (Right/Left), Chronicity (Chronic/Acute), Severity (Moderate/Severe), and Stage (Stage 1-5).
3. NAMING RULES:
   - Use FULL MEDICAL TERMINOLOGY: Never return abbreviations (HTN -> Essential Hypertension, etc.).
   - COMBINE QUALIFIERS: If the note says "Depression, chronic, recurrent" return "Major Depressive Disorder, Chronic, Recurrent".

INCLUDE: Diseases explicitly named in Assessment, Plan, Problem List, PMH, or HPI.
EXCLUDE: Routine screening, normal findings, isolated symptoms, and any disease not explicitly written in the document.

Return a JSON array of objects:
- "disease_name": specific clinical name exactly as documented but expanded.
- "section_sources": array of header names where this disease is explicitly named.
- "confidence": 0.0 to 1.0 based on clarity of documentation.
- "status": "active" (in Assessment/Plan/Problem List) or "history" (PMH only with no current evidence).

Return ONLY the JSON array.
"""
        return prompt

    # ------------------------------------------------------------------
    # Assessment post-processing — deterministic section source patching
    # ------------------------------------------------------------------
    @staticmethod
    def _patch_assessment_sources(
        raw_diseases: List[Dict], sections: Dict[str, Dict[str, Any]]
    ) -> List[Dict]:
        """
        After the LLM response arrives, deterministically patch section_sources
        for diseases that appear (by name fragment) in Assessment-family sections.

        This catches the common failure mode where the LLM attributes an
        Assessment item to HPI or PMH because those sections contain the
        same term.

        Also applies name-normalisation for Assessment-listed diseases:
        - "Depression" from PMH that is listed as "Major depression/moderate/recurrent"
          in Assessment gets renamed to the full Assessment name.
        """
        # Build a lookup of assessment text (lower-cased) for substring search
        assessment_text_lower = ""
        assessment_raw_text = ""
        for key in _ASSESSMENT_KEYS:
            sec = sections.get(key)
            _st = (sec.get("text", "") if isinstance(sec, dict) else str(sec or ""))
            if sec and _st.strip():
                assessment_text_lower += " " + _st.lower()
                assessment_raw_text += " " + _st

        # Pattern to find numbered Assessment items and their comma-separated diseases
        # e.g. "1.Essential hypertension: ..." or "3.Major depression/moderate/recurrent, anxiety, insomnia:"
        numbered_item_pattern = re.compile(
            r'\d+\.\s*([A-Za-z][^:\n]+?)(?:\s*:|$)',
            re.IGNORECASE
        )
        # Collect disease name fragments known to be in Assessment
        # Also build a map: short_fragment → full_item_text (for name upgrading)
        assessment_disease_fragments: set = set()
        # Maps a short/generic name → the full Assessment item description
        # e.g. "depression" → "Major Depressive Disorder, Recurrent, Moderate"
        name_upgrade_map: Dict[str, str] = {}

        for key in _ASSESSMENT_KEYS:
            sec = sections.get(key)
            _st = (sec.get("text", "") if isinstance(sec, dict) else str(sec or ""))
            if sec and _st.strip():
                for m in numbered_item_pattern.finditer(_st):
                    raw_item = m.group(1).strip()  # e.g. "Major depression/moderate/recurrent, anxiety, insomnia"
                    fragment = raw_item.lower()
                    # Split on ", " to catch comma-separated lists
                    for part in re.split(r',\s*', fragment):
                        raw_part = part.strip()
                        # Split on "/" to catch slash-combined entries like "Major depression/moderate/recurrent"
                        sub_parts = [s.strip() for s in re.split(r'/', raw_part)]
                        # The first sub_part is the primary name; rest are qualifiers
                        primary_part = sub_parts[0]
                        if len(primary_part) >= 3:
                            assessment_disease_fragments.add(primary_part)
                            # Build an expanded name by joining slash-parts with space
                            if len(sub_parts) > 1:
                                expanded_name = " ".join(sub_parts).title()
                                # Register short name → expanded name mapping
                                name_upgrade_map[primary_part] = expanded_name
                        for sub in sub_parts[1:]:
                            if len(sub) >= 3:
                                assessment_disease_fragments.add(sub)

        patched = []
        for item in raw_diseases:
            name = (item.get("disease_name") or "").strip()
            name_lower = name.lower()
            sources = list(item.get("section_sources") or [])
            sources_lower = {s.lower() for s in sources}

            # Check if this disease name (or any core word) appears in Assessment text
            in_assessment = (
                name_lower in assessment_text_lower
                or any(frag in name_lower or name_lower in frag
                       for frag in assessment_disease_fragments if len(frag) >= 4)
            )

            if in_assessment:
                if "assessment" not in sources_lower:
                    sources = ["assessment"] + [s for s in sources if s.lower() not in _ASSESSMENT_KEYS]
                    item = dict(item)
                    item["section_sources"] = sources
                    item["section"] = "assessment"   # Upgrade canonical section too

                # Also upgrade vague disease name to the full Assessment item name
                # e.g. "depression" → "Major Depression Moderate Recurrent"
                # Only upgrade if current name is shorter/less specific
                for short_frag, expanded_name in name_upgrade_map.items():
                    if (name_lower == short_frag
                            or name_lower in short_frag
                            or short_frag in name_lower):
                        if len(expanded_name) > len(name):
                            item = dict(item)
                            item["disease_name"] = expanded_name
                            item["normalized_name"] = expanded_name.lower()
                            break

            patched.append(item)

        return patched

    # ------------------------------------------------------------------
    # Name normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_name(name: str) -> str:
        """Lowercase, strip, and clean a disease name."""
        normalized = name.lower().strip()
        normalized = re.sub(r'\n\n.*$', '', normalized).strip()
        normalized = re.sub(r'[\n\r]+', ' ', normalized)
        normalized = re.sub(r'^[,.\s]+|[,.\s]+$', '', normalized)
        normalized = " ".join(normalized.split())
        return normalized


# Singleton instance
llm_disease_extractor = LLMDiseaseExtractor()
