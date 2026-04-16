"""
Single-pass Clinical Document Analyzer.

Sends the ENTIRE clinical document to Gemini in ONE LLM call with a
comprehensive prompt that performs:
  1. Section detection
  2. Clinical reasoning-based disease extraction
  3. MEAT validation with evidence grading
  4. ICD-10-CM code assignment
  5. Exclusion logging

This eliminates the rate-limit cascade of the multi-step pipeline
(3+ LLM calls before a single disease is found) and gives the model
full document context for cross-referencing.
"""

import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

from types import SimpleNamespace

from ..core.config import settings
from .icd_lookup import ICDLookupService
from ..utils.abbreviation_expander import MEDICAL_ABBREVIATIONS, expand_abbreviations
from ..utils.section_candidate_extractor import SectionCandidateExtractor
from ..utils.fuzzy_section_matcher import match_canonical
from ..utils.gemini_retry import call_gemini_safe as call_gemini
from .smart_section_detector import ContentBasedSectionInferrer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abbreviation ↔ expansion lookup for matching (disease abbrevs only)
# ---------------------------------------------------------------------------
# Build a bidirectional map so "ckd" matches "chronic kidney disease" and vice versa.
_ABBR_TO_EXPANSION = {k.lower(): v.lower() for k, v in MEDICAL_ABBREVIATIONS.items()}
_EXPANSION_TO_ABBR = {v.lower(): k.lower() for k, v in MEDICAL_ABBREVIATIONS.items()}


def _normalise_for_match(name: str) -> set:
    """Return a set of matching keys for a disease name.

    Given "CKD 3" → {"ckd 3", "chronic kidney disease 3", "ckd", "chronic kidney disease"}
    Given "Chronic Kidney Disease Stage 3" → same set + the full form.

    This lets the assessment verifier recognise that "CKD 3" and
    "Chronic Kidney Disease Stage 3" refer to the same condition.
    """
    low = name.lower().strip()
    keys = {low}

    # Split into leading clinical term and trailing qualifier (stage/type/grade + number)
    # e.g. "CKD 3" → term="ckd", qualifier=" 3"
    # e.g. "Type 2 Diabetes Mellitus" → stays whole initially
    m = re.match(r"^(.+?)\s*((?:stage|type|grade|class|step)?\s*\d[\w.]*)$", low, re.IGNORECASE)
    base = m.group(1).strip() if m else low
    qualifier = m.group(2).strip() if m else ""

    # Add abbreviation expansions for the base term
    if base in _ABBR_TO_EXPANSION:
        expanded = _ABBR_TO_EXPANSION[base]
        keys.add(expanded + (" " + qualifier if qualifier else ""))
        keys.add(expanded)
    if base in _EXPANSION_TO_ABBR:
        abbr = _EXPANSION_TO_ABBR[base]
        keys.add(abbr + (" " + qualifier if qualifier else ""))
        keys.add(abbr)

    # Also try the full string
    if low in _ABBR_TO_EXPANSION:
        keys.add(_ABBR_TO_EXPANSION[low])
    if low in _EXPANSION_TO_ABBR:
        keys.add(_EXPANSION_TO_ABBR[low])

    return keys


def _abbreviation_aware_match(candidate: str, existing_names: set) -> bool:
    """Return True if `candidate` matches any name in `existing_names`.

    Uses substring matching PLUS abbreviation expansion so that:
      "ckd 3"  matches  "chronic kidney disease stage 3"
      "htn"    matches  "essential hypertension"
      "dm2"    matches  "type 2 diabetes mellitus"
    Also uses word-overlap matching so that:
      "pressure injury of skin of right buttock"  matches  "pressure injury of right buttock"
    """
    cand_keys = _normalise_for_match(candidate)

    # Extract significant words (>=4 chars) for overlap matching
    _STOP_WORDS = {"with", "from", "that", "this", "type", "stage", "grade", "unspecified", "specified"}
    def _sig_words(text: str) -> set:
        return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOP_WORDS}

    cand_sig = _sig_words(candidate)

    for existing in existing_names:
        # Fast substring check (original behaviour)
        if candidate in existing or existing in candidate:
            return True
        # Abbreviation-expanded check
        exist_keys = _normalise_for_match(existing)
        for ck in cand_keys:
            for ek in exist_keys:
                if ck in ek or ek in ck:
                    return True
        # Word-overlap check: if >=70% of significant words overlap,
        # these are the same condition (e.g. "pressure injury of skin of
        # right buttock" vs "pressure injury of right buttock stage 3")
        if cand_sig:
            exist_sig = _sig_words(existing)
            if exist_sig:
                overlap = cand_sig & exist_sig
                smaller = min(len(cand_sig), len(exist_sig))
                if smaller > 0 and len(overlap) / smaller >= 0.70:
                    return True
    return False


# ---------------------------------------------------------------------------
# Safety settings — disable content filtering for medical documents
# ---------------------------------------------------------------------------
SAFETY_SETTINGS = []

# Response schema not used with OpenAI (uses json_object mode instead)
_ANALYSIS_RESPONSE_SCHEMA = None


class ClinicalDocumentAnalyzer:
    """
    Single-pass LLM analyzer that extracts diseases, validates MEAT,
    and assigns ICD-10 codes in one Gemini call.

    This is the primary extraction engine.  The multi-step pipeline
    (Smart Section Detector → Regex → LLM Disease Extractor → MEAT Validator
    → ICD Ranker) remains as the fallback when this call fails.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._prompt_path = Path(__file__).resolve().parents[2] / "prompt.json"
        self._client = None
        # Use active model (OpenAI or Gemini)
        from ..core.config import get_active_model
        self.model_name = get_active_model()
        self.logger.info(
            f"ClinicalDocumentAnalyzer initialized with model: {self.model_name}"
        )

        # ICD lookup for post-LLM code validation
        try:
            self._icd_lookup = ICDLookupService()
        except Exception:
            self._icd_lookup = None

    @property
    def client(self):
        """Lazy-init LLM client on first use. Returns a sentinel for OpenAI."""
        if self._client is None:
            from ..core.config import create_openai_client
            oai = create_openai_client()
            self._client = oai if oai else "__openai_sentinel__"
        return self._client

    def _parse_json_robust(self, text: str) -> dict:
        """Parse JSON with progressive repair strategies.

        Handles common LLM output issues:
        - Trailing commas before } or ]
        - Missing commas between elements
        - Truncated output (incomplete JSON)
        - Single quotes instead of double quotes
        """
        # Strategy 1: direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: trailing commas
        repaired = re.sub(r',\s*([}\]])', r'\1', text)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # Strategy 3: missing commas — insert comma between }{, ]["  etc.
        repaired = re.sub(r'}\s*{', '},{', repaired)
        repaired = re.sub(r']\s*\[', '],[', repaired)
        repaired = re.sub(r'"\s*\n\s*"', '",\n"', repaired)
        repaired = re.sub(r'"\s*\n\s*{', '",\n{', repaired)
        repaired = re.sub(r'}\s*\n\s*"', '},\n"', repaired)
        repaired = re.sub(r']\s*\n\s*"', '],\n"', repaired)
        repaired = re.sub(r'"\s*\n\s*\[', '",\n[', repaired)
        # Missing comma after number before quote/brace
        repaired = re.sub(r'(\d)\s*\n\s*"', r'\1,\n"', repaired)
        repaired = re.sub(r'(\d)\s*\n\s*{', r'\1,\n{', repaired)
        # true/false/null followed by quote or brace without comma
        repaired = re.sub(r'(true|false|null)\s*\n\s*"', r'\1,\n"', repaired)
        repaired = re.sub(r'(true|false|null)\s*\n\s*{', r'\1,\n{', repaired)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # Strategy 4: truncated output — find the last complete object
        # Count braces/brackets to find where the JSON becomes valid
        depth_brace = 0
        depth_bracket = 0
        last_valid_pos = 0
        in_string = False
        escape = False
        for i, ch in enumerate(repaired):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth_brace += 1
            elif ch == '}':
                depth_brace -= 1
            elif ch == '[':
                depth_bracket += 1
            elif ch == ']':
                depth_bracket -= 1
            if depth_brace == 0 and depth_bracket == 0 and i > 10:
                last_valid_pos = i + 1
                break
        if last_valid_pos > 10:
            try:
                result = json.loads(repaired[:last_valid_pos])
                self.logger.info("Single-pass JSON repaired (balanced truncation).")
                return result
            except json.JSONDecodeError:
                pass

        # Strategy 5: brute force close — close all open braces/brackets
        trimmed = repaired.rstrip()
        open_braces = trimmed.count('{') - trimmed.count('}')
        open_brackets = trimmed.count('[') - trimmed.count(']')
        # Remove any trailing comma
        trimmed = re.sub(r',\s*$', '', trimmed)
        trimmed += ']' * max(0, open_brackets) + '}' * max(0, open_braces)
        try:
            result = json.loads(trimmed)
            self.logger.info("Single-pass JSON repaired (force-closed).")
            return result
        except json.JSONDecodeError:
            pass

        # All strategies failed — raise so outer handler catches it
        raise json.JSONDecodeError("All JSON repair strategies failed", text, 0)

    def _build_config(self) -> types.GenerateContentConfig:
        """Build a fresh GenerateContentConfig, loading system_prompt from prompt.json each time."""
        system_instr = (
            "You are a senior clinical coding specialist with 20+ years of "
            "ICD-10-CM and HCC risk adjustment experience."
        )
        try:
            with self._prompt_path.open("r", encoding="utf-8") as f:
                prompt_config = json.load(f)
            system_instr = prompt_config.get("system_prompt", system_instr)
        except Exception:
            pass

        model_name = (settings.OPENAI_MODEL or "").lower()

        return SimpleNamespace(
            temperature=0.1,
            top_p=0.95,
            max_output_tokens=16384,
            response_mime_type="application/json",
            response_schema=_ANALYSIS_RESPONSE_SCHEMA,
            safety_settings=SAFETY_SETTINGS,
            system_instruction=system_instr,
        )

    # ------------------------------------------------------------------
    # Pre-LLM section marker injection
    # ------------------------------------------------------------------

    def _inject_section_markers(self, text: str) -> str:
        """Inject visible section markers into document text before LLM call.

        Two-layer detection:
          Layer 1 — Regex header detection (SectionCandidateExtractor):
            Uses 8 regex patterns to find explicit section headers like
            "Assessment:", "[Medications]", "PHYSICAL EXAM", etc.
            Results are filtered for false positives (tiny content windows).

          Layer 2 — Content-based inference (ContentSectionInferrer):
            When regex misses sections (no headers in PDF, unusual formatting),
            scans content patterns to infer sections. E.g. lines with drug
            names + dosages → medications section, even without a header.

        Handles all known failure modes:
          - Headerless PDFs → content inference fills the gaps
          - False positives → short content window filter removes them
          - Merged sections → fuzzy matcher maps "Assessment and Plan" correctly
          - Novel formats → content patterns work regardless of header style
        """
        # ── Layer 1: Regex-based header detection ────────────────────
        extractor = SectionCandidateExtractor()
        candidates = extractor.extract_candidates(text)
        high_conf, _ = extractor.filter_by_confidence(candidates, threshold=0.65)

        regex_markers: List[Dict] = []
        seen_positions: Set[int] = set()
        for cand in high_conf:
            canonical, score = match_canonical(cand["header"])
            if canonical and score >= 0.60:
                pos = cand["position"]
                if any(abs(pos - sp) < 10 for sp in seen_positions):
                    continue
                seen_positions.add(pos)
                regex_markers.append({
                    "position": pos,
                    "canonical": canonical,
                    "original": cand["header"],
                    "confidence": score,
                    "source": "regex",
                })

        # ── False positive filter: remove regex markers with tiny content ─
        regex_markers = self._filter_short_sections(text, regex_markers)

        self.logger.info(
            f"Regex detection: {len(regex_markers)} section markers — "
            + ", ".join(m["canonical"] for m in sorted(regex_markers, key=lambda x: x["position"]))
        )

        # ── Layer 2: Content-based inference (SmartSectionDetector engine) ─
        inferrer = ContentBasedSectionInferrer()
        inferred_result = inferrer.infer_sections(text)
        inferred_sections = inferred_result.get("sections", {})

        # Convert to marker format: {position, canonical, confidence, source}
        content_markers = []
        for section_name, section_data in inferred_sections.items():
            content_markers.append({
                "position": section_data.get("start", 0),
                "canonical": section_name,
                "confidence": inferred_result.get("confidence", 0.5),
                "source": "content_inference",
            })

        # Merge: content markers only fill gaps not covered by regex
        all_markers = list(regex_markers)
        regex_canonicals = {m["canonical"] for m in regex_markers}

        for cm in content_markers:
            # Skip if regex already found this section type
            if cm["canonical"] in regex_canonicals:
                continue
            # Skip if too close to an existing marker OF THE SAME type
            if any(
                abs(cm["position"] - m["position"]) < 50
                and cm["canonical"] == m["canonical"]
                for m in all_markers
            ):
                continue
            all_markers.append(cm)

        if not all_markers:
            self.logger.info("No sections detected — wrapping as General Clinical Narrative.")
            return f"\n═══ [SECTION: General Clinical Narrative (inferred)] ═══\n{text}"

        # ── Inject markers into text ─────────────────────────────────
        # Sort descending so insertions don't shift later positions
        all_markers.sort(key=lambda m: m["position"], reverse=True)

        # Fill large uncovered gap at the start of the document
        first_marker_pos = min(m["position"] for m in all_markers)
        if first_marker_pos > 500:
            all_markers.append({
                "position": 0,
                "canonical": "General Clinical Narrative",
                "confidence": 0.50,
                "source": "content_inference",
            })
            all_markers.sort(key=lambda m: m["position"], reverse=True)

        result = text
        for m in all_markers:
            pos = m["position"]
            inferred_tag = " (inferred)" if m["source"] == "content_inference" else ""
            tag = f"\n═══ [SECTION: {m['canonical']}{inferred_tag}] ═══\n"
            result = result[:pos] + tag + result[pos:]

        self.logger.info(
            f"Injected {len(all_markers)} total markers: "
            + ", ".join(
                f"{m['canonical']}({'inf' if m['source'] == 'content_inference' else 'rgx'})"
                for m in sorted(all_markers, key=lambda x: x["position"])
            )
        )
        return result

    def _filter_short_sections(
        self, text: str, markers: List[Dict]
    ) -> List[Dict]:
        """Remove regex markers whose content window is too short.

        A detected "header" followed by < 40 chars before the next header
        is likely a false positive (e.g. "Assessment: stable" as inline text).
        Exception: allergies sections can legitimately be very short ("NKDA").
        """
        if len(markers) < 2:
            return markers

        sorted_m = sorted(markers, key=lambda m: m["position"])
        filtered: List[Dict] = []
        text_len = len(text)

        for i, m in enumerate(sorted_m):
            next_pos = sorted_m[i + 1]["position"] if i + 1 < len(sorted_m) else text_len
            window_size = next_pos - m["position"]

            if window_size < 40 and m["canonical"] != "allergies":
                self.logger.debug(
                    f"Filtering short-section false positive: "
                    f"'{m.get('original', m['canonical'])}' — only {window_size} chars"
                )
                continue
            filtered.append(m)

        return filtered

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze_document(self, raw_text: str, dietary_analysis: bool = False) -> Optional[Dict[str, Any]]:
        """
        Run the full single-pass analysis on a clinical document.

        Returns
        -------
        dict or None
            Parsed JSON result with keys: sections_found, summary, diseases,
            excluded_conditions.  Returns None on failure (caller should
            fall back to the multi-step pipeline).
        """
        if not self.client:
            self.logger.warning("Gemini client not initialized — skipping single-pass analysis.")
            return None

        if not raw_text or len(raw_text.strip()) < 50:
            self.logger.warning("Document text too short for analysis.")
            return None

        # Expand abbreviations so the LLM sees full clinical terms
        # e.g. "HTN" → "Hypertension (HTN)", "CKD" → "Chronic Kidney Disease (CKD)"
        expanded_text = expand_abbreviations(raw_text, context_aware=True)

        # Pre-LLM section marker injection: detect section headers via regex
        # and inject ═══ [SECTION: X] ═══ markers so the LLM can identify
        # section boundaries regardless of PDF formatting
        expanded_text = self._inject_section_markers(expanded_text)

        # Approach 5: Pre-LLM NER candidate extraction
        # Extract disease candidates from document text using regex patterns
        # Pass to LLM as hints to constrain extraction and reduce hallucination
        ner_candidates = self._extract_ner_candidates(raw_text)

        prompt = self._build_prompt(expanded_text, ner_candidates=ner_candidates, dietary_analysis=dietary_analysis)

        try:
            config = self._build_config()
            response = await call_gemini(
                client=self.client,
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            # Extract text from response parts (skip thinking parts)
            response_text = ""
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for i, part in enumerate(response.candidates[0].content.parts):
                    is_thought = getattr(part, "thought", False)
                    has_text = bool(part.text) if hasattr(part, 'text') else False
                    text_len = len(part.text) if has_text else 0
                    self.logger.info(
                        f"Response part {i}: thought={is_thought}, has_text={has_text}, len={text_len}"
                    )
                    if not is_thought and part.text:
                        response_text += part.text
            else:
                # Log what we got instead
                if response.candidates:
                    cand = response.candidates[0]
                    self.logger.error(
                        f"Response candidate: finish_reason={getattr(cand, 'finish_reason', 'N/A')}, "
                        f"has_content={cand.content is not None if hasattr(cand, 'content') else 'N/A'}"
                    )
                else:
                    self.logger.error("Response has no candidates!")

            # Log finish reason and token usage for debugging truncation
            if response.candidates:
                cand = response.candidates[0]
                self.logger.info(f"Finish reason: {getattr(cand, 'finish_reason', 'N/A')}")
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                um = response.usage_metadata
                self.logger.info(
                    f"Token usage: prompt={getattr(um, 'prompt_token_count', '?')}, "
                    f"candidates={getattr(um, 'candidates_token_count', '?')}, "
                    f"total={getattr(um, 'total_token_count', '?')}, "
                    f"thoughts={getattr(um, 'thoughts_token_count', '?')}"
                )

            self.logger.info(f"Extracted response_text length: {len(response_text)}")

            response_text = response_text.strip()

            # Clean JSON formatting
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()

            result = self._parse_json_robust(response_text)

            # Store the raw count before multi-step cleaning (The "38 potential diseases")
            raw_diseases = result.get("diseases", [])
            result.setdefault("summary", {})["total_raw_detected"] = len(raw_diseases)

            # Approach 4: Two-step verification — remove diseases not in document
            result = self._verify_diseases_in_document(raw_text, result)

            # Approach 7: Evidence-in-document validation — strip fabricated evidence
            result = self._verify_evidence_in_document(raw_text, result)

            # Post-LLM evidence quality: deduplicate and strip generic evidence
            result = self._deduplicate_meat_evidence(result)

            # Strict evidence validation — reject MEAT entries without real evidence
            result = self._strict_evidence_validation(result, raw_text)

            # Approach 10: Remove diseases with NO MEAT after all cleaning
            # EXCEPTION: Z68.x (BMI) codes are supplementary codes that
            # accompany an obesity/overweight diagnosis and do not require
            # independent MEAT evidence.
            diseases = result.get("diseases", [])
            active_diseases = []
            for d in diseases:
                icd = d.get("icd10_code") or ""
                is_bmi_supplementary = icd.startswith("Z68.")
                if d.get("meat_score", 0) >= 1 or is_bmi_supplementary:
                    active_diseases.append(d)
                else:
                    result.setdefault("excluded_conditions", []).append({
                        "term": d.get("disease_name", ""),
                        "reason_excluded": "NO_MEAT_EVIDENCE",
                        "source_section": d.get("source_section", ""),
                        "exclusion_reasoning": (
                            "All MEAT evidence was stripped by post-LLM validation "
                            "(hallucinated or irrelevant evidence). No verifiable "
                            "clinical activity for this condition."
                        ),
                    })
            result["diseases"] = active_diseases

            # Deterministic correction: Respiratory failure type must match
            # Assessment text, not HPI or Problem List wording
            result = self._correct_resp_failure_from_assessment(raw_text, result)

            # Deterministic safety net: catch any Assessment items the LLM missed
            # or that were removed by evidence stripping above
            result = self._verify_assessment_coverage(raw_text, result)

            diseases = result.get("diseases", [])
            excluded = result.get("excluded_conditions", [])
            self.logger.info(
                f"Single-pass analysis: {len(diseases)} active diseases, "
                f"{len(excluded)} excluded conditions."
            )
            return result

        except json.JSONDecodeError as e:
            self.logger.error(f"Single-pass JSON parse failed: {e}")
            if response_text:
                self.logger.error(f"Raw response (first 3000): {response_text[:3000]}")
                # Write full response to temp file for debugging
                try:
                    from pathlib import Path
                    debug_path = Path(__file__).resolve().parents[2] / "logs" / "last_raw_response.txt"
                    debug_path.parent.mkdir(exist_ok=True)
                    debug_path.write_text(response_text, encoding="utf-8")
                    self.logger.error(f"Full response written to {debug_path}")
                except Exception:
                    pass
            return None
        except Exception as e:
            # After retries exhausted by call_gemini, propagate the error
            self.logger.error(f"Single-pass analysis failed after retries: {e}")
            raise

    # ------------------------------------------------------------------
    # Post-LLM Assessment coverage verifier
    # ------------------------------------------------------------------
    # Deterministic correction: Respiratory failure type from Assessment
    # ------------------------------------------------------------------

    _RESP_FAILURE_RE = re.compile(
        r"(?:acute\s+(?:on\s+chronic\s+)?(?:hypoxic\s+|hypercapnic\s+|hypoxic\s+and\s+hypercapnic\s+)?"
        r"respiratory\s+failure|"
        r"chronic\s+(?:hypoxic\s+|hypercapnic\s+)?respiratory\s+failure)",
        re.IGNORECASE,
    )

    def _correct_resp_failure_from_assessment(
        self, raw_text: str, result: dict
    ) -> dict:
        """
        If the LLM coded respiratory failure (J96.xx), verify the type
        matches the Assessment/Plan text.  The Assessment wording is
        authoritative: 'Acute respiratory failure' → J96.0x, 'Chronic' →
        J96.1x, 'Acute on chronic' → J96.2x.
        """
        # Extract the Assessment/Plan section
        text_lower = raw_text.lower()
        assessment_start = -1
        for marker in ["assessment and plan", "assessment/plan", "assessment & plan",
                        "assessment\n", "a/p\n", "a/p:"]:
            idx = text_lower.rfind(marker)
            if idx > assessment_start:
                assessment_start = idx

        if assessment_start < 0:
            return result

        assessment_text = text_lower[assessment_start:]

        # Find all respiratory failure mentions in Assessment
        matches = list(self._RESP_FAILURE_RE.finditer(assessment_text))
        if not matches:
            return result

        # Determine the type from Assessment
        for m in matches:
            phrase = m.group(0).lower()
            if "acute on chronic" in phrase or "acute-on-chronic" in phrase:
                correct_prefix = "J96.2"
            elif "chronic" in phrase and "acute" not in phrase:
                correct_prefix = "J96.1"
            elif "acute" in phrase:
                correct_prefix = "J96.0"
            else:
                continue

            # Now check diseases for mismatched respiratory failure codes
            for disease in result.get("diseases", []):
                icd = disease.get("icd10_code") or ""
                if not icd.startswith("J96."):
                    continue

                current_prefix = icd[:5]  # e.g., "J96.0", "J96.1", "J96.2"
                if current_prefix != correct_prefix:
                    # Determine the correct suffix (hypoxia/hypercapnia)
                    suffix = icd[5:] if len(icd) > 5 else "1"  # default to hypoxia
                    new_code = correct_prefix + suffix
                    self.logger.info(
                        f"Resp failure correction: Assessment says '{phrase}' "
                        f"→ changing {icd} to {new_code}"
                    )
                    disease["icd10_code"] = new_code
                    old_name = disease.get("disease_name", "")
                    # Also fix the disease name
                    if correct_prefix == "J96.0":
                        disease["disease_name"] = re.sub(
                            r"(?i)acute\s+on\s+chronic", "Acute", old_name
                        )
                    elif correct_prefix == "J96.1":
                        disease["disease_name"] = re.sub(
                            r"(?i)acute\s+on\s+chronic", "Chronic", old_name
                        )
            break  # Use first Assessment match

        return result

    # ------------------------------------------------------------------

    def _verify_assessment_coverage(self, raw_text: str, result: dict) -> dict:
        """
        Deterministic safety net that runs AFTER the LLM call.

        Finds every numbered/bulleted item in the Assessment (or Assessment &
        Plan) section and checks that each one is represented in
        result["diseases"].  Any item the LLM missed is appended as a
        LOW_MEAT entry so it is never silently dropped.
        """
        # Find the Assessment / Assessment & Plan section
        section_header = re.compile(
            r"(?:^|\n)[ \t]*"
            r"(?:assessment\s*(?:and\s*plan|&\s*plan|/\s*plan)?|a\s*/\s*p)"
            r"\s*[:\n]",
            re.IGNORECASE | re.MULTILINE,
        )
        m = section_header.search(raw_text)
        if not m:
            return result

        section_start = m.end()

        # Stop at the next major section header
        next_header = re.compile(
            r"\n[ \t]*(?:plan|medications?|physical\s*exam|review\s*of\s*systems?"
            r"|follow|labs?|imaging|referral|problem\s*list|past\s*medical"
            r"|hpi|history\s*of\s*present|chief\s*complaint|vitals?"
            r"|subjective|objective|sign)[^\n]*\n",
            re.IGNORECASE,
        )
        nm = next_header.search(raw_text, section_start)
        section_end = nm.start() if nm else len(raw_text)
        assessment_text = raw_text[section_start:section_end]

        # Extract numbered/bulleted items that look like clinical conditions
        items = re.findall(
            r"(?:^|\n)[ \t]*(?:\d+[.):]\s*|[\u2022\-\*]\s*)([A-Z][^\n]{4,120})",
            assessment_text,
            re.MULTILINE,
        )

        # Fallback: if no numbered/bulleted items found, try plain lines that
        # start with a capital letter (common in free-text Assessment sections).
        # Only lines that look clinical: start with uppercase, contain at least
        # one word >=4 chars (filters out "OK", "See above", etc.)
        if not items:
            plain_lines = re.findall(
                r"(?:^|\n)[ \t]*([A-Z][A-Za-z].{3,120})",
                assessment_text,
                re.MULTILINE,
            )
            # Keep only lines where the first word is likely a clinical term
            # (>= 3 chars, not a common sentence starter)
            _SENTENCE_STARTERS = {
                "the", "this", "that", "there", "these", "those", "she", "her",
                "his", "patient", "will", "should", "please", "note", "also",
                "today", "currently", "recommend", "discussed", "plan", "need",
                "would", "could", "may", "might", "has", "had", "was", "were",
                "are", "been", "being", "have", "does", "did", "can",
                # Also exclude common Assessment sub-headers and comment prefixes
                "comments", "orders", "diagnoses", "referral", "ambulatory",
            }
            for line in plain_lines:
                first_word = line.split()[0].lower().rstrip(",:;")
                if first_word not in _SENTENCE_STARTERS and len(first_word) >= 3:
                    items.append(line)

        if not items:
            return result

        # Build a set of already-captured disease names (lowercase)
        # Include ALL diseases (not just ACTIVE) to avoid re-adding excluded ones
        existing_names = {
            d.get("disease_name", "").lower().strip()
            for d in result.get("diseases", [])
        }
        # Also include excluded conditions to avoid resurresting them
        for exc in result.get("excluded_conditions", []):
            term = exc.get("term", "").lower().strip()
            if term:
                existing_names.add(term)

        # Track existing ICD codes to prevent duplicate coding via different names
        existing_icds = {
            d.get("icd10_code", "").strip().upper()
            for d in result.get("diseases", [])
            if d.get("icd10_code", "").strip()
        }

        skip_first_words = {
            "see", "f/u", "follow", "consult", "refer", "order", "schedule",
            "return", "review", "continue", "discuss", "monitoring", "education",
            "discussed", "informed", "counseled", "labs", "imaging",
        }

        # Patterns that are NOT diseases — vaccines, lab orders, etc.
        _NON_DISEASE_RE = re.compile(
            r"(?i)(?:vaccine|vaccination|immunization|immunized|DTaP|MMR|IPV|Hep\s*[AB]|PENTACEL|INFANRIX|Rotavirus|Poliovirus|Pneumococcal|Varicella|PCV13|PROQUAD|ACTHIB|HAVRIX|VAQTA|IPOL|ROTATEQ|VARIVAX|^CBC|^CMP|^Comprehensive Metabolic|^Iron Profile|^MRI\b|^CT\b|^X-ray|^Ultrasound|^Answers submitted|^Overview$|^Blood pressure today|^Advised him|^ACE inhibitor|^VNA nursing|^Improvement or symptoms|^Still unclear)"
        )

        # Stable/not-managed conditions: items that are just listed with
        # "stable", "continue", "no change" annotations should not be added
        # by the coverage verifier — they are not actively managed this visit.
        _STABLE_ONLY_RE = re.compile(
            r"(?i)(?:^|\s*[-–—:,]\s*)"
            r"(?:stable|well\s+controlled|controlled|unchanged|"
            r"continue\s+(?:current|same|home)\s+(?:meds|medications?|regimen|treatment)|"
            r"no\s+change|at\s+goal|resolved|quiescent|"
            r"not\s+active|no\s+acute\s+issues?|asymptomatic|"
            r"no\s+(?:new\s+)?(?:symptoms?|complaints?|concerns?)|"
            r"doing\s+well|improved|improving|"
            r"noted\s+on\s+(?:colonoscopy|imaging|ct|mri|x-ray))"
            r"(?:\s*$|\s*[,;.])",
        )

        # Background/incidental phrases that should not be added by verifier
        # unless encounter-level management action is explicitly documented.
        # Only truly generic terms — NOT specific disease names.
        _BACKGROUND_OR_INCIDENTAL_RE = re.compile(
            r"(?i)\b(?:incidental|old\s+finding|stable\s+from\s+prior|"
            r"chronic\s+appearing|degenerative)\b"
        )
        _ACTIONABLE_RE = re.compile(
            r"(?i)\b(?:start|started|initiat|prescrib|adjust|increase|decrease|"
            r"taper|stop|discontinu|hold|order|follow\s*up|f/?u|monitor|"
            r"recheck|refer|discuss|counsel|reviewed|evaluat|treat|therapy|plan)\b"
        )
        _ORDER_ONLY_RE = re.compile(
            r"(?i)\b(?:future|profile|panel|order(?:ed)?|test(?:ing)?|ige|"
            r"lab\s+order|screening\s+order)\b"
        )

        next_idx = max(
            (d.get("index", 0) for d in result.get("diseases", [])), default=0
        )
        new_entries = []

        for item_text in items:
            item_clean = item_text.strip().rstrip(":")
            if not item_clean or len(item_clean) < 5 or len(item_clean) > 120:
                continue
            if item_clean.split()[0].lower() in skip_first_words:
                continue
            if self._is_non_disease_phrase(item_clean):
                continue
            if _NON_DISEASE_RE.search(item_clean):
                continue
            # Skip stable/not-managed items — they are listed but not actively
            # addressed this visit. E.g. "HTN - stable, continue current meds"
            if _STABLE_ONLY_RE.search(item_text):
                self.logger.debug(
                    f"Assessment verifier: skipping stable/not-managed item '{item_clean}'"
                )
                continue

            # Skip order-only testing/allergy workup lines that are not
            # diagnoses (e.g. "Allergy profile IgE; Future").
            if _ORDER_ONLY_RE.search(item_clean) and not _ACTIONABLE_RE.search(item_clean):
                self.logger.debug(
                    f"Assessment verifier: skipping order-only item '{item_clean}'"
                )
                continue

            # Skip chronic/incidental background items unless there is explicit
            # encounter-level management action in the line.
            if _BACKGROUND_OR_INCIDENTAL_RE.search(item_clean) and not _ACTIONABLE_RE.search(item_clean):
                self.logger.debug(
                    f"Assessment verifier: skipping background/incidental item '{item_clean}'"
                )
                continue

            norm = item_clean.lower().strip()
            # Abbreviation-aware match: "CKD 3" matches "chronic kidney disease stage 3"
            already_found = _abbreviation_aware_match(norm, existing_names)
            if already_found:
                continue

            next_idx += 1
            # Try to assign ICD code via lookup
            icd_code = ""
            icd_desc = ""
            if self._icd_lookup:
                try:
                    result_icd = self._icd_lookup.search_combined(item_clean, item_clean)
                    best = result_icd.get("best_match")
                    if best:
                        icd_code = best.get("icd_code", "")
                        icd_desc = best.get("description", "")
                except Exception:
                    pass

            # Skip if this ICD code was already captured by the LLM
            if icd_code and icd_code.upper() in existing_icds:
                self.logger.debug(
                    f"Assessment verifier: skipping '{item_clean}' — ICD {icd_code} already present"
                )
                continue
            new_entries.append({
                "index": next_idx,
                "disease_name": item_clean,
                "active_status": "ACTIVE",
                "meat_grade": "Partial MEAT",
                "meat_score": 1,
                "priority_level": "1-Assessment",
                "source_section": "Assessment_and_Plan",
                "all_sections_mentioned": ["Assessment_and_Plan"],
                "context_snippets": [item_clean],
                "meat_validation": {
                    "M_monitor": "",
                    "E_evaluate": "",
                    "A_assess": item_clean,
                    "T_treat": "",
                    "meat_score": 1,
                    "meat_grade": "Partial MEAT",
                    "meat_grade_reasoning": (
                        "Found in Assessment section but not captured by LLM — "
                        "added by coverage verifier."
                    ),
                    "primary_activation_basis": f"ASSESS: {item_clean}",
                },
                "icd10_code": icd_code,
                "icd10_description": icd_desc,
                "icd10_selection_reasoning": "Assigned via ICD lookup (coverage verifier)." if icd_code else "Missed by LLM; requires ICD assignment.",
                "hcc_status": "",
            })

        if new_entries:
            self.logger.warning(
                f"Assessment coverage verifier added {len(new_entries)} missed item(s): "
                + ", ".join(e["disease_name"] for e in new_entries)
            )
            result.setdefault("diseases", []).extend(new_entries)

        # ── BMI supplement code (Z68.x) ─────────────────────────────────
        # When E66.x (overweight/obesity) is coded and a BMI value appears
        # in the document, add the appropriate Z68.x supplementary code.
        existing_icds_final = {
            (d.get("icd10_code") or "").strip().upper()
            for d in result.get("diseases", [])
            if (d.get("icd10_code") or "").strip()
        }
        has_e66 = any(icd.startswith("E66") for icd in existing_icds_final)
        has_z68 = any(icd.startswith("Z68") for icd in existing_icds_final)

        if has_e66 and not has_z68:
            bmi_match = re.search(
                r'\b(?:BMI|Body\s+Mass\s+Index)\s*(?:of\s*|:\s*|=\s*|is\s*|>\s*)?'
                r'(\d{2,3}(?:\.\d+)?)',
                raw_text, re.IGNORECASE,
            )
            # Fallback: extract BMI from disease names (LLM often includes it)
            if not bmi_match:
                for d in result.get("diseases", []):
                    dname = d.get("disease_name", "")
                    bmi_match = re.search(
                        r'(?:BMI|body\s+mass\s+index)\s*(?:of\s*|:\s*|=\s*|is\s*)?'
                        r'(\d{2,3}(?:\.\d+)?)',
                        dname, re.IGNORECASE,
                    )
                    if bmi_match:
                        break
            if bmi_match:
                bmi_val = float(bmi_match.group(1))
                z68_code = self._bmi_to_z68(bmi_val)
                if z68_code and z68_code not in existing_icds_final:
                    next_idx += 1
                    bmi_phrase = f"BMI {bmi_val}"
                    result.setdefault("diseases", []).append({
                        "index": next_idx,
                        "disease_name": bmi_phrase,
                        "active_status": "ACTIVE",
                        "meat_grade": "Partial MEAT",
                        "meat_score": 1,
                        "priority_level": "1-Assessment",
                        "source_section": "Assessment_and_Plan",
                        "all_sections_mentioned": ["Assessment_and_Plan"],
                        "context_snippets": [bmi_phrase],
                        "meat_validation": {
                            "M_monitor": "",
                            "E_evaluate": bmi_match.group(0),
                            "A_assess": bmi_phrase,
                            "T_treat": "",
                            "meat_score": 2,
                            "meat_grade": "Partial MEAT",
                            "meat_grade_reasoning": (
                                "BMI supplement code paired with E66.x obesity/overweight."
                            ),
                            "primary_activation_basis": f"EVAL: {bmi_match.group(0)}",
                        },
                        "icd10_code": z68_code,
                        "icd10_description": f"Body mass index [BMI] {bmi_val}",
                        "icd10_selection_reasoning": "Deterministic BMI supplement code.",
                        "hcc_status": "",
                    })
                    self.logger.warning(
                        f"BMI supplement code added: {bmi_phrase} -> {z68_code}"
                    )

        # ── Personal / family history Z-codes ───────────────────────────
        _history_patterns = [
            (re.compile(r"(?i)\b(?:personal\s+)?history\s+of\s+(?:\w+\s+)*?"
                        r"(?:cancer|carcinoma|malignant\s+neoplasm|malignancy|lymphoma|"
                        r"leukemia|melanoma|sarcoma)\b"),
             "Z85.819", "Personal history of malignant neoplasm, unspecified"),
        ]
        for hpat, hcode, hdesc in _history_patterns:
            if hcode in existing_icds_final:
                continue
            hm = hpat.search(assessment_text)
            if hm:
                next_idx += 1
                hphrase = hm.group(0).strip()
                result.setdefault("diseases", []).append({
                    "index": next_idx,
                    "disease_name": hphrase,
                    "active_status": "ACTIVE",
                    "meat_grade": "Partial MEAT",
                    "meat_score": 1,
                    "priority_level": "1-Assessment",
                    "source_section": "Assessment_and_Plan",
                    "all_sections_mentioned": ["Assessment_and_Plan"],
                    "context_snippets": [hphrase],
                    "meat_validation": {
                        "M_monitor": "",
                        "E_evaluate": "",
                        "A_assess": hphrase,
                        "T_treat": "",
                        "meat_score": 1,
                        "meat_grade": "Partial MEAT",
                        "meat_grade_reasoning": "History code deterministic match.",
                        "primary_activation_basis": f"ASSESS: {hphrase}",
                    },
                    "icd10_code": hcode,
                    "icd10_description": hdesc,
                    "icd10_selection_reasoning": "Deterministic history code pattern.",
                    "hcc_status": "",
                })
                existing_icds_final.add(hcode)
                self.logger.warning(
                    f"History Z-code fallback added: {hphrase} -> {hcode}"
                )

        return result

    # ------------------------------------------------------------------
    # BMI value → Z68.x ICD code mapping
    # ------------------------------------------------------------------
    @staticmethod
    def _bmi_to_z68(bmi: float) -> str:
        """Map a BMI numeric value to the appropriate Z68.x ICD-10 code."""
        if bmi < 20:
            return "Z68.1"
        elif bmi < 40:
            return f"Z68.{int(bmi)}"
        elif bmi < 45:
            return "Z68.41"
        elif bmi < 50:
            return "Z68.42"
        elif bmi < 60:
            return "Z68.43"
        elif bmi < 70:
            return "Z68.44"
        elif bmi >= 70:
            return "Z68.45"
        return ""

    # ------------------------------------------------------------------
    # Post-LLM MEAT evidence deduplication and quality enforcement
    # ------------------------------------------------------------------

    # Generic evidence strings that are NOT disease-specific and must be removed
    _GENERIC_EVIDENCE = re.compile(
        r"^(?:medications?|procedures?|not\s*documented|problem\s*list|"
        r"height\s*weight\s*bmi|vital\s*signs?|"
        r"n/?a|none|see\s*above|as\s*above|see\s*plan|per\s*plan)$",
        re.IGNORECASE,
    )

    # Mapping of monitoring/evaluation evidence to the disease categories they are relevant to
    _EVIDENCE_DISEASE_RELEVANCE = {
        # ECG/EKG/Electrocardiogram → only cardiac conditions
        r"(?i)(?:ecg|ekg|electrocardiogram|12.lead|rhythm\s*strip)": {
            "coronary", "artery", "cardiac", "heart", "atrial", "fibrillation",
            "flutter", "arrhythmia", "tachycardia", "bradycardia", "murmur",
            "pacemaker", "valve", "aortic", "mitral", "tricuspid", "angina",
            "ischemic", "cardiomyopathy", "sinus", "hypertensive heart",
        },
        # Height/Weight/BMI → only obesity/nutrition
        r"(?i)(?:height\s*weight|bmi|body\s*mass|weight\s*\d)": {
            "obesity", "obese", "overweight", "underweight", "malnutrition",
            "cachexia", "bmi", "morbid",
        },
        # Lipid panel → only lipid conditions
        r"(?i)(?:lipid|cholesterol|ldl|hdl|triglyceride)": {
            "lipid", "cholesterol", "hyperlipidemia", "dyslipidemia",
            "hypercholesterolemia", "hypertriglyceridemia",
        },
        # HbA1c / glucose → only diabetes
        r"(?i)(?:hba1c|a1c|glucose|glycated|blood\s*sugar|fasting\s*glucose)": {
            "diabet", "glucose", "hyperglycemia", "insulin", "dm", "a1c",
        },
        # Renal function → only kidney conditions
        r"(?i)(?:creatinine|gfr|egfr|bun|renal\s*function)": {
            "kidney", "renal", "ckd", "nephro", "dialysis", "uremia",
        },
        # Thyroid labs → only thyroid conditions
        r"(?i)(?:tsh|t3|t4|thyroid\s*function|free\s*t4)": {
            "thyroid", "hypothyroid", "hyperthyroid", "hashimoto", "graves",
        },
    }

    def _is_evidence_relevant(self, evidence: str, disease_name: str) -> bool:
        """Check if evidence is clinically relevant to the named disease."""
        if not evidence:
            return True  # empty is fine
        disease_lower = disease_name.lower()
        for evidence_pattern, relevant_disease_terms in self._EVIDENCE_DISEASE_RELEVANCE.items():
            if re.search(evidence_pattern, evidence):
                # This evidence matches a specific clinical test — check relevance
                if any(term in disease_lower for term in relevant_disease_terms):
                    return True  # relevant
                return False  # evidence matched a specific test but disease doesn't match
        return True  # not a recognized pattern — assume relevant

    def _deduplicate_meat_evidence(self, result: dict) -> dict:
        """
        Post-LLM safety net that ensures MEAT evidence is:
        1. Disease-specific (not generic labels)
        2. Clinically relevant to each disease
        3. Not duplicated across diseases (first disease keeps evidence)
        """
        diseases = result.get("diseases", [])
        if not diseases:
            return result

        # Track which evidence strings have been used (per MEAT component)
        used_evidence = {
            "M_monitor": {},   # evidence_normalized → disease_name
            "E_evaluate": {},
            "T_treat": {},
        }
        # A_assess is exempt from dedup — each disease legitimately has its own assessment text

        meat_keys = ["M_monitor", "E_evaluate", "A_assess", "T_treat"]
        cleaned_count = 0

        for disease in diseases:
            meat_val = disease.get("meat_validation", {})
            disease_name = disease.get("disease_name", "")

            for key in meat_keys:
                evidence = (meat_val.get(key) or "").strip()
                if not evidence:
                    continue

                # Rule 1: Remove generic/useless evidence
                if self._GENERIC_EVIDENCE.match(evidence):
                    self.logger.debug(
                        f"Stripped generic evidence '{evidence}' from '{disease_name}'.{key}"
                    )
                    meat_val[key] = ""
                    cleaned_count += 1
                    continue

                # Rule 2: Check clinical relevance (skip for A_assess)
                if key != "A_assess" and not self._is_evidence_relevant(evidence, disease_name):
                    self.logger.debug(
                        f"Stripped irrelevant evidence '{evidence}' from '{disease_name}'.{key}"
                    )
                    meat_val[key] = ""
                    cleaned_count += 1
                    continue

                # Rule 3: Deduplication (skip for A_assess)
                if key != "A_assess":
                    evidence_norm = evidence.lower().strip()
                    if evidence_norm in used_evidence[key]:
                        original_owner = used_evidence[key][evidence_norm]
                        self.logger.debug(
                            f"Dedup: '{evidence}' already used by '{original_owner}', "
                            f"clearing from '{disease_name}'.{key}"
                        )
                        meat_val[key] = ""
                        cleaned_count += 1
                        continue
                    used_evidence[key][evidence_norm] = disease_name

            # Recalculate meat_score after cleaning
            score = sum(1 for k in meat_keys if meat_val.get(k, "").strip())
            meat_val["meat_score"] = score

            # Update meat_grade based on new score
            grade_map = {4: "Full MEAT", 3: "Medium MEAT", 2: "Half MEAT", 1: "Partial MEAT", 0: "No MEAT"}
            meat_val["meat_grade"] = grade_map.get(score, "No MEAT")
            disease["meat_score"] = score
            disease["meat_grade"] = meat_val["meat_grade"]

        if cleaned_count:
            self.logger.info(f"MEAT evidence cleanup: {cleaned_count} generic/duplicate/irrelevant entries removed.")

        return result

    # ------------------------------------------------------------------
    # Post-LLM: Strict evidence-only MEAT validation (NO fallback/inference)
    # ------------------------------------------------------------------

    # Minimum evidence length to be considered valid
    _MIN_EVIDENCE_LENGTH = 3

    def _strict_evidence_validation(self, result: dict, raw_text: str) -> dict:
        """
        Strict evidence validation layer: reject any MEAT entry where the
        evidence text is absent, too short, or not verifiable in the source document.

        Rules:
          1. Evidence must be non-empty and >= _MIN_EVIDENCE_LENGTH chars.
          2. Evidence must be found in the raw document text (substring or
             high-word-overlap).
          3. No inference, no fallback, no disease-name-only entries.
          4. Diseases with zero valid MEAT components get status markers.
          5. Confidence is downgraded when only a diagnosis is documented.

        NO artificial enrichment is applied.
        """
        diseases = result.get("diseases", [])
        rejected_count = 0
        insufficient_count = 0

        for d in diseases:
            meat_val = d.get("meat_validation", {})
            components_valid = {"M_monitor": False, "E_evaluate": False,
                                "A_assess": False, "T_treat": False}

            for key, label in [
                ("M_monitor", "monitoring"),
                ("E_evaluate", "evaluation"),
                ("A_assess", "assessment"),
                ("T_treat", "treatment"),
            ]:
                evidence = (meat_val.get(key) or "").strip()

                # Rule 1: evidence must exist and be substantive
                if not evidence or len(evidence) < self._MIN_EVIDENCE_LENGTH:
                    if evidence:  # too short — reject silently
                        rejected_count += 1
                    meat_val[key] = ""
                    continue

                # Rule 2: evidence must be present in source document
                if not self._evidence_in_document(evidence, raw_text):
                    self.logger.debug(
                        f"Rejected {label} evidence for '{d.get('disease_name', '')}': "
                        f"not found in document."
                    )
                    meat_val[key] = ""
                    rejected_count += 1
                    continue

                components_valid[key] = True

            # Recalculate score from verified components only
            valid_count = sum(components_valid.values())
            meat_val["meat_score"] = valid_count

            if valid_count == 0:
                meat_val["meat_grade"] = "No MEAT"
                meat_val["meat_status"] = "insufficient clinical evidence"
                meat_val["no_monitoring"] = "No Monitoring evidence found"
                meat_val["no_treatment"] = "No Treatment documented"
                meat_val["no_evaluation"] = "No Evaluation evidence found"
                meat_val["no_assessment"] = "Diagnosis documented without supporting MEAT criteria"
                meat_val["evidence_based"] = False
                d["meat_score"] = 0
                d["meat_grade"] = "No MEAT"
                # Downgrade confidence for diagnosis-only entries
                current_conf = d.get("confidence_score", 0.8)
                d["confidence_score"] = min(current_conf, 0.42)
                insufficient_count += 1
            elif valid_count == 1:
                meat_val["meat_grade"] = "Weak evidence"
                meat_val["meat_status"] = "single MEAT component documented"
                meat_val["evidence_based"] = True
                d["meat_score"] = 1
                d["meat_grade"] = "Weak evidence"
            elif valid_count == 2:
                meat_val["meat_grade"] = "Moderate evidence"
                meat_val["meat_status"] = "two MEAT components documented"
                meat_val["evidence_based"] = True
                d["meat_score"] = 2
                d["meat_grade"] = "Moderate evidence"
            else:  # 3 or 4
                meat_val["meat_grade"] = "Strong evidence"
                meat_val["meat_status"] = "strong MEAT documentation"
                meat_val["evidence_based"] = True
                d["meat_score"] = valid_count
                d["meat_grade"] = "Strong evidence"

            d["meat_validation"] = meat_val

        if rejected_count:
            self.logger.info(
                f"Strict MEAT validation: {rejected_count} evidence entries rejected "
                f"(absent or not found in document)."
            )
        if insufficient_count:
            self.logger.info(
                f"Strict MEAT validation: {insufficient_count} disease(s) marked as "
                f"'insufficient clinical evidence' (no valid MEAT)."
            )

        return result

    def _evidence_in_document(self, evidence: str, raw_text: str) -> bool:
        """
        Verify that an evidence string actually appears in the source document.
        Uses three techniques:
          1. Exact substring (case-insensitive, normalized whitespace)
          2. Partial sentence overlap >= 30% word match
          3. Key phrase from first 80 chars present
        """
        if not evidence or not raw_text:
            return False

        ev_norm = " ".join(evidence.lower().split())
        doc_norm = " ".join(raw_text.lower().split())

        # 1. Exact substring
        if ev_norm in doc_norm:
            return True

        # 2. Word overlap on key part (first 80 chars of evidence)
        key_part = ev_norm[:80]
        ev_words = set(w for w in key_part.split() if len(w) > 3)
        if not ev_words:
            return False

        # Search for any window in doc containing >= 30% of ev_words
        words_found = sum(1 for w in ev_words if w in doc_norm)
        if len(ev_words) > 0 and (words_found / len(ev_words)) >= 0.30:
            return True

        return False

    # ------------------------------------------------------------------
    # REMOVED: Old fallback-based MEAT enrichment (_TREAT_SIGNALS,
    # _MONITOR_SIGNALS, _EVALUATE_SIGNALS, _enrich_meat_from_context).
    # Evidence must come directly from LLM extraction, verified against
    # the source document. No artificial inference is allowed.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pre-LLM: NER-style candidate extraction (Approach 5)
    # ------------------------------------------------------------------

    # Medical suffixes that signal a disease/condition term
    _DISEASE_SUFFIX_RE = re.compile(
        r"(?:disease|disorder|syndrome|deficiency|failure|insufficiency|"
        r"infection|inflammation|neoplasm|carcinoma|tumor|malignancy|"
        r"stenosis|occlusion|obstruction|fibrosis|sclerosis|neuropathy|"
        r"myopathy|arthritis|itis|emia|osis|opathy|penia|algia|uria|"
        r"plegia|paresis|ectasia|edema|dysplasia|regurgitation|prolapse)\b",
        re.IGNORECASE,
    )

    # Numbered/bulleted list item in Assessment/Plan
    _NUMBERED_RE = re.compile(
        r"(?:^|\n)\s*(?:\d{1,2}[.):\s]+|[\u2022\-\*]\s+)([A-Z][^\n]{3,100})",
        re.MULTILINE,
    )

    # Contextual disease mention patterns
    _CONTEXT_DISEASE_RE = re.compile(
        r"(?:history\s+of|h/o|diagnosed\s+with|diagnosis(?:\s+of)?:?\s*|"
        r"assessment:?\s*|impression:?\s*|problem:?\s*)"
        r"\s+([A-Z][A-Za-z0-9,/\s\-()]{2,80})",
        re.IGNORECASE,
    )

    def _extract_ner_candidates(self, text: str) -> List[str]:
        """
        Content-based NER candidate extraction using regex patterns.

        Scans the document text for disease-like entities using:
        1. Medical suffix patterns (e.g. words ending in -itis, -emia, disease, etc.)
        2. Numbered/bulleted list items from Assessment/Plan sections
        3. Contextual patterns (h/o X, diagnosed with X, etc.)

        Returns a deduplicated list of candidate disease names found in the text.
        These are passed to the LLM as hints to constrain extraction.
        """
        candidates: Set[str] = set()

        # 1. Find Assessment/Plan section and extract numbered items
        section_header = re.compile(
            r"(?:^|\n)[ \t]*"
            r"(?:assessment\s*(?:and\s*plan|&\s*plan|/\s*plan)?|a\s*/\s*p|"
            r"problem\s*list|active\s*problems?|diagnosis|impression)"
            r"\s*[:\n]",
            re.IGNORECASE | re.MULTILINE,
        )
        for m in section_header.finditer(text):
            sec_start = m.end()
            # Find end of section (next major header)
            next_hdr = re.search(
                r"\n[ \t]*(?:medications?|physical\s*exam|review\s*of\s*systems?"
                r"|labs?|imaging|referral|vitals?|subjective|objective"
                r"|family\s*history|social\s*history|allergies)[^\n]*\n",
                text[sec_start:], re.IGNORECASE,
            )
            sec_end = sec_start + next_hdr.start() if next_hdr else len(text)
            sec_text = text[sec_start:sec_end]

            # Extract numbered/bulleted items
            for item_m in self._NUMBERED_RE.finditer(sec_text):
                item = item_m.group(1).strip().rstrip(":.,;")
                # Truncate at plan annotations (stable, counsel, continue, etc.)
                plan_cut = re.search(
                    r"\s*[-–=,]\s*(?:stable|controlled|counsel|monitor|continue|"
                    r"consider|start|check|order|increase|decrease|refer)\b",
                    item, re.IGNORECASE,
                )
                if plan_cut:
                    item = item[:plan_cut.start()].strip()
                if 3 < len(item) < 100:
                    candidates.add(item)

        # 2. Find disease-suffix words in the entire text
        for m in self._DISEASE_SUFFIX_RE.finditer(text):
            # Get surrounding context (the phrase containing this term)
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 10)
            chunk = text[start:end]
            # Try to extract a clean phrase
            phrase_m = re.search(
                r"([A-Z][A-Za-z\s\-,()]{2,60}?" + re.escape(m.group()) + r")",
                chunk,
            )
            if phrase_m:
                cand = phrase_m.group(1).strip().rstrip(":.,;")
                if 3 < len(cand) < 80:
                    candidates.add(cand)

        # 3. Contextual patterns
        for m in self._CONTEXT_DISEASE_RE.finditer(text):
            cand = m.group(1).strip().rstrip(":.,;")
            # Truncate at common stopwords
            stop_cut = re.search(
                r"\b(?:and|with|due\s+to|secondary|on|per|since|for)\b",
                cand, re.IGNORECASE,
            )
            if stop_cut and stop_cut.start() > 5:
                cand = cand[:stop_cut.start()].strip()
            if 3 < len(cand) < 80:
                candidates.add(cand)

        # 4. Abbreviation expansion — also add expanded forms
        expanded_candidates: Set[str] = set()
        for cand in candidates:
            expanded_candidates.add(cand)
            for word in cand.split():
                w_lower = word.lower().strip("(),")
                if w_lower in _ABBR_TO_EXPANSION:
                    expanded_candidates.add(
                        cand.replace(word, _ABBR_TO_EXPANSION[w_lower].title())
                    )

        # Deduplicate and sort
        result = sorted(expanded_candidates)
        self.logger.info(
            f"NER pre-extraction found {len(result)} candidate diseases"
        )
        return result

    # ------------------------------------------------------------------
    # Post-LLM: Two-step verification (Approach 4)
    # ------------------------------------------------------------------

    def _verify_diseases_in_document(
        self, raw_text: str, result: dict
    ) -> dict:
        """
        Two-step verification: check each extracted disease actually
        exists in the document text.

        For each disease in results['diseases']:
        1. Check if disease_name (or a close variant) appears in raw_text
        2. Check if at least one context_snippet appears in raw_text
        3. If NEITHER is found → mark as INVALID and move to excluded_conditions

        Uses fuzzy matching (SequenceMatcher) to handle minor LLM
        paraphrasing of disease names.
        """
        diseases = result.get("diseases", [])
        if not diseases:
            return result

        text_lower = raw_text.lower()
        # Build word set for fast partial matching
        text_words = set(re.findall(r"[a-z]{3,}", text_lower))

        verified = []
        newly_excluded = []

        for disease in diseases:
            name = (disease.get("disease_name") or "").strip()
            if not name:
                continue

            name_lower = name.lower()

            # Check 1: Direct substring match (fastest)
            if name_lower in text_lower:
                verified.append(disease)
                continue

            # Check 2: Abbreviation-aware match
            name_keys = _normalise_for_match(name)
            found_via_abbr = False
            for key in name_keys:
                if key in text_lower:
                    found_via_abbr = True
                    break
            if found_via_abbr:
                verified.append(disease)
                continue

            # Check 3: Word-overlap match — at least 80% of disease name words
            # must appear in the document
            name_words = set(re.findall(r"[a-z]{3,}", name_lower))
            if name_words:
                overlap = len(name_words & text_words) / len(name_words)
                if overlap >= 0.8:
                    verified.append(disease)
                    continue

            # Check 4: Context snippet verification — if the LLM quoted text
            # that actually exists in the document, trust the disease
            snippets = disease.get("context_snippets", [])
            snippet_found = False
            for snippet in snippets:
                if not snippet:
                    continue
                snippet_lower = snippet.lower().strip()
                # Direct substring
                if snippet_lower[:50] in text_lower:
                    snippet_found = True
                    break
                # Fuzzy match on first 80 chars
                ratio = SequenceMatcher(
                    None, snippet_lower[:80], text_lower
                ).find_longest_match(0, min(80, len(snippet_lower)), 0, len(text_lower))
                if ratio.size >= 30:
                    snippet_found = True
                    break

            if snippet_found:
                verified.append(disease)
                continue

            # Check 5: Word overlap — if most significant words of the disease
            # name appear in the document, accept it
            name_sig_words = {w for w in re.findall(r"[a-z]{3,}", name_lower)
                              if w not in {"the", "and", "with", "type", "stage", "grade"}}
            if name_sig_words and len(name_sig_words & text_words) / len(name_sig_words) >= 0.7:
                verified.append(disease)
                continue

            # Disease NOT found in document → hallucination
            self.logger.warning(
                f"Two-step verification REJECTED: '{name}' not found in document"
            )
            newly_excluded.append({
                "term": name,
                "reason_excluded": "HALLUCINATED",
                "source_section": disease.get("source_section", "unknown"),
                "exclusion_reasoning": (
                    f"Disease name not found in document text. "
                    f"Removed by two-step verification."
                ),
            })

        if newly_excluded:
            self.logger.info(
                f"Two-step verification removed {len(newly_excluded)} "
                f"hallucinated disease(s): "
                + ", ".join(e["term"] for e in newly_excluded)
            )
            result["diseases"] = verified
            result.setdefault("excluded_conditions", []).extend(newly_excluded)

        return result

    # ------------------------------------------------------------------
    # Post-LLM: Evidence-in-document validation (Approach 7)
    # ------------------------------------------------------------------

    def _verify_evidence_in_document(
        self, raw_text: str, result: dict
    ) -> dict:
        """
        Validate that MEAT evidence strings actually appear in the document.

        For each MEAT component (M_monitor, E_evaluate, A_assess, T_treat):
        - Check if the evidence text can be found in the document
        - If similarity is below threshold → clear the evidence
        - Recalculate meat_score after cleaning
        """
        diseases = result.get("diseases", [])
        if not diseases:
            return result

        text_lower = raw_text.lower()
        meat_keys = ["M_monitor", "E_evaluate", "A_assess", "T_treat"]
        cleaned_count = 0

        for disease in diseases:
            meat_val = disease.get("meat_validation", {})

            for key in meat_keys:
                evidence = (meat_val.get(key) or "").strip()
                if not evidence or len(evidence) < 5:
                    continue

                evidence_lower = evidence.lower()

                # Quick check: direct substring match
                if evidence_lower[:40] in text_lower:
                    continue

                # Check individual key phrases (split by commas/semicolons)
                phrases = re.split(r"[,;]+", evidence_lower)
                any_phrase_found = False
                for phrase in phrases:
                    phrase = phrase.strip()
                    if len(phrase) < 4:
                        continue
                    if phrase in text_lower:
                        any_phrase_found = True
                        break

                if any_phrase_found:
                    continue

                # Normalized word-overlap check (faster than SequenceMatcher)
                ev_words = set(re.findall(r"[a-z]{3,}", evidence_lower))
                if ev_words:
                    doc_words = set(re.findall(r"[a-z]{3,}", text_lower))
                    overlap = len(ev_words & doc_words) / len(ev_words)
                    if overlap < 0.5:
                        self.logger.debug(
                            f"Evidence validation: stripped '{evidence[:50]}...' from "
                            f"'{disease.get('disease_name', '')}' .{key} "
                            f"(word_overlap={overlap:.2f})"
                        )
                        meat_val[key] = ""
                        cleaned_count += 1

            # Recalculate meat_score after cleaning
            score = sum(1 for k in meat_keys if meat_val.get(k, "").strip())
            meat_val["meat_score"] = score
            grade_map = {4: "Full MEAT", 3: "Medium MEAT", 2: "Half MEAT", 1: "Partial MEAT", 0: "No MEAT"}
            meat_val["meat_grade"] = grade_map.get(score, "No MEAT")
            disease["meat_score"] = score
            disease["meat_grade"] = meat_val["meat_grade"]

        if cleaned_count:
            self.logger.info(
                f"Evidence-in-document validation: stripped {cleaned_count} "
                f"fabricated evidence entries."
            )

        return result

    # ------------------------------------------------------------------
    # Result conversion — map single-pass output to pipeline format
    # ------------------------------------------------------------------

    def convert_to_pipeline_format(
        self, analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert the single-pass LLM output into the format expected by
        the existing pipeline (detected_diseases, sections, meat, icd).

        This allows the rest of the system (DB storage, frontend display)
        to work unchanged.
        """
        diseases_raw = analysis_result.get("diseases", [])
        sections_found = analysis_result.get("sections_found", [])
        excluded = analysis_result.get("excluded_conditions", [])

        # Build sections dict (lightweight — positions not available in single-pass)
        sections = {}
        for sec_name in sections_found:
            canonical = self._canonicalize_section(sec_name)
            if canonical not in sections:
                sections[canonical] = {
                    "text": f"[Detected by single-pass LLM: {sec_name}]",
                    "start": 0,
                    "end": 0,
                }

        # Build detected_diseases list
        detected_diseases = []
        for d in diseases_raw:
            name = (d.get("disease_name") or "").strip()
            if not name:
                continue
            if d.get("active_status", "").upper() != "ACTIVE":
                continue

            source_section = self._canonicalize_section(
                d.get("source_section", "unknown")
            )
            all_sections = [
                self._canonicalize_section(s)
                for s in d.get("all_sections_mentioned", [source_section])
            ]

            detected_diseases.append({
                "disease_name": name,
                "normalized_name": name.lower().strip(),
                "confidence_score": 0.95,
                "negated": False,
                "section": source_section,
                "section_sources": all_sections,
                "entity_type": "CONDITION",
                "sentence_number": 0,
                "icd_code": d.get("icd10_code", ""),
                "icd_description": d.get("icd10_description", ""),
                "icd_selection_reasoning": d.get("icd10_selection_reasoning", ""),
                "meat_grade": d.get("meat_grade", ""),
                "meat_score": d.get("meat_score", 0),
                "meat_validation": d.get("meat_validation", {}),
                "priority_level": d.get("priority_level", ""),
                "context_snippets": d.get("context_snippets", []),
                "hcc_status": d.get("hcc_status", ""),
                "active_status": "ACTIVE",
            })

        self.logger.info(
            f"Converted single-pass result: {len(detected_diseases)} active diseases, "
            f"{len(sections)} sections."
        )

        return {
            "detected_diseases": detected_diseases,
            "sections": sections,
            "excluded_conditions": excluded,
            "analysis_source": "single_pass_llm",
        }

    # Non-disease phrases that should never appear as extracted diseases
    _NON_DISEASE_PHRASES = {
        "chronic disease management", "disease management", "medication management",
        "care coordination", "health maintenance", "preventive care", "wellness visit",
        "follow-up visit", "medication reconciliation", "care plan", "patient education",
        "counseling", "screening", "routine exam", "annual physical", "office visit",
        "established patient", "new patient", "telephone encounter", "telehealth visit",
        "referral", "consultation", "discharge planning", "case management",
        "pain management", "weight management", "stress management",
        "chronic care management", "transitional care management",
    }

    @classmethod
    def _is_non_disease_phrase(cls, name: str, has_icd: bool = False) -> bool:
        """Return True if the name is a clinical workflow phrase, not a disease.

        Parameters
        ----------
        has_icd : bool
            When True, the LLM assigned an ICD code to this entry.  In that
            case only the exact-match blocklist is checked — the fuzzy suffix
            heuristic is skipped because a real ICD code is strong evidence
            that the LLM identified an actual condition, not a workflow phrase.
        """
        low = name.lower().strip()
        # Exact match — always block these regardless of ICD
        if low in cls._NON_DISEASE_PHRASES:
            return True
        # When the LLM already assigned an ICD code, trust it — skip fuzzy rule
        if has_icd:
            return False
        # Fuzzy match — only for entries without ICD (e.g. assessment verifier items)
        # If the name is <=4 words and ends with a workflow suffix, block it.
        workflow_suffixes = ("management", "visit", "coordination", "planning",
                            "education", "counseling", "screening", "reconciliation")
        words = low.split()
        if len(words) <= 4 and any(low.endswith(s) for s in workflow_suffixes):
            return True
        return False

    def convert_to_unified_results(
        self, analysis_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Convert the single-pass LLM output directly into the unified_results
        format used by the frontend, bypassing the separate MEAT/ICD steps.
        """
        diseases_raw = analysis_result.get("diseases", [])
        unified = []

        for idx, d in enumerate(diseases_raw, 1):
            name = (d.get("disease_name") or "").strip()
            if not name:
                continue
            if d.get("active_status", "").upper() != "ACTIVE":
                continue

            # Filter out clinical workflow phrases that aren't diseases
            # Pass has_icd=True when LLM assigned an ICD code — stronger signal it's real
            entry_has_icd = bool(d.get("icd10_code", "").strip())
            if self._is_non_disease_phrase(name, has_icd=entry_has_icd):
                self.logger.debug(f"Filtered non-disease phrase: '{name}'")
                continue

            meat_val = d.get("meat_validation", {})
            meat_grade = d.get("meat_grade", meat_val.get("meat_grade", ""))
            meat_score = d.get("meat_score", meat_val.get("meat_score", 0))

            # Evidence-based MEAT tier: only count components with real
            # documented evidence from the source PDF.
            _m = bool(meat_val.get("M_monitor"))
            _e = bool(meat_val.get("E_evaluate"))
            _a = bool(meat_val.get("A_assess"))
            _t = bool(meat_val.get("T_treat"))
            _components = sum([_m, _e, _a, _t])

            # New evidence-based scoring: 0=no_meat, 1=weak, 2=moderate, 3+=strong
            if _components >= 3:
                tier = "strong_evidence"
            elif _components == 2:
                tier = "moderate_evidence"
            elif _components == 1:
                tier = "weak_evidence"
            else:
                tier = "no_meat"

            tier_label = {
                "strong_evidence": "Strong Evidence",
                "moderate_evidence": "Moderate Evidence",
                "weak_evidence": "Weak Evidence",
                "no_meat": "No MEAT",
            }.get(tier, "No MEAT")

            # Evidence-based flags from strict validation
            meat_status = d.get("meat_status", meat_val.get("meat_status", ""))
            evidence_based = d.get("evidence_based", meat_val.get("evidence_based", _components > 0))

            source_section = d.get("source_section", "unknown")
            all_sections = d.get("all_sections_mentioned", [source_section])

            icd_code = d.get("icd10_code") or ""
            icd_desc = d.get("icd10_description") or ""
            icd_confidence = 0.95

            # Skip diseases without ICD codes — unless from Assessment section
            # (coverage verifier items may need a second lookup attempt)
            if not icd_code:
                is_assessment_source = source_section.lower().replace(" ", "_") in {
                    "assessment", "assessment_and_plan",
                }
                if is_assessment_source and self._icd_lookup:
                    # Second-chance ICD lookup for Assessment items the verifier added
                    try:
                        retry = self._icd_lookup.search_combined(name, name)
                        best = retry.get("best_match")
                        if best:
                            icd_code = best.get("icd_code", "")
                            icd_desc = best.get("description", "")
                    except Exception:
                        pass
                if not icd_code:
                    continue

            # Validate ICD code against database — catch hallucinated codes
            if self._icd_lookup:
                try:
                    db_hit = self._icd_lookup.search_by_code(icd_code)
                    if db_hit is None:
                        # Try without the decimal (some DBs store raw codes)
                        db_hit = self._icd_lookup.search_by_code(icd_code.replace(".", ""))
                    if db_hit:
                        # Prefer the DB description — it's ground truth
                        icd_code = db_hit.get("icd_code", icd_code)
                        icd_desc = db_hit.get("description", icd_desc) or icd_desc
                    else:
                        icd_confidence = 0.7
                        self.logger.warning(
                            f"ICD code '{icd_code}' not found in DB for '{name}' — "
                            f"marking reduced confidence."
                        )
                except Exception as _icd_err:
                    self.logger.debug(f"ICD validation skipped: {_icd_err}")

            mon = bool(meat_val.get("M_monitor"))
            eva = bool(meat_val.get("E_evaluate"))
            ass = bool(meat_val.get("A_assess"))
            tre = bool(meat_val.get("T_treat"))

            # Evidence-based confidence: no MEAT → cap at 0.42
            meat_components = sum([mon, eva, ass, tre])
            if meat_components == 0:
                confidence = round(min(0.42, icd_confidence * 0.4), 2)
            else:
                confidence = round(
                    (meat_components / 4) * 0.6 + icd_confidence * 0.4, 2
                )

            unified.append({
                "number": idx,
                "disease": name,
                "icd_code": icd_code,
                "icd_description": icd_desc,
                "segment": source_section,
                "segment_source_raw": all_sections,
                "disease_status": "Active",
                "monitoring": mon,
                "evaluation": eva,
                "assessment": ass,
                "treatment": tre,
                "meat_level": tier_label,
                "meat_tier": tier,
                "meat_score": _components,
                "meat_status": meat_status,
                "evidence_based": evidence_based,
                "confidence": confidence,
                "icd_confidence": icd_confidence,
                "icd_method": "single_pass_llm",
                "monitoring_evidence": meat_val.get("M_monitor") or "",
                "evaluation_evidence": meat_val.get("E_evaluate") or "",
                "assessment_evidence": meat_val.get("A_assess") or "",
                "treatment_evidence": meat_val.get("T_treat") or "",
                "llm_reasoning": meat_val.get("meat_grade_reasoning", ""),
                "primary_activation_basis": meat_val.get("primary_activation_basis", ""),
                "icd_selection_reasoning": d.get("icd10_selection_reasoning", ""),
                "hcc_status": d.get("hcc_status", ""),
                "context_snippets": d.get("context_snippets", []),
            })

        # Re-number
        for i, r in enumerate(unified, 1):
            r["number"] = i

        self.logger.info(f"Built {len(unified)} unified results from single-pass analysis.")
        return unified

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    @classmethod
    def _load_prompt_config(cls) -> dict:
        """Load the full prompt config from backend/prompt.json (fresh every call)."""
        prompt_path = Path(__file__).resolve().parents[2] / "prompt.json"
        try:
            with prompt_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error(f"Failed to load prompt.json from {prompt_path}: {exc}")
            return {}

    @classmethod
    def _flatten_to_text(cls, obj, indent: int = 0) -> str:
        """Recursively convert a nested dict/list into readable indented text."""
        lines = []
        prefix = "  " * indent
        if isinstance(obj, dict):
            for key, val in obj.items():
                label = key.replace("_", " ").upper() if indent == 0 else key.replace("_", " ").title()
                if isinstance(val, str):
                    lines.append(f"{prefix}{label}: {val}")
                elif isinstance(val, list):
                    lines.append(f"{prefix}{label}:")
                    for item in val:
                        if isinstance(item, str):
                            lines.append(f"{prefix}  - {item}")
                        else:
                            lines.append(cls._flatten_to_text(item, indent + 2))
                elif isinstance(val, dict):
                    lines.append(f"{prefix}{label}:")
                    lines.append(cls._flatten_to_text(val, indent + 1))
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    lines.append(f"{prefix}- {item}")
                else:
                    lines.append(cls._flatten_to_text(item, indent + 1))
        elif isinstance(obj, str):
            lines.append(f"{prefix}{obj}")
        return "\n".join(lines)

    def _build_prompt(self, text: str, ner_candidates: List[str] = None, dietary_analysis: bool = False) -> str:
        """Build a clear natural-language prompt from prompt.json phases.

        Converts the structured JSON into readable instructions with the
        clinical document in a clearly delimited block. This gives Gemini
        clear instructions + clear document boundaries for accurate extraction.

        Parameters
        ----------
        text : str
            The clinical document text (already abbreviation-expanded).
        ner_candidates : list[str], optional
            Disease candidates pre-extracted via NER/regex from the document.
            Injected into the prompt to constrain LLM extraction.
        """
        doc_text = text[:25000] if len(text) > 25000 else text

        config = self._load_prompt_config()

        # ── Detect prompt.json structure ──────────────────────────────────────
        # New structure: top-level STEP_x keys (e.g. STEP_0_dynamic_document_reading)
        # Old structure: nested under `user_prompt` with phase_x keys
        # We support BOTH formats.
        step_keys = [k for k in config if k.startswith("STEP_")]
        user_prompt = config.get("user_prompt", {})
        has_new_structure = len(step_keys) > 0

        if not has_new_structure and not user_prompt:
            self.logger.warning("prompt.json has no recognized structure — using inline fallback")

        # ── Build NER hint string ─────────────────────────────────────────────
        ner_hint = ""
        if ner_candidates:
            ner_hint = (
                "The following disease candidates were pre-extracted via NER. "
                "Use as a REFERENCE — extract these AND any others in the document, "
                "but NEVER hallucinate conditions not in the text.\n"
                "Candidates:\n"
                + "\n".join(f"  - {c}" for c in ner_candidates[:40])
            )

        # ── Clinical document block (always included) ─────────────────────────
        doc_block = (
            f"\n{'='*60}\n"
            f"CLINICAL DOCUMENT TO ANALYZE:\n"
            f"{'='*60}\n"
            f"{doc_text}\n"
            f"{'='*60}\n"
            f"END OF CLINICAL DOCUMENT\n"
            f"{'='*60}"
        )

        # ── Build prompt from NEW structure (STEP_x keys) ─────────────────────
        if has_new_structure:
            parts = []

            # System philosophy header
            core = config.get("core_philosophy")
            if core:
                parts.append(f"CORE PHILOSOPHY:\n{self._flatten_to_text(core)}")

            if ner_hint:
                parts.append(f"NER PRE-EXTRACTION HINT:\n{ner_hint}")

            # Document block
            parts.append(doc_block)

            # All STEP instructions in order
            ordered_steps = [
                ("STEP_0_dynamic_document_reading",  "STEP 0 — DYNAMIC DOCUMENT READING"),
                ("STEP_1_section_mapping",            "STEP 1 — SECTION MAPPING & TIERS"),
                ("STEP_2_disease_extraction",         "STEP 2 — DISEASE EXTRACTION"),
                ("STEP_3_zcode_extraction_with_encounter_gate", "STEP 3 — Z-CODE EXTRACTION GATE"),
                ("STEP_4_meat_validation",            "STEP 4 — MEAT VALIDATION"),
                ("STEP_4A_mdm_activation_override",   "STEP 4A — MDM ACTIVATION OVERRIDE"),
                ("STEP_5_icd10_coding",               "STEP 5 — ICD-10 CODING"),
                ("STEP_6_exclusion_logging",          "STEP 6 — EXCLUSION LOGGING"),
                ("STEP_7_self_verification",          "STEP 7 — SELF-VERIFICATION CHECKLIST"),
                ("STEP_8_output_format",              "STEP 8 — OUTPUT FORMAT"),
            ]

            for key, label in ordered_steps:
                step = config.get(key)
                if not step:
                    continue
                parts.append(f"\n{label}:\n{self._flatten_to_text(step)}")

            # Usage instructions (final instruction)
            usage = config.get("usage_instructions", {}).get("how_to_use_this_prompt", "")
            if usage:
                parts.append(f"\nFINAL INSTRUCTION:\n{usage}")

            prompt = "\n\n".join(parts)
            self.logger.info(f"[prompt.json STEP_x] Built prompt: {len(prompt)} chars")
            return prompt

        # ── Build prompt from OLD structure (user_prompt.phase_x keys) ────────
        if user_prompt:
            parts = []
            if "core_philosophy" in user_prompt:
                parts.append(f"CORE PHILOSOPHY:\n{user_prompt['core_philosophy']}")
            anti_hall = user_prompt.get("ANTI_HALLUCINATION_RULES")
            if anti_hall:
                parts.append("\nANTI-HALLUCINATION RULES:")
                anti_hall_text = self._flatten_to_text(anti_hall)
                anti_hall_text = anti_hall_text.replace("{{NER_CANDIDATES}}", ner_hint)
                parts.append(anti_hall_text)
            parts.append(doc_block)
            phase_keys = [
                ("phase_1_section_mapping",  "PHASE 1 — SECTION MAPPING"),
                ("phase_2_disease_detection", "PHASE 2 — DISEASE DETECTION"),
                ("phase_3_meat_validation",  "PHASE 3 — MEAT VALIDATION"),
                ("phase_4_icd10_coding",     "PHASE 4 — ICD-10 CODING"),
                ("phase_5_exclusion_logging", "PHASE 5 — EXCLUSION LOGGING"),
                ("phase_6_self_verification", "PHASE 6 — SELF-VERIFICATION"),
                ("phase_7_output_format",    "PHASE 7 — OUTPUT FORMAT"),
            ]
            if dietary_analysis:
                phase_keys.append(("phase_8_dietary_and_meat_based_analysis", "PHASE 8 — DIETARY"))
            for key, label in phase_keys:
                phase = user_prompt.get(key)
                if not phase:
                    continue
                parts.append(f"\n{label}:\n{self._flatten_to_text(phase)}")
            if "final_instruction" in user_prompt:
                parts.append(f"\n{user_prompt['final_instruction']}")
            prompt = "\n\n".join(parts)
            self.logger.info(f"[prompt.json phase_x] Built prompt: {len(prompt)} chars")
            return prompt

        # ── Last-resort inline fallback (kept minimal — triggers alert) ───────
        self.logger.error(
            "CRITICAL: prompt.json not loaded AND no structure found. "
            "Using minimal inline fallback. FIX prompt.json immediately."
        )
        return (
            "You are a senior clinical coding specialist. Extract every clinically "
            "active, billable disease from the clinical document below. For each "
            "disease: validate MEAT (Monitor, Evaluate, Assess, Treat) with verbatim "
            "evidence, assign the most specific ICD-10-CM code, and log excluded "
            "conditions.\n\n"
            "CRITICAL RULES:\n"
            "1. SLASH SPLIT: If Assessment has 'Condition A/Condition B', extract EACH "
            "as a SEPARATE disease entry with its own ICD code. "
            "Example: 'Atrial fibrillation/flutter' → entry 1: I48.91 (Afib), "
            "entry 2: I48.92 (Flutter). NEVER group slash-separated diagnoses.\n"
            "2. DEVICE STATUS Z-CODES: If a device (e.g. Foley catheter, tracheostomy, "
            "pacemaker, ostomy) is in the Assessment, extract a status Z-code: "
            "Foley/urethral catheter → Z97.8, Tracheostomy → Z93.0, "
            "Gastrostomy → Z93.1, Colostomy → Z93.3, Ileostomy → Z93.2.\n"
            "3. FAMILY HISTORY Z-CODES: Extract Z80.x/Z82.x/Z83.x when driving clinical decisions.\n"
            "4. BMI CODES: Extract Z68.x alongside E66.x (Overweight/Obesity).\n"
            "5. SURVEILLANCE: Active Problem List items with a future follow-up plan are ACTIVE.\n"
            "6. MDM RULE: Condition mentioned in Assessment = ACTIVE, even if treatment declined.\n"
            "7. THROMBOCYTOSIS: Prefer D75.839 (Thrombocytosis, other) over generic D75.x.\n"
            "8. INCONTINENCE + FOLEY: Code BOTH R32 (incontinence) AND Z97.8 (Foley status) "
            "when both are in Assessment. They are separate codeable conditions.\n\n"
            f"{doc_block}\n\n"
            "Return ONLY valid JSON: sections_found, summary, diseases (disease_name, "
            "active_status, meat_grade, meat_score, source_section, all_sections_mentioned, "
            "context_snippets, meat_validation[M_monitor,E_evaluate,A_assess,T_treat], "
            "icd10_code, icd10_description, icd10_selection_reasoning, hcc_status), "
            "excluded_conditions."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _canonicalize_section(name: str) -> str:
        """Convert a section name to a canonical key."""
        if not name:
            return "unknown"
        canonical = re.sub(r"[^a-z0-9]+", "_", name.lower().strip())
        canonical = canonical.strip("_")

        # Map common variations
        mapping = {
            "assessment_and_plan": "assessment_and_plan",
            "assessment_plan": "assessment_and_plan",
            "a_p": "assessment_and_plan",
            "assessment": "assessment",
            "plan": "plan",
            "past_medical_history": "past_medical_history",
            "pmh": "past_medical_history",
            "medical_history": "past_medical_history",
            "history_of_present_illness": "history_present_illness",
            "hpi": "history_present_illness",
            "chief_complaint": "chief_complaint",
            "cc": "chief_complaint",
            "medications": "medications",
            "current_medications": "medications",
            "active_medications": "medications",
            "review_of_systems": "review_of_systems",
            "ros": "review_of_systems",
            "physical_exam": "physical_exam",
            "physical_examination": "physical_exam",
            "pe": "physical_exam",
            "vitals": "vitals",
            "vital_signs": "vitals",
            "social_history": "social_history",
            "family_history": "family_history",
            "allergies": "allergies",
            "immunizations": "immunizations",
            "imaging": "imaging",
            "lab_results": "lab_results",
            "problem_list": "problem_list",
            "active_problems": "active_problems",
            "general_clinical_narrative": "general_clinical_narrative",
        }
        return mapping.get(canonical, canonical)


# Singleton
clinical_document_analyzer = ClinicalDocumentAnalyzer()
