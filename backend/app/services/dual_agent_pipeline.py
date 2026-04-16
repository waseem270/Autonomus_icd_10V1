"""
Dual-Agent Medical Coding Pipeline
===================================

Agent 1 (Precision Agent):
    Uses the existing full pipeline (analyze_document → output_filter →
    medical_coding_rules → confidence_scorer) to extract high-precision codes.

Agent 2 (Verifier + Recall Agent):
    Receives Agent 1's codes + the full clinical document and performs:
    1. VERIFICATION — checks each Agent 1 code is real, correct, and has MEAT
    2. RECALL — finds additional codes Agent 1 missed

Flow:  PDF → text → Agent 1 (precision extract) → Agent 2 (verify + recall) → apply verdicts → final
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..core.config import settings, get_llm_provider, get_active_model
from ..utils.gemini_retry import call_gemini_safe as call_gemini
from .clinical_document_analyzer import ClinicalDocumentAnalyzer
from .output_filter import OutputFilter, output_filter
from .medical_coding_rules import MedicalCodingRules, medical_coding_rules
from .confidence_scorer import ConfidenceScorer, confidence_scorer

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Agent 2 — Recall Sweep Prompt
# ═══════════════════════════════════════════════════════════════════════════

_AGENT2_SYSTEM = """You are a senior CCS performing VERIFICATION AND RECALL on first-pass ICD-10 coding.

PART 1 — VERIFY each code: (a) condition actually in document? (b) ICD-10 correct and maximally specific? (c) real MEAT evidence? (d) actively managed THIS encounter? Mark: KEEP, FIX (provide corrected code), or REMOVE.

PART 2 — RECALL missed codes: Check for Assessment/Plan items not captured, symptom codes with workup, active chronic conditions with medication management, Z-codes (BMI Z68.x, cancer history Z85.x, device status Z93/Z95/Z97).

RULES:
1. SAME-FAMILY: Patient CAN have multiple codes from same ICD family (F11.20+F11.288, E11.9+E11.649, I48.91+I48.92). NEVER skip because another same-chapter code exists.
2. SLASH-SPLIT: 'fibrillation/flutter' = TWO codes (I48.91 + I48.92).
3. SPECIFICITY: Always use the most specific code documented.
4. Every code needs MEAT evidence — quote EXACT text.
5. Do NOT add conditions only in PMH without active management.
6. Code what Assessment SAYS — do not infer from other sections."""

_AGENT2_USER_TEMPLATE = """CLINICAL DOCUMENT:
{doc_separator}
{document_text}
{doc_separator}

CODES FROM FIRST-PASS CODER (verify each one):
{existing_codes_block}

TASK:
1. VERIFY each code above — is it correct, wrong ICD, or should be removed?
2. FIND any ADDITIONAL codes that were missed.

Return ONLY valid JSON in this exact format:
{{
  "verified_codes": [
    {{
      "icd_code": "original ICD code",
      "disease_name": "original disease name",
      "verdict": "KEEP or FIX or REMOVE",
      "corrected_icd_code": "new code if FIX, otherwise null",
      "corrected_disease_name": "corrected name if FIX, otherwise null",
      "reason": "brief explanation for FIX or REMOVE verdicts",
      "meat_evidence": {{
        "M_monitor": "verbatim quote or empty string",
        "E_evaluate": "verbatim quote or empty string",
        "A_assess": "verbatim quote or empty string",
        "T_treat": "verbatim quote or empty string"
      }}
    }}
  ],
  "additional_codes": [
    {{
      "icd_code": "X00.0",
      "disease_name": "Condition name",
      "reason_missed": "Brief explanation",
      "meat_evidence": {{
        "M_monitor": "verbatim quote or empty string",
        "E_evaluate": "verbatim quote or empty string",
        "A_assess": "verbatim quote or empty string",
        "T_treat": "verbatim quote or empty string"
      }},
      "source_section": "assessment_and_plan or other section name"
    }}
  ]
}}"""


class DualAgentPipeline:
    """Two-agent pipeline: Agent 1 (precision) + Agent 2 (recall sweep)."""

    def __init__(self):
        self.analyzer = ClinicalDocumentAnalyzer()
        self.logger = logging.getLogger(self.__class__.__name__)

    async def process_document(
        self,
        pdf_path: str,
        raw_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full 2-agent pipeline on a clinical document."""
        # ── Step 1: Text extraction ─────────────────────────────────────
        if raw_text is None:
            from .text_extraction import TextExtractionService
            extractor = TextExtractionService()
            extraction = await extractor.extract_text(pdf_path)
            raw_text = extraction.get("raw_text", "") or extraction.get("text", "")
            if not raw_text or len(raw_text.strip()) < 50:
                return {"diseases": [], "agent1_count": 0, "agent2_removed": 0,
                        "agent2_fixed": 0, "agent2_added": 0,
                        "summary": "Document too short or unreadable"}

        # ── Step 2: Agent 1 — Full precision pipeline ───────────────────
        self.logger.info("=== AGENT 1: Precision pipeline ===")
        agent1_diseases = await self._agent1_precision(raw_text)
        agent1_codes = {d.get("icd_code", "") for d in agent1_diseases if d.get("icd_code")}
        self.logger.info(f"Agent 1 produced {len(agent1_codes)} codes: {sorted(agent1_codes)}")

        # ── Step 3: Agent 2 — Verify + Recall ───────────────────────────
        self.logger.info("=== AGENT 2: Verify + Recall ===")
        verified, new_codes = await self._agent2_verify_and_recall(raw_text, agent1_diseases)
        removed_count = sum(1 for v in verified if v.get("verdict") == "REMOVE")
        fixed_count = sum(1 for v in verified if v.get("verdict") == "FIX")
        self.logger.info(
            f"Agent 2: {removed_count} removed, {fixed_count} fixed, "
            f"{len(new_codes)} additional codes"
        )

        # ── Step 4: Apply verdicts + Merge ──────────────────────────────
        merged = self._apply_verdicts_and_merge(agent1_diseases, verified, new_codes)

        # ── Step 4b: Z68 BMI code enforcement ───────────────────────────
        # Z68.x supplementary codes must be paired with an E66.x obesity code.
        # • No E66.x present → remove spurious Z68.x (LLM hallucination from vitals).
        # • E66.x present but no Z68.x → re-add the BMI supplement code.
        _merge_icds = {(d.get("icd_code") or "").upper() for d in merged}
        _has_e66 = any(ic.startswith("E66") for ic in _merge_icds)
        _has_z68 = any(ic.startswith("Z68") for ic in _merge_icds)
        if _has_z68 and not _has_e66:
            _before = len(merged)
            merged = [
                d for d in merged
                if not (d.get("icd_code") or "").upper().startswith("Z68")
            ]
            self.logger.info(
                f"Z68 enforcement: removed {_before - len(merged)} "
                f"spurious BMI code(s) (no E66.x present)"
            )
        elif _has_e66 and not _has_z68:
            _bmi_m = output_filter._BMI_VALUE_RE.search(raw_text)
            if _bmi_m:
                _bmi_val = float(_bmi_m.group(1))
                _z68 = output_filter._bmi_to_icd_code(_bmi_val)
                if _z68:
                    merged.append({
                        "number": len(merged) + 1,
                        "disease": f"BMI {_bmi_val}",
                        "icd_code": _z68,
                        "segment": "assessment_and_plan",
                        "segment_source_raw": ["assessment_and_plan"],
                        "monitoring_evidence": "",
                        "evaluation_evidence": _bmi_m.group(0),
                        "assessment_evidence": f"BMI {_bmi_val}",
                        "treatment_evidence": "",
                        "meat_score": 1,
                        "meat_tier": "bmi_supplement",
                    })
                    self.logger.info(
                        f"Z68 enforcement: re-added {_z68} "
                        f"(BMI {_bmi_val}, E66.x paired)"
                    )

        # ── Step 5: Confidence scoring ──────────────────────────────────
        merged = confidence_scorer.score_batch(merged)

        return {
            "diseases": merged,
            "agent1_count": len(agent1_codes),
            "agent2_removed": removed_count,
            "agent2_fixed": fixed_count,
            "agent2_added": len(new_codes),
            "summary": (
                f"Agent 1: {len(agent1_codes)} codes → "
                f"Agent 2: -{removed_count} removed, ~{fixed_count} fixed, "
                f"+{len(new_codes)} added → Total: {len(merged)}"
            ),
        }

    # ═══════════════════════════════════════════════════════════════════
    # Agent 1 — Full precision pipeline (existing proven pathway)
    # ═══════════════════════════════════════════════════════════════════

    async def _agent1_precision(self, raw_text: str) -> List[Dict]:
        """Run the existing full pipeline: analyze → filter → rules → score."""
        result = await self.analyzer.analyze_document(raw_text)

        if not result or not result.get("diseases"):
            self.logger.warning("Agent 1: LLM returned no diseases")
            return []

        unified = self.analyzer.convert_to_unified_results(result)
        filtered = output_filter.apply(unified, single_pass=True)
        rules_out = medical_coding_rules.apply(filtered, raw_text=raw_text)
        scored = confidence_scorer.score_batch(rules_out["results"])
        return scored

    # ═══════════════════════════════════════════════════════════════════
    # Agent 2 — Verify existing codes + Recall missed codes
    # ═══════════════════════════════════════════════════════════════════

    async def _agent2_verify_and_recall(
        self,
        raw_text: str,
        agent1_diseases: List[Dict],
    ) -> tuple:
        """Ask LLM to verify Agent 1's codes and find missed ones.
        
        Returns:
            (verified_codes: List[Dict], additional_codes: List[Dict])
        """
        # Build existing codes block showing each code for verification
        lines = []
        for i, d in enumerate(agent1_diseases, 1):
            icd = d.get("icd_code", "")
            name = d.get("disease", "")
            if icd:
                lines.append(f"  {i}. {icd}: {name}")
        existing_codes_block = "\n".join(lines) if lines else "  (none captured)"

        # Use full document — don't truncate
        sep = "=" * 60

        user_msg = _AGENT2_USER_TEMPLATE.format(
            document_text=raw_text,
            existing_codes_block=existing_codes_block,
            doc_separator=sep,
        )

        # Build config
        from types import SimpleNamespace
        _max_tokens = 4096  # Sufficient for verification JSON
        config = SimpleNamespace(
            temperature=0.2,
            max_output_tokens=_max_tokens,
            top_p=0.95,
            system_instruction=_AGENT2_SYSTEM,
            response_mime_type="application/json",
            response_schema=None,
            safety_settings=[],
        )

        try:
            response = await call_gemini(
                client=self.analyzer.client,
                model=self.analyzer.model_name,
                contents=user_msg,
                config=config,
            )

            # Extract response text
            response_text = ""
            if (response.candidates and
                response.candidates[0].content and
                response.candidates[0].content.parts):
                for part in response.candidates[0].content.parts:
                    if not getattr(part, "thought", False) and part.text:
                        response_text += part.text

            return self._parse_verify_response(response_text, raw_text)

        except Exception as e:
            self.logger.error(f"Agent 2 verify+recall failed: {e}")
            # Safe fallback — keep all Agent 1 results unchanged
            return ([], [])

    def _parse_verify_response(
        self,
        response_text: str,
        raw_text: str,
    ) -> tuple:
        """Parse Agent 2's verification + recall response.
        
        Returns:
            (verified_codes: List[Dict], additional_codes: List[Dict])
        """
        try:
            text = response_text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)

            data = json.loads(text)

            # Parse verified codes
            verified = []
            for item in data.get("verified_codes", []):
                if not isinstance(item, dict):
                    continue
                verdict = (item.get("verdict") or "KEEP").upper()
                if verdict not in ("KEEP", "FIX", "REMOVE"):
                    verdict = "KEEP"
                
                entry = {
                    "icd_code": (item.get("icd_code") or "").strip(),
                    "disease_name": (item.get("disease_name") or "").strip(),
                    "verdict": verdict,
                    "corrected_icd_code": (item.get("corrected_icd_code") or "").strip() or None,
                    "corrected_disease_name": (item.get("corrected_disease_name") or "").strip() or None,
                    "reason": (item.get("reason") or "").strip(),
                    "meat_evidence": item.get("meat_evidence", {}),
                }
                verified.append(entry)
                if verdict != "KEEP":
                    self.logger.info(
                        f"Agent 2 {verdict}: {entry['icd_code']} — {entry['reason']}"
                    )

            # Parse additional codes
            additional = []
            existing_icds = {v["icd_code"].upper() for v in verified if v.get("icd_code")}
            for item in data.get("additional_codes", []):
                if not isinstance(item, dict):
                    continue
                icd = (item.get("icd_code") or "").strip()
                if not icd:
                    continue
                if icd.upper() in existing_icds:
                    self.logger.debug(f"Agent 2 additional {icd} already in verified set — skipping")
                    continue

                name = (item.get("disease_name") or "").strip()
                meat = item.get("meat_evidence", {})
                source = item.get("source_section", "assessment_and_plan")
                reason = item.get("reason_missed", "")

                unified_item = {
                    "number": 0,
                    "disease": name,
                    "icd_code": icd,
                    "segment": source,
                    "segment_source_raw": [source],
                    "monitoring_evidence": (meat.get("M_monitor") or "").strip(),
                    "evaluation_evidence": (meat.get("E_evaluate") or "").strip(),
                    "assessment_evidence": (meat.get("A_assess") or "").strip() or reason,
                    "treatment_evidence": (meat.get("T_treat") or "").strip(),
                    "meat_tier": "agent2_recall",
                    "agent2_added": True,
                    "agent2_reason": reason,
                }
                
                # Ensure Agent 2 codes pass Rules 6 / 9 by guaranteeing a minimum meat_score of 1
                unified_item["meat_score"] = max(1, sum(1 for k in ["monitoring", "evaluation", "assessment", "treatment"]
                                                  if unified_item.get(f"{k}_evidence")))
                additional.append(unified_item)
                existing_icds.add(icd.upper())
                self.logger.info(f"Agent 2 ADD: {icd} — {name} (reason: {reason})")

            return (verified, additional)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.error(f"Agent 2 JSON parse failed: {e}")
            return ([], [])

    # ═══════════════════════════════════════════════════════════════════
    # Apply verdicts + Merge: Agent 1 results + Agent 2 new codes
    # ═══════════════════════════════════════════════════════════════════

    def _apply_verdicts_and_merge(
        self,
        agent1_diseases: List[Dict],
        verified: List[Dict],
        new_codes: List[Dict],
    ) -> List[Dict]:
        """Apply Agent 2's verification verdicts to Agent 1 codes, then add new codes."""
        # Build verdict lookup by ICD code
        verdict_map = {}
        for v in verified:
            icd = (v.get("icd_code") or "").upper()
            if icd:
                verdict_map[icd] = v

        # Apply verdicts to Agent 1 results
        kept = []
        for d in agent1_diseases:
            icd = (d.get("icd_code") or "").upper()
            v = verdict_map.get(icd)
            
            if v is None:
                # Not mentioned by Agent 2 verification — keep as-is
                kept.append(d)
                continue

            verdict = v.get("verdict", "KEEP")
            if verdict == "REMOVE":
                self.logger.info(
                    f"Agent 2 REMOVED: {icd} — {v.get('reason', 'no reason')}"
                )
                continue  # Drop this code
            elif verdict == "FIX":
                new_icd = v.get("corrected_icd_code")
                new_name = v.get("corrected_disease_name")
                if new_icd:
                    self.logger.info(
                        f"Agent 2 FIX: {icd} → {new_icd} — {v.get('reason', '')}"
                    )
                    d["icd_code"] = new_icd
                if new_name:
                    d["disease"] = new_name
                # Update MEAT evidence if Agent 2 provided better evidence
                meat = v.get("meat_evidence", {})
                if meat:
                    for field_key, field_name in [
                        ("M_monitor", "monitoring_evidence"),
                        ("E_evaluate", "evaluation_evidence"),
                        ("A_assess", "assessment_evidence"),
                        ("T_treat", "treatment_evidence"),
                    ]:
                        new_ev = (meat.get(field_key) or "").strip()
                        old_ev = (d.get(field_name) or "").strip()
                        if new_ev and not old_ev:
                            d[field_name] = new_ev
                kept.append(d)
            else:
                # KEEP
                kept.append(d)

        # Add new codes from recall sweep
        existing_icds = {d.get("icd_code", "").upper() for d in kept}
        for nc in new_codes:
            icd = (nc.get("icd_code") or "").upper()
            if icd not in existing_icds:
                kept.append(nc)
                existing_icds.add(icd)

        # Expand any comma-separated ICD codes (e.g., "I48.91, I48.92") into separate entries
        expanded = []
        seen_icds: set = set()
        for d in kept:
            icd = (d.get("icd_code") or "").strip()
            if "," in icd:
                sub_codes = [c.strip() for c in icd.split(",") if c.strip()]
                for sub in sub_codes:
                    if sub.upper() not in seen_icds:
                        entry = dict(d)
                        entry["icd_code"] = sub
                        expanded.append(entry)
                        seen_icds.add(sub.upper())
            elif icd and icd.upper() not in seen_icds:
                expanded.append(d)
                seen_icds.add(icd.upper())
            elif not icd:
                expanded.append(d)
        kept = expanded

        # Renumber
        for i, r in enumerate(kept, 1):
            r["number"] = i

        return kept


# Singleton — used by pipeline_orchestrator
dual_agent_pipeline = DualAgentPipeline()
