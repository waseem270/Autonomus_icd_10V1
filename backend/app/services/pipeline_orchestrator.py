"""
Pipeline Orchestrator — single entry point for the full clinical coding pipeline.

Coordinates:
    1. Text extraction
    2. Text structuring (section detection + disease extraction)
    3. MEAT validation
    4. ICD-10 mapping
    5. Output filtering & confidence scoring
    6. Database persistence

Replaces the duplicated business logic previously spread across route handlers.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ..core.config import settings
from ..models.document import Document
from ..models.disease import DetectedDisease
from ..models.enums import DocumentStatus, MappingMethod, MappingStatus
from ..models.meat import MEATValidation
from ..models.mapping import ICDMapping
from ..services.text_extraction import text_extraction_service
from ..services.text_structuring import text_structuring_service
from ..services.meat_processor import meat_processor
from ..services.icd_mapper import icd_mapper
from ..services.clinical_document_analyzer import clinical_document_analyzer
from ..services.output_filter import output_filter
from ..services.confidence_scorer import confidence_scorer
from ..services.deterministic_validator import deterministic_validator
from ..services.medical_coding_rules import medical_coding_rules
from ..utils.text_preprocessor import segment_sentences, normalize_whitespace, fix_line_breaks
from ..utils.abbreviation_expander import expand_abbreviations

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrate the complete clinical document analysis pipeline."""

    # ──────────────────────────────────────────────────────────────────
    # Text Extraction
    # ──────────────────────────────────────────────────────────────────
    async def extract_text(self, document: Document, db: Session) -> Dict[str, Any]:
        """Extract text from uploaded PDF and persist to DB."""
        start = time.time()
        document.status = DocumentStatus.PROCESSING
        db.commit()

        try:
            result = await text_extraction_service.extract_text(document.file_path)
            document.raw_text = result["raw_text"]
            document.page_count = result["page_count"]
            document.processed = True
            document.status = DocumentStatus.COMPLETED
            db.commit()
            return {"document_id": document.id, "processing_time": time.time() - start, **result}
        except Exception:
            document.status = DocumentStatus.FAILED
            db.commit()
            raise

    # ──────────────────────────────────────────────────────────────────
    # Structure (section detection + disease extraction)
    # ──────────────────────────────────────────────────────────────────
    async def structure_text(self, document: Document, db: Session) -> Dict[str, Any]:
        """Run the text-structuring pipeline and persist detected diseases."""
        structure_result = await text_structuring_service.structure_text(document.raw_text)

        # Clear & persist diseases
        db.query(DetectedDisease).filter(DetectedDisease.document_id == document.id).delete()
        for dd in structure_result["detected_diseases"]:
            db.add(DetectedDisease(
                document_id=document.id,
                disease_name=dd["disease_name"],
                normalized_name=dd["normalized_name"],
                confidence_score=dd["confidence_score"],
                negated=dd["negated"],
                section=dd.get("section"),
                sentence_number=dd.get("sentence_number"),
            ))
        document.status = DocumentStatus.COMPLETED
        db.commit()
        return structure_result

    # ──────────────────────────────────────────────────────────────────
    # MEAT Validation
    # ──────────────────────────────────────────────────────────────────
    async def validate_meat(
        self, document: Document, db: Session
    ) -> Dict[str, Any]:
        """Run MEAT validation on detected diseases and persist results."""
        diseases = db.query(DetectedDisease).filter(
            DetectedDisease.document_id == document.id
        ).all()
        if not diseases:
            raise ValueError("No diseases detected. Run structuring first.")

        disease_dicts = [
            {
                "disease_name": d.disease_name,
                "normalized_name": d.normalized_name,
                "confidence_score": d.confidence_score,
                "negated": d.negated,
                "section": d.section,
                "section_sources": [d.section] if d.section else ["unknown"],
                "sentence_number": d.sentence_number,
            }
            for d in diseases
        ]

        preprocessed = self._preprocess_for_meat(document.raw_text)
        fast_sections = self._build_fast_sections(diseases, preprocessed["sentences"])

        meat_result = await meat_processor.process_meat_validation(
            detected_diseases=disease_dicts,
            full_text=preprocessed["text"],
            sentences=preprocessed["sentences"],
            sections=fast_sections,
        )

        self._persist_meat_results(diseases, meat_result, db)
        db.commit()
        return meat_result

    # ──────────────────────────────────────────────────────────────────
    # ICD Mapping
    # ──────────────────────────────────────────────────────────────────
    async def map_icd(self, document: Document, db: Session) -> Dict[str, Any]:
        """Map validated diseases to ICD-10 codes and persist results."""
        diseases = db.query(DetectedDisease).filter(
            DetectedDisease.document_id == document.id,
            DetectedDisease.negated == False,  # noqa: E712
        ).all()
        if not diseases:
            raise ValueError("No diseases to map. Run structuring first.")

        diseases_with_meat = self._load_diseases_with_meat(diseases, db)
        mapping_result = await icd_mapper.map_multiple_diseases(diseases_with_meat)

        # Deterministic ICD validation — strip codes not in the ICD dictionary
        for mapping in mapping_result["mappings"]:
            if mapping.get("icd_code"):
                validation = deterministic_validator.validate_icd_code(mapping["icd_code"])
                if not validation["valid"]:
                    logger.warning(
                        "ICD code rejected by deterministic validator",
                        extra={
                            "stage": "icd_validation",
                            "disease": mapping["disease"],
                            "rejected_code": mapping["icd_code"],
                            "reason": validation["reason"],
                        },
                    )
                    mapping["icd_code"] = None
                    mapping["status"] = "not_found"
                    mapping["confidence"] = 0.0

        self._persist_icd_mappings(diseases, diseases_with_meat, mapping_result, db)
        db.commit()
        return mapping_result

    # ──────────────────────────────────────────────────────────────────
    # Full Pipeline (process-all)
    # ──────────────────────────────────────────────────────────────────
    async def process_all(
        self,
        document: Document,
        db: Session,
        dietary_analysis: bool = False,
    ) -> Dict[str, Any]:
        """
        Run end-to-end: dual-agent pipeline → fallback single-pass → fallback multi-step.
        """
        start = time.time()
        
        from ..utils.token_tracker import token_tracker
        token_tracker.reset()

        # ── Try dual-agent pipeline first (highest accuracy) ────────
        dual_result = await self._try_dual_agent(document)
        if dual_result is not None:
            all_diseases = dual_result["diseases"]

            # Simplified flow: pass extracted diseases directly to output_filter
            # NOTE: medical_coding_rules already ran inside Agent 1's precision
            # pipeline; re-applying would double-add coverage-verifier codes.
            filtered = output_filter.apply(all_diseases, single_pass=True)

            scored = confidence_scorer.score_batch(filtered)
            rules_result = {"results": scored, "audit_log": [], "stats": {"total_output": len(scored)}}

            logger.info(
                "Dual-agent pipeline complete",
                extra={
                    "stage": "process_all",
                    "source": "dual_agent_pipeline",
                    "document_id": document.id,
                    "agent1_count": dual_result["agent1_count"],
                    "agent2_removed": dual_result.get("agent2_removed", 0),
                    "agent2_fixed": dual_result.get("agent2_fixed", 0),
                    "agent2_added": dual_result.get("agent2_added", 0),
                    "before_filter": len(all_diseases),
                    "after_filter": len(scored),
                    "processing_time": round(time.time() - start, 2),
                },
            )
            return self._build_response(
                document, scored, {}, start,
                source="dual_agent_pipeline",
                all_results=all_diseases,
                rules_audit=rules_result.get("audit_log"),
                rules_stats=rules_result.get("stats"),
            )

        # ── Fallback: single-pass LLM ────────────────────────────────
        single_pass = await self._try_single_pass(document, db, dietary_analysis)
        if single_pass is not None:
            filtered = output_filter.apply(single_pass["unified_results"], single_pass=True)
            rules_result = medical_coding_rules.apply(
                filtered, raw_text=document.raw_text or ""
            )
            scored = confidence_scorer.score_batch(rules_result["results"])
            logger.info(
                "Pipeline complete",
                extra={
                    "stage": "process_all", "source": "single_pass_llm",
                    "document_id": document.id,
                    "detected": len(single_pass["unified_results"]),
                    "after_rules": rules_result["stats"]["total_output"],
                    "filtered": len(scored),
                    "processing_time": round(time.time() - start, 2),
                },
            )
            return self._build_response(
                document, scored, single_pass, start,
                source="single_pass_llm",
                all_results=single_pass["unified_results"],
                rules_audit=rules_result["audit_log"],
                rules_stats=rules_result["stats"],
            )

        # ── Fallback: multi-step pipeline ───────────────────────────
        logger.info("Single-pass returned no results; falling back to multi-step pipeline.")
        structure_result = await self.structure_text(document, db)
        meat_result = await self.validate_meat(document, db)
        mapping_result = await self.map_icd(document, db)

        unified = self._build_unified_table(meat_result, mapping_result)
        filtered = output_filter.apply(unified)
        rules_result = medical_coding_rules.apply(
            filtered, raw_text=document.raw_text or ""
        )
        scored = confidence_scorer.score_batch(rules_result["results"])

        logger.info(
            "Pipeline complete",
            extra={
                "stage": "process_all", "source": "multi_step_pipeline",
                "document_id": document.id,
                "detected": len(unified),
                "after_rules": rules_result["stats"]["total_output"],
                "filtered": len(scored),
                "processing_time": round(time.time() - start, 2),
            },
        )

        return self._build_response(
            document, scored, {"structure": structure_result}, start,
            source="multi_step_pipeline",
            all_results=unified,
            rules_audit=rules_result["audit_log"],
            rules_stats=rules_result["stats"],
        )

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────
    async def _try_dual_agent(self, document: Document) -> Optional[Dict[str, Any]]:
        """Run dual-agent pipeline; return None on failure."""
        try:
            from .dual_agent_pipeline import dual_agent_pipeline as _dual
            result = await _dual.process_document(
                pdf_path=document.file_path,
                raw_text=document.raw_text,
            )
            if not result or not result.get("diseases"):
                return None

            # Normalise Agent 2 items — add boolean MEAT fields & proper meat_tier
            _MEAT_TIER = {4: "strong_evidence", 3: "strong_evidence",
                          2: "moderate_evidence", 1: "weak_evidence", 0: "no_meat"}
            for r in result["diseases"]:
                if r.get("agent2_added"):
                    r.setdefault("monitoring", bool(r.get("monitoring_evidence", "").strip()))
                    r.setdefault("evaluation", bool(r.get("evaluation_evidence", "").strip()))
                    r.setdefault("assessment", bool(r.get("assessment_evidence", "").strip()))
                    r.setdefault("treatment",  bool(r.get("treatment_evidence", "").strip()))
                    r["meat_tier"] = _MEAT_TIER.get(r.get("meat_score", 0), "no_meat")

            return result
        except Exception as e:
            logger.warning(f"Dual-agent pipeline failed: {e}")
            return None

    async def _try_single_pass(
        self, document: Document, db: Session, dietary_analysis: bool
    ) -> Optional[Dict[str, Any]]:
        """Attempt single-pass LLM; return None on failure."""
        try:
            sp = await clinical_document_analyzer.analyze_document(
                document.raw_text, dietary_analysis=dietary_analysis,
            )
        except Exception as e:
            logger.warning(f"Single-pass LLM failed: {e}")
            return None

        if not sp or not sp.get("diseases"):
            return None

        logger.info(f"Single-pass: {len(sp['diseases'])} diseases found.")
        unified = clinical_document_analyzer.convert_to_unified_results(sp)
        pipeline_data = clinical_document_analyzer.convert_to_pipeline_format(sp)

        # Persist to DB
        self._persist_single_pass(document, pipeline_data, db)

        return {"unified_results": unified, "single_pass": sp, "pipeline_data": pipeline_data}

    def _persist_single_pass(
        self, document: Document, pipeline_data: Dict, db: Session
    ) -> None:
        """Store single-pass diseases, MEAT, and ICD mappings."""
        doc_id = document.id

        db.query(DetectedDisease).filter(DetectedDisease.document_id == doc_id).delete()
        for dd in pipeline_data["detected_diseases"]:
            db.add(DetectedDisease(
                document_id=doc_id,
                disease_name=dd["disease_name"],
                normalized_name=dd["normalized_name"],
                confidence_score=dd["confidence_score"],
                negated=dd["negated"],
                section=dd.get("section"),
                sentence_number=dd.get("sentence_number", 0),
            ))
        document.status = DocumentStatus.COMPLETED
        db.commit()

        diseases_db = db.query(DetectedDisease).filter(
            DetectedDisease.document_id == doc_id
        ).all()
        disease_ids = [d.id for d in diseases_db]

        # MEAT
        db.query(MEATValidation).filter(
            MEATValidation.disease_id.in_(disease_ids)
        ).delete(synchronize_session=False)

        seen = set()
        for dd in pipeline_data["detected_diseases"]:
            rec = next((d for d in diseases_db if d.disease_name == dd["disease_name"]), None)
            if not rec or rec.id in seen or not dd.get("meat_validation"):
                continue
            seen.add(rec.id)
            mv = dd["meat_validation"]
            _has_m = bool(mv.get("M_monitor"))
            _has_e = bool(mv.get("E_evaluate"))
            _has_a = bool(mv.get("A_assess"))
            _has_t = bool(mv.get("T_treat"))
            _evidence_count = sum([_has_m, _has_e, _has_a, _has_t])
            db.add(MEATValidation(
                disease_id=rec.id,
                monitoring=_has_m,
                monitoring_evidence=mv.get("M_monitor") or "",
                monitoring_confidence=0.85 if _has_m else 0.0,
                evaluation=_has_e,
                evaluation_evidence=mv.get("E_evaluate") or "",
                evaluation_confidence=0.85 if _has_e else 0.0,
                assessment=_has_a,
                assessment_evidence=mv.get("A_assess") or "",
                assessment_confidence=0.85 if _has_a else 0.0,
                treatment=_has_t,
                treatment_evidence=mv.get("T_treat") or "",
                treatment_confidence=0.85 if _has_t else 0.0,
                meat_valid=_evidence_count >= 2,
                overall_confidence=round(_evidence_count / 4, 2) if _evidence_count > 0 else 0.0,
                llm_reasoning=mv.get("meat_grade_reasoning", ""),
            ))

        # ICD
        db.query(ICDMapping).filter(
            ICDMapping.disease_id.in_(disease_ids)
        ).delete(synchronize_session=False)

        for dd in pipeline_data["detected_diseases"]:
            rec = next((d for d in diseases_db if d.disease_name == dd["disease_name"]), None)
            if rec and dd.get("icd_code"):
                # Validate ICD code deterministically
                validation = deterministic_validator.validate_icd_code(dd["icd_code"])
                if validation["valid"]:
                    db.add(ICDMapping(
                        disease_id=rec.id,
                        icd_code=dd["icd_code"],
                        icd_description=dd.get("icd_description", ""),
                        mapping_method=MappingMethod.LLM_RANKED,
                        confidence_score=0.95,
                        status=MappingStatus.AUTO_ASSIGNED,
                        llm_reasoning=dd.get("icd_selection_reasoning", ""),
                    ))
                else:
                    logger.warning(
                        f"ICD {dd['icd_code']} rejected by deterministic validator "
                        f"for '{dd['disease_name']}'"
                    )
        db.commit()

    def _persist_meat_results(
        self,
        diseases: List[DetectedDisease],
        meat_result: Dict[str, Any],
        db: Session,
    ) -> None:
        """Store MEAT validation results, deduplicating by disease_id."""
        disease_ids = [d.id for d in diseases]
        db.query(MEATValidation).filter(
            MEATValidation.disease_id.in_(disease_ids)
        ).delete(synchronize_session=False)

        seen = set()
        for result in meat_result["meat_results"]:
            rec = next((d for d in diseases if d.disease_name == result["disease"]), None)
            if not rec or rec.id in seen:
                continue
            seen.add(rec.id)
            vdata = result["validation"]["final_meat"]
            db.add(MEATValidation(
                disease_id=rec.id,
                monitoring=vdata.get("monitoring", False),
                monitoring_evidence=vdata.get("monitoring_evidence", ""),
                monitoring_confidence=vdata.get("monitoring_confidence", 0.0),
                evaluation=vdata.get("evaluation", False),
                evaluation_evidence=vdata.get("evaluation_evidence", ""),
                evaluation_confidence=vdata.get("evaluation_confidence", 0.0),
                assessment=vdata.get("assessment", False),
                assessment_evidence=vdata.get("assessment_evidence", ""),
                assessment_confidence=vdata.get("assessment_confidence", 0.0),
                treatment=vdata.get("treatment", False),
                treatment_evidence=vdata.get("treatment_evidence", ""),
                treatment_confidence=vdata.get("treatment_confidence", 0.0),
                meat_valid=result["validation"]["meat_valid"],
                overall_confidence=vdata.get("overall_confidence", 0.0),
                llm_reasoning=result.get("meat_data", {}).get("llm_reasoning", ""),
            ))

    def _persist_icd_mappings(
        self,
        diseases: List[DetectedDisease],
        diseases_with_meat: List[Dict],
        mapping_result: Dict[str, Any],
        db: Session,
    ) -> None:
        disease_ids = [d.id for d in diseases]
        db.query(ICDMapping).filter(
            ICDMapping.disease_id.in_(disease_ids)
        ).delete(synchronize_session=False)

        for i, mapping in enumerate(mapping_result["mappings"]):
            if i >= len(diseases_with_meat):
                break
            did = diseases_with_meat[i]["disease_id"]
            if mapping.get("icd_code"):
                db.add(ICDMapping(
                    disease_id=did,
                    icd_code=mapping["icd_code"],
                    icd_description=mapping["icd_description"],
                    mapping_method=mapping["mapping_method"],
                    confidence_score=mapping["confidence"],
                    status=mapping["status"],
                    llm_reasoning=mapping.get("llm_reasoning", ""),
                ))

    def _load_diseases_with_meat(
        self, diseases: List[DetectedDisease], db: Session
    ) -> List[Dict[str, Any]]:
        result = []
        for d in diseases:
            meat = db.query(MEATValidation).filter(MEATValidation.disease_id == d.id).first()
            meat_data = {}
            if meat:
                meat_data = {
                    "meat_valid": meat.meat_valid,
                    "assessment_evidence": meat.assessment_evidence,
                    "treatment_evidence": meat.treatment_evidence,
                    "evaluation_evidence": meat.evaluation_evidence,
                    "monitoring_evidence": meat.monitoring_evidence,
                    "overall_confidence": meat.overall_confidence,
                }
            result.append({
                "disease_id": d.id,
                "disease_name": d.disease_name,
                "normalized_name": d.normalized_name,
                "meat_validation": meat_data,
            })
        return result

    def _preprocess_for_meat(self, raw_text: str) -> Dict[str, Any]:
        text = fix_line_breaks(normalize_whitespace(raw_text))
        text = expand_abbreviations(text, context_aware=True)
        sentences = segment_sentences(text)
        return {"text": text, "sentences": sentences}

    def _build_fast_sections(
        self, diseases: List[DetectedDisease], sentences: List[Dict]
    ) -> Dict[str, Dict]:
        sections: Dict[str, Dict] = {}
        for d in diseases:
            sec = d.section or "unknown"
            if sec not in sections:
                sections[sec] = {"text": "", "section_name": sec}
            dn_lower = d.disease_name.lower()
            for sent in sentences:
                if dn_lower in sent.get("text", "").lower():
                    sections[sec]["text"] += " " + sent["text"]
        return sections

    def _build_unified_table(
        self, meat_result: Dict, mapping_result: Dict
    ) -> List[Dict[str, Any]]:
        unified = []
        for idx, mr in enumerate(meat_result.get("meat_results", [])):
            disease_name = mr["disease"]
            icd_info = next(
                (m for m in mapping_result.get("mappings", []) if m["disease"] == disease_name),
                None,
            )
            seg_sources = mr.get("segment_source", [])
            final_meat = mr.get("validation", {}).get("final_meat", mr.get("meat_data", {}))
            tier = mr.get("meat_tier", "invalid")

            _mon = bool(final_meat.get("monitoring", False))
            _eva = bool(final_meat.get("evaluation", False))
            _ass = bool(final_meat.get("assessment", False))
            _tre = bool(final_meat.get("treatment", False))

            unified.append({
                "number": idx + 1,
                "disease": disease_name,
                "icd_code": (icd_info["icd_code"] if icd_info and icd_info.get("icd_code") else "—"),
                "icd_description": (icd_info["icd_description"] if icd_info and icd_info.get("icd_description") else ""),
                "segment": self._format_segment_label(seg_sources),
                "segment_source_raw": seg_sources,
                "disease_status": mr.get("disease_status", "Chronic"),
                "monitoring": _mon,
                "evaluation": _eva,
                "assessment": _ass,
                "treatment": _tre,
                "meat_level": {
                    "strong_evidence": "Strong Evidence", "moderate_evidence": "Moderate Evidence",
                    "weak_evidence": "Weak Evidence", "no_meat": "No MEAT",
                }.get(tier, "No MEAT"),
                "meat_tier": tier,
                "confidence": 0.0,  # will be scored by confidence_scorer
                "icd_confidence": icd_info["confidence"] if icd_info else 0.0,
                "icd_method": icd_info["mapping_method"] if icd_info else "none",
                "monitoring_evidence": final_meat.get("monitoring_evidence", ""),
                "evaluation_evidence": final_meat.get("evaluation_evidence", ""),
                "assessment_evidence": final_meat.get("assessment_evidence", ""),
                "treatment_evidence": final_meat.get("treatment_evidence", ""),
                "llm_reasoning": mr.get("meat_data", {}).get("llm_reasoning", ""),
            })
        return unified

    @staticmethod
    def _format_segment_label(seg_sources: list) -> str:
        if not seg_sources:
            return "Unknown"
        _ASSESSMENT_DISPLAY = {
            "assessment_and_plan", "assessment",
            "active_problems", "active_problem_list", "problem_list",
        }
        sources_lower = {s.lower() for s in seg_sources}
        if sources_lower & _ASSESSMENT_DISPLAY:
            for key in ("assessment_and_plan", "assessment", "active_problems",
                        "active_problem_list", "problem_list"):
                if key in sources_lower:
                    return key.replace("_", " ").title()
        return "/".join(s.replace("_", " ").title() for s in seg_sources)

    def _build_response(
        self,
        document: Document,
        filtered: List[Dict],
        extra: Dict,
        start_time: float,
        source: str,
        all_results: List[Dict],
        rules_audit: Optional[List[Dict]] = None,
        rules_stats: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        processing_time = round(time.time() - start_time, 2)
        
        from ..utils.token_tracker import token_tracker
        logger.info(token_tracker.get_summary())
        
        return {
            "document_id": document.id,
            "processing_time": processing_time,
            "total_diseases": len(filtered),
            "total_detected": len(all_results),
            "unified_results": filtered,
            "analysis_source": source,
            "summary": {
                "strong_evidence": sum(1 for r in filtered if r.get("meat_tier") == "strong_evidence"),
                "moderate_evidence": sum(1 for r in filtered if r.get("meat_tier") == "moderate_evidence"),
                "weak_evidence": sum(1 for r in filtered if r.get("meat_tier") == "weak_evidence"),
                "no_meat": sum(1 for r in filtered if r.get("meat_tier") == "no_meat"),
                "icd_mapped": sum(1 for r in filtered if r.get("icd_code", "—") != "—"),
            },
            "excluded_summary": {
                "total_excluded": len(all_results) - len(filtered),
            },
            "medical_coding_rules": {
                "audit_log": rules_audit or [],
                "stats": rules_stats or {},
            },
        }


# Singleton
pipeline_orchestrator = PipelineOrchestrator()
