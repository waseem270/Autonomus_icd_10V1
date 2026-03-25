import os
import re
import uuid
import shutil
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from ...core.config import settings
from ..dependencies import get_db
from ...models.document import Document
from ...models.disease import DetectedDisease
from ...models.enums import DocumentStatus, MappingMethod, MappingStatus
from ...models.meat import MEATValidation
from ...models.mapping import ICDMapping
from ...schemas.document import (
    DocumentUploadResponse,
    TextExtractionResponse,
    DocumentDetailResponse,
    DocumentSummary,
    DocumentListResponse,
    PDFTypeInfo,
    TextStructureResponse,
    MEATValidationResponse,
    ICDMappingResponse
)
from ...services.text_extraction import text_extraction_service
from ...services.text_structuring import text_structuring_service
from ...services.meat_processor import meat_processor
from ...services.icd_mapper import icd_mapper
from ...services.clinical_document_analyzer import clinical_document_analyzer
from ...utils.pdf_detector import detect_pdf_type
from ...utils.abbreviation_expander import MEDICAL_ABBREVIATIONS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# ═══════════════════════════════════════════════════════════════════════════
# Strict Output Filter — 3 Rules
# ═══════════════════════════════════════════════════════════════════════════
# Rule 1: Assessment section priority — only diseases from Assessment/A&P
# Rule 2: MEAT required — disease must have valid MEAT
# Rule 3: Active disease only — exclude History Only, resolved, ruled-out, etc.
#
# EXCLUDED categories:
#   - Past Medical History diseases (PMH-only)
#   - Family History diseases
#   - Resolved / ruled-out diseases
#   - Symptoms when a diagnosis already exists
#   - Diseases without MEAT
# ═══════════════════════════════════════════════════════════════════════════

_ASSESSMENT_SECTIONS = {"assessment", "assessment_and_plan", "active_problems", "active_problem_list", "problem_list"}
_EXCLUDED_SECTIONS = {"family_history", "fh", "screening"}
_EXCLUDED_STATUSES = {"History Only"}
_EXCLUDED_TIERS = {"history_only", "invalid"}

# Sections that should NOT be sole source of an active disease (Approach 10)
_RESTRICTED_SOLE_SECTIONS = {"family_history", "fh", "screening", "social_history"}

# Plan annotation patterns that trail disease names in Assessment items.
# e.g. "Morbid obesity stable, counsel diet and exercise" → "Morbid obesity"
# Pattern handles both "Disease stable" and "Disease, stable on meds"
_PLAN_ANNOTATION_RE = re.compile(
    r'(?i)(?:'
    r'\s+(?:stable|controlled|uncontrolled|well controlled)\b(?:\s*[,=].*)?$'
    r'|\s*[=,]\s*(?:stable|controlled|uncontrolled|well controlled|'
    r'counsel|monitor|consider|continue|increase|decrease|'
    r'refer|serial|start|check|order|worsening|improving|'
    r'due to dm|on meds|diet and exercise|fiber/fluids)\b.*$'
    r')'
)
# Fragments too short or generic to be real disease names
_FRAGMENT_RE = re.compile(
    r'^(?:due to \w+|stable.*|on meds|single episode.*|psy support|'
    r'serial \w+|consider \w+|increase \w+|monitor \w+)$',
    re.IGNORECASE,
)


def _clean_disease_display_name(name: str) -> str:
    """Strip trailing plan annotations from Assessment-derived disease names."""
    cleaned = _PLAN_ANNOTATION_RE.sub('', name).strip().rstrip('.,;:= ')
    return cleaned if len(cleaned) >= 3 else name


def _apply_strict_output_filter(unified_results: list, single_pass: bool = False) -> list:
    """
    Apply the 3 strict output rules + deduplication:
      Rule 1 — Assessment section priority: when assessment diseases exist, only
               diseases from Assessment/A&P pass. When NO assessment diseases
               exist (graceful fallback), allow other active clinical sections.
               EXCEPTION: diseases with treatment evidence (active medication)
               pass regardless of section, since medication = actively managed.
               When single_pass=True (LLM already did clinical reasoning),
               Rule 1 is skipped entirely.
      Rule 2 — MEAT required: meat_tier must be full/medium/half
      Rule 3 — Active disease only: exclude history-only, family history, resolved
      Rule 4 — ICD code required: disease must map to a real ICD-10 code
      Rule 5 — Deduplication: remove overlapping/duplicate disease names

    Returns a new list of qualifying results, re-numbered sequentially.
    """
    filtered = []

    # Normalise section names: "Review of Systems" → "review_of_systems"
    def _normalise_sections(raw: list) -> set:
        return {s.lower().replace(" ", "_") for s in raw}

    # The actual Assessment/A&P section (this visit's assessment)
    _VISIT_ASSESSMENT = {"assessment", "assessment_and_plan"}

    # Check if ANY disease came from an assessment section
    has_assessment_diseases = any(
        _normalise_sections(r.get("segment_source_raw", [])) & _ASSESSMENT_SECTIONS
        for r in unified_results
    )

    # Primary clinical sections that count as disease sources
    _PRIMARY_DISEASE_SECTIONS = _ASSESSMENT_SECTIONS | {
        "chief_complaint", "cc", "history_present_illness", "hpi",
        "past_medical_history", "pmh", "medical_history",
        "plan", "impression", "physical_exam", "pe", "objective",
    }

    # Sections that indicate an active/current disease (for fallback only)
    _ACTIVE_CLINICAL_SECTIONS = _PRIMARY_DISEASE_SECTIONS | {
        "medications", "current_medications", "review_of_systems", "ros",
    }

    # Sections that indicate active clinical management (supports disease
    # from Problem List even without treatment evidence field)
    _ACTIVE_EVIDENCE_SECTIONS = {
        "medications", "current_medications", "vital_signs", "vitals",
        "physical_exam", "pe", "labs", "imaging", "procedures",
    }

    # Patterns that indicate a historical cancer (should not be reported as active
    # unless explicitly discussed in the Assessment section of this visit)
    _CANCER_PATTERN = re.compile(
        r"(?i)\b(?:neoplasm|malignant|cancer|carcinoma|melanoma|lymphoma|sarcoma|"
        r"leukemia|myeloma|oncolog)\b"
    )

    for r in unified_results:
        seg_sources = r.get("segment_source_raw", [])
        seg_lower = _normalise_sections(seg_sources)

        # Also normalise the primary segment field
        primary_seg = r.get("segment", "").lower().replace(" ", "_")

        # Rule 1: Assessment-section priority
        in_assessment = bool(seg_lower & _ASSESSMENT_SECTIONS)
        in_visit_assessment = bool(seg_lower & _VISIT_ASSESSMENT)
        has_treatment = bool(r.get("treatment", False) or r.get("treatment_evidence"))

        # Sections that indicate this-visit focus (Chief Complaint, HPI)
        _VISIT_FOCUS_SECTIONS = {
            "chief_complaint", "cc", "history_present_illness", "hpi",
        }

        if single_pass:
            # Single-pass: Actual Assessment/A&P diseases always pass.
            # Chief Complaint / HPI diseases always pass (visit focus).
            # Other primary-section diseases need treatment or active
            # evidence from clinical sections (Meds, Vitals, PE, Labs).
            # Diseases on Problem List without supporting evidence are
            # considered inactive/historical for this visit.
            if has_assessment_diseases:
                has_active_evidence = bool(seg_lower & _ACTIVE_EVIDENCE_SECTIONS)
                in_visit_focus = bool(seg_lower & _VISIT_FOCUS_SECTIONS)
                if in_visit_assessment:
                    pass  # Assessment diseases always pass
                elif in_visit_focus:
                    pass  # Chief Complaint / HPI diseases always pass
                elif has_treatment or has_active_evidence:
                    in_primary_section = bool(seg_lower & _PRIMARY_DISEASE_SECTIONS)
                    if in_primary_section:
                        pass  # Primary section + treatment/active evidence
                    else:
                        continue  # Supporting sections only — skip
                else:
                    continue  # No treatment or active evidence — skip
        else:
            if has_assessment_diseases:
                if not in_assessment and not has_treatment:
                    continue
            else:
                if not (seg_lower & _ACTIVE_CLINICAL_SECTIONS):
                    continue

        # Rule 1b: Historical cancer exclusion — cancers NOT in this visit's
        # Assessment are treated as historical (problem list cancers with
        # post-cancer medication are still historical)
        dname = r.get("disease", r.get("disease_name", ""))
        if _CANCER_PATTERN.search(dname) and not in_visit_assessment:
            logger.debug(f"Rule 1b: Excluding historical cancer '{dname}'")
            continue

        # Rule 2: Must have valid MEAT (full, medium, or half)
        if r.get("meat_tier") in _EXCLUDED_TIERS:
            continue

        # Rule 3: Must be an active disease
        if r.get("disease_status") in _EXCLUDED_STATUSES:
            continue

        # Rule 3b: Skip if segment is ONLY from restricted sections (Approach 10)
        # Diseases from family history, screening, social history, or ROS
        # should not be active unless also found in a primary section
        if seg_lower and seg_lower.issubset(_RESTRICTED_SOLE_SECTIONS):
            continue

        # Rule 4: Must have a valid ICD-10 code (no "—" or empty)
        icd = r.get("icd_code", "")
        if not icd or icd == "—" or icd == "-":
            continue

        # Rule 5: Clean plan annotations from disease name & drop fragments
        cleaned_name = _clean_disease_display_name(dname)
        if _FRAGMENT_RE.match(cleaned_name):
            continue
        r["disease"] = cleaned_name  # update display name

        # Rule 6 (Approach 10): Evidence requirement — at least one MEAT
        # evidence field must contain verifiable text (not just boolean)
        has_real_evidence = any([
            (r.get("monitoring_evidence") or "").strip(),
            (r.get("evaluation_evidence") or "").strip(),
            (r.get("assessment_evidence") or "").strip(),
            (r.get("treatment_evidence") or "").strip(),
        ])
        if not has_real_evidence:
            logger.debug(f"Rule 6: Skipping '{cleaned_name}' — no verifiable evidence text")
            continue

        filtered.append(r)

    # Grace rule: When STILL nothing passed (section detector failure),
    # fall back to any diseases with valid MEAT from clinical sections.
    if not filtered and unified_results:
        _GRACE_SECTIONS = {"history_present_illness", "plan", "chief_complaint",
                           "physical_exam", "review_of_systems", "objective",
                           "past_medical_history", "pmh", "medical_history",
                           "medications", "current_medications", "hpi", "cc"}
        logger.info("Grace rule activated: no diseases passed filter, "
                     "allowing MEAT-valid diseases from clinical sections")
        for r in unified_results:
            seg_sources = r.get("segment_source_raw", [])
            seg_lower = _normalise_sections(seg_sources)

            # Only from clinical sections
            if not (seg_lower & _GRACE_SECTIONS):
                continue
            # Must have at least half MEAT
            if r.get("meat_tier") in {"history_only", "invalid"}:
                continue
            # Still must be active
            if r.get("disease_status") in _EXCLUDED_STATUSES:
                continue
            if seg_lower and seg_lower.issubset(_EXCLUDED_SECTIONS):
                continue
            # ICD code preferred but not required in grace mode
            filtered.append(r)

    # Rule 5: Deduplicate overlapping disease names
    # If "hypertension" and "Essential hypertension" both pass, keep more specific.
    # If two diseases share the same ICD code, keep the more specific name.
    filtered = _deduplicate_diseases(filtered)

    # Re-number
    for i, r in enumerate(filtered, 1):
        r["number"] = i

    return filtered


def _format_segment_label(seg_sources: list) -> str:
    """
    Format segment sources into a human-readable label.

    When assessment sections are present, show ONLY the primary assessment
    label (e.g. just "Assessment" instead of
    "Assessment/History Present Illness/Past Medical History").
    This keeps the output table clean and matches the ideal output format.
    """
    if not seg_sources:
        return "Unknown"

    _ASSESSMENT_DISPLAY = {
        "assessment_and_plan", "assessment",
        "active_problems", "active_problem_list", "problem_list"
    }
    sources_lower = {s.lower() for s in seg_sources}

    # If any assessment key is present, display only the canonical assessment label
    if sources_lower & _ASSESSMENT_DISPLAY:
        # Pick the most specific assessment key present
        for key in ("assessment_and_plan", "assessment", "active_problems",
                    "active_problem_list", "problem_list"):
            if key in sources_lower:
                return key.replace("_", " ").title()

    # Otherwise show all sources joined
    return "/".join(s.replace("_", " ").title() for s in seg_sources)


def _deduplicate_diseases(results: list) -> list:
    """
    Remove duplicate/overlapping diseases from final results.
    
    Strategy:
    1. Group by ICD code — keep the most specific disease name per code.
    2. Substring check — if one disease name is a substring of another, keep the longer one.
    """
    if len(results) <= 1:
        return results

    # Pass 1: Group by ICD code, keep the most specific (longest name) per code
    by_icd: dict = {}
    no_icd_dupes = []
    for r in results:
        icd = r.get("icd_code", "")
        if icd and icd != "—":
            if icd not in by_icd:
                by_icd[icd] = r
            else:
                # Keep the one with the longer (more specific) disease name
                existing_name = by_icd[icd]["disease"].lower()
                new_name = r["disease"].lower()
                if len(new_name) > len(existing_name):
                    by_icd[icd] = r
                # If same length, keep higher MEAT tier
                elif len(new_name) == len(existing_name):
                    tier_priority = {"full_meat": 0, "medium_meat": 1, "half_meat": 2}
                    if tier_priority.get(r["meat_tier"], 9) < tier_priority.get(by_icd[icd]["meat_tier"], 9):
                        by_icd[icd] = r
        else:
            no_icd_dupes.append(r)

    deduped = list(by_icd.values()) + no_icd_dupes

    # Pass 1.5: Abbreviation-aware grouping — "HTN" and "Essential Hypertension"
    # both refer to the same condition. Keep the longer (more specific) name.
    _abbr_map = {k.lower(): v.lower() for k, v in MEDICAL_ABBREVIATIONS.items()}
    _rev_map = {v.lower(): k.lower() for k, v in MEDICAL_ABBREVIATIONS.items()}

    def _expand_name(name: str) -> set:
        """Return all known expansions/abbreviations for a disease name."""
        low = name.lower().strip()
        variants = {low}
        # Try expanding abbreviation → full name
        if low in _abbr_map:
            variants.add(_abbr_map[low])
        # Try collapsing full name → abbreviation
        if low in _rev_map:
            variants.add(_rev_map[low])
        # Also check if any known abbreviation is a whole-word inside the name
        for abbr, expansion in _abbr_map.items():
            if abbr in low.split():
                variants.add(expansion)
            if expansion in low:
                variants.add(abbr)
        return variants

    abbr_deduped = []
    seen_variant_groups: list = []  # list of sets of variant keys
    for r in deduped:
        rname = r["disease"].lower().strip()
        r_variants = _expand_name(rname)
        merged = False
        for idx, group in enumerate(seen_variant_groups):
            # If any variant from this result overlaps with an existing group
            if r_variants & group:
                # Keep the longer (more specific) disease name
                existing = abbr_deduped[idx]
                if len(rname) > len(existing["disease"]):
                    abbr_deduped[idx] = r
                elif len(rname) == len(existing["disease"]):
                    tier_priority = {"full_meat": 0, "medium_meat": 1, "half_meat": 2}
                    if tier_priority.get(r["meat_tier"], 9) < tier_priority.get(existing["meat_tier"], 9):
                        abbr_deduped[idx] = r
                seen_variant_groups[idx] |= r_variants
                merged = True
                break
        if not merged:
            abbr_deduped.append(r)
            seen_variant_groups.append(r_variants)

    deduped = abbr_deduped

    # Pass 2: Substring check — remove entries whose name is a substring of another
    final = []
    names_lower = [r["disease"].lower().strip() for r in deduped]
    for i, r in enumerate(deduped):
        name_i = names_lower[i]
        is_subset = False
        for j, name_j in enumerate(names_lower):
            if i != j and name_i in name_j and len(name_i) < len(name_j):
                is_subset = True
                break
        if not is_subset:
            final.append(r)

    # Pass 3: Stem-based overlap — catch "neuropathies" vs "diabetic polyneuropathy"
    # If a disease's root word appears inside another more specific disease, remove it.
    # ONLY when both share the same ICD code — different ICD codes mean clinically
    # distinct conditions that should both be reported.
    import re as _re
    _MEDICAL_SUFFIXES = ['ies', 'es', 'y', 's', 'ia', 'ic', 'al', 'ous']
    def _word_roots(name: str) -> list:
        """Extract word roots (>= 5 chars) from disease name."""
        roots = []
        for w in name.lower().split():
            if len(w) < 5:
                continue
            root = w
            for suf in sorted(_MEDICAL_SUFFIXES, key=len, reverse=True):
                if root.endswith(suf) and len(root) - len(suf) >= 4:
                    root = root[:-len(suf)]
                    break
            if len(root) >= 5:
                roots.append(root)
        return roots

    to_remove = set()
    final_names = [r["disease"].lower().strip() for r in final]
    for i, r_i in enumerate(final):
        if i in to_remove:
            continue
        roots_i = _word_roots(final_names[i])
        if not roots_i:
            continue
        icd_i = r_i.get("icd_code", "")
        for j, r_j in enumerate(final):
            if i == j or j in to_remove:
                continue
            if len(final_names[i]) >= len(final_names[j]):
                continue  # Only remove the shorter/less specific one
            # Only stem-dedup when ICD codes match — different ICD = different condition
            icd_j = r_j.get("icd_code", "")
            if icd_i and icd_j and icd_i != icd_j:
                continue
            # Check if any root from the shorter name appears in the longer name
            longer_joined = final_names[j].replace(" ", "")
            if any(root in longer_joined for root in roots_i):
                to_remove.add(i)
                break
    final = [r for i, r in enumerate(final) if i not in to_remove]

    # Sort by disease name for consistent output
    final.sort(key=lambda r: r["disease"].lower())
    return final


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload PDF document.
    """
    # 1. Validate file extension
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    # 2. Validate file size (check headers first if available, otherwise read)
    # Note: Stream content if very large, but here we check settings
    content = await file.read()
    if len(content) > settings.UPLOAD_MAX_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {settings.UPLOAD_MAX_SIZE / (1024*1024)}MB.")
    
    await file.seek(0) # Reset to start after reading for size check

    # 3. Setup upload directory
    upload_dir = Path(settings.UPLOAD_FOLDER)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Generate unique filename
    doc_id = str(uuid.uuid4())
    safe_filename = f"{doc_id}.pdf"
    file_path = upload_dir / safe_filename
    
    # 5. Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail="Could not save file.")

    # 6. Create database record
    new_doc = Document(
        id=doc_id,
        filename=file.filename,
        file_path=str(file_path),
        status=DocumentStatus.UPLOADED,
        processed=False
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)

    return DocumentUploadResponse(
        document_id=new_doc.id,
        filename=new_doc.filename,
        status=new_doc.status.value,
        message="Document uploaded successfully."
    )

@router.post("/{document_id}/extract", response_model=TextExtractionResponse)
async def extract_document_text(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Extract text from uploaded PDF.
    """
    # 1. Get document from database
    doc = db.query(Document).filter(Document.id == document_id, Document.deleted == False).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # 2. Check file exists
    if not doc.file_path or not os.path.exists(doc.file_path):
        raise HTTPException(status_code=404, detail="Physical file missing on server.")

    # 3. Extract text
    import time
    try:
        doc.status = DocumentStatus.PROCESSING
        db.commit()
        
        start_time = time.time()
        result = await text_extraction_service.extract_text(doc.file_path)
        processing_time = time.time() - start_time
        
        # 4. Update database
        doc.raw_text = result["raw_text"]
        doc.page_count = result["page_count"]
        doc.processed = True
        doc.status = DocumentStatus.COMPLETED
        db.commit()
        
        return {
            "document_id": doc.id,
            "processing_time": processing_time,
            **result
        }
    except Exception as e:
        doc.status = DocumentStatus.FAILED
        db.commit()
        logger.error(f"Extraction failed for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{document_id}/structure", response_model=TextStructureResponse)
async def structure_document_text(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Structure extracted text - detect sections and diseases.
    """
    # 1. Get document
    doc = db.query(Document).filter(Document.id == document_id, Document.deleted == False).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # 2. Check if text exists
    if not doc.raw_text:
        raise HTTPException(
            status_code=400, 
            detail="No text to structure. Run /extract endpoint first."
        )
    
    try:
        # 3. Run structuring pipeline
        structure_result = await text_structuring_service.structure_text(
            doc.raw_text
        )
        
        # 4. Store detected diseases in database
        # Clear existing diseases for this document to avoid duplicates on re-run
        db.query(DetectedDisease).filter(
            DetectedDisease.document_id == document_id
        ).delete()
        
        # Add new detected diseases
        for disease_data in structure_result["detected_diseases"]:
            new_disease = DetectedDisease(
                document_id=document_id,
                disease_name=disease_data["disease_name"],
                normalized_name=disease_data["normalized_name"],
                confidence_score=disease_data["confidence_score"],
                negated=disease_data["negated"],
                section=disease_data.get("section"),
                sentence_number=disease_data.get("sentence_number")
            )
            db.add(new_disease)
        
        # Update document status
        doc.status = DocumentStatus.COMPLETED
        db.commit()
        
        return {
            "document_id": document_id,
            "total_sentences": structure_result["total_sentences"],
            "total_diseases": structure_result["total_diseases"],
            "sections_found": len(structure_result["sections"]),
            "detected_diseases": structure_result["detected_diseases"],
            "sections": structure_result["sections"],
            "processing_stats": structure_result["processing_stats"]
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Structuring failed for {document_id}: {e}")
        error_str = str(e).lower()
        if any(x in error_str for x in ["503", "429", "unavailable", "rate limit", "quota"]):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Gemini API is temporarily unavailable (high demand). "
                    "Retries were exhausted. Please try again in a few minutes."
                ),
            )
        raise HTTPException(status_code=500, detail=f"Structuring failed: {str(e)}")


@router.post("/{document_id}/validate-meat", response_model=MEATValidationResponse)
async def validate_meat_for_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Validate MEAT criteria for all detected diseases.
    
    Steps:
    1. Get document and verify structured data exists
    2. Load detected diseases, sentences, sections
    3. Run MEAT processing pipeline
    4. Store MEAT results in database
    5. Return validation results
    """
    # 1. Get document
    document = db.query(Document).filter(Document.id == document_id, Document.deleted == False).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # 2. Get detected diseases
    diseases = db.query(DetectedDisease).filter(
        DetectedDisease.document_id == document_id
    ).all()
    
    if not diseases:
        raise HTTPException(
            status_code=400,
            detail="No diseases detected. Run /structure endpoint first."
        )
    
    try:
        # Convert diseases to dict format for the processor
        disease_dicts = [
            {
                "disease_name": d.disease_name,
                "normalized_name": d.normalized_name,
                "confidence_score": d.confidence_score,
                "negated": d.negated,
                "section": d.section,
                "section_sources": [d.section] if d.section else ["unknown"],
                "sentence_number": d.sentence_number
            }
            for d in diseases
        ]
        
        # 3. Build lightweight context (sentence segmentation only — no Gemini re-detection)
        #    Full re-structuring is too expensive and causes timeouts.
        from ...utils.text_preprocessor import segment_sentences, normalize_whitespace, fix_line_breaks
        from ...utils.abbreviation_expander import expand_abbreviations

        _text = fix_line_breaks(normalize_whitespace(document.raw_text))
        _text = expand_abbreviations(_text, context_aware=True)
        fast_sentences = segment_sentences(_text)

        # Reconstruct a lightweight sections map from the stored disease records
        # so context_builder can still provide section-level context.
        fast_sections: dict = {}
        for d in diseases:
            sec = d.section or "unknown"
            if sec not in fast_sections:
                fast_sections[sec] = {"text": "", "section_name": sec}
            # Append any sentence from the global text that mentions this disease
            dn_lower = d.disease_name.lower()
            for sent in fast_sentences:
                if dn_lower in sent.get("text", "").lower():
                    fast_sections[sec]["text"] += " " + sent["text"]

        # 4. Run MEAT validation pipeline
        meat_result = await meat_processor.process_meat_validation(
            detected_diseases=disease_dicts,
            full_text=_text,
            sentences=fast_sentences,
            sections=fast_sections
        )
        
        # 5. Store MEAT results in database
        # Clear existing MEAT validations
        disease_ids = [d.id for d in diseases]
        db.query(MEATValidation).filter(
            MEATValidation.disease_id.in_(disease_ids)
        ).delete(synchronize_session=False)
        
        # Add new MEAT validations (hybrid: skip duplicates, preserve history)
        seen_disease_ids = set()
        for result in meat_result["meat_results"]:
            disease_record = next(
                (d for d in diseases if d.disease_name == result["disease"]),
                None
            )
            
            if disease_record:
                # Skip if already processed (duplicate in results)
                if disease_record.id in seen_disease_ids:
                    logger.info(f"Skipping duplicate MEAT for disease_id={disease_record.id}")
                    continue
                seen_disease_ids.add(disease_record.id)

                meat_data = result["meat_data"]
                # The gate validates and stores final adjusted data in 'final_meat'
                validation_data = result["validation"]["final_meat"]
                
                meat_validation = MEATValidation(
                    disease_id=disease_record.id,
                    monitoring=validation_data.get("monitoring", False),
                    monitoring_evidence=validation_data.get("monitoring_evidence", ""),
                    monitoring_confidence=validation_data.get("monitoring_confidence", 0.0),
                    evaluation=validation_data.get("evaluation", False),
                    evaluation_evidence=validation_data.get("evaluation_evidence", ""),
                    evaluation_confidence=validation_data.get("evaluation_confidence", 0.0),
                    assessment=validation_data.get("assessment", False),
                    assessment_evidence=validation_data.get("assessment_evidence", ""),
                    assessment_confidence=validation_data.get("assessment_confidence", 0.0),
                    treatment=validation_data.get("treatment", False),
                    treatment_evidence=validation_data.get("treatment_evidence", ""),
                    treatment_confidence=validation_data.get("treatment_confidence", 0.0),
                    meat_valid=result["validation"]["meat_valid"],
                    overall_confidence=validation_data.get("overall_confidence", 0.0),
                    llm_reasoning=meat_data.get("llm_reasoning", "")
                )
                
                db.add(meat_validation)
        
        db.commit()
        
        return {
            "document_id": document_id,
            "total_diseases": meat_result["total_diseases"],
            "valid_meat_count": meat_result["valid_meat_count"],
            "requires_review_count": meat_result["requires_review_count"],
            "meat_results": meat_result["meat_results"],
            "processing_summary": meat_result["processing_summary"]
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"MEAT validation failed for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"MEAT validation failed: {str(e)}")


@router.post("/{document_id}/map-icd", response_model=ICDMappingResponse)
async def map_icd_codes(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Map MEAT-validated diseases to ICD-10 codes.
    
    Steps:
    1. Get document and verify MEAT validation exists
    2. Load diseases with MEAT validation
    3. Run ICD mapping pipeline:
       - Exact match lookup
       - Fuzzy matching
       - LLM ranking (using MEAT evidence)
       - Confidence gating
    4. Store ICD mappings in database
    5. Return mapping results
    
    Errors:
    - 404: Document not found
    - 400: No MEAT validation (run /validate-meat first)
    - 500: ICD mapping failed
    """
    
    # Get document
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get diseases for mapping (non-negated)
    from ...models.disease import DetectedDisease
    
    diseases = db.query(DetectedDisease).filter(
        DetectedDisease.document_id == document_id,
        DetectedDisease.negated == False
    ).all()
    
    if not diseases:
        raise HTTPException(
            status_code=400,
            detail="No diseases to map. Run /structure first."
        )
    
    try:
        # Prepare diseases with MEAT data
        diseases_with_meat = []
        
        for disease in diseases:
            # Get MEAT validation
            meat = db.query(MEATValidation).filter(
                MEATValidation.disease_id == disease.id
            ).first()
            
            meat_data = {}
            if meat:
                meat_data = {
                    "meat_valid": meat.meat_valid,
                    "assessment_evidence": meat.assessment_evidence,
                    "treatment_evidence": meat.treatment_evidence,
                    "evaluation_evidence": meat.evaluation_evidence,
                    "monitoring_evidence": meat.monitoring_evidence,
                    "overall_confidence": meat.overall_confidence
                }
            
            diseases_with_meat.append({
                "disease_id": disease.id,
                "disease_name": disease.disease_name,
                "normalized_name": disease.normalized_name,
                "meat_validation": meat_data
            })
        
        # Run ICD mapping
        mapping_result = await icd_mapper.map_multiple_diseases(diseases_with_meat)
        
        # Store mappings in database
        # Clear existing mappings for this document's diseases
        disease_ids = [d.id for d in diseases]
        db.query(ICDMapping).filter(
            ICDMapping.disease_id.in_(disease_ids)
        ).delete(synchronize_session=False)
        
        # Add new mappings
        for i, mapping in enumerate(mapping_result["mappings"]):
            disease_id = diseases_with_meat[i]["disease_id"]
            
            if mapping["icd_code"]:
                icd_mapping = ICDMapping(
                    disease_id=disease_id,
                    icd_code=mapping["icd_code"],
                    icd_description=mapping["icd_description"],
                    mapping_method=mapping["mapping_method"],
                    confidence_score=mapping["confidence"],
                    status=mapping["status"],
                    llm_reasoning=mapping.get("llm_reasoning", "")
                )
                
                db.add(icd_mapping)
        
        db.commit()
        
        return ICDMappingResponse(
            document_id=document_id,
            total_diseases=mapping_result["total_diseases"],
            auto_assigned=mapping_result["auto_assigned"],
            manual_review=mapping_result["manual_review"],
            not_found=mapping_result["not_found"],
            mappings=mapping_result["mappings"]
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"ICD Mapping failed for {document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"ICD mapping failed: {str(e)}"
        )


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """Get document details by ID."""
    doc = db.query(Document).filter(Document.id == document_id, Document.deleted == False).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc

@router.get("", response_model=DocumentListResponse)
async def list_documents(
    skip: int = 0,
    limit: int = 20,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List documents with pagination.
    """
    query = db.query(Document).filter(Document.deleted == False)
    
    if status:
        query = query.filter(Document.status == status)
    
    total = query.count()
    items = query.offset(skip).limit(min(limit, 100)).all()
    
    return DocumentListResponse(
        total=total,
        skip=skip,
        limit=limit,
        documents=items
    )


@router.post("/{document_id}/process-all")
async def process_all_steps(
    document_id: str,
    dietary_analysis: bool = Query(default=False, description="Include Phase 8 dietary/nutritional analysis annotations"),
    db: Session = Depends(get_db)
):
    """
    Run the complete pipeline: Structure → MEAT → ICD in one call.
    Returns a unified result combining all disease data into a single table format.

    Requires: Document must already be uploaded and text extracted.
    """
    import time
    
    # 1. Get document
    document = db.query(Document).filter(
        Document.id == document_id, Document.deleted == False
    ).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if not document.raw_text:
        raise HTTPException(
            status_code=400,
            detail="No text to process. Run /extract endpoint first."
        )
    
    start_time = time.time()
    
    try:
        # ══════════════════════════════════════════════════════════════════
        # PRIMARY PATH: Single-pass LLM analysis (1 call does everything)
        # ══════════════════════════════════════════════════════════════════
        single_pass_result = None
        try:
            single_pass_result = await clinical_document_analyzer.analyze_document(
                document.raw_text,
                dietary_analysis=dietary_analysis,
            )
        except Exception as sp_err:
            logger.warning(
                f"Single-pass LLM failed after retries: {sp_err}. "
                f"Attempting multi-step pipeline."
            )

        if single_pass_result and single_pass_result.get("diseases"):
            logger.info(
                f"Single-pass analysis succeeded: "
                f"{len(single_pass_result['diseases'])} diseases found."
            )

            # Convert to unified results (already includes MEAT + ICD)
            unified_results = clinical_document_analyzer.convert_to_unified_results(
                single_pass_result
            )
            pipeline_data = clinical_document_analyzer.convert_to_pipeline_format(
                single_pass_result
            )

            # ── Store in DB ──────────────────────────────────────────
            # Clear existing records
            db.query(DetectedDisease).filter(
                DetectedDisease.document_id == document_id
            ).delete()

            for dd in pipeline_data["detected_diseases"]:
                new_disease = DetectedDisease(
                    document_id=document_id,
                    disease_name=dd["disease_name"],
                    normalized_name=dd["normalized_name"],
                    confidence_score=dd["confidence_score"],
                    negated=dd["negated"],
                    section=dd.get("section"),
                    sentence_number=dd.get("sentence_number", 0),
                )
                db.add(new_disease)

            document.status = DocumentStatus.COMPLETED
            db.commit()

            # Store MEAT validations from single-pass results
            diseases_db = db.query(DetectedDisease).filter(
                DetectedDisease.document_id == document_id
            ).all()
            disease_ids = [d.id for d in diseases_db]

            db.query(MEATValidation).filter(
                MEATValidation.disease_id.in_(disease_ids)
            ).delete(synchronize_session=False)

            seen_disease_ids_sp = set()
            for dd in pipeline_data["detected_diseases"]:
                disease_record = next(
                    (d for d in diseases_db if d.disease_name == dd["disease_name"]),
                    None,
                )
                if disease_record and dd.get("meat_validation"):
                    # Skip if already processed (duplicate in results)
                    if disease_record.id in seen_disease_ids_sp:
                        logger.info(f"MEAT already validated for {disease_record.id}, skipping...")
                        continue
                    # Skip if already exists in DB
                    existing = db.query(MEATValidation).filter_by(disease_id=disease_record.id).first()
                    if existing:
                        logger.info(f"MEAT already exists in DB for {disease_record.id}, skipping...")
                        seen_disease_ids_sp.add(disease_record.id)
                        continue
                    seen_disease_ids_sp.add(disease_record.id)

                    mv = dd["meat_validation"]
                    meat_validation = MEATValidation(
                        disease_id=disease_record.id,
                        monitoring=bool(mv.get("M_monitor")),
                        monitoring_evidence=mv.get("M_monitor") or "",
                        monitoring_confidence=0.9 if mv.get("M_monitor") else 0.0,
                        evaluation=bool(mv.get("E_evaluate")),
                        evaluation_evidence=mv.get("E_evaluate") or "",
                        evaluation_confidence=0.9 if mv.get("E_evaluate") else 0.0,
                        assessment=bool(mv.get("A_assess")),
                        assessment_evidence=mv.get("A_assess") or "",
                        assessment_confidence=0.9 if mv.get("A_assess") else 0.0,
                        treatment=bool(mv.get("T_treat")),
                        treatment_evidence=mv.get("T_treat") or "",
                        treatment_confidence=0.9 if mv.get("T_treat") else 0.0,
                        meat_valid=dd.get("meat_score", 0) >= 1,
                        overall_confidence=0.9,
                        llm_reasoning=mv.get("meat_grade_reasoning", ""),
                    )
                    db.add(meat_validation)

            # Store ICD mappings from single-pass results
            db.query(ICDMapping).filter(
                ICDMapping.disease_id.in_(disease_ids)
            ).delete(synchronize_session=False)

            for dd in pipeline_data["detected_diseases"]:
                disease_record = next(
                    (d for d in diseases_db if d.disease_name == dd["disease_name"]),
                    None,
                )
                if disease_record and dd.get("icd_code"):
                    icd_mapping = ICDMapping(
                        disease_id=disease_record.id,
                        icd_code=dd["icd_code"],
                        icd_description=dd.get("icd_description", ""),
                        mapping_method=MappingMethod.LLM_RANKED,
                        confidence_score=0.95,
                        status=MappingStatus.AUTO_ASSIGNED,
                        llm_reasoning=dd.get("icd_selection_reasoning", ""),
                    )
                    db.add(icd_mapping)

            db.commit()

            # ── Apply filter with single_pass=True (skip Rule 1) ──
            all_results = unified_results
            filtered_results = _apply_strict_output_filter(unified_results, single_pass=True)

            processing_time = time.time() - start_time

            return {
                "document_id": document_id,
                "processing_time": round(processing_time, 2),
                "total_diseases": len(filtered_results),
                "total_detected": len(all_results),
                "sections_found": len(single_pass_result.get("sections_found", [])),
                "unified_results": filtered_results,
                "analysis_source": "single_pass_llm",
                "summary": {
                    "full_meat": sum(1 for r in filtered_results if r["meat_tier"] == "full_meat"),
                    "medium_meat": sum(1 for r in filtered_results if r["meat_tier"] == "medium_meat"),
                    "half_meat": sum(1 for r in filtered_results if r["meat_tier"] == "half_meat"),
                    "history_only": 0,
                    "invalid": 0,
                    "icd_mapped": sum(1 for r in filtered_results if r["icd_code"] != "—"),
                },
                "excluded_summary": {
                    "total_excluded": len(all_results) - len(filtered_results),
                    "excluded_conditions": single_pass_result.get("excluded_conditions", []),
                },
            }

        # ══════════════════════════════════════════════════════════════════
        # FALLBACK PATH: Multi-step fragmented pipeline
        # ══════════════════════════════════════════════════════════════════
        logger.info(
            "Single-pass analysis returned no results or failed. "
            "Falling back to multi-step pipeline."
        )

        # ── Step 1: Structure ──────────────────────────────────────────
        structure_result = await text_structuring_service.structure_text(
            document.raw_text
        )
        
        # Store detected diseases in database
        db.query(DetectedDisease).filter(
            DetectedDisease.document_id == document_id
        ).delete()
        
        for disease_data in structure_result["detected_diseases"]:
            new_disease = DetectedDisease(
                document_id=document_id,
                disease_name=disease_data["disease_name"],
                normalized_name=disease_data["normalized_name"],
                confidence_score=disease_data["confidence_score"],
                negated=disease_data["negated"],
                section=disease_data.get("section"),
                sentence_number=disease_data.get("sentence_number")
            )
            db.add(new_disease)
        
        document.status = DocumentStatus.COMPLETED
        db.commit()
        
        # ── Step 2: MEAT Validation ────────────────────────────────────
        diseases = db.query(DetectedDisease).filter(
            DetectedDisease.document_id == document_id
        ).all()
        
        disease_dicts = []
        for d in diseases:
            # Find section_sources from structure_result
            original = next(
                (dd for dd in structure_result["detected_diseases"]
                 if dd["disease_name"] == d.disease_name),
                None
            )
            disease_dicts.append({
                "disease_name": d.disease_name,
                "normalized_name": d.normalized_name,
                "confidence_score": d.confidence_score,
                "negated": d.negated,
                "section": d.section,
                "sentence_number": d.sentence_number,
                "section_sources": original.get("section_sources", [d.section or "unknown"]) if original else [d.section or "unknown"]
            })
        
        # Build context for MEAT
        from ...utils.text_preprocessor import segment_sentences, normalize_whitespace, fix_line_breaks
        from ...utils.abbreviation_expander import expand_abbreviations
        
        _text = fix_line_breaks(normalize_whitespace(document.raw_text))
        _text = expand_abbreviations(_text, context_aware=True)
        fast_sentences = segment_sentences(_text)
        
        fast_sections: dict = {}
        for d in diseases:
            sec = d.section or "unknown"
            if sec not in fast_sections:
                fast_sections[sec] = {"text": "", "section_name": sec}
            dn_lower = d.disease_name.lower()
            for sent in fast_sentences:
                if dn_lower in sent.get("text", "").lower():
                    fast_sections[sec]["text"] += " " + sent["text"]
        
        # Also add the actual section texts from structure_result
        for sec_name, sec_data in structure_result.get("sections", {}).items():
            if sec_name not in fast_sections:
                fast_sections[sec_name] = sec_data
            else:
                fast_sections[sec_name]["text"] = sec_data.get("text", "") + " " + fast_sections[sec_name].get("text", "")
        
        meat_result = await meat_processor.process_meat_validation(
            detected_diseases=disease_dicts,
            full_text=_text,
            sentences=fast_sentences,
            sections=fast_sections
        )
        
        # Store MEAT results
        disease_ids = [d.id for d in diseases]
        db.query(MEATValidation).filter(
            MEATValidation.disease_id.in_(disease_ids)
        ).delete(synchronize_session=False)
        
        seen_disease_ids_fb = set()
        for result in meat_result["meat_results"]:
            disease_record = next(
                (d for d in diseases if d.disease_name == result["disease"]),
                None
            )
            if disease_record:
                # Skip if already processed (duplicate in results)
                if disease_record.id in seen_disease_ids_fb:
                    logger.info(f"MEAT already validated for {disease_record.id}, skipping...")
                    continue
                # Skip if already exists in DB
                existing = db.query(MEATValidation).filter_by(disease_id=disease_record.id).first()
                if existing:
                    logger.info(f"MEAT already exists in DB for {disease_record.id}, skipping...")
                    seen_disease_ids_fb.add(disease_record.id)
                    continue
                seen_disease_ids_fb.add(disease_record.id)

                meat_data = result["meat_data"]
                validation_data = result["validation"]["final_meat"]
                meat_validation = MEATValidation(
                    disease_id=disease_record.id,
                    monitoring=validation_data.get("monitoring", False),
                    monitoring_evidence=validation_data.get("monitoring_evidence", ""),
                    monitoring_confidence=validation_data.get("monitoring_confidence", 0.0),
                    evaluation=validation_data.get("evaluation", False),
                    evaluation_evidence=validation_data.get("evaluation_evidence", ""),
                    evaluation_confidence=validation_data.get("evaluation_confidence", 0.0),
                    assessment=validation_data.get("assessment", False),
                    assessment_evidence=validation_data.get("assessment_evidence", ""),
                    assessment_confidence=validation_data.get("assessment_confidence", 0.0),
                    treatment=validation_data.get("treatment", False),
                    treatment_evidence=validation_data.get("treatment_evidence", ""),
                    treatment_confidence=validation_data.get("treatment_confidence", 0.0),
                    meat_valid=result["validation"]["meat_valid"],
                    overall_confidence=validation_data.get("overall_confidence", 0.0),
                    llm_reasoning=meat_data.get("llm_reasoning", "")
                )
                db.add(meat_validation)
        
        db.commit()
        
        # ── Step 3: ICD Mapping ────────────────────────────────────────
        diseases_with_meat = []
        for disease in diseases:
            meat = db.query(MEATValidation).filter(
                MEATValidation.disease_id == disease.id
            ).first()
            
            meat_data = {}
            if meat:
                meat_data = {
                    "meat_valid": meat.meat_valid,
                    "assessment_evidence": meat.assessment_evidence,
                    "treatment_evidence": meat.treatment_evidence,
                    "evaluation_evidence": meat.evaluation_evidence,
                    "monitoring_evidence": meat.monitoring_evidence,
                    "overall_confidence": meat.overall_confidence
                }
            
            diseases_with_meat.append({
                "disease_id": disease.id,
                "disease_name": disease.disease_name,
                "normalized_name": disease.normalized_name,
                "meat_validation": meat_data
            })
        
        mapping_result = await icd_mapper.map_multiple_diseases(diseases_with_meat)
        
        # Store ICD mappings
        db.query(ICDMapping).filter(
            ICDMapping.disease_id.in_(disease_ids)
        ).delete(synchronize_session=False)
        
        for i, mapping in enumerate(mapping_result["mappings"]):
            disease_id = diseases_with_meat[i]["disease_id"]
            if mapping["icd_code"]:
                icd_mapping = ICDMapping(
                    disease_id=disease_id,
                    icd_code=mapping["icd_code"],
                    icd_description=mapping["icd_description"],
                    mapping_method=mapping["mapping_method"],
                    confidence_score=mapping["confidence"],
                    status=mapping["status"],
                    llm_reasoning=mapping.get("llm_reasoning", "")
                )
                db.add(icd_mapping)
        
        db.commit()
        
        # ── Step 4: Build Unified Table ────────────────────────────────
        unified_results = []
        meat_results_list = meat_result.get("meat_results", [])
        mappings_list = mapping_result.get("mappings", [])
        
        for idx, mr in enumerate(meat_results_list):
            disease_name = mr["disease"]
            
            # Find ICD mapping for this disease
            icd_info = next(
                (m for m in mappings_list if m["disease"] == disease_name),
                None
            )
            
            # Format segment sources — show only primary Assessment label when applicable
            seg_sources = mr.get("segment_source", [])
            segment_label = _format_segment_label(seg_sources)
            
            # MEAT tier display label
            tier = mr.get("meat_tier", "invalid")
            tier_label = {
                "full_meat": "Full MEAT",
                "medium_meat": "Medium MEAT",
                "half_meat": "Half MEAT",
                "history_only": "History Only",
                "invalid": "Invalid",
            }.get(tier, "Unknown")
            
            meat_data = mr.get("meat_data", {})
            # Use gate-adjusted final_meat values for display (not raw LLM)
            final_meat = mr.get("validation", {}).get("final_meat", meat_data)

            _mon = bool(final_meat.get("monitoring", False))
            _eva = bool(final_meat.get("evaluation", False))
            _ass = bool(final_meat.get("assessment", False))
            _tre = bool(final_meat.get("treatment", False))
            _icd_conf = icd_info["confidence"] if icd_info else 0.0
            _meat_count = sum([_mon, _eva, _ass, _tre])
            _confidence = round((_meat_count / 4) * 0.6 + _icd_conf * 0.4, 2)

            unified_results.append({
                "number": idx + 1,
                "disease": disease_name,
                "icd_code": icd_info["icd_code"] if icd_info and icd_info.get("icd_code") else "—",
                "icd_description": icd_info["icd_description"] if icd_info and icd_info.get("icd_description") else "",
                "segment": segment_label,
                "segment_source_raw": seg_sources,
                "disease_status": mr.get("disease_status", "Chronic"),
                "monitoring": _mon,
                "evaluation": _eva,
                "assessment": _ass,
                "treatment": _tre,
                "meat_level": tier_label,
                "meat_tier": tier,
                "confidence": _confidence,
                "icd_confidence": _icd_conf,
                "icd_method": icd_info["mapping_method"] if icd_info else "none",
                "monitoring_evidence": final_meat.get("monitoring_evidence", ""),
                "evaluation_evidence": final_meat.get("evaluation_evidence", ""),
                "assessment_evidence": final_meat.get("assessment_evidence", ""),
                "treatment_evidence": final_meat.get("treatment_evidence", ""),
                "llm_reasoning": meat_data.get("llm_reasoning", ""),
            })

        # ── Apply Strict Output Filter (3 Rules) ───────────────────────
        all_results = unified_results  # keep reference for summary
        filtered_results = _apply_strict_output_filter(unified_results)

        processing_time = time.time() - start_time

        return {
            "document_id": document_id,
            "processing_time": round(processing_time, 2),
            "total_diseases": len(filtered_results),
            "total_detected": len(all_results),
            "sections_found": len(structure_result.get("sections", {})),
            "unified_results": filtered_results,
            "summary": {
                "full_meat": sum(1 for r in filtered_results if r["meat_tier"] == "full_meat"),
                "medium_meat": sum(1 for r in filtered_results if r["meat_tier"] == "medium_meat"),
                "half_meat": sum(1 for r in filtered_results if r["meat_tier"] == "half_meat"),
                "history_only": 0,
                "invalid": 0,
                "icd_mapped": sum(1 for r in filtered_results if r["icd_code"] != "—"),
            },
            "excluded_summary": {
                "total_excluded": len(all_results) - len(filtered_results),
                "history_only": sum(1 for r in all_results if r["meat_tier"] == "history_only"),
                "invalid_meat": sum(1 for r in all_results if r["meat_tier"] == "invalid"),
                "non_assessment": sum(1 for r in all_results if not ({s.lower() for s in r.get('segment_source_raw', [])} & _ASSESSMENT_SECTIONS)),
            }
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Process-all failed for {document_id}: {e}")
        error_str = str(e).lower()
        if any(x in error_str for x in ["503", "429", "unavailable", "rate limit", "quota"]):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Gemini API is currently unavailable (503/429). "
                    "All retry attempts exhausted. Please try again later."
                ),
            )
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/{document_id}/validate-and-map")
async def validate_and_map(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Run MEAT validation + ICD mapping on already-structured diseases.
    Requires: /structure endpoint must have been run first (diseases in DB).
    Returns unified result table.
    """
    import time

    # 1. Get document
    document = db.query(Document).filter(
        Document.id == document_id, Document.deleted == False
    ).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if not document.raw_text:
        raise HTTPException(status_code=400, detail="No text available. Run /extract first.")

    # 2. Get diseases from DB (created by /structure)
    diseases = db.query(DetectedDisease).filter(
        DetectedDisease.document_id == document_id
    ).all()

    if not diseases:
        raise HTTPException(
            status_code=400,
            detail="No diseases found. Run /structure endpoint first."
        )

    start_time = time.time()

    try:
        # Build disease dicts with section_sources
        disease_dicts = []
        for d in diseases:
            disease_dicts.append({
                "disease_name": d.disease_name,
                "normalized_name": d.normalized_name,
                "confidence_score": d.confidence_score,
                "negated": d.negated,
                "section": d.section,
                "sentence_number": d.sentence_number,
                "section_sources": [d.section or "unknown"]
            })

        # Build context for MEAT
        from ...utils.text_preprocessor import segment_sentences, normalize_whitespace, fix_line_breaks
        from ...utils.abbreviation_expander import expand_abbreviations

        _text = fix_line_breaks(normalize_whitespace(document.raw_text))
        _text = expand_abbreviations(_text, context_aware=True)
        fast_sentences = segment_sentences(_text)

        fast_sections: dict = {}
        for d in diseases:
            sec = d.section or "unknown"
            if sec not in fast_sections:
                fast_sections[sec] = {"text": "", "section_name": sec}
            dn_lower = d.disease_name.lower()
            for sent in fast_sentences:
                if dn_lower in sent.get("text", "").lower():
                    fast_sections[sec]["text"] += " " + sent["text"]

        # ── MEAT Validation ────────────────────────────────────────────
        meat_result = await meat_processor.process_meat_validation(
            detected_diseases=disease_dicts,
            full_text=_text,
            sentences=fast_sentences,
            sections=fast_sections
        )

        # Store MEAT results
        disease_ids = [d.id for d in diseases]
        db.query(MEATValidation).filter(
            MEATValidation.disease_id.in_(disease_ids)
        ).delete(synchronize_session=False)

        seen_disease_ids_re = set()
        for result in meat_result["meat_results"]:
            disease_record = next(
                (d for d in diseases if d.disease_name == result["disease"]),
                None
            )
            if disease_record:
                # Skip if already processed (duplicate in results)
                if disease_record.id in seen_disease_ids_re:
                    logger.info(f"MEAT already validated for {disease_record.id}, skipping...")
                    continue
                # Skip if already exists in DB
                existing = db.query(MEATValidation).filter_by(disease_id=disease_record.id).first()
                if existing:
                    logger.info(f"MEAT already exists in DB for {disease_record.id}, skipping...")
                    seen_disease_ids_re.add(disease_record.id)
                    continue
                seen_disease_ids_re.add(disease_record.id)

                meat_data = result["meat_data"]
                validation_data = result["validation"]["final_meat"]
                meat_validation = MEATValidation(
                    disease_id=disease_record.id,
                    monitoring=validation_data.get("monitoring", False),
                    monitoring_evidence=validation_data.get("monitoring_evidence", ""),
                    monitoring_confidence=validation_data.get("monitoring_confidence", 0.0),
                    evaluation=validation_data.get("evaluation", False),
                    evaluation_evidence=validation_data.get("evaluation_evidence", ""),
                    evaluation_confidence=validation_data.get("evaluation_confidence", 0.0),
                    assessment=validation_data.get("assessment", False),
                    assessment_evidence=validation_data.get("assessment_evidence", ""),
                    assessment_confidence=validation_data.get("assessment_confidence", 0.0),
                    treatment=validation_data.get("treatment", False),
                    treatment_evidence=validation_data.get("treatment_evidence", ""),
                    treatment_confidence=validation_data.get("treatment_confidence", 0.0),
                    meat_valid=result["validation"]["meat_valid"],
                    overall_confidence=validation_data.get("overall_confidence", 0.0),
                    llm_reasoning=meat_data.get("llm_reasoning", "")
                )
                db.add(meat_validation)

        db.commit()

        # ── ICD Mapping ────────────────────────────────────────────────
        diseases_with_meat = []
        for disease in diseases:
            meat = db.query(MEATValidation).filter(
                MEATValidation.disease_id == disease.id
            ).first()

            meat_data = {}
            if meat:
                meat_data = {
                    "meat_valid": meat.meat_valid,
                    "assessment_evidence": meat.assessment_evidence,
                    "treatment_evidence": meat.treatment_evidence,
                    "evaluation_evidence": meat.evaluation_evidence,
                    "monitoring_evidence": meat.monitoring_evidence,
                    "overall_confidence": meat.overall_confidence
                }

            diseases_with_meat.append({
                "disease_id": disease.id,
                "disease_name": disease.disease_name,
                "normalized_name": disease.normalized_name,
                "meat_validation": meat_data
            })

        mapping_result = await icd_mapper.map_multiple_diseases(diseases_with_meat)

        # Store ICD mappings
        db.query(ICDMapping).filter(
            ICDMapping.disease_id.in_(disease_ids)
        ).delete(synchronize_session=False)

        for i, mapping in enumerate(mapping_result["mappings"]):
            disease_id = diseases_with_meat[i]["disease_id"]
            if mapping["icd_code"]:
                icd_mapping = ICDMapping(
                    disease_id=disease_id,
                    icd_code=mapping["icd_code"],
                    icd_description=mapping["icd_description"],
                    mapping_method=mapping["mapping_method"],
                    confidence_score=mapping["confidence"],
                    status=mapping["status"],
                    llm_reasoning=mapping.get("llm_reasoning", "")
                )
                db.add(icd_mapping)

        db.commit()

        # ── Build Unified Table ────────────────────────────────────────
        unified_results = []
        meat_results_list = meat_result.get("meat_results", [])
        mappings_list = mapping_result.get("mappings", [])

        for idx, mr in enumerate(meat_results_list):
            disease_name = mr["disease"]

            icd_info = next(
                (m for m in mappings_list if m["disease"] == disease_name),
                None
            )

            seg_sources = mr.get("segment_source", [])
            segment_label = _format_segment_label(seg_sources)

            tier = mr.get("meat_tier", "invalid")
            tier_label = {
                "full_meat": "Full MEAT",
                "medium_meat": "Medium MEAT",
                "half_meat": "Half MEAT",
                "history_only": "History Only",
                "invalid": "Invalid",
            }.get(tier, "Unknown")

            meat_data = mr.get("meat_data", {})
            # Use gate-adjusted values (final_meat) for MEAT flags, not raw LLM output
            final_meat = mr.get("validation", {}).get("final_meat", meat_data)

            _mon = bool(final_meat.get("monitoring", False))
            _eva = bool(final_meat.get("evaluation", False))
            _ass = bool(final_meat.get("assessment", False))
            _tre = bool(final_meat.get("treatment", False))
            _icd_conf = icd_info["confidence"] if icd_info else 0.0
            _meat_count = sum([_mon, _eva, _ass, _tre])
            _confidence = round((_meat_count / 4) * 0.6 + _icd_conf * 0.4, 2)

            unified_results.append({
                "number": idx + 1,
                "disease": disease_name,
                "icd_code": icd_info["icd_code"] if icd_info and icd_info.get("icd_code") else "—",
                "icd_description": icd_info["icd_description"] if icd_info and icd_info.get("icd_description") else "",
                "segment": segment_label,
                "segment_source_raw": seg_sources,
                "disease_status": mr.get("disease_status", "Chronic"),
                "monitoring": _mon,
                "evaluation": _eva,
                "assessment": _ass,
                "treatment": _tre,
                "meat_level": tier_label,
                "meat_tier": tier,
                "confidence": _confidence,
                "icd_confidence": _icd_conf,
                "icd_method": icd_info["mapping_method"] if icd_info else "none",
                "monitoring_evidence": final_meat.get("monitoring_evidence", ""),
                "evaluation_evidence": final_meat.get("evaluation_evidence", ""),
                "assessment_evidence": final_meat.get("assessment_evidence", ""),
                "treatment_evidence": final_meat.get("treatment_evidence", ""),
                "llm_reasoning": meat_data.get("llm_reasoning", ""),
            })

        # ── Apply Strict Output Filter (3 Rules) ───────────────────────
        all_results = unified_results
        filtered_results = _apply_strict_output_filter(unified_results)

        processing_time = time.time() - start_time

        return {
            "document_id": document_id,
            "processing_time": round(processing_time, 2),
            "total_diseases": len(filtered_results),
            "total_detected": len(all_results),
            "unified_results": filtered_results,
            "summary": {
                "full_meat": sum(1 for r in filtered_results if r["meat_tier"] == "full_meat"),
                "medium_meat": sum(1 for r in filtered_results if r["meat_tier"] == "medium_meat"),
                "half_meat": sum(1 for r in filtered_results if r["meat_tier"] == "half_meat"),
                "history_only": 0,
                "invalid": 0,
                "icd_mapped": sum(1 for r in filtered_results if r["icd_code"] != "—"),
            },
            "excluded_summary": {
                "total_excluded": len(all_results) - len(filtered_results),
                "history_only": sum(1 for r in all_results if r["meat_tier"] == "history_only"),
                "invalid_meat": sum(1 for r in all_results if r["meat_tier"] == "invalid"),
                "non_assessment": sum(1 for r in all_results if not ({s.lower() for s in r.get('segment_source_raw', [])} & _ASSESSMENT_SECTIONS)),
            }
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Validate-and-map failed for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"MEAT+ICD processing failed: {str(e)}")


@router.get("/{document_id}/export-csv")
async def export_diseases_csv(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Export active MEAT-validated diseases as an 8-column CSV.
    Columns: Disease, ICD-10 Code, Segment, Monitor, Evaluate, Assess, Treatment, MEAT Level
    Requires: /validate-and-map (or /process-all) must have been run first.
    """
    import io
    import csv

    document = db.query(Document).filter(
        Document.id == document_id, Document.deleted == False
    ).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    diseases = db.query(DetectedDisease).filter(
        DetectedDisease.document_id == document_id
    ).all()
    if not diseases:
        raise HTTPException(
            status_code=404,
            detail="No diseases found. Run /structure and /validate-and-map first."
        )

    _TIER_LABELS = {
        4: "Full MEAT",
        3: "Medium MEAT",
        2: "Half MEAT",
        1: "Partial MEAT",
    }

    rows = []
    for d in diseases:
        meat = db.query(MEATValidation).filter(
            MEATValidation.disease_id == d.id
        ).first()
        icd = db.query(ICDMapping).filter(
            ICDMapping.disease_id == d.id
        ).first()

        # Only export MEAT-valid diseases that have an ICD code
        if not meat or not meat.meat_valid:
            continue
        icd_code = (icd.icd_code if icd and icd.icd_code else "—")
        if not icd_code or icd_code == "—":
            continue

        score = sum([
            bool(meat.monitoring),
            bool(meat.evaluation),
            bool(meat.assessment),
            bool(meat.treatment),
        ])
        if score == 0:
            continue  # No MEAT components — skip
        meat_level = _TIER_LABELS.get(score, "Partial MEAT")

        rows.append({
            "Disease": d.disease_name,
            "ICD-10 Code": icd_code,
            "Segment": (d.section or "Unknown").replace("_", " ").title(),
            "Monitor": meat.monitoring_evidence or ("Yes" if meat.monitoring else "—"),
            "Evaluate": meat.evaluation_evidence or ("Yes" if meat.evaluation else "—"),
            "Assess": meat.assessment_evidence or ("Yes" if meat.assessment else "—"),
            "Treatment": meat.treatment_evidence or ("Yes" if meat.treatment else "—"),
            "MEAT Level": meat_level,
        })

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No MEAT-validated diseases with ICD codes found. Run /validate-and-map first."
        )

    fieldnames = ["Disease", "ICD-10 Code", "Segment", "Monitor", "Evaluate", "Assess", "Treatment", "MEAT Level"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    filename_stem = Path(document.filename).stem if getattr(document, "filename", None) else f"document_{document_id}"
    csv_bytes = output.getvalue().encode("utf-8")

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename_stem}_diseases.csv"'}
    )
