"""
Document API routes — thin controller layer.

All business logic is delegated to ``pipeline_orchestrator``.
This file handles HTTP concerns only: request parsing, DB lookup, error mapping.
"""

import io
import csv
import logging
import uuid
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ...core.config import settings
from ..dependencies import get_db
from ...models.document import Document
from ...models.disease import DetectedDisease
from ...models.enums import DocumentStatus
from ...models.meat import MEATValidation
from ...models.mapping import ICDMapping
from ...schemas.document import (
    DocumentUploadResponse,
    TextExtractionResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    TextStructureResponse,
    MEATValidationResponse,
    ICDMappingResponse,
)
from ...services.pipeline_orchestrator import pipeline_orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


# ═══════════════════════════════════════════════════════════════════════════
# Upload
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    content = await file.read()
    if len(content) > settings.UPLOAD_MAX_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.UPLOAD_MAX_SIZE / (1024*1024):.0f}MB.",
        )
    await file.seek(0)

    upload_dir = Path(settings.UPLOAD_FOLDER)
    upload_dir.mkdir(parents=True, exist_ok=True)

    doc_id = str(uuid.uuid4())
    safe_filename = f"{doc_id}.pdf"
    file_path = upload_dir / safe_filename

    try:
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail="Could not save file.")

    new_doc = Document(
        id=doc_id,
        filename=file.filename,
        file_path=str(file_path),
        status=DocumentStatus.UPLOADED,
        processed=False,
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)

    return DocumentUploadResponse(
        document_id=new_doc.id,
        filename=new_doc.filename,
        status=new_doc.status.value,
        message="Document uploaded successfully.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Extract
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/{document_id}/extract", response_model=TextExtractionResponse)
async def extract_document_text(document_id: str, db: Session = Depends(get_db)):
    doc = _get_document(document_id, db)
    if not doc.file_path or not Path(doc.file_path).exists():
        raise HTTPException(status_code=404, detail="Physical file missing on server.")
    try:
        return await pipeline_orchestrator.extract_text(doc, db)
    except Exception as e:
        logger.error(f"Extraction failed for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Structure
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/{document_id}/structure", response_model=TextStructureResponse)
async def structure_document_text(document_id: str, db: Session = Depends(get_db)):
    doc = _get_document(document_id, db)
    _require_text(doc)
    try:
        result = await pipeline_orchestrator.structure_text(doc, db)
        return {
            "document_id": document_id,
            "total_sentences": result["total_sentences"],
            "total_diseases": result["total_diseases"],
            "sections_found": len(result["sections"]),
            "detected_diseases": result["detected_diseases"],
            "sections": result["sections"],
            "processing_stats": result["processing_stats"],
        }
    except Exception as e:
        db.rollback()
        _raise_service_error("Structuring", document_id, e)


# ═══════════════════════════════════════════════════════════════════════════
# MEAT Validation
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/{document_id}/validate-meat", response_model=MEATValidationResponse)
async def validate_meat_for_document(document_id: str, db: Session = Depends(get_db)):
    doc = _get_document(document_id, db)
    try:
        result = await pipeline_orchestrator.validate_meat(doc, db)
        return {
            "document_id": document_id,
            "total_diseases": result["total_diseases"],
            "valid_meat_count": result["valid_meat_count"],
            "requires_review_count": result["requires_review_count"],
            "meat_results": result["meat_results"],
            "processing_summary": result["processing_summary"],
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        db.rollback()
        _raise_service_error("MEAT validation", document_id, e)


# ═══════════════════════════════════════════════════════════════════════════
# ICD Mapping
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/{document_id}/map-icd", response_model=ICDMappingResponse)
async def map_icd_codes(document_id: str, db: Session = Depends(get_db)):
    doc = _get_document(document_id, db)
    try:
        result = await pipeline_orchestrator.map_icd(doc, db)
        return ICDMappingResponse(
            document_id=document_id,
            total_diseases=result["total_diseases"],
            auto_assigned=result["auto_assigned"],
            manual_review=result["manual_review"],
            not_found=result["not_found"],
            mappings=result["mappings"],
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        db.rollback()
        _raise_service_error("ICD mapping", document_id, e)


# ═══════════════════════════════════════════════════════════════════════════
# Process All (full pipeline)
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/{document_id}/process-all")
async def process_all_steps(
    document_id: str,
    dietary_analysis: bool = Query(default=False, description="Include dietary analysis"),
    db: Session = Depends(get_db),
):
    doc = _get_document(document_id, db)
    _require_text(doc)
    try:
        return await pipeline_orchestrator.process_all(doc, db, dietary_analysis=dietary_analysis)
    except Exception as e:
        db.rollback()
        _raise_service_error("Processing", document_id, e)


# ═══════════════════════════════════════════════════════════════════════════
# Validate + Map (MEAT + ICD on already-structured diseases)
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/{document_id}/validate-and-map")
async def validate_and_map(document_id: str, db: Session = Depends(get_db)):
    doc = _get_document(document_id, db)
    _require_text(doc)

    diseases = db.query(DetectedDisease).filter(
        DetectedDisease.document_id == document_id
    ).all()
    if not diseases:
        raise HTTPException(
            status_code=400, detail="No diseases found. Run /structure first."
        )

    try:
        meat_result = await pipeline_orchestrator.validate_meat(doc, db)
        mapping_result = await pipeline_orchestrator.map_icd(doc, db)

        from ...services.output_filter import output_filter
        from ...services.confidence_scorer import confidence_scorer

        unified = pipeline_orchestrator._build_unified_table(meat_result, mapping_result)
        filtered = output_filter.apply(unified)
        scored = confidence_scorer.score_batch(filtered)

        return pipeline_orchestrator._build_response(
            doc, scored, {}, 0.0,
            source="validate_and_map",
            all_results=unified,
        )
    except Exception as e:
        db.rollback()
        _raise_service_error("Validate-and-map", document_id, e)


# ═══════════════════════════════════════════════════════════════════════════
# Read endpoints
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(document_id: str, db: Session = Depends(get_db)):
    return _get_document(document_id, db)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    skip: int = 0,
    limit: int = 20,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    query = db.query(Document).filter(Document.deleted == False)  # noqa: E712
    if status:
        query = query.filter(Document.status == status)
    total = query.count()
    items = query.offset(skip).limit(min(limit, 100)).all()
    return DocumentListResponse(total=total, skip=skip, limit=limit, documents=items)


# ═══════════════════════════════════════════════════════════════════════════
# CSV Export
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/{document_id}/export-csv")
async def export_diseases_csv(document_id: str, db: Session = Depends(get_db)):
    doc = _get_document(document_id, db)
    diseases = db.query(DetectedDisease).filter(
        DetectedDisease.document_id == document_id
    ).all()
    if not diseases:
        raise HTTPException(status_code=404, detail="No diseases found.")

    _TIER_LABELS = {4: "Full MEAT", 3: "Medium MEAT", 2: "Half MEAT", 1: "Partial MEAT"}
    rows = []
    for d in diseases:
        meat = db.query(MEATValidation).filter(MEATValidation.disease_id == d.id).first()
        icd = db.query(ICDMapping).filter(ICDMapping.disease_id == d.id).first()
        if not meat or not meat.meat_valid:
            continue
        icd_code = icd.icd_code if icd and icd.icd_code else "—"
        if not icd_code or icd_code == "—":
            continue
        score = sum([bool(meat.monitoring), bool(meat.evaluation), bool(meat.assessment), bool(meat.treatment)])
        if score == 0:
            continue
        rows.append({
            "Disease": d.disease_name,
            "ICD-10 Code": icd_code,
            "Segment": (d.section or "Unknown").replace("_", " ").title(),
            "Monitor": meat.monitoring_evidence or ("Yes" if meat.monitoring else "—"),
            "Evaluate": meat.evaluation_evidence or ("Yes" if meat.evaluation else "—"),
            "Assess": meat.assessment_evidence or ("Yes" if meat.assessment else "—"),
            "Treatment": meat.treatment_evidence or ("Yes" if meat.treatment else "—"),
            "MEAT Level": _TIER_LABELS.get(score, "Partial MEAT"),
        })

    if not rows:
        raise HTTPException(status_code=404, detail="No exportable diseases found.")

    fieldnames = ["Disease", "ICD-10 Code", "Segment", "Monitor", "Evaluate", "Assess", "Treatment", "MEAT Level"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    stem = Path(doc.filename).stem if doc.filename else f"document_{document_id}"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{stem}_diseases.csv"'},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_document(document_id: str, db: Session) -> Document:
    doc = db.query(Document).filter(
        Document.id == document_id, Document.deleted == False  # noqa: E712
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


def _require_text(doc: Document):
    if not doc.raw_text:
        raise HTTPException(status_code=400, detail="No text available. Run /extract first.")


def _raise_service_error(stage: str, document_id: str, exc: Exception):
    logger.error(f"{stage} failed for {document_id}: {exc}")
    error_str = str(exc).lower()
    if any(x in error_str for x in ("503", "429", "unavailable", "rate limit", "quota")):
        raise HTTPException(
            status_code=503,
            detail=f"Gemini API temporarily unavailable. Please try again later.",
        )
    raise HTTPException(status_code=500, detail=f"{stage} failed: {exc}")
