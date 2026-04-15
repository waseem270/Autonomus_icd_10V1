from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import re
from ..models.enums import DocumentStatus

class DocumentUploadResponse(BaseModel):
    """Response after uploading a document."""
    document_id: str = Field(..., description="Unique UUID for the uploaded document")
    filename: str = Field(..., description="Original name of the uploaded file")
    upload_date: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the upload")
    status: str = Field(..., description="Current status of the document")
    message: str = Field("Document uploaded successfully", description="Informational message")
    
    model_config = ConfigDict(from_attributes=True)

    @field_validator('document_id')
    @classmethod
    def validate_uuid(cls, v):
        try:
            uuid.UUID(str(v))
            return str(v)
        except ValueError:
            raise ValueError("Invalid UUID format for document_id")

    @field_validator('filename')
    @classmethod
    def validate_pdf_extension(cls, v):
        if not v.lower().endswith('.pdf'):
            raise ValueError("Filename must have .pdf extension")
        return v

class TextExtractionResponse(BaseModel):
    """Response after text extraction."""
    document_id: str = Field(..., description="Unique UUID of the document")
    raw_text: str = Field(..., description="The complete extracted text content")
    page_count: int = Field(..., description="Total number of pages in the PDF")
    extraction_method: str = Field(..., description="Method used: 'digital' or 'ocr'")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="LLM/System confidence in extraction")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Calculated text quality score")
    processing_time: float = Field(..., description="Time taken to process in seconds")
    
    model_config = ConfigDict(from_attributes=True)

class DocumentDetailResponse(BaseModel):
    """Detailed document information."""
    id: str = Field(..., description="Document unique identifier")
    filename: str = Field(..., description="Original filename")
    upload_date: datetime = Field(..., description="Date document was uploaded")
    patient_id: Optional[str] = Field(None, description="Inferred or assigned Patient ID")
    raw_text: Optional[str] = Field(None, description="Extracted text content if available")
    processed: bool = Field(..., description="Whether text extraction has been performed")
    status: str = Field(..., description="Current status (uploaded, processing, completed, failed)")
    created_at: datetime = Field(..., description="Record creation timestamp")
    updated_at: datetime = Field(..., description="Record last update timestamp")
    
    model_config = ConfigDict(from_attributes=True)

class DocumentSummary(BaseModel):
    """Summary info for document list item."""
    id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Filename")
    upload_date: datetime = Field(..., description="Upload timestamp")
    status: str = Field(..., description="Current status")
    processed: bool = Field(..., description="Processing flag")
    
    model_config = ConfigDict(from_attributes=True)

class DocumentListResponse(BaseModel):
    """Paginated list of documents."""
    total: int = Field(..., description="Total documents matching the filter")
    skip: int = Field(..., description="Offset used for pagination")
    limit: int = Field(..., description="Page size used for pagination")
    documents: List[DocumentSummary] = Field(..., description="List of document summaries")

class PDFTypeInfo(BaseModel):
    """PDF type detection info."""
    type: str = Field(..., description="Detected type: digital, scanned, or hybrid")
    page_count: int = Field(..., description="Total pages")
    extractable_chars: int = Field(..., description="Total characters extracted during detection")
    avg_chars_per_page: float = Field(..., description="Average characters per page")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    needs_ocr: bool = Field(..., description="Whether OCR is recommended for this document")

class DetectedDiseaseSchema(BaseModel):
    """Schema for detected disease entity."""
    disease_name: str = Field(..., description="Original text of the disease/condition")
    normalized_name: str = Field(..., description="Normalized version of the name")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="NLP confidence score")
    negated: bool = Field(..., description="Whether the condition is negated")
    section: Optional[str] = Field(None, description="Section where found")
    section_sources: Optional[List[str]] = Field(None, description="All sections this disease was found in")
    sentence_number: Optional[int] = Field(None, description="Sentence index")
    entity_type: str = Field(..., description="Type of entity (DISEASE/CONDITION)")
    start_char: Optional[int] = Field(None, description="Starting character position")
    end_char: Optional[int] = Field(None, description="Ending character position")
    
    model_config = ConfigDict(from_attributes=True)

class SectionInfo(BaseModel):
    """Schema for clinical note section."""
    text: str = Field(..., description="Section content text")
    start: int = Field(..., description="Start character index")
    end: int = Field(..., description="End character index")

    model_config = ConfigDict(from_attributes=True)

class TextStructureResponse(BaseModel):
    """Response after text structuring."""
    document_id: str = Field(..., description="Document ID")
    total_sentences: int = Field(..., description="Total sentences analyzed")
    total_diseases: int = Field(..., description="Total diseases detected")
    sections_found: int = Field(..., description="Number of clinical sections identified")
    detected_diseases: List[DetectedDiseaseSchema] = Field(..., description="List of detected disease entities")
    sections: Dict[str, SectionInfo] = Field(..., description="Detected sections and their metadata")
    processing_stats: Dict[str, Any] = Field(..., description="Execution statistics")
    
    model_config = ConfigDict(from_attributes=True)

class SentenceInfo(BaseModel):
    """Schema for sentence metadata."""
    sentence_number: int = Field(..., description="Index of the sentence")
    text: str = Field(..., description="Sentence text")
    start_char: int = Field(..., description="Start character index")
    end_char: int = Field(..., description="End character index")
    length: int = Field(..., description="Length of the sentence")

    model_config = ConfigDict(from_attributes=True)

class MEATElementData(BaseModel):
    """MEAT element validation data."""
    present: bool
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)

class MEATValidationData(BaseModel):
    """Complete MEAT validation for a disease."""
    disease: str
    monitoring: bool
    monitoring_evidence: str
    monitoring_confidence: float
    evaluation: bool
    evaluation_evidence: str
    evaluation_confidence: float
    assessment: bool
    assessment_evidence: str
    assessment_confidence: float
    treatment: bool
    treatment_evidence: str
    treatment_confidence: float
    overall_confidence: float
    llm_reasoning: str

class MEATValidationResult(BaseModel):
    """Validation result with gate checks."""
    meat_valid: bool
    validation_rule: str
    confidence_tier: str
    requires_manual_review: bool
    issues_found: List[str]
    final_meat: Optional[Dict[str, Any]] = None

class MEATDiseaseResult(BaseModel):
    """Combined MEAT result for a disease."""
    disease: str
    meat_data: MEATValidationData
    validation: MEATValidationResult
    final_status: str  # "valid", "invalid", "review", "history_only"
    meat_tier: Optional[str] = None  # "strong_evidence", "moderate_evidence", "weak_evidence", "no_meat"
    meat_status: Optional[str] = None  # Human-readable status message
    evidence_based: Optional[bool] = None  # True if all MEAT components backed by document evidence
    disease_status: Optional[str] = None  # "Active Chronic", "Active Acute", "Chronic", "History Only"
    meat_criteria_count: Optional[int] = None
    segment_source: Optional[List[str]] = None
    context_summary: Dict[str, Any]

class MEATValidationResponse(BaseModel):
    """Response after MEAT validation."""
    document_id: str
    total_diseases: int
    valid_meat_count: int
    requires_review_count: int
    meat_results: List[MEATDiseaseResult]
    processing_summary: Dict[str, Any]
    
    model_config = ConfigDict(from_attributes=True)

class ICDCandidate(BaseModel):
    """ICD code candidate."""
    icd_code: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    match_type: str
    llm_rank: Optional[int] = None
    llm_score: Optional[float] = None
    llm_reasoning: Optional[str] = None

class ICDMappingResult(BaseModel):
    """ICD mapping result for a disease."""
    disease: str
    icd_code: Optional[str]
    icd_description: Optional[str]
    mapping_method: str
    confidence: float = Field(ge=0.0, le=1.0)
    status: str  # "auto_assigned", "manual_review", "not_found"
    candidates: List[ICDCandidate]
    llm_reasoning: Optional[str]

class ICDMappingResponse(BaseModel):
    """Response after ICD mapping."""
    document_id: str
    total_diseases: int
    auto_assigned: int
    manual_review: int
    not_found: int
    mappings: List[ICDMappingResult]
    
    class Config:
        from_attributes = True
