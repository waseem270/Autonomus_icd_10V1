from typing import List, Optional
from datetime import datetime
from sqlalchemy import String, Text, Boolean, Enum as SQLEnum, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base, TimestampMixin
from .enums import DocumentStatus

class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=True)
    upload_date: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    patient_id: Mapped[Optional[str]] = mapped_column(String(50))
    raw_text: Mapped[Optional[str]] = mapped_column(Text)
    page_count: Mapped[Optional[int]] = mapped_column(default=0)
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[DocumentStatus] = mapped_column(
        SQLEnum(DocumentStatus), 
        default=DocumentStatus.UPLOADED
    )
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    detected_diseases: Mapped[List["DetectedDisease"]] = relationship(
        back_populates="document", 
        cascade="all, delete-orphan"
    )
    rejected_diagnoses: Mapped[List["RejectedDiagnosis"]] = relationship(
        back_populates="document", 
        cascade="all, delete-orphan"
    )
    audit_logs: Mapped[List["AuditLog"]] = relationship(
        back_populates="document"
    )

    __table_args__ = (
        Index("ix_documents_status_processed_deleted", "status", "processed", "deleted"),
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.status}')>"
