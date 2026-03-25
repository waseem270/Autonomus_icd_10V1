import uuid
from typing import List, Optional
from sqlalchemy import String, Float, Boolean, Enum as SQLEnum, ForeignKey, Index, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base, TimestampMixin
from .enums import TemporalStatus

class DetectedDisease(Base, TimestampMixin):
    __tablename__ = "detected_diseases"

    document_id: Mapped[str] = mapped_column(
        String(36), 
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False
    )
    disease_name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(255), index=True)
    confidence_score: Mapped[float] = mapped_column(Float)
    negated: Mapped[bool] = mapped_column(Boolean, default=False)
    temporal_status: Mapped[TemporalStatus] = mapped_column(
        SQLEnum(TemporalStatus), 
        default=TemporalStatus.CURRENT
    )
    section: Mapped[Optional[str]] = mapped_column(String(100))
    sentence_number: Mapped[Optional[int]] = mapped_column()

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="detected_diseases")
    meat_validation: Mapped[Optional["MEATValidation"]] = relationship(
        back_populates="disease", 
        uselist=False, 
        cascade="all, delete-orphan"
    )
    icd_mappings: Mapped[List["ICDMapping"]] = relationship(
        back_populates="disease", 
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("confidence_score >= 0.0 AND confidence_score <= 1.0"),
        Index("ix_detected_diseases_doc_id_norm_name", "document_id", "normalized_name"),
    )

    def __repr__(self) -> str:
        return f"<DetectedDisease(id={self.id}, name='{self.disease_name}', score={self.confidence_score})>"
