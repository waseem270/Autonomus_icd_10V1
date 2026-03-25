import uuid
from typing import Optional
from sqlalchemy import Float, Text, Boolean, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base, TimestampMixin

class MEATValidation(Base, TimestampMixin):
    __tablename__ = "meat_validation"

    disease_id: Mapped[str] = mapped_column(
        String(36), 
        ForeignKey("detected_diseases.id", ondelete="CASCADE"),
        unique=True,
        nullable=False
    )
    
    monitoring: Mapped[bool] = mapped_column(Boolean, default=False)
    monitoring_evidence: Mapped[Optional[str]] = mapped_column(Text)
    monitoring_confidence: Mapped[Optional[float]] = mapped_column(Float)
    
    evaluation: Mapped[bool] = mapped_column(Boolean, default=False)
    evaluation_evidence: Mapped[Optional[str]] = mapped_column(Text)
    evaluation_confidence: Mapped[Optional[float]] = mapped_column(Float)
    
    assessment: Mapped[bool] = mapped_column(Boolean, default=False)
    assessment_evidence: Mapped[Optional[str]] = mapped_column(Text)
    assessment_confidence: Mapped[Optional[float]] = mapped_column(Float)
    
    treatment: Mapped[bool] = mapped_column(Boolean, default=False)
    treatment_evidence: Mapped[Optional[str]] = mapped_column(Text)
    treatment_confidence: Mapped[Optional[float]] = mapped_column(Float)
    
    meat_valid: Mapped[bool] = mapped_column(Boolean, nullable=False)
    overall_confidence: Mapped[Optional[float]] = mapped_column(Float)
    llm_reasoning: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    disease: Mapped["DetectedDisease"] = relationship(back_populates="meat_validation")

    def __repr__(self) -> str:
        return f"<MEATValidation(id={self.id}, valid={self.meat_valid})>"
