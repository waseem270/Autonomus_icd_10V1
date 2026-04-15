import uuid
from typing import Optional
from datetime import datetime
from sqlalchemy import String, Float, Text, Enum as SQLEnum, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base, TimestampMixin
from .enums import MappingMethod, MappingStatus

class ICDMapping(Base, TimestampMixin):
    __tablename__ = "icd_mappings"

    disease_id: Mapped[str] = mapped_column(
        String(36), 
        ForeignKey("detected_diseases.id", ondelete="CASCADE"),
        nullable=False
    )
    icd_code: Mapped[str] = mapped_column(String(10), nullable=False)
    icd_description: Mapped[Optional[str]] = mapped_column(String(500))
    mapping_method: Mapped[MappingMethod] = mapped_column(SQLEnum(MappingMethod), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[MappingStatus] = mapped_column(
        SQLEnum(MappingStatus), 
        default=MappingStatus.PENDING
    )
    llm_reasoning: Mapped[Optional[str]] = mapped_column(Text)
    reviewed_by: Mapped[Optional[str]] = mapped_column(String(100))
    reviewed_at: Mapped[Optional[datetime]] = mapped_column()

    # Relationships
    disease: Mapped["DetectedDisease"] = relationship(back_populates="icd_mappings")

    __table_args__ = (
        Index("ix_icd_mappings_disease_id_code_status", "disease_id", "icd_code", "status"),
    )

    def __repr__(self) -> str:
        return f"<ICDMapping(id={self.id}, code='{self.icd_code}', status='{self.status}')>"
