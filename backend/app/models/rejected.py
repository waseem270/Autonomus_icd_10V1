import uuid
from typing import Optional
from sqlalchemy import String, Text, Enum as SQLEnum, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base, TimestampMixin
from .enums import RejectionReason

class RejectedDiagnosis(Base, TimestampMixin):
    __tablename__ = "rejected_diagnoses"

    document_id: Mapped[str] = mapped_column(
        String(36), 
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False
    )
    disease_name: Mapped[str] = mapped_column(String(255), nullable=False)
    rejection_reason: Mapped[RejectionReason] = mapped_column(SQLEnum(RejectionReason), nullable=False)
    section: Mapped[Optional[str]] = mapped_column(String(100))
    evidence: Mapped[Optional[str]] = mapped_column(Text)
    sentence_number: Mapped[Optional[int]] = mapped_column()

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="rejected_diagnoses")

    def __repr__(self) -> str:
        return f"<RejectedDiagnosis(id={self.id}, name='{self.disease_name}', reason='{self.rejection_reason}')>"
