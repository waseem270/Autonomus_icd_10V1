import uuid
from typing import Optional
from datetime import datetime, timezone
from sqlalchemy import String, Text, ForeignKey, Index, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base, TimestampMixin

class AuditLog(Base, TimestampMixin):
    __tablename__ = "audit_log"

    # id, created_at, updated_at from mixin, 
    # but the request asks for 'timestamp' indexed, default now
    # We'll use created_at as the primary timestamp source but fulfill the requirement:
    # timestamp: Mapped[datetime] = mapped_column(index=True, default=datetime.utcnow)
    # However, since TimestampMixin already has created_at, I'll add timestamp explicitly as requested.
    
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        index=True, 
        default=lambda: datetime.now(timezone.utc)
    )

    document_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True
    )
    action: Mapped[str] = mapped_column(String(255), nullable=False)
    user_id: Mapped[Optional[str]] = mapped_column(String(100))
    changes: Mapped[Optional[str]] = mapped_column(Text)
    reasoning: Mapped[Optional[str]] = mapped_column(Text)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))

    # Relationships
    document: Mapped[Optional["Document"]] = relationship(back_populates="audit_logs")

    __table_args__ = (
        Index("ix_audit_log_timestamp_doc_id", "timestamp", "document_id"),
    )

    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, action='{self.action}', user='{self.user_id}')>"
