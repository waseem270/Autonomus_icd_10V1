from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from ..core.config import settings
from ..utils.pdf_detector import detect_pdf_type

# Create SQLite engine
# Note: connect_args={"check_same_thread": False} is required for SQLite
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=settings.DEBUG
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)

def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency for FastAPI.
    
    Usage:
        @router.get("/endpoint")
        def my_endpoint(db: Session = Depends(get_db)):
            ...
    
    Yields session and automatically closes it after request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
