import logging
import time
from urllib.parse import quote_plus, urlparse, urlunparse

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from .config import settings
from ..models.base import Base

logger = logging.getLogger(__name__)


def _safe_url(url: str) -> str:
    """
    Re-encode the password portion of a database URL so special characters
    (%  ?  @  # etc.) do not break SQLAlchemy's URL parser.
    """
    if "://" not in url:
        return url
    parsed = urlparse(url)
    if parsed.password:
        safe_pass = quote_plus(parsed.password)
        # Rebuild netloc: user:encoded_pass@host:port
        host_part = parsed.hostname or ""
        if parsed.port:
            host_part += f":{parsed.port}"
        netloc = f"{parsed.username}:{safe_pass}@{host_part}"
        return urlunparse(parsed._replace(netloc=netloc))
    return url


def _build_engine(max_retries: int = 3, retry_delay: int = 2):
    """Create the SQLAlchemy engine based on DATABASE_URL with retry logic."""
    url = settings.DATABASE_URL
    is_sqlite = url.startswith("sqlite")

    connect_args = {}
    pool_kwargs = {}

    if is_sqlite:
        connect_args["check_same_thread"] = False
    else:
        # Ensure psycopg2 driver is specified for PostgreSQL
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        # Re-encode password to handle special chars
        url = _safe_url(url)
        pool_kwargs.update(
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
        )

    eng = create_engine(
        url,
        connect_args=connect_args,
        pool_pre_ping=True,
        echo=False,  # Do NOT echo SQL in production pipeline runs
        **pool_kwargs,
    )

    # Retry connection for PostgreSQL
    if not is_sqlite:
        for attempt in range(1, max_retries + 1):
            try:
                with eng.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info(
                    "PostgreSQL connected",
                    extra={"attempt": attempt, "host": urlparse(url).hostname},
                )
                break
            except Exception as e:
                logger.warning(
                    f"DB connection attempt {attempt}/{max_retries} failed: {e}"
                )
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Cannot connect to PostgreSQL after {max_retries} attempts: {e}"
                    ) from e
                time.sleep(retry_delay)
    else:
        logger.info("SQLite engine created", extra={"path": url})

    return eng


engine = _build_engine()

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def create_tables():
    """Create all database tables."""
    from .. import models  # noqa: F401  — register all models with Base
    Base.metadata.create_all(bind=engine)
    # Log which tables exist
    table_names = list(Base.metadata.tables.keys())
    logger.info("Tables created/verified", extra={"tables": table_names})


def get_db() -> Generator[Session, None, None]:
    """Database session dependency — single source of truth."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
