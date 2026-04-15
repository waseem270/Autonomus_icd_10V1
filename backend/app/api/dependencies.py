"""
API dependencies — re-exports the canonical DB session from core.database.

All routes should use ``get_db`` from this module (or directly from
``backend.app.core.database``).  There is intentionally only ONE engine
and ONE SessionLocal for the entire application.
"""

from ..core.database import get_db, create_tables  # noqa: F401
