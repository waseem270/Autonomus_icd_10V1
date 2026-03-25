from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
from pathlib import Path
import os

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATABASE_DIR = BASE_DIR / "database"
UPLOADS_DIR = BASE_DIR / "uploads"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATABASE_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

class Settings(BaseSettings):
    """Application settings loaded from .env file."""
    
    # API Configuration
    API_VERSION: str = "v1"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = True
    
    # Database - Absolute path
    DATABASE_URL: str = f"sqlite:///{DATABASE_DIR}/medical_icd.db"
    
    # CORS
    CORS_ORIGINS: str = "http://localhost:8501,http://localhost:3000"
    
    # File Upload - Absolute path
    UPLOAD_MAX_SIZE: int = 26214400  # 25MB in bytes
    UPLOAD_FOLDER: str = str(UPLOADS_DIR)
    ALLOWED_EXTENSIONS: str = ".pdf"
    
    # Gemini / Vertex AI
    GEMINI_API_KEY: Optional[str] = None
    GCP_PROJECT: str = "gen-lang-client-0574304931"
    GCP_LOCATION: str = "us-central1"
    GEMINI_MODEL: str = "gemini-3-flash-preview"
    GEMINI_MAX_TOKENS: int = 8000
    GEMINI_TEMPERATURE: float = 0.5
    GEMINI_TOP_P: float = 0.0
    GEMINI_TIMEOUT: int = 60
    GEMINI_MAX_RETRIES: int = 3
    GEMINI_RETRY_DELAY: int = 2
    
    # OCR Configuration
    TESSERACT_CMD: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    OCR_LANGUAGE: str = "eng"
    OCR_CONFIG: str = "--psm 6"
    PDF2IMAGE_POPPLLER_PATH: str = ""
    
    # NLP
    SCISPACY_MODEL: str = "en_core_sci_md"
    NEGSPACY_ENABLED: bool = True
    DISEASE_CONFIDENCE_THRESHOLD: float = 0.60  # Raised from 0.70 — now meaningful (was passing everything at 0.75 base)
    
    # MEAT Validation
    MEAT_CONFIDENCE_HIGH: float = 0.85
    MEAT_CONFIDENCE_MEDIUM: float = 0.65
    MEAT_REQUIRE_EVIDENCE: bool = True
    
    # ICD Mapping
    ICD_EXACT_MATCH_PRIORITY: bool = False  # Allow high-confidence AI matches to auto-assign
    ICD_FUZZY_THRESHOLD: float = 0.70  # Lowered from 0.85 — multi-strategy search recasts broader net
    ICD_AUTO_ASSIGN_THRESHOLD: float = 0.90  # New: auto-assign if confidence >= 0.90
    ICD_MANUAL_REVIEW_THRESHOLD: float = 0.70
    
    # Logging - Absolute path
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = str(LOGS_DIR / "app.log")
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
    # Determine .env path: prefer root .env, fall back to config/.env
    _root_env = os.path.join(BASE_DIR, ".env")
    _config_env = os.path.join(BASE_DIR, "config", ".env")
    _env_path = _root_env if os.path.isfile(_root_env) else _config_env

    model_config = SettingsConfigDict(
        env_file=_env_path,
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Create global settings instance
settings = Settings()

# Cached Gemini client — created lazily on first use
_genai_client = None
_genai_client_initialized = False


def create_genai_client():
    """Create or return the cached google.genai Client using API key."""
    global _genai_client, _genai_client_initialized
    
    if _genai_client_initialized:
        return _genai_client
    
    from google import genai
    from dotenv import load_dotenv
    import logging

    # Ensure .env vars are in os.environ (pydantic doesn't always export them)
    load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)

    _logger = logging.getLogger(__name__)

    if settings.GEMINI_API_KEY:
        try:
            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            _logger.info("Gemini client initialised via API key.")
            _genai_client = client
            _genai_client_initialized = True
            return client
        except Exception as e:
            _logger.error(f"API key init failed: {e}")

    _logger.error("No GEMINI_API_KEY configured.")
    _genai_client_initialized = True
    return None
