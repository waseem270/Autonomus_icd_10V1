from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path

from .core.config import settings
from .core.database import create_tables
from .api.routes import documents
import logging
import sys

# Configure logging (UTF-8 to handle Unicode in clinical text on Windows)
_log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
_stream_handler = logging.StreamHandler(
    stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)
)
_stream_handler.setFormatter(logging.Formatter(_log_fmt))

# Ensure log directory exists before file handler is created
log_path = Path(settings.LOG_FILE)
log_path.parent.mkdir(parents=True, exist_ok=True)

_file_handler = logging.FileHandler(settings.LOG_FILE, encoding='utf-8')
_file_handler.setFormatter(logging.Formatter(_log_fmt))
logging.basicConfig(
    level=logging.INFO,
    format=_log_fmt,
    handlers=[_stream_handler, _file_handler]
)
logger = logging.getLogger(__name__)

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown events.
    """
    # Startup
    print("🚀 Starting Medical ICD Mapper API...")
    
    # 1. Create database tables
    try:
        create_tables()
        print("✅ Database tables created/verified")
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")

    # 2. Create required folders
    required_folders = [settings.UPLOAD_FOLDER, "logs"]
    for folder in required_folders:
        Path(folder).mkdir(exist_ok=True)
    print("✅ System folders ready")
    
    print(f"📚 Database URL: {settings.DATABASE_URL}")
    print(f"📂 Uploads directory: {settings.UPLOAD_FOLDER}")
    
    yield
    
    # Shutdown
    print("👋 Shutting down Medical ICD Mapper API...")

# Create FastAPI app
app = FastAPI(
    title="Medical ICD Mapper API",
    description="Extract diseases from prescriptions and map to ICD-10 codes",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
# Note: uses .split(",") to convert string from config to list
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    documents.router,
    prefix=settings.API_PREFIX
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """API health check."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "database": "connected"
    }

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Medical ICD Mapper API",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }
