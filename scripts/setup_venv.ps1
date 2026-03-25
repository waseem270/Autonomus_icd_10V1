# Medical ICD Mapper - Setup Virtual Environment
# This script initializes the Python environment on Windows

Write-Host "--- Medical ICD Mapper Setup ---" -ForegroundColor Cyan

# 1. Check Python version (Must be 3.10.x)
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "Python 3\.10") {
    Write-Host "Check: Python 3.10 detected. [OK]" -ForegroundColor Green
} else {
    Write-Host "Error: Python 3.10 is required. Found: $pythonVersion" -ForegroundColor Red
    Write-Host "Please install Python 3.10 from python.org"
    exit 1
}

# 2. Create virtual environment
if (!(Test-Path .\venv)) {
    Write-Host "Creating virtual environment 'venv'..." -ForegroundColor Yellow
    python -m venv venv
} else {
    Write-Host "Virtual environment 'venv' already exists. [Skipping]" -ForegroundColor Gray
}

# 3. Activate virtual environment
Write-Host "Activating environment..." -ForegroundColor Yellow
. .\venv\Scripts\Activate.ps1

# 4. Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 5. Install backend requirements
Write-Host "Installing Backend dependencies..." -ForegroundColor Yellow
pip install -r backend/requirements.txt

# 6. Install frontend requirements
Write-Host "Installing Frontend dependencies..." -ForegroundColor Yellow
pip install -r frontend/requirements.txt

# 7. Install dev requirements
Write-Host "Installing Development dependencies..." -ForegroundColor Yellow
pip install -r requirements-dev.txt

# 8. Download scispaCy medical model
Write-Host "Installing scispaCy model (en_core_sci_md v0.5.3)..." -ForegroundColor Yellow
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz

# 9. Create necessary folders
Write-Host "Creating project directories (logs, uploads, database)..." -ForegroundColor Yellow
if (!(Test-Path .\logs)) { New-Item -ItemType Directory -Path .\logs | Out-Null }
if (!(Test-Path .\uploads)) { New-Item -ItemType Directory -Path .\uploads | Out-Null }
if (!(Test-Path .\database)) { New-Item -ItemType Directory -Path .\database | Out-Null }

# 10. Success Message
Write-Host "`n--- Setup Complete! ---" -ForegroundColor Green
Write-Host "To start working:"
Write-Host "1. Activate:   .\scripts\activate.ps1"
Write-Host "2. Run Backend: uvicorn backend.main:app --reload"
Write-Host "3. Run Frontend: streamlit run frontend/app.py"
