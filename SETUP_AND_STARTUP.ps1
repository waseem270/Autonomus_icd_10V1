# ═══════════════════════════════════════════════════════════════════════════
# Medical ICD Mapper - Complete Setup & Startup Script
# ═══════════════════════════════════════════════════════════════════════════
# Purpose: Fully automated setup, configuration, and startup of the project
# Platform: Windows PowerShell 5.1+
# Usage: .\SETUP_AND_STARTUP.ps1
# ═══════════════════════════════════════════════════════════════════════════

param(
    [Parameter(Mandatory = $false)]
    [ValidateSet("setup", "run", "all", "clean", "help")]
    [string]$Mode = "all",
    
    [Parameter(Mandatory = $false)]
    [switch]$SkipVenv,
    
    [Parameter(Mandatory = $false)]
    [switch]$SkipDeps,
    
    [Parameter(Mandatory = $false)]
    [switch]$Verbose
)

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

$SCRIPT_NAME = "Medical ICD Mapper Setup"
$SCRIPT_VERSION = "1.0.0"
$PROJECT_ROOT = Get-Location
$VENV_PATH = Join-Path $PROJECT_ROOT "venv"
$BACKEND_PORT = 8000
$FRONTEND_PORT = 8501
$PYTHON_MIN_VERSION = "3.10"

# Color codes for terminal output
$Colors = @{
    "Reset"   = "`e[0m"
    "Bold"    = "`e[1m"
    "Success" = "`e[32m"  # Green
    "Error"   = "`e[31m"  # Red
    "Warning" = "`e[33m"  # Yellow
    "Info"    = "`e[36m"  # Cyan
    "Header"  = "`e[35m"  # Magenta
}

# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

function Write-ColorOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message,
        
        [Parameter(Mandatory = $false)]
        [ValidateSet("Success", "Error", "Warning", "Info", "Header", "Reset")]
        [string]$Color = "Reset"
    )
    
    Write-Host "$($Colors[$Color])$Message$($Colors['Reset'])"
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-ColorOutput "═" Header
    Write-ColorOutput "  $Title" Header
    Write-ColorOutput "═" Header
    Write-Host ""
}

function Write-Step {
    param([string]$Step)
    Write-ColorOutput "→ $Step" Info
}

function Test-ExecutableExists {
    param([string]$Command)
    try {
        $null = Get-Command $Command -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function Test-PythonVersion {
    param([string]$PythonPath = "python")
    try {
        $version = & $PythonPath --version 2>&1
        if ($version -match "(\d+\.\d+)") {
            $versionNum = [version]$matches[1]
            $minVersion = [version]$PYTHON_MIN_VERSION
            if ($versionNum -ge $minVersion) {
                return $true
            }
        }
        return $false
    } catch {
        return $false
    }
}

function Test-FileExists {
    param([string]$Path)
    return Test-Path $Path -PathType Leaf
}

function Test-DirectoryExists {
    param([string]$Path)
    return Test-Path $Path -PathType Container
}

# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

function Validate-Prerequisites {
    Write-Section "Checking Prerequisites"
    
    $allValid = $true
    
    # Check Python
    Write-Step "Checking Python installation..."
    if (Test-ExecutableExists "python") {
        $version = & python --version 2>&1
        Write-ColorOutput "✓ Python found: $version" Success
        if (-not (Test-PythonVersion)) {
            Write-ColorOutput "✗ Python version must be $PYTHON_MIN_VERSION or higher" Error
            $allValid = $false
        }
    } else {
        Write-ColorOutput "✗ Python not found. Please install Python $PYTHON_MIN_VERSION+" Error
        Write-ColorOutput "  Download from: https://www.python.org/downloads/" Warning
        $allValid = $false
    }
    
    # Check Git (optional but recommended)
    Write-Step "Checking Git installation..."
    if (Test-ExecutableExists "git") {
        Write-ColorOutput "✓ Git found" Success
    } else {
        Write-ColorOutput "⚠ Git not found (optional, but recommended for version control)" Warning
    }
    
    # Check project structure
    Write-Step "Checking project structure..."
    $requiredDirs = @("backend", "frontend", "database", "config")
    foreach ($dir in $requiredDirs) {
        if (Test-DirectoryExists $dir) {
            Write-ColorOutput "✓ Directory exists: .\$dir" Success
        } else {
            Write-ColorOutput "✗ Missing directory: .\$dir" Error
            $allValid = $false
        }
    }
    
    # Check requirements file
    Write-Step "Checking requirements file..."
    if (Test-FileExists "requirements.txt") {
        Write-ColorOutput "✓ requirements.txt found" Success
    } else {
        Write-ColorOutput "✗ requirements.txt not found" Error
        $allValid = $false
    }
    
    if (-not $allValid) {
        Write-ColorOutput "Prerequisites check failed!" Error
        exit 1
    }
    
    Write-ColorOutput "✓ All prerequisites validated!" Success
}

# ═══════════════════════════════════════════════════════════════════════════
# VIRTUAL ENVIRONMENT SETUP
# ═══════════════════════════════════════════════════════════════════════════

function Setup-VirtualEnvironment {
    Write-Section "Setting Up Virtual Environment"
    
    if ($SkipVenv) {
        Write-ColorOutput "⊘ Skipping virtual environment setup (--SkipVenv)" Warning
        return
    }
    
    # Check if venv already exists
    if (Test-DirectoryExists $VENV_PATH) {
        Write-ColorOutput "Virtual environment already exists at: $VENV_PATH" Info
        $continue = Read-Host "Do you want to recreate it? (y/n)"
        if ($continue -eq "y") {
            Write-Step "Removing existing virtual environment..."
            Remove-Item -Recurse -Force $VENV_PATH
            Write-ColorOutput "✓ Removed" Success
        } else {
            Write-ColorOutput "✓ Using existing virtual environment" Info
            return
        }
    }
    
    Write-Step "Creating Python virtual environment..."
    & python -m venv $VENV_PATH
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "✓ Virtual environment created" Success
    } else {
        Write-ColorOutput "✗ Failed to create virtual environment" Error
        exit 1
    }
    
    # Verify activation script exists
    $activateScript = Join-Path $VENV_PATH "Scripts\Activate.ps1"
    if (-not (Test-FileExists $activateScript)) {
        Write-ColorOutput "✗ Activation script not found" Error
        exit 1
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# DEPENDENCY INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════

function Install-Dependencies {
    Write-Section "Installing Dependencies"
    
    if ($SkipDeps) {
        Write-ColorOutput "⊘ Skipping dependency installation (--SkipDeps)" Warning
        return
    }
    
    # Activate virtual environment
    Write-Step "Activating virtual environment..."
    $activateScript = Join-Path $VENV_PATH "Scripts\Activate.ps1"
    
    if (-not (Test-FileExists $activateScript)) {
        Write-ColorOutput "✗ Activation script not found" Error
        exit 1
    }
    
    & $activateScript
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "✗ Failed to activate virtual environment" Error
        exit 1
    }
    
    Write-ColorOutput "✓ Virtual environment activated" Success
    
    # Upgrade pip
    Write-Step "Upgrading pip..."
    & python -m pip install --upgrade pip | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "✓ Pip upgraded" Success
    } else {
        Write-ColorOutput "⚠ Warning: Pip upgrade had issues (continuing anyway)" Warning
    }
    
    # Install requirements
    Write-Step "Installing Python packages from requirements.txt..."
    Write-ColorOutput "This may take 3-5 minutes..." Info
    & pip install -r requirements.txt
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "✓ All dependencies installed successfully" Success
    } else {
        Write-ColorOutput "✗ Failed to install dependencies" Error
        exit 1
    }
    
    # Verify critical packages
    Write-Step "Verifying critical packages..."
    $criticalPackages = @("fastapi", "streamlit", "sqlalchemy", "scispacy", "pydantic")
    foreach ($package in $criticalPackages) {
        & python -c "import $package" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✓ $package installed" Success
        } else {
            Write-ColorOutput "✗ $package not found" Error
            exit 1
        }
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

function Setup-Environment {
    Write-Section "Configuring Environment"
    
    # Check for .env file
    $envFile = Join-Path $PROJECT_ROOT "config\.env"
    if (-not (Test-FileExists $envFile)) {
        Write-ColorOutput "⚠ .env file not found at config\.env" Warning
        Write-ColorOutput "Creating template .env file..." Info
        
        $envTemplate = @"
# Medical ICD Mapper Environment Configuration
# Last Updated: $(Get-Date -Format 'yyyy-MM-dd')

# ═══ API Configuration ═══
API_ENV=development
API_DEBUG=true
API_HOST=0.0.0.0
API_PORT=8000

# ═══ Database Configuration ═══
DATABASE_URL=sqlite:///./medical_icd_mapper.db
DB_ECHO=false

# ═══ LLM Configuration (Claude/Anthropic) ═══
# Get your API key from: https://console.anthropic.com/
ANTHROPIC_API_KEY=your-api-key-here

# ═══ Google Gemini Configuration (Optional) ═══
# For advanced disease extraction features
GOOGLE_GENAI_API_KEY=your-google-api-key-here

# ═══ CORS Configuration ═══
CORS_ORIGINS=http://localhost:3000,http://localhost:8501,http://localhost:8000

# ═══ File Upload Configuration ═══
UPLOAD_FOLDER=uploads
MAX_UPLOAD_SIZE_MB=25

# ═══ Logging Configuration ═══
LOG_FILE=backend/logs/app.log
LOG_LEVEL=INFO

# ═══ Application Paths ═══
ICD_DATA_PATH=database/seeds/data/icd10cm-codes-April-2025.txt
PROMPT_FILE=backend/prompt.json

# ═══ Security Configuration ═══
SECRET_KEY=your-secret-key-change-in-production
ALGORITHM=HS256

# ═══ Processing Configuration ═══
MAX_WORKERS=4
TIMEOUT_SECONDS=300
"@
        
        # Create directory if doesn't exist
        $configDir = Join-Path $PROJECT_ROOT "config"
        if (-not (Test-DirectoryExists $configDir)) {
            New-Item -ItemType Directory -Path $configDir | Out-Null
        }
        
        $envTemplate | Out-File -FilePath $envFile -Encoding UTF8
        Write-ColorOutput "✓ Template .env file created at: $envFile" Success
        Write-ColorOutput "⚠ IMPORTANT: Update the API keys in .env file before running the application" Warning
        
    } else {
        Write-ColorOutput "✓ .env file already exists" Success
    }
    
    # Create required directories
    Write-Step "Creating required directories..."
    $requiredDirs = @(
        "uploads",
        "backend/logs",
        "backend/uploads"
    )
    
    foreach ($dir in $requiredDirs) {
        $fullPath = Join-Path $PROJECT_ROOT $dir
        if (-not (Test-DirectoryExists $fullPath)) {
            New-Item -ItemType Directory -Path $fullPath | Out-Null
            Write-ColorOutput "✓ Created: $dir" Success
        } else {
            Write-ColorOutput "✓ Exists: $dir" Success
        }
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# DATABASE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

function Initialize-Database {
    Write-Section "Initializing Database"
    
    Write-Step "Checking ICD-10 codes database..."
    $icdDataFile = Join-Path $PROJECT_ROOT "database\seeds\data\icd10cm-codes-April-2025.txt"
    
    if (-not (Test-FileExists $icdDataFile)) {
        Write-ColorOutput "⚠ ICD-10 data file not found: $icdDataFile" Warning
        Write-ColorOutput "  The system will still work but without ICD code lookup" Warning
        return
    }
    
    $icdCount = (Get-Content $icdDataFile).Length
    Write-ColorOutput "✓ ICD-10 database found: $icdCount codes" Success
    
    Write-Step "Database tables will be created on first API run"
    Write-ColorOutput "✓ Database initialization ready" Success
}

# ═══════════════════════════════════════════════════════════════════════════
# APPLICATION STARTUP
# ═══════════════════════════════════════════════════════════════════════════

function Start-Backend {
    Write-Section "Starting Backend API Server"
    
    $activateScript = Join-Path $VENV_PATH "Scripts\Activate.ps1"
    & $activateScript
    
    Write-ColorOutput "Starting FastAPI on http://localhost:$BACKEND_PORT" Info
    Write-ColorOutput "Press Ctrl+C to stop the server" Warning
    Write-Host ""
    
    # Run uvicorn
    & python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port $BACKEND_PORT
}

function Start-Frontend {
    Write-Section "Starting Frontend Web Interface"
    
    $activateScript = Join-Path $VENV_PATH "Scripts\Activate.ps1"
    & $activateScript
    
    Write-ColorOutput "Starting Streamlit on http://localhost:$FRONTEND_PORT" Info
    Write-ColorOutput "Press Ctrl+C to stop the server" Warning
    Write-Host ""
    
    # Run streamlit
    & streamlit run frontend/app.py --logger.level=info
}

function Start-Both {
    Write-Section "Starting Both Backend and Frontend"
    
    Write-ColorOutput "This will open two separate terminal windows" Info
    Write-Host ""
    
    Write-Step "Starting Backend in new terminal..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PROJECT_ROOT'; & '$VENV_PATH\Scripts\Activate.ps1'; python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port $BACKEND_PORT"
    
    Write-Host ""
    Start-Sleep -Seconds 3
    
    Write-Step "Starting Frontend in new terminal..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PROJECT_ROOT'; & '$VENV_PATH\Scripts\Activate.ps1'; streamlit run frontend/app.py"
    
    Write-ColorOutput "✓ Both servers starting in separate terminals" Success
    Write-ColorOutput "   Backend: http://localhost:$BACKEND_PORT" Info
    Write-ColorOutput "   Frontend: http://localhost:$FRONTEND_PORT" Info
}

# ═══════════════════════════════════════════════════════════════════════════
# CLEANUP AND UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

function Clean-Project {
    Write-Section "Cleaning Up Project"
    
    $confirm = Read-Host "This will remove virtual environment and cache files. Continue? (y/n)"
    if ($confirm -ne "y") {
        Write-ColorOutput "Cleanup cancelled" Warning
        return
    }
    
    Write-Step "Removing virtual environment..."
    if (Test-DirectoryExists $VENV_PATH) {
        Remove-Item -Recurse -Force $VENV_PATH
        Write-ColorOutput "✓ Removed: $VENV_PATH" Success
    }
    
    Write-Step "Removing Python cache directories..."
    Get-ChildItem -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
        ForEach-Object { Remove-Item -Recurse -Force $_ }
    Write-ColorOutput "✓ Removed Python cache" Success
    
    Write-Step "Removing compiled files (.pyc)..."
    Get-ChildItem -Recurse -Filter "*.pyc" -ErrorAction SilentlyContinue |
        ForEach-Object { Remove-Item -Force $_ }
    Write-ColorOutput "✓ Removed compiled files" Success
    
    Write-ColorOutput "✓ Project cleanup complete" Success
}

function Show-Help {
    Write-ColorOutput "Medical ICD Mapper - Setup & Startup Script" Header
    Write-ColorOutput "Version $SCRIPT_VERSION" Info
    Write-Host ""
    Write-ColorOutput "USAGE:" Bold
    Write-Host "  .\SETUP_AND_STARTUP.ps1 -Mode <mode> [options]"
    Write-Host ""
    Write-ColorOutput "MODES:" Bold
    Write-Host "  setup      - Only setup (prerequisites, venv, dependencies)"
    Write-Host "  run        - Only run (requires prior setup)"
    Write-Host "  all        - Setup and run everything (default)"
    Write-Host "  clean      - Remove virtual environment and cache"
    Write-Host "  help       - Show this help message"
    Write-Host ""
    Write-ColorOutput "OPTIONS:" Bold
    Write-Host "  -SkipVenv   - Skip virtual environment creation"
    Write-Host "  -SkipDeps   - Skip dependency installation"
    Write-Host "  -Verbose    - Show detailed output"
    Write-Host ""
    Write-ColorOutput "EXAMPLES:" Bold
    Write-Host "  # Full setup and run"
    Write-Host "  .\SETUP_AND_STARTUP.ps1"
    Write-Host ""
    Write-Host "  # Only setup"
    Write-Host "  .\SETUP_AND_STARTUP.ps1 -Mode setup"
    Write-Host ""
    Write-Host "  # Run after setup is complete"
    Write-Host "  .\SETUP_AND_STARTUP.ps1 -Mode run"
    Write-Host ""
    Write-Host "  # Clean and start fresh"
    Write-Host "  .\SETUP_AND_STARTUP.ps1 -Mode clean"
    Write-Host "  .\SETUP_AND_STARTUP.ps1"
    Write-Host ""
    Write-ColorOutput "AFTER STARTUP:" Bold
    Write-Host "  Backend API:  http://localhost:8000"
    Write-Host "  Frontend UI:  http://localhost:8501"
    Write-Host "  API Docs:     http://localhost:8000/docs"
    Write-Host ""
}

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

function Main {
    # Display header
    Write-ColorOutput "🏥 $SCRIPT_NAME" Header
    Write-ColorOutput "Version $SCRIPT_VERSION | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" Info
    Write-Host ""
    
    # Handle help
    if ($Mode -eq "help") {
        Show-Help
        exit 0
    }
    
    # Handle different modes
    switch ($Mode) {
        "setup" {
            Write-ColorOutput "MODE: Setup Only" Warning
            Validate-Prerequisites
            Setup-VirtualEnvironment
            Install-Dependencies
            Setup-Environment
            Initialize-Database
            Write-ColorOutput "✓ Setup complete! Run the script with -Mode run to start the servers" Success
        }
        
        "run" {
            Write-ColorOutput "MODE: Run Only" Warning
            Write-ColorOutput "Choose what to start:" Info
            Write-Host "  1 - Backend API (FastAPI)"
            Write-Host "  2 - Frontend UI (Streamlit)"
            Write-Host "  3 - Both (separate terminals)"
            $choice = Read-Host "Enter choice (1-3)"
            
            switch ($choice) {
                "1" { Start-Backend }
                "2" { Start-Frontend }
                "3" { Start-Both }
                default {
                    Write-ColorOutput "Invalid choice" Error
                    exit 1
                }
            }
        }
        
        "all" {
            Write-ColorOutput "MODE: Full Setup & Run" Warning
            Validate-Prerequisites
            Setup-VirtualEnvironment
            Install-Dependencies
            Setup-Environment
            Initialize-Database
            
            Write-Section "Ready to Start"
            Write-ColorOutput "✓ Setup complete!" Success
            Write-Host ""
            Write-ColorOutput "Choose what to start:" Info
            Write-Host "  1 - Backend API (FastAPI)"
            Write-Host "  2 - Frontend UI (Streamlit)"
            Write-Host "  3 - Both (separate terminals)"
            $choice = Read-Host "Enter choice (1-3)"
            
            switch ($choice) {
                "1" { Start-Backend }
                "2" { Start-Frontend }
                "3" { Start-Both }
                default {
                    Write-ColorOutput "Invalid choice" Error
                    exit 1
                }
            }
        }
        
        "clean" {
            Write-ColorOutput "MODE: Clean" Warning
            Clean-Project
        }
        
        default {
            Write-ColorOutput "Unknown mode: $Mode" Error
            Show-Help
            exit 1
        }
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

Main
