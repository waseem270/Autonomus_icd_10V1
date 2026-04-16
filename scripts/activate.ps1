# Medical ICD Mapper - Activate Environment
# Simple shortcut script to activate the venv

if (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "Activating Medical ICD Mapper environment..." -ForegroundColor Cyan
    . .\venv\Scripts\Activate.ps1
} else {
    Write-Host "Error: Virtual environment not found. Please run .\scripts\setup_venv.ps1 first." -ForegroundColor Red
}
