# setup.ps1 — Automated setup for Facial Recognition System
# Run from the project root:
#   .\setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "" 
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Facial Recognition System | Setup Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ── 1. Create required folders ──────────────────────────────────────────────
$folders = @("images", "models")
foreach ($folder in $folders) {
    if (-not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder | Out-Null
        Write-Host "[+] Created folder: $folder" -ForegroundColor Green
    } else {
        Write-Host "[=] Folder already exists: $folder" -ForegroundColor Yellow
    }
}

# ── 2. Create virtual environment ──────────────────────────────────────────
if (-not (Test-Path "venv")) {
    Write-Host ""
    Write-Host "[*] Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
    Write-Host "[+] Virtual environment created: venv/" -ForegroundColor Green
} else {
    Write-Host "[=] Virtual environment already exists: venv/" -ForegroundColor Yellow
}

# ── 3. Install dependencies ─────────────────────────────────────────────────
Write-Host ""
Write-Host "[*] Installing dependencies from requirements.txt..." -ForegroundColor Cyan
& ".\venv\Scripts\pip.exe" install --upgrade pip --quiet
& ".\venv\Scripts\pip.exe" install -r requirements.txt
Write-Host "[+] Dependencies installed." -ForegroundColor Green

# ── 4. Done ─────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Add expression feedback images to the images/ folder" -ForegroundColor White
Write-Host "  2. Activate the virtual environment:" -ForegroundColor White
Write-Host "       .\venv\Scripts\activate" -ForegroundColor Yellow
Write-Host "  3. Run the application:" -ForegroundColor White
Write-Host "       python src/main.py" -ForegroundColor Yellow
Write-Host ""
