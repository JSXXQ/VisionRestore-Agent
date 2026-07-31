$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (!(Test-Path ".venv")) {
  python -m venv .venv
}
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -e ".[dev]"
Push-Location "apps\web"
npm.cmd install
Pop-Location
Write-Host "Setup complete. Start with: powershell -ExecutionPolicy Bypass -File scripts/start.ps1"
