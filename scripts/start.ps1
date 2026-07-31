$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force -Path ".task-pids" | Out-Null
if (!(Test-Path "config\models.local.yaml")) { throw "Missing config/models.local.yaml" }
if (!(Test-Path ".venv")) {
  Write-Host "Missing .venv; running setup.ps1 first"
  & "$PSScriptRoot\setup.ps1"
}
$ApiPort = 8000
$WebPort = 5173
$apiUsed = Get-NetTCPConnection -LocalPort $ApiPort -ErrorAction SilentlyContinue
$webUsed = Get-NetTCPConnection -LocalPort $WebPort -ErrorAction SilentlyContinue
if ($apiUsed) { throw "Port $ApiPort is already in use" }
if ($webUsed) { throw "Port $WebPort is already in use" }
$api = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m","uvicorn","visionrestore.main:app","--app-dir","apps/api","--host","127.0.0.1","--port","8000" -WorkingDirectory $Root -PassThru -WindowStyle Hidden
$web = Start-Process -FilePath "npm.cmd" -ArgumentList "run","dev" -WorkingDirectory (Join-Path $Root "apps\web") -PassThru -WindowStyle Hidden
$api.Id | Set-Content ".task-pids\api.pid"
$web.Id | Set-Content ".task-pids\web.pid"
Write-Host "VisionRestore Agent started"
Write-Host "Web: http://127.0.0.1:5173"
Write-Host "API docs: http://127.0.0.1:8000/docs"

