$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Root = Split-Path -Parent $PSScriptRoot
$PidDirectory = Join-Path $Root ".task-pids"
$WebDirectory = Join-Path $Root "apps\web"
$ApiPort = 8000
$WebPort = 5173

function Resolve-NodeExecutable {
  $command = Get-Command "node.exe" -ErrorAction SilentlyContinue
  if ($command) {
    return $command.Source
  }

  $candidates = @(
    (Join-Path $env:ProgramFiles "nodejs\node.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "nodejs\node.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\nodejs\node.exe"),
    (Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
  )
  foreach ($candidate in $candidates) {
    if ($candidate -and (Test-Path -LiteralPath $candidate)) {
      return $candidate
    }
  }
  throw "Node.js was not found. Install Node.js 20+ before starting VisionRestore Agent."
}

function Get-ListeningProcessId([int]$Port) {
  $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($connection) {
    return [int]$connection.OwningProcess
  }
  return $null
}

function Wait-ForPort([string]$Name, [int]$Port, [int]$TimeoutSeconds = 45) {
  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    $processId = Get-ListeningProcessId -Port $Port
    if ($processId) {
      return $processId
    }
    Start-Sleep -Milliseconds 500
  }
  throw "$Name did not start on port $Port within $TimeoutSeconds seconds."
}

Set-Location $Root
New-Item -ItemType Directory -Force -Path $PidDirectory | Out-Null

if (!(Test-Path -LiteralPath (Join-Path $Root "config\models.local.yaml"))) {
  throw "Missing config/models.local.yaml"
}
if (!(Test-Path -LiteralPath (Join-Path $Root ".venv\Scripts\python.exe"))) {
  Write-Host "Missing .venv; running setup.ps1 first"
  & (Join-Path $PSScriptRoot "setup.ps1")
}

$NodeExecutable = Resolve-NodeExecutable
$ViteScript = Join-Path $WebDirectory "node_modules\vite\bin\vite.js"
if (!(Test-Path -LiteralPath $ViteScript)) {
  throw "Frontend dependencies are missing. Run npm install in apps/web first."
}

if (Get-ListeningProcessId -Port $ApiPort) {
  throw "Port $ApiPort is already in use"
}
if (Get-ListeningProcessId -Port $WebPort) {
  throw "Port $WebPort is already in use"
}

$ApiOutLog = Join-Path $PidDirectory "api_stdout.log"
$ApiErrorLog = Join-Path $PidDirectory "api_stderr.log"
$WebOutLog = Join-Path $PidDirectory "web_stdout.log"
$WebErrorLog = Join-Path $PidDirectory "web_stderr.log"
$startedProcesses = @()

try {
  $api = Start-Process `
    -FilePath (Join-Path $Root ".venv\Scripts\python.exe") `
    -ArgumentList "-m", "uvicorn", "visionrestore.main:app", "--app-dir", "apps/api", "--host", "127.0.0.1", "--port", "$ApiPort" `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $ApiOutLog `
    -RedirectStandardError $ApiErrorLog `
    -PassThru `
    -WindowStyle Hidden
  $startedProcesses += $api

  $web = Start-Process `
    -FilePath $NodeExecutable `
    -ArgumentList "node_modules/vite/bin/vite.js", "--host", "127.0.0.1", "--port", "$WebPort" `
    -WorkingDirectory $WebDirectory `
    -RedirectStandardOutput $WebOutLog `
    -RedirectStandardError $WebErrorLog `
    -PassThru `
    -WindowStyle Hidden
  $startedProcesses += $web

  $apiProcessId = Wait-ForPort -Name "Backend" -Port $ApiPort
  $webProcessId = Wait-ForPort -Name "Frontend" -Port $WebPort
  $apiProcessId | Set-Content -LiteralPath (Join-Path $PidDirectory "api.pid")
  $webProcessId | Set-Content -LiteralPath (Join-Path $PidDirectory "web.pid")

  Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$ApiPort/api/v2/health" -TimeoutSec 10 | Out-Null
  Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$WebPort" -TimeoutSec 10 | Out-Null

  Write-Host "VisionRestore Agent started"
  Write-Host "Web: http://127.0.0.1:$WebPort"
  Write-Host "API docs: http://127.0.0.1:$ApiPort/docs"
  Write-Host "Logs: $PidDirectory"
}
catch {
  foreach ($port in @($ApiPort, $WebPort)) {
    $processId = Get-ListeningProcessId -Port $port
    if ($processId) {
      Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
  }
  foreach ($process in $startedProcesses) {
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
  }
  Remove-Item -LiteralPath (Join-Path $PidDirectory "api.pid") -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $PidDirectory "web.pid") -Force -ErrorAction SilentlyContinue
  Write-Host "Startup failed. Logs are in: $PidDirectory" -ForegroundColor Red
  throw
}
