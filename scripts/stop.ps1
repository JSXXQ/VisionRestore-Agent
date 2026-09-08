function Stop-ProcessTree([int]$RootProcessId) {
  $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $RootProcessId" -ErrorAction SilentlyContinue
  foreach ($child in $children) {
    Stop-ProcessTree -RootProcessId $child.ProcessId
  }
  Stop-Process -Id $RootProcessId -Force -ErrorAction SilentlyContinue
}

function Stop-KnownPortProcess([int]$Port, [string]$CommandPattern) {
  $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  foreach ($connection in $connections) {
    $processId = [int]$connection.OwningProcess
    $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
    if ($processInfo -and $processInfo.CommandLine -match $CommandPattern) {
      Stop-ProcessTree -RootProcessId $processId
    }
    elseif ($processInfo) {
      Write-Warning "Port $Port is used by another application; process $processId was not stopped."
    }
  }
}

function Wait-ForPortRelease([int]$Port, [int]$TimeoutSeconds = 10) {
  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (!$connection) {
      return
    }
    Start-Sleep -Milliseconds 250
  }
  Write-Warning "Port $Port is still listening after $TimeoutSeconds seconds."
}

$Root = Split-Path -Parent $PSScriptRoot
$PidDirectory = Join-Path $Root ".task-pids"
foreach ($name in @("api", "web")) {
  $pidFile = Join-Path $PidDirectory "$name.pid"
  if (Test-Path -LiteralPath $pidFile) {
    $processId = Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue
    if ($processId) {
      Stop-ProcessTree -RootProcessId ([int]$processId)
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
  }
}

# Recover safely from stale PID files or a previous partial startup.
Stop-KnownPortProcess -Port 8000 -CommandPattern "uvicorn\s+visionrestore\.main:app"
Stop-KnownPortProcess -Port 5173 -CommandPattern "vite.+(?:--port\s+)?5173"
Wait-ForPortRelease -Port 8000
Wait-ForPortRelease -Port 5173

Write-Host "VisionRestore Agent stopped"
