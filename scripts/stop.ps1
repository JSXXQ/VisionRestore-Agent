$Root = Split-Path -Parent $PSScriptRoot
foreach ($name in @("api","web")) {
  $pidFile = Join-Path $Root ".task-pids\$name.pid"
  if (Test-Path $pidFile) {
    $procId = Get-Content $pidFile
    Stop-Process -Id $procId -ErrorAction SilentlyContinue
    Remove-Item $pidFile -ErrorAction SilentlyContinue
  }
}
Write-Host "VisionRestore Agent stopped"
