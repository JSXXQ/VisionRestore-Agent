function Stop-ProcessTree([int]$RootProcessId) {
  $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $RootProcessId" -ErrorAction SilentlyContinue
  foreach ($child in $children) {
    Stop-ProcessTree -RootProcessId $child.ProcessId
  }
  Stop-Process -Id $RootProcessId -Force -ErrorAction SilentlyContinue
}

$Root = Split-Path -Parent $PSScriptRoot
foreach ($name in @("api","web")) {
  $pidFile = Join-Path $Root ".task-pids\$name.pid"
  if (Test-Path $pidFile) {
    $procId = Get-Content $pidFile
    Stop-ProcessTree -RootProcessId ([int]$procId)
    Remove-Item $pidFile -ErrorAction SilentlyContinue
  }
}
Write-Host "VisionRestore Agent stopped"
