Write-Host "Python:" (python --version)
Write-Host "Node:" (node --version)
Write-Host "npm:" (npm.cmd --version)
Write-Host "Git:" (git --version)
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { nvidia-smi } else { Write-Host "nvidia-smi not found" }
