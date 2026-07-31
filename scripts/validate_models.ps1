$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (!(Test-Path "config\models.local.yaml")) { throw "Missing config/models.local.yaml" }
$code = "from visionrestore.adapters.registry import ModelRegistry;`nfor m in ModelRegistry().list():`n print(m['model_id'], m['status_message']);`n [print('  ', w['checkpoint_id'], w['status'], w.get('size_bytes')) for w in m['capabilities'].get('weights', [])]"
& ".\.venv\Scripts\python.exe" -c $code
