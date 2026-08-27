$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (!(Test-Path "data\cache\test_images\low_light.png")) {
  & ".\.venv\Scripts\python.exe" "scripts\generate_test_images.py"
}
$Py = "E:\anconda\envs\pytorch\python.exe"
$InputPath = Join-Path $Root "data\cache\test_images\low_light.png"
& $Py scripts\model_infer_runner.py --model retinexformer --checkpoint-id lol_v2_real --source-path (Join-Path $Root "third_party\retinexformer") --weight-path (Join-Path $Root "weights\retinexformer\lol_v2_real.pth") --input $InputPath --output (Join-Path $Root "data\outputs\smoke_retinex_lol_v2_real.png") --device cuda --precision fp32
& $Py scripts\model_infer_runner.py --model sci --checkpoint-id medium --source-path (Join-Path $Root "third_party\sci\CVPR") --weight-path (Join-Path $Root "weights\sci\medium.pt") --input $InputPath --output (Join-Path $Root "data\outputs\smoke_sci_medium.png") --device cuda --precision fp32
& $Py scripts\model_infer_runner.py --model zero_dce --checkpoint-id epoch99 --source-path (Join-Path $Root "third_party\zero_dce\Zero-DCE_code") --weight-path (Join-Path $Root "weights\zero_dce\epoch99.pth") --input $InputPath --output (Join-Path $Root "data\outputs\smoke_zero_dce_epoch99.png") --device cuda --precision fp32
Write-Host "Smoke test finished. Outputs are in data/outputs."
