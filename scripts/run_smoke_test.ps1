$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (!(Test-Path "data\cache\test_images\low_light.png")) {
  & ".\.venv\Scripts\python.exe" "scripts\generate_test_images.py"
}
$Py = "E:\anconda\envs\pytorch\python.exe"
& $Py scripts\model_infer_runner.py --model retinexformer --checkpoint-id lol_v2_real --source-path "E:\codex_project\VisionRestore-Agent\master\Retinexformer-master\Retinexformer-master" --weight-path "E:\codex_project\VisionRestore-Agent\master\Retinexformer-master\Retinexformer-master\MST_Plus_Plus_NTIRE-20260731T071519Z-1-001\MST_Plus_Plus_NTIRE\LOL_v2_real.pth" --input "E:\codex_project\VisionRestore-Agent\data\cache\test_images\low_light.png" --output "E:\codex_project\VisionRestore-Agent\data\outputs\smoke_retinex_lol_v2_real.png" --device cuda --precision fp32
& $Py scripts\model_infer_runner.py --model sci --checkpoint-id medium --source-path "E:\codex_project\VisionRestore-Agent\master\SCI-main\SCI-main\CVPR" --weight-path "E:\codex_project\VisionRestore-Agent\master\SCI-main\SCI-main\CVPR\weights\medium.pt" --input "E:\codex_project\VisionRestore-Agent\data\cache\test_images\low_light.png" --output "E:\codex_project\VisionRestore-Agent\data\outputs\smoke_sci_medium.png" --device cuda --precision fp32
& $Py scripts\model_infer_runner.py --model zero_dce --checkpoint-id epoch99 --source-path "E:\codex_project\VisionRestore-Agent\master\Zero-DCE-master\Zero-DCE-master\Zero-DCE_code" --weight-path "E:\codex_project\VisionRestore-Agent\master\Zero-DCE-master\Zero-DCE-master\Zero-DCE_code\snapshots\Epoch99.pth" --input "E:\codex_project\VisionRestore-Agent\data\cache\test_images\low_light.png" --output "E:\codex_project\VisionRestore-Agent\data\outputs\smoke_zero_dce_epoch99.png" --device cuda --precision fp32
Write-Host "Smoke test finished. Outputs are in data/outputs."
