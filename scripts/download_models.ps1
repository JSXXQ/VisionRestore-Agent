param([ValidateSet("zero_dce","sci","retinexformer","snr_aware","all")] [string]$Model = "all")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
function CloneIfMissing($Path, $Url) {
  if (!(Test-Path $Path)) { git clone $Url $Path } else { Write-Host "$Path 已存在，跳过 clone" }
  if (Test-Path $Path) { Push-Location $Path; git rev-parse HEAD; Pop-Location }
}
if ($Model -eq "zero_dce" -or $Model -eq "all") {
  CloneIfMissing "third_party\zero_dce" "https://github.com/Li-Chongyi/Zero-DCE.git"
  Write-Host "请将 Zero-DCE 官方权重 Epoch99.pth 放入 weights\zero_dce\"
}
if ($Model -eq "sci" -or $Model -eq "all") {
  CloneIfMissing "third_party\sci" "https://github.com/vis-opt-group/SCI.git"
  Write-Host "请将 SCI 官方权重 medium.pt 放入 weights\sci\"
}
if ($Model -eq "retinexformer" -or $Model -eq "all") {
  CloneIfMissing "third_party\retinexformer" "https://github.com/caiyuanhao1998/Retinexformer.git"
  Write-Host "请按官方说明下载 Retinexformer 权重到 weights\retinexformer\retinexformer.pth"
}
if ($Model -eq "snr_aware" -or $Model -eq "all") {
  CloneIfMissing "third_party\snr_aware" "https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance.git"
  Write-Host "请按官方说明下载 SNR-Aware 权重到 weights\snr_aware\snr_aware.pth"
}
