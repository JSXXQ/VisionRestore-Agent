[CmdletBinding(SupportsShouldProcess)]
param(
    [switch]$CopyFallback
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

function Remove-EmptyDirectory([string]$Path) {
    if ((Test-Path -LiteralPath $Path -PathType Container) -and
        (Get-ChildItem -LiteralPath $Path -Force | Measure-Object).Count -eq 0) {
        Remove-Item -LiteralPath $Path
    }
}

$sourceMappings = @(
    @{ Legacy = "master\Retinexformer-master\Retinexformer-master"; Canonical = "third_party\retinexformer" },
    @{ Legacy = "master\DarkIR-main\DarkIR-main"; Canonical = "third_party\darkir" },
    @{ Legacy = "master\HVI-CIDNet-master\HVI-CIDNet-master"; Canonical = "third_party\hvi_cidnet" },
    @{ Legacy = "master\FLOL-main\FLOL-main"; Canonical = "third_party\flol" },
    @{ Legacy = "master\SCI-main\SCI-main"; Canonical = "third_party\sci" },
    @{ Legacy = "master\Zero-DCE-master\Zero-DCE-master"; Canonical = "third_party\zero_dce" },
    @{ Legacy = "master\LPDM-main\LPDM-main"; Canonical = "third_party\lpdm" },
    @{ Legacy = "master\NAFNet-main\NAFNet-main"; Canonical = "third_party\nafnet" },
    @{ Legacy = "master\Real-ESRGAN-master\Real-ESRGAN-master"; Canonical = "third_party\realesrgan" },
    @{ Legacy = "master\MambaIR-main\MambaIR-main"; Canonical = "third_party\mambair" }
)

foreach ($mapping in $sourceMappings) {
    $legacy = Join-Path $Root $mapping.Legacy
    $canonical = Join-Path $Root $mapping.Canonical
    if (Test-Path -LiteralPath $legacy -PathType Container) {
        Remove-EmptyDirectory $canonical
        if (Test-Path -LiteralPath $canonical) {
            throw "Canonical source directory already contains files: $canonical"
        }
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $canonical) | Out-Null
        if ($PSCmdlet.ShouldProcess($legacy, "Move model source to $canonical")) {
            Move-Item -LiteralPath $legacy -Destination $canonical
        }
    } elseif (-not (Test-Path -LiteralPath $canonical -PathType Container)) {
        throw "Neither legacy nor canonical source directory exists for $($mapping.Canonical)"
    }
}

$legacyWrappers = @(
    "master\Retinexformer-master",
    "master\DarkIR-main",
    "master\HVI-CIDNet-master",
    "master\FLOL-main",
    "master\SCI-main",
    "master\Zero-DCE-master",
    "master\LPDM-main",
    "master\NAFNet-main",
    "master\Real-ESRGAN-master",
    "master\MambaIR-main",
    "master",
    "third_party\snr_aware"
)
foreach ($relativePath in $legacyWrappers) {
    Remove-EmptyDirectory (Join-Path $Root $relativePath)
}

$weights = @(
    @{ Model = "retinexformer"; Checkpoint = "lol_v2_real"; Source = "third_party\retinexformer\MST_Plus_Plus_NTIRE-20260731T071519Z-1-001\MST_Plus_Plus_NTIRE\LOL_v2_real.pth"; Target = "weights\retinexformer\lol_v2_real.pth"; Required = $true },
    @{ Model = "retinexformer"; Checkpoint = "sdsd_indoor"; Source = "third_party\retinexformer\MST_Plus_Plus_NTIRE-20260731T071519Z-1-001\MST_Plus_Plus_NTIRE\SDSD_indoor.pth"; Target = "weights\retinexformer\sdsd_indoor.pth"; Required = $true },
    @{ Model = "retinexformer"; Checkpoint = "sdsd_outdoor"; Source = "third_party\retinexformer\MST_Plus_Plus_NTIRE-20260731T071519Z-1-001\MST_Plus_Plus_NTIRE\SDSD_outdoor.pth"; Target = "weights\retinexformer\sdsd_outdoor.pth"; Required = $true },
    @{ Model = "retinexformer"; Checkpoint = "ntire"; Source = "third_party\retinexformer\MST_Plus_Plus_NTIRE-20260731T071519Z-1-001\MST_Plus_Plus_NTIRE\NTIRE.pth"; Target = "weights\retinexformer\ntire.pth"; Required = $true },
    @{ Model = "darkir"; Checkpoint = "real_lsrw"; Source = "third_party\darkir\OneDrive_2_2026-8-1\DarkIR_1k_cr_mt.pt"; Target = "weights\darkir\real_lsrw.pt"; Required = $true },
    @{ Model = "darkir"; Checkpoint = "lol_blur"; Source = "third_party\darkir\OneDrive_2_2026-8-1\DarkIR_384.pt"; Target = "weights\darkir\lol_blur.pt"; Required = $true },
    @{ Model = "darkir"; Checkpoint = "lol_blur_w64"; Source = "third_party\darkir\OneDrive_2_2026-8-1\DarkIR_64width.pt"; Target = "weights\darkir\lol_blur_w64.pt"; Required = $true },
    @{ Model = "darkir"; Checkpoint = "all_lol"; Source = "third_party\darkir\OneDrive_2_2026-8-1\DarkIR_allLOL.pt"; Target = "weights\darkir\all_lol.pt"; Required = $true },
    @{ Model = "hvi_cidnet"; Checkpoint = "sice"; Source = "third_party\hvi_cidnet\weights\SICE.pth"; Target = "weights\hvi_cidnet\sice.pth"; Required = $true },
    @{ Model = "hvi_cidnet"; Checkpoint = "fivek"; Source = "third_party\hvi_cidnet\weights\fivek.pth"; Target = "weights\hvi_cidnet\fivek.pth"; Required = $true },
    @{ Model = "hvi_cidnet"; Checkpoint = "lol_blur"; Source = "third_party\hvi_cidnet\weights\LOL-Blur.pth"; Target = "weights\hvi_cidnet\lol_blur.pth"; Required = $true },
    @{ Model = "hvi_cidnet"; Checkpoint = "sid"; Source = "third_party\hvi_cidnet\weights\SID.pth"; Target = "weights\hvi_cidnet\sid.pth"; Required = $true },
    @{ Model = "flol"; Checkpoint = "lol_v2_real"; Source = "third_party\flol\weights\flolv2_all_111439.pt"; Target = "weights\flol\lol_v2_real.pt"; Required = $true },
    @{ Model = "flol"; Checkpoint = "uhd_ll"; Source = "third_party\flol\weights\flolv2_UHDLL.pt"; Target = "weights\flol\uhd_ll.pt"; Required = $true },
    @{ Model = "sci"; Checkpoint = "easy"; Source = "third_party\sci\CVPR\weights\easy.pt"; Target = "weights\sci\easy.pt"; Required = $true },
    @{ Model = "sci"; Checkpoint = "medium"; Source = "third_party\sci\CVPR\weights\medium.pt"; Target = "weights\sci\medium.pt"; Required = $true },
    @{ Model = "sci"; Checkpoint = "difficult"; Source = "third_party\sci\CVPR\weights\difficult.pt"; Target = "weights\sci\difficult.pt"; Required = $true },
    @{ Model = "zero_dce"; Checkpoint = "epoch99"; Source = "third_party\zero_dce\Zero-DCE_code\snapshots\Epoch99.pth"; Target = "weights\zero_dce\epoch99.pth"; Required = $true },
    @{ Model = "lpdm"; Checkpoint = "lpdm_lol"; Source = "third_party\lpdm\pre_weight\lpdm_lol.ckpt"; Target = "weights\lpdm\lpdm_lol.ckpt"; Required = $true },
    @{ Model = "nafnet"; Checkpoint = "sidd_width32"; Source = "third_party\nafnet\pre_weight\NAFNet-SIDD-width32.pth"; Target = "weights\nafnet\sidd_width32.pth"; Required = $true },
    @{ Model = "nafnet"; Checkpoint = "sidd_width64"; Source = "third_party\nafnet\pre_weight\NAFNet-SIDD-width64.pth"; Target = "weights\nafnet\sidd_width64.pth"; Required = $true },
    @{ Model = "realesrgan"; Checkpoint = "realesrgan_x2plus"; Source = "third_party\realesrgan\pre_weight\RealESRGAN_x2plus.pth"; Target = "weights\realesrgan\realesrgan_x2plus.pth"; Required = $true },
    @{ Model = "realesrgan"; Checkpoint = "realesrgan_x4plus"; Source = "third_party\realesrgan\pre_weight\RealESRGAN_x4plus.pth"; Target = "weights\realesrgan\realesrgan_x4plus.pth"; Required = $true },
    @{ Model = "realesrgan"; Checkpoint = "realesr_general_x4v3"; Source = "third_party\realesrgan\pre_weight\realesr-general-x4v3.pth"; Target = "weights\realesrgan\realesr_general_x4v3.pth"; Required = $false },
    @{ Model = "mambair"; Checkpoint = "real_sr"; Source = "third_party\mambair\pre_weight\MambaIR-real.pth"; Target = "weights\mambair\real_sr.pth"; Required = $false },
    @{ Model = "mambair"; Checkpoint = "real_dn"; Source = "third_party\mambair\pre_weight\realDN.pth"; Target = "weights\mambair\real_dn.pth"; Required = $false }
)

$manifest = @()
foreach ($weight in $weights) {
    $source = Join-Path $Root $weight.Source
    $target = Join-Path $Root $weight.Target
    $status = "missing"
    if (Test-Path -LiteralPath $source -PathType Leaf) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
        if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
            try {
                if ($PSCmdlet.ShouldProcess($target, "Create hard link to $source")) {
                    New-Item -ItemType HardLink -Path $target -Target $source | Out-Null
                }
            } catch {
                if (-not $CopyFallback) {
                    throw "Could not create hard link for $($weight.Model)/$($weight.Checkpoint). Re-run with -CopyFallback to copy instead. $($_.Exception.Message)"
                }
                Copy-Item -LiteralPath $source -Destination $target
            }
        }
        if (Test-Path -LiteralPath $target -PathType Leaf) {
            $sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
            $targetHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
            if ($sourceHash -ne $targetHash) {
                throw "Canonical weight differs from source: $target"
            }
            $status = "ready"
        }
    } elseif ($weight.Required) {
        throw "Required source weight is missing: $source"
    }
    $size = if (Test-Path -LiteralPath $target -PathType Leaf) { (Get-Item -LiteralPath $target).Length } else { $null }
    $manifest += [ordered]@{
        model_id = $weight.Model
        checkpoint_id = $weight.Checkpoint
        path = $weight.Target.Replace("\", "/")
        required = [bool]$weight.Required
        status = $status
        size_bytes = $size
    }
}

$manifestPath = Join-Path $Root "weights\manifest.local.json"
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding utf8
Write-Host "Model layout standardized."
Write-Host "Manifest: $manifestPath"
Write-Host "Ready weights: $((@($manifest | Where-Object status -eq 'ready')).Count)"
Write-Host "Missing optional weights: $((@($manifest | Where-Object status -eq 'missing')).Count)"
