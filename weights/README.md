# Model Weights

This directory is the canonical runtime entry point for model checkpoints.

Layout:

```text
weights/
├── retinexformer/
├── darkir/
├── hvi_cidnet/
├── flol/
├── sci/
├── zero_dce/
├── lpdm/
├── nafnet/
├── realesrgan/
└── mambair/
```

Rules:

- Application configuration references project-relative paths under `weights/`.
- Third-party model source remains under `third_party/`.
- `scripts/standardize_model_layout.ps1` creates same-volume hard links from vendor checkpoint files, so the canonical layout does not duplicate large binaries.
- Model binaries and `manifest.local.json` are local-only and ignored by Git.
- Missing optional checkpoints must remain explicitly missing; never create placeholder model files.

Validate the resulting registry with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/validate_models.ps1
```
