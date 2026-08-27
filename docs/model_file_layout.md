# Model File Layout

## Purpose

VisionRestore separates application code, third-party source, runtime checkpoints, generated engines, and task artifacts. This prevents model downloads from becoming hidden implementation details inside nested vendor repositories.

## Canonical layout

```text
VisionRestore-Agent/
├── apps/                     application code
├── config/
│   ├── models.example.yaml   portable configuration template
│   └── models.local.yaml     machine-local runtime configuration
├── workers/                  isolated model worker entry points
├── third_party/<model>/      third-party source and original configs
├── weights/<model>/          canonical runtime checkpoint entry points
├── data/trt/                 generated ONNX and TensorRT engines
└── data/tasks/               task inputs, candidates, outputs and reports
```

## Path rules

- `external_python` may be an absolute machine path because it points outside the project.
- `source_path`, `worker_script`, checkpoint `path`, and `config_path` should be project-relative.
- Relative model paths are resolved against `PROJECT_ROOT` by `visionrestore.core.model_config`.
- Workers receive resolved absolute paths at runtime.
- Model binaries remain ignored by Git.
- `models.example.yaml` documents the portable shape; `models.local.yaml` records the current machine's enabled checkpoints.

## Weight policy

The canonical `weights/<model>/` files are same-volume hard links to the original checkpoint files retained with third-party sources. This provides a clear runtime directory without duplicating large files such as LPDM and NAFNet weights.

Configured enhancement checkpoints:

- Retinexformer: `lol_v2_real`, `sdsd_indoor`, `sdsd_outdoor`, `ntire`
- DarkIR: `real_lsrw`, `lol_blur`, `lol_blur_w64`, `all_lol`
- HVI-CIDNet: `sice`, `fivek`, `lol_blur`, `sid`
- FLOL: `lol_v2_real`, `uhd_ll`
- SCI: `easy`, `medium`, `difficult`
- Zero-DCE: `epoch99`

Configured postprocess checkpoints:

- LPDM: `lpdm_lol`
- NAFNet: `sidd_width32`, `sidd_width64`
- Real-ESRGAN: `realesrgan_x2plus`, `realesrgan_x4plus`, `realesr_general_x4v3`

Additional unregistered Real-ESRGAN family assets may be retained under
`third_party/realesrgan/pre_weight/`. They are not exposed to the Agent until a
checkpoint profile, architecture mapping, and real health check are added.

## Maintenance

Run the idempotent layout script after adding or restoring local model repositories:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/standardize_model_layout.ps1
```

Then validate model registration and real small-image health checks:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/validate_models.ps1
```

Do not report a model as ready solely because a file exists. Registry availability still requires source, environment, worker, checkpoint, configuration, and real health validation.
