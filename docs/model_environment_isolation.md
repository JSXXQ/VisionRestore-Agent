# Model Environment Isolation

Different third-party models can use separate Python environments and worker scripts.

Implemented foundation:

- `ModelRuntimeConfig` and `WeightProfile` parse per-model runtime settings from `config/models.local.yaml`.
- `SubprocessBackend` uses argument arrays, not `shell=True`.
- Each worker call writes `request.json`, expects `response.json`, captures stdout/stderr logs, enforces a timeout, and supports cancellation.
- `workers/` contains protocol files for Retinexformer, SCI, Zero-DCE, DarkIR, HVI-CIDNet, FLOL, LPDM, and MambaIR.
- Third-party source is normalized under `third_party/<model>/`.
- Runtime checkpoint paths are normalized under `weights/<model>/` and resolved relative to the project root.
- `scripts/standardize_model_layout.ps1` creates canonical same-volume hard links without duplicating large model files.
- `data/python_packages/` is prepended to child-process `PYTHONPATH` as a
  project-local compatibility layer for small pure-Python runtime dependencies.
  The active Retinexformer and worker launch paths share this rule; `einops
  0.6.1` is also declared in `pyproject.toml` for reproducible project setup.

Current runtime boundary:

- DarkIR, HVI-CIDNet, FLOL, LPDM, NAFNet, and Real-ESRGAN have real small-image health checks.
- LPDM remains excluded from the default full-size postprocess path when the current Windows/PyTorch runtime is unstable.
- MambaIR weights are inventoried locally but MambaIR is not registered in the active default model registry.

## Real-ESRGAN and IQA environment

Real-ESRGAN and PyIQA are configured to use `E:/anconda/envs/pytorch/python.exe`.
Real-ESRGAN x2/x4 checkpoints, including `realesr_general_x4v3`, are available.
BasicSR 1.4.2 is isolated under `third_party/basicsr_runtime/` because the
configured Conda environment cannot safely resolve its Unicode user-site path.
