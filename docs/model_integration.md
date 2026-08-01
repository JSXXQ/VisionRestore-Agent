# Model Integration

Use `scripts/download_models.ps1` to clone official repositories. Place weights under `weights/<model>/`. Complete and verify adapter-specific wrappers before marking a model available for production inference.

## FLOL Worker

FLOL is integrated through the isolated subprocess worker path. The adapter only marks FLOL available after a real 32x32 image health inference succeeds with a configured local source path, worker script, Python executable, checkpoint, and CUDA runtime. Successful health checks are cached under `data/cache/model_health/` and used as evidence for model availability and auto-routing.

The FLOL worker prepares the local conda CUDA DLL search paths before importing PyTorch and includes a minimal `kornia.filters.gaussian_blur2d` compatibility shim so the existing local environment can run without downloading extra packages. Formal enhancement calls pass absolute input and output paths into the worker because the worker process runs with the external model source directory as its working directory.

## HVI-CIDNet Worker

HVI-CIDNet is integrated through the same isolated subprocess worker contract. The worker loads `net.CIDNet.CIDNet` from the configured local source path, loads the selected local `.pth` checkpoint, pads RGB images to a multiple of 8, runs CUDA inference, crops back to the original dimensions, and writes the enhanced RGB output. Availability and routing are gated by the same real small-image health cache used by the worker adapter.

## DarkIR Worker

DarkIR is integrated through the isolated subprocess worker contract without using the repository's DDP/complexity-measurement inference entrypoint. The worker loads `archs/DarkIR.py` directly, reads network parameters from the configured YAML file, loads the selected local `params` checkpoint, runs CUDA inference, and saves a clamped RGB output. This avoids optional `ptflops` and distributed-runtime dependencies while still executing the original DarkIR network and local checkpoints.

## Remaining Dependency-Blocked Workers

MambaIR and LPDM have local source and weight paths configured, but they are not marked available until their required runtime packages exist in the selected local Python environment. MambaIR currently reports missing `mamba_ssm`, `causal_conv1d`, and `timm`. LPDM currently reports missing `omegaconf` and `pytorch_lightning`. Their workers return structured dependency-blocked failures during health checks instead of mock results or generic unsupported messages.
