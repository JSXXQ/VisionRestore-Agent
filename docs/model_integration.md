# Model Integration

Use `scripts/download_models.ps1` to clone official repositories. Place weights under `weights/<model>/`. Complete and verify adapter-specific wrappers before marking a model available for production inference.

## FLOL Worker

FLOL is integrated through the isolated subprocess worker path. The adapter only marks FLOL available after a real 32x32 image health inference succeeds with a configured local source path, worker script, Python executable, checkpoint, and CUDA runtime. Successful health checks are cached under `data/cache/model_health/` and used as evidence for model availability and auto-routing.

The FLOL worker prepares the local conda CUDA DLL search paths before importing PyTorch and includes a minimal `kornia.filters.gaussian_blur2d` compatibility shim so the existing local environment can run without downloading extra packages. Formal enhancement calls pass absolute input and output paths into the worker because the worker process runs with the external model source directory as its working directory.
