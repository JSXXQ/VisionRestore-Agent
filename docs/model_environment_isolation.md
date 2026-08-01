# Model Environment Isolation

Different third-party models can use separate Python environments and worker scripts.

Implemented foundation:

- `ModelRuntimeConfig` and `WeightProfile` parse per-model runtime settings from `config/models.local.yaml`.
- `SubprocessBackend` uses argument arrays, not `shell=True`.
- Each worker call writes `request.json`, expects `response.json`, captures stdout/stderr logs, enforces a timeout, and supports cancellation.
- `workers/` contains protocol files for Retinexformer, SCI, Zero-DCE, DarkIR, HVI-CIDNet, FLOL, LPDM, and MambaIR.
- Unimplemented workers return `success=false` and never mark a model ready.

Current limitation:

- DarkIR, HVI-CIDNet, FLOL, LPDM, and MambaIR still need real adapter logic and real small-image health checks before they can participate in formal routing.
