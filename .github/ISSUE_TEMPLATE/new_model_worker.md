---
name: New model worker
about: Propose or add a new model worker (Retinexformer / DarkIR / SCI / etc.)
title: "[Worker] Add "
labels: ["enhancement", "model-worker"]
assignees: []
---

## Model

- **Name / repo**:
- **Paper / year**:
- **License**:
- **Input shape expected**:
- **Output shape produced**:
- **Approx. VRAM**:
- **Public weights**:

## Why it fits VisionRestore Agent

<!-- Which category: enhancement / postprocess / manual baseline? -->

## Implementation plan

- [ ] Add worker under `workers/<name>_worker.py`
- [ ] Register in `apps/api/visionrestore/adapters/registry.py`
- [ ] Add weights folder under `weights/<name>/`
- [ ] Update `config/models.example.yaml` and routing rules
- [ ] Add `tests/test_<name>.py`
- [ ] Add health-check & smoke test
- [ ] Update `docs/model_adapter.md` and `README*`

## Verification

<!-- How will you prove real inference works on the same hardware? -->

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_<name>.py -v
```
