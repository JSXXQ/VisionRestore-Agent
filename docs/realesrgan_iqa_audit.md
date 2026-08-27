# Real-ESRGAN and IQA Audit

## Git state

- Branch: `feature/multi-expert-agent-v2`
- Recent commits:
  - `038e7bf feat: enable real LPDM worker inference`
  - `d5daec4 chore: report blocked postprocess worker dependencies`
  - `e30a7db feat: enable real DarkIR worker inference`
  - `137c316 feat: enable real HVI-CIDNet worker inference`
  - `2b5ad40 feat: enable real FLOL worker inference`
- Working tree: dirty. There are existing modified files from the V2 multi-candidate work and model integrations. Do not use `git reset --hard`, `git clean -fd`, or directory replacement.

## Current V2 task chain

Default `/api/v2/tasks` currently enters:

`TaskService.create` -> `EnhancementAgentV2.run` -> `ImageAnalyzer` -> `IntentParser` -> optional multimodal analysis -> `HardwareInspector` -> `CandidatePlanner` -> `MultiCandidateExecutor` -> `ResultSelector/CandidateRanker` -> `ResidualDegradationAnalyzer` -> postprocess confirmation states.

Compatibility mode exists through `parameters.task_mode = "single_candidate"`, which keeps the older `EnhancementAgent` route.

This chain must be preserved. Real-ESRGAN must be added only as a super-resolution postprocess tool. IQA must be added only as an evaluation layer.

## Current super-resolution state

- `PostprocessController._super_resolution_unavailable` currently returns a placeholder message saying MambaIR realSR is not configured.
- `ResidualDegradationAnalyzer` can recommend SR by setting `super_resolution_recommended` and `preferred_sr_model`, but currently uses `preferred_sr_model = "none"`.
- Existing states include `awaiting_sr_confirmation`, `super_resolving`, `evaluating_super_resolution`, `rolling_back`, `finalizing`, and `completed`.
- There is no dedicated `POST /api/v2/tasks/{task_id}/super-resolution/confirm` endpoint yet.

## Current MambaIR status

No active MambaIR adapter was found in the current backend registry. Any historical records must remain readable. The new work should explicitly expose MambaIR realSR as disabled/unsupported if it appears in model center or legacy records.

Required display status:

- status: `disabled`
- reason: `unsupported_in_current_environment`
- auto route: off
- executable: no
- replacement: `Real-ESRGAN`

## Real-ESRGAN source and weights

Source path:

`third_party/realesrgan`

Weight directory:

`weights/realesrgan`

Actual files found:

- `RealESRGAN_x2plus.pth`, size `67061725`
- `RealESRGAN_x4plus.pth`, size `67040989`
- `RealESRGAN_x4plus_anime_6B.pth`, size `17938799`

Expected but missing:

- `realesr-general-x4v3.pth`

The missing general x4v3 weight must be reported as missing checkpoint. Do not create an empty file, mock weight, or random checkpoint.

## Real-ESRGAN environment status

Current inference Python:

`E:/anconda/envs/pytorch/python.exe`

Installed module check:

- `torch`: available
- `cv2`: available
- `realesrgan`: missing
- `basicsr`: missing
- `pyiqa`: missing

Implication: PyTorch backend cannot be marked ready until Real-ESRGAN dependencies are installed or the source is made importable with its dependencies. NCNN backend must be independently checked for `realesrgan-ncnn-vulkan.exe`; do not assume it exists.

## Current IQA state

`config/scoring_rules.yaml` currently reserves:

- `topiq_nr`
- `musiq`
- `clipiqa`
- `niqe`
- `brisque`

but `iqa.enabled` is false and `CandidateEvaluator._iqa_score` only consumes values already present in `metrics.iqa`. No real PyIQA execution exists yet.

Current `MultiCandidateExecutor` writes `iqa: {status: not_configured}` for candidate outputs.

## Files prepared for modification

Likely backend files:

- `apps/api/visionrestore/adapters/registry.py`
- `apps/api/visionrestore/adapters/worker_model.py`
- `apps/api/visionrestore/services/postprocess_controller.py`
- `apps/api/visionrestore/services/residual_analyzer.py`
- `apps/api/visionrestore/services/candidate_evaluator.py`
- `apps/api/visionrestore/services/multi_candidate_executor.py`
- `apps/api/visionrestore/api/v2_routes.py`
- `apps/api/visionrestore/storage/database.py`
- `apps/api/visionrestore/schemas/postprocess.py`
- `apps/api/visionrestore/schemas/task.py`

Likely new backend files:

- `workers/realesrgan_worker.py`
- `workers/pyiqa_worker.py`
- `apps/api/visionrestore/services/realesrgan_service.py`
- `apps/api/visionrestore/services/iqa_service.py`
- `apps/api/visionrestore/schemas/iqa.py`
- `config/iqa_scoring.yaml`
- `config/iqa_calibration.yaml`
- optional `scripts/warmup_iqa.py`
- optional `scripts/check_iqa_offline.py`

Frontend files:

- `apps/web/src/main.tsx`
- `apps/web/src/styles.css`

Docs:

- `docs/realesrgan_integration.md`
- `docs/iqa_integration.md`
- `docs/postprocess_workflow.md`
- `docs/candidate_scoring.md`
- `docs/model_environment.md`
- `docs/current_v2_call_chain.md`
- `docs/development_log.md`
- `THIRD_PARTY_NOTICES.md`

## Compatibility logic to preserve

- V2 multi-candidate enhancement pool stays: Retinexformer, DarkIR, HVI-CIDNet, FLOL, SCI.
- NAFNet stays postprocess/manual-only denoising.
- LPDM rollback behavior stays intact.
- Old tasks must remain readable even if they lack new IQA/SR fields.
- Formal mode must not fabricate Real-ESRGAN or IQA success.

## Data migration approach

No destructive migration is needed. Add compatible entity records and optional task fields only. Old tasks should display `no data` for missing IQA/SR records.

Suggested entity types if implemented:

- `iqa_metric`
- `super_resolution_result`
- expanded `model_health_record`

If table changes are needed, use additive schema creation with SQLAlchemy metadata, not deletion.

## Test plan

- Unit tests for Real-ESRGAN missing checkpoint status.
- Unit tests for MambaIR realSR disabled status.
- Unit tests for PyTorch/NCNN backend command construction and path safety.
- Unit tests for x2/x4 dimension assertions.
- Unit tests for SR failure rollback.
- Unit tests for IQA metric normalization, lower-better handling, partial failure, and full failure downgrade.
- Existing V2 task tests must keep passing.
- Run `pytest -q`.
- Run `npm run build`.

## Known environment conflicts

- Real-ESRGAN PyTorch backend is not ready because `realesrgan` and `basicsr` are missing in the current PyTorch environment.
- PyIQA is not installed.
- Installing these packages into the shared PyTorch environment may affect existing model workers. Prefer a separate IQA environment for PyIQA. For Real-ESRGAN, either install official dependencies into the model environment after verification or configure a separate Real-ESRGAN environment.
