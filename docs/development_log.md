# Development Log

## 2026-08-24 - Scene-aware color scoring and proportional result previews

- Corrected a model-agnostic FinalScore failure where global RGB channel
  imbalance treated naturally dominant scene colors as an artificial cast.
- Scene-adaptive color handling is enabled only when MUSIQ and CLIP-IQA are both
  available and each reaches the configured quality threshold; otherwise the
  validated conservative absolute-color fallback remains active.
- Kept all model names, PlanningScore, LLMScore, runtime, and hardware outside
  FinalScore. Severe color casts remain a hard-validity failure.
- For the reproduced `r0e04cc91t.png` case, DarkIR changed from 80.78/second to
  86.45/first; paired GT verification was 35.2452 dB PSNR and 0.958061 SSIM,
  versus Retinexformer at 16.3884 dB and 0.792665.
- Historical 100-output offline rescoring retained the validated aggregate:
  +0.510261 dB PSNR and +0.027875 SSIM versus fixed Retinexformer.
- Changed upload, comparison, and candidate images from crop-fill to proportional
  contain rendering. Full pytest passed 101/101 and the frontend production
  bundle completed in an isolated validation output directory.

## 2026-08-24 - Model-agnostic FinalScore correction and real fixed-30 validation

- Replaced the overly smooth-output preference in the local IQA fallback with
  configured target brightness, absolute color-cast, effective-detail,
  structure, artifact, and highlight evidence.
- Kept FinalScore independent of LocalScore, LLMScore, Knowledge, model name,
  runtime, and hardware. No HVI-CIDNet penalty or DarkIR bonus was introduced.
- Fixed failed/unavailable denoise handling so LangGraph keeps the previous best
  result and advances instead of re-entering the same confirmation loop.
- Added a project-local `einops 0.6.1` fallback for isolated model processes and
  declared the dependency in `pyproject.toml`.
- Real fixed-30 LOLv2 run: 30/30 Agent tasks and 30/30 baselines succeeded;
  selected Retinexformer 21, DarkIR 8, HVI-CIDNet 1. Against fixed
  Retinexformer, PSNR improved by 0.816071 dB and SSIM by 0.013658; both 95%
  confidence intervals were above zero. DarkIR's eight selected samples averaged
  +3.070321 dB PSNR and +0.057880 SSIM.
- One close-score false positive remains and is reported as no-reference
  uncertainty rather than hidden with a model-specific rule.

## 2026-08-23 - Simplified Local/LLM 50-50 planning and calibrated FinalScore

- Simplified active candidate planning to three score concepts: LocalScore, LLMScore, and post-inference FinalScore.
- Active planning now uses `0.5 * local_score + 0.5 * llm_score` when validated external scoring is available and LocalScore-only fallback otherwise.
- Repositioned allowlisted model-role Knowledge as the LLM capability reference instead of an independent KnowledgeAdjustment.
- Converted hardware/model readiness to gates and automatic checkpoint selection to each family default healthy checkpoint.
- Updated the multimodal prompt to score every available model family from 0-100 and return no checkpoint choice.
- Calibrated FinalScore with target-aware brightness recovery, hybrid IQA/local perceptual scoring, and stronger underexposure/structure safeguards.
- Offline rescoring of 100 stored real candidate outputs improved the selector to +0.324001 dB PSNR and +0.022668 SSIM over fixed Retinexformer, with FinalScore/reference false positives reduced from 20 to 8. Ground truth remains offline evaluation only.

## 2026-08-23 - V2.2 decision-quality optimization

- Preserved the single-Agent LangGraph architecture, typed state, tools, ModelRegistry, local knowledge boundary, SQLite checkpointing, and existing model algorithms/weights.
- Added `config/planning_rules.yaml` and changed active planning to `ModelPriorScore + InputMatchScore + KnowledgeAdjustment + HardwareAdjustment`.
- Added per-signal planning evidence from illumination, noise, blur, color shift, detail loss, resolution pressure, user constraints, semantic advice, and hardware observations.
- Kept allowlisted knowledge provenance and `[-5,+5]` bound; knowledge still cannot add candidates and never enters final quality scoring.
- Replaced the old overlapping final layers with image quality, restoration, constraint, and stability layers; planning, knowledge, runtime, and memory are explicitly excluded from `final_score`.
- Kept the existing `CandidateEvaluator -> CandidateRanker -> ResultSelector` authority chain.
- Kept the LangGraph postprocess subgraph and interrupt/resume mechanism, while making denoise-to-SR routing conditional on the residual recommendation.
- Unified denoise and Real-ESRGAN re-evaluation with `CandidateEvaluator`; any score decrease or hard validity failure rolls back to the prior best artifact.
- Added scaled-output support for SR quality/ROI evaluation without changing Real-ESRGAN inference behavior.

## 2026-08-23 - Completed Real-ESRGAN checkpoint inventory

- Moved `realesr-general-x4v3.pth` from the temporary `超分/` staging directory to `third_party/realesrgan/pre_weight/`.
- Created the canonical runtime hard link at `weights/realesrgan/realesr_general_x4v3.pth`.
- Preserved additional ESRGAN and RealESRNet assets under the third-party weight inventory without exposing them to automatic Agent routing.
- Removed three staging duplicates only after their SHA-256 hashes matched the canonical x2, x4, and anime files.
- Regenerated the local weight manifest: 26 ready checkpoints and 0 missing checkpoints.
- Added a project-local BasicSR 1.4.2 runtime fallback for the configured read-only Conda environment and its broken Unicode user-site path.
- Verified `realesr_general_x4v3` with real CUDA fp16 inference: 16x16 input to 64x64 output, SRVGGNetCompact, native 4x scale.

## 2026-08-23 - Standardized model source and weight layout

- Moved local third-party source trees from nested `master/<archive>/<repo>` paths to canonical `third_party/<model>/` directories.
- Added canonical `weights/<model>/` runtime checkpoint directories.
- Used same-volume hard links so large LPDM, NAFNet, MambaIR, and Real-ESRGAN checkpoints are not duplicated.
- Converted `config/models.local.yaml` to project-relative source, worker, checkpoint, and model-config paths.
- Added project-relative path resolution in `visionrestore.core.model_config` while preserving absolute external Python paths.
- Added `scripts/standardize_model_layout.ps1`, local manifest generation, layout documentation, and a relative-path regression test.
- Updated the launcher, smoke test, README, third-party notices, Real-ESRGAN docs, and portable model config example to remove stale `master/` and old project-root references.
- Registry and real small-image health checks passed for every configured available model. The later checkpoint inventory update installed the optional `realesr_general_x4v3` asset as well.

## 2026-08-22 - Lightweight knowledge-augmented planning

- Connected the existing allowlisted local retrieval node to `CandidatePlanner` through a deterministic `PlanningKnowledgeAdapter`.
- Added retrieval provenance fields: `item_id` and `matched_terms`.
- Added candidate planning fields: `knowledge_adjustment` and structured `knowledge_evidence`.
- Historical V2.1 formula was `local_score + knowledge_adjustment + ai_semantic_bonus + hardware_adjustment`; superseded by the V2.2 layered formula documented above.
- Limited knowledge adjustment to `[-5, +5]` and preserved manual-choice priority.
- Restricted score-changing knowledge to model-specific `model_roles` and `eval_history`; workflow and postprocess context remain trace/report-only.
- Preserved ModelRegistry, checkpoint, auto-route, hardware, original-input-only, ROI, post-inference ranking, and rollback boundaries.
- Added graph event/report observability and regression tests for identity-only matches, source boundaries, score composition, provenance, and adjustment capping.

## 2026-07-31 Audit Snapshot

Backup before this audit:

- Commit: `114739f backup: save current local ui and inference state`
- Tag: `backup-before-multimodal-ai-20260731`

This is an incremental optimization project. The current working local upload, image analysis, Retinexformer, SCI, Zero-DCE, CUDA inference, SQLite task storage, and task management paths must be preserved.

## Completed

- Local FastAPI service exists at `http://127.0.0.1:8000` with `/api/v1/*` routes.
- React/Vite frontend exists at `http://127.0.0.1:5173`.
- Local RGB image upload and validation are implemented in `visionrestore.utils.file_security.safe_image_upload`.
- Local image statistics are implemented in `visionrestore.services.image_analyzer.ImageAnalyzer`.
- Local user intent parsing is implemented in `visionrestore.agent.intent_parser.IntentParser`.
- Local hardware inspection is implemented in `visionrestore.services.hardware.HardwareInspector`.
- Model registry is implemented in `visionrestore.adapters.registry.ModelRegistry`.
- Hierarchical local routing is implemented in `visionrestore.routers.hierarchical_router.HierarchicalRouter`.
- Real local inference is implemented through `visionrestore.adapters.base.ModelAdapter.enhance`, which calls `scripts/model_infer_runner.py`.
- Real adapters exist for `RetinexformerAdapter`, `SCIAdapter`, and `ZeroDCEAdapter`.
- Local no-reference quality evaluation is implemented in `visionrestore.services.evaluator.QualityEvaluator`.
- SQLite persistence for files, tasks, and settings is implemented in `visionrestore.storage.database.Database`.
- Agent task loop is implemented in `visionrestore.agent.enhancement_agent.EnhancementAgent`.
- Reports are generated by `visionrestore.services.report.ReportService`.
- Frontend has been moved away from raw debug panels into cards, tables, timeline, candidate cards, metrics table, and settings cards.
- Top bar GPU and model counts come from `/api/v1/system` and `/api/v1/models`, not from hardcoded UI reference values.
- Real result metadata is returned on candidates: `is_mock`, `adapter_class`, `model_id`, `checkpoint_id`, `checkpoint_path`, `checkpoint_sha256`, `device`, `precision`, `runtime_ms`, `peak_memory_mb`, `input_sha256`, and `output_sha256`.
- Optional multimodal AI provider layer exists with `disabled`, `openai`, `openai_compatible`, `anthropic`, and `gemini` provider slots.
- `openai_compatible` supports any vendor that exposes an OpenAI-style `/chat/completions` endpoint.
- `/api/v1/ai/*` endpoints exist for provider listing, provider health check, analysis, public settings, and current-provider testing.
- `MultimodalAnalysisResult` validates provider output with strict model/checkpoint enums.
- Frontend workbench includes analysis mode selection: local, text AI, and multimodal AI.
- Screenshots exist:
  - `docs/screenshots/final-workbench.png`
  - `docs/screenshots/final-model-center.png`
  - `docs/screenshots/final-system-settings.png`

## Partially Completed

- UI is close to the provided reference layout, but drag compare, synchronized zoom, and image pan are still static/simple.
- Model health check can run real small-image inference, but checkpoint health status is not persisted into the model list.
- Fallback currently tries one real fallback candidate when routing provides one; richer historical-failure-aware fallback is not implemented.
- Route candidates expose one score and reasons; they do not yet separate `local_rule_score`, `semantic_bonus`, `resource_adjustment`, and `final_score`.
- Settings page shows local system information, but AI provider settings UI does not exist yet.
- Development docs exist, but `docs/multimodal_ai.md`, `docs/privacy_modes.md`, and `docs/hybrid_routing.md` are not yet created.

## Not Completed

- Native Anthropic Claude and Gemini request formats are not implemented yet; their provider slots are present but not healthy.
- `HybridRoutingEngine` does not exist yet.
- Multimodal AI suggestions are not integrated into routing.
- User privacy modes are not represented in the frontend yet:
  - local only
  - text and metrics only
  - multimodal preview image
- Automated tests for timeout fallback, auth failure fallback, EXIF stripping, preview max edge, hybrid routing, and frontend no-raw-JSON are not yet added.

## Mock Audit

Formal mode defaults:

- `.env.example` contains `ALLOW_MOCK_MODEL=false` and `ALLOW_MOCK_MODELS=false`.
- `Settings.mock_models_enabled` returns `allow_mock_model or allow_mock_models`.
- `ModelRegistry` only registers `mock_model` when `mock_models_enabled` is true.
- Formal runtime model list currently contains only real model IDs: `retinexformer`, `sci`, `zero_dce`.
- `MockModelAdapter` is kept for tests only and returns `is_mock=True`.
- Frontend can label mock candidates as a yellow Mock result, but formal routing must not fall back to mock.
- Existing pytest configuration sets `ALLOW_MOCK_MODELS=true` in `tests/conftest.py`, so unit/smoke tests can use the mock path when needed.

Mock keyword locations:

- `.env.example`
- `apps/api/visionrestore/core/config.py`
- `apps/api/visionrestore/adapters/mock.py`
- `apps/api/visionrestore/adapters/registry.py`
- `apps/web/src/main.tsx`
- `tests/conftest.py`

## Real Model Audit

Real model registry entries:

- `retinexformer`
- `sci`
- `zero_dce`

Real adapters:

- `apps/api/visionrestore/adapters/retinexformer.py`
- `apps/api/visionrestore/adapters/sci.py`
- `apps/api/visionrestore/adapters/zero_dce.py`

Real execution path:

1. `TaskService.create`
2. `EnhancementAgent.run`
3. `ImageAnalyzer.analyze`
4. `IntentParser.parse`
5. `HardwareInspector.inspect`
6. `ModelRegistry.list`
7. `HierarchicalRouter.route`
8. `ModelRegistry.get(model_id)`
9. `ModelAdapter.enhance`
10. `scripts/model_infer_runner.py`
11. `QualityEvaluator.evaluate`
12. SQLite task update and report generation

Verified real local environment:

- Inference Python: `E:\anconda\envs\pytorch\python.exe`
- PyTorch: 2.5.1
- CUDA: 12.1
- GPU observed through API: NVIDIA GeForce RTX 3060
- Real Retinexformer endpoint validation completed with `is_mock=false`, `adapter_class=RetinexformerAdapter`, `device=cuda`, checkpoint path present, output SHA present, and output size equal to input size.

## Frontend Data Audit

Search result:

- No `<pre>` usage remains in `apps/web/src`.
- `JSON.stringify` remains only in request body serialization for `/api/v1/intent/parse` and `/api/v1/tasks`.
- No `RTX 4060` hardcoded value remains in `apps/web/src`.

Current frontend pages:

- Workbench
- Model and weight center
- History
- System settings

Current missing frontend pieces for the next phase:

- Analysis mode selector: local only / text AI / multimodal AI.
- AI provider status card.
- Scene semantics card backed by `MultimodalAnalysisResult`.
- Local rule score, semantic bonus, resource adjustment, and final score split.
- Clear fallback message when AI provider is unavailable.
- Privacy notice before sending text, metrics, or preview image to a configured cloud API.

## Current API Audit

Implemented:

- `GET /api/v1/health`
- `GET /api/v1/system`
- `GET /api/v1/models`
- `GET /api/v1/models/{model_id}`
- `GET /api/v1/models/{model_id}/weights`
- `POST /api/v1/models/{model_id}/refresh`
- `POST /api/v1/models/{model_id}/load`
- `POST /api/v1/models/{model_id}/unload`
- `POST /api/v1/models/{model_id}/install`
- `POST /api/v1/models/{model_id}/health-check`
- `POST /api/v1/images/upload`
- `POST /api/v1/images/analyze`
- `POST /api/v1/intent/parse`
- `POST /api/v1/tasks`
- `GET /api/v1/tasks/{task_id}`
- `POST /api/v1/tasks/{task_id}/cancel`
- `GET /api/v1/tasks/{task_id}/results`
- `GET /api/v1/tasks/{task_id}/report`
- `GET /api/v1/history`
- `GET /api/v1/history/{task_id}`
- `DELETE /api/v1/history/{task_id}`
- `GET /api/v1/files/{file_id}`
- `GET /api/v1/settings`
- `PUT /api/v1/settings`
- `WebSocket /api/v1/tasks/{task_id}/stream`

Not implemented:

- Any `/api/v1/ai/*` endpoint.
- Provider health checks.
- Provider settings with secret redaction.
- Provider test call.

## Current Tests

Existing tests cover:

- API health.
- Backend imports/basic behavior.
- Image analyzer.
- No-reference evaluator.
- Model registry honesty.
- Deterministic planner.
- SQLite persistence.
- Task smoke path.
- Upload security.
- AI provider listing and secret redaction.
- Disabled provider local fallback.
- Structured AI output validation.
- OpenAI-compatible provider with mocked HTTP.

Known gaps:

- Invalid model/checkpoint rejection.
- Provider timeout fallback.
- Provider auth failure fallback.
- No-key fallback.
- Local mode network-call prevention.
- Text-only mode no-image guarantee.
- Multimodal mode preview-image behavior.
- EXIF removal.
- Preview max-edge behavior.
- Hybrid routing scoring.
- Manual choice priority.
- Formal-mode mock disabled.
- Frontend no-raw-JSON regression.
- Real Retinexformer regression.
- Real SCI regression.

## Known Risks

- Some terminal output displays mojibake in PowerShell, but previous frontend build and screenshots rendered correctly in Chrome.
- `DeterministicPlanner` is legacy and still contains mojibake display strings in test expectations; the active Agent flow uses `HierarchicalRouter`.
- `TaskService` creates a fresh `EnhancementAgent` per task; this is simple but does not reuse loaded model state.
- Long-running inference cancellation is cooperative and only checked between candidate runs.
- API secrets must never be returned to frontend when multimodal settings are added.
- Cloud multimodal analysis must never be required for upload, local inference, output save, or local result evaluation.

## Next Implementation Order

1. Add multimodal environment variables to `.env.example` and `Settings`.
2. Add provider/result schemas with strict enums.
3. Add `MultimodalAnalysisProvider`, `DisabledAnalysisProvider`, provider registry, and health response models.
4. Add OpenAI provider behind explicit configuration, timeout, retries, and strict result validation.
5. Add local preview-image generation with metadata removal and max-edge limit.
6. Add `/api/v1/ai/*` routes with secret redaction.
7. Add `HybridRoutingEngine` without replacing local `HierarchicalRouter`.
8. Add frontend analysis mode selector and AI status/semantic cards.
9. Add provider and hybrid routing tests.
10. Re-run local Retinexformer and SCI regression tests.

## 2026-08-01 - Phase 1: model runtime config and subprocess worker base

- Added structured worker request/response schema for `health_check`, `enhance`, `denoise`, and `super_resolve` operations.
- Added `SubprocessBackend` and `SubprocessModelRuntime` as the safe execution base for per-model isolated Python environments. It uses argument arrays, writes `request.json`/`response.json`, captures stdout/stderr logs, enforces timeout, supports cancellation, and rejects outputs outside the task directory.
- Extended model config parsing with `ModelRuntimeConfig` and `WeightProfile`, while keeping compatibility with the existing `weights:` format used by Retinexformer/SCI/Zero-DCE.
- Extended `config/models.example.yaml` with placeholder isolated-environment entries for DarkIR, HVI-CIDNet, FLOL, LPDM, and MambaIR. No real absolute paths or keys were added.
- Added subprocess backend tests.
- Verification: `.\.venv\Scripts\python -m pytest` passed: 23 passed, 3 warnings.

## 2026-08-01 - Phase 2/3: honest model registry and candidate planning base

- Added `WorkerModelAdapter` for DarkIR, HVI-CIDNet, FLOL, LPDM, and MambaIR. These models are visible in the model registry but remain `available=false` until source, worker, environment, weights/config, and real health checks are complete.
- Extended `ModelRegistry` groups for enhancement models and postprocess models without enabling unverified models for routing.
- Added `CandidatePlanner` and candidate plan schemas. The planner enforces candidate budgets: speed=1, balanced=2, quality/compare=3, honors manual choices, and marks all candidates as `original_input_only`.
- Added tests for honest model status and candidate planning behavior.
- Verification: `.\.venv\Scripts\python -m pytest` passed: 28 passed, 3 warnings.

## 2026-08-01 - Phase 4: candidate scoring base

- Added `config/scoring_rules.yaml` for score layer weights, hard validity thresholds, runtime cost normalization, and IQA placeholder policy.
- Added `CandidateEvaluator` and `CandidateRanker` with hard elimination, layered scoring, close-result detection, and a clear note that the score is an Agent recommendation score rather than an absolute image quality percentage.
- Added tests for invalid output elimination, ranking, and close-candidate warnings.
- Verification: `.\.venv\Scripts\python -m pytest` passed: 31 passed, 3 warnings.

## 2026-08-01 - Phase 7: residual degradation analyzer base

- Added postprocess recommendation schema and `config/postprocess_rules.yaml`.
- Added `ResidualDegradationAnalyzer` to recommend denoise and super-resolution interactively with reasons, confidence, risks, preferred model, and scale.
- DarkIR outputs use a higher denoise threshold because DarkIR is treated as a joint restoration model.
- High-resolution inputs do not receive default SR recommendations unless the user explicitly asks.
- Verification: `.\.venv\Scripts\python -m pytest` passed: 35 passed, 3 warnings.

## 2026-08-01 - Phase 25: /api/v2 foundation

- Added `/api/v2` base routes while preserving `/api/v1` compatibility.
- Implemented `GET /api/v2/health`, `GET /api/v2/system`, `GET /api/v2/models`, `POST /api/v2/models/scan`, `POST /api/v2/models/{model_id}/health-check`, `POST /api/v2/images/upload`, and `POST /api/v2/images/analyze`.
- `/api/v2/models` returns model groups and counts only non-mock ready models.
- Added API tests for v2 health, model grouping, and scan.
- Verification: `.\.venv\Scripts\python -m pytest` passed: 38 passed, 3 warnings.

## 2026-08-01 - Phase 5/8/11/25 foundation: workers, artifact lineage, and v2 task APIs

- Added the required `workers/` protocol files for Retinexformer, SCI, Zero-DCE, DarkIR, HVI-CIDNet, FLOL, LPDM, and MambaIR.
- Retinexformer/SCI/Zero-DCE workers wrap the existing local `model_infer_runner.py` protocol. New model workers explicitly return `success=false` until real adapter logic is implemented, so they cannot create fake results.
- Added artifact lineage schema and service. Task creation now creates `data/tasks/<task_id>/` with input, previews, candidates, selected, postprocess, final, reports, and logs folders.
- Expanded `/api/v2` with task create/get/cancel, plan, candidates, ranking, candidate selection, recommendations, postprocess decision recording, and artifacts endpoints.
- Postprocess decisions are recorded but not falsely executed until the corresponding real worker is ready.
- Verification: `.\.venv\Scripts\python -m pytest` passed: 41 passed, 3 warnings.

## 2026-08-01 - Phase 19: additive database foundation

- Added additive v2 database tables without deleting or migrating away the existing `files`, `tasks`, and `settings` payload tables.
- Added `schema_migrations` version tracking.
- Added v2 entity tables for candidate plans/results/metrics/rankings, postprocess recommendations/decisions/results, artifact lineages, model health records, and AI analysis records.
- Added generic `put_entity`, `get_entity`, and `list_entities` helpers for the new v2 entities.
- Verification: `.\.venv\Scripts\python -m pytest` passed: 44 passed, 3 warnings.

## 2026-08-01 - Phase 3: multi-candidate executor base

- Added `MultiCandidateExecutor` to execute candidate plans serially while forcing every candidate to read the same original input image.
- Candidate failures are recorded per candidate and do not stop remaining candidates.
- Formal multi-candidate execution rejects mock outputs instead of ranking them.
- Added tests for original-input isolation, failure isolation, and mock rejection.
- Verification: `.\.venv\Scripts\python -m pytest` passed: 47 passed, 3 warnings.

## 2026-08-01 - Phase 8/9: postprocess decision controller base

- Extended task statuses with denoise/SR confirmation and execution states.
- Added `PostprocessDecision` and `PostprocessResult` schemas.
- Added `PostprocessController` to handle accept/skip/choose_model decisions without falsely executing unavailable postprocess models.
- `/api/v2/tasks/{task_id}/postprocess/decision` now validates structured decisions, updates task status, records logs, and persists the decision in the v2 database entity table.
- Verification: `.\.venv\Scripts\python -m pytest` passed: 50 passed, 3 warnings.

## 2026-08-01 - Phase 6: result selector base

- Added `ResultSelector` and selection schema to keep best candidate, second-best candidate, successful candidates, failed/eliminated candidates, close-score warnings, and explanation text.
- Reused `CandidateRanker` as the scoring authority instead of duplicating ranking logic.
- Added tests for best/second/failed retention, close competition messaging, and all-failed handling.
- Verification: `.\.venv\Scripts\python -m pytest` passed: 53 passed, 3 warnings.

## 2026-08-01 - Phase 11/25: v2 trace persistence and compatibility endpoints

- `/api/v2/models/{model_id}/health-check` now persists model health records.
- `/api/v2/tasks/{task_id}/ranking` now uses `ResultSelector` and persists the latest candidate ranking.
- `/api/v2/tasks/{task_id}/recommendations` persists postprocess recommendations.
- Added `/api/v2/tasks/{task_id}/report` and websocket `/api/v2/tasks/{task_id}/stream` compatibility endpoints.
- Updated v2 tests for persisted ranking and structured postprocess decision behavior.
- Verification: `.\.venv\Scripts\python -m pytest` passed: 53 passed, 3 warnings.

## 2026-08-01 - Documentation refresh for v2 foundations

- Added docs for multi-candidate planning, candidate scoring, postprocess workflow, model environment isolation, artifact lineage, and model download/check workflow.
- Updated API documentation with the implemented `/api/v2` foundation endpoints.
- Documentation explicitly marks unavailable worker models and incomplete execution phases to avoid presenting placeholders as finished real model integration.

## 2026-08-01 - V2 Multi-Candidate Main Flow

- Documented the old `/api/v2/tasks` call chain and the new active V2 chain in `docs/current_v2_call_chain.md`.
- Added `EnhancementAgentV2` and switched default V2 tasks to the multi-candidate pipeline.
- Kept `EnhancementAgent` as `parameters.task_mode = "single_candidate"` compatibility mode.
- Updated `CandidatePlanner` so automatic V2 enhancement candidates are Retinexformer, DarkIR, HVI-CIDNet, FLOL, and SCI.
- Kept NAFNet as postprocess/manual-only and excluded it from automatic enhancement planning.
- Updated `MultiCandidateExecutor` so every candidate reads the original input and writes inside the task candidate directory.
- Registered real candidate outputs for API file serving.
- Separated planning score from final result score.
- Routed real outputs through `ResultSelector`, `CandidateEvaluator`, and `CandidateRanker` before selecting the best result.
- Added residual degradation diagnosis before completion.
- Updated postprocess default to LPDM and added re-score plus automatic rollback on worse denoise output.
- Updated the UI to show V2 candidate planning, candidate result scoring, and LPDM/NAFNet manual postprocess actions.
- Validation: backend pytest passed; frontend build passed.

## 2026-08-02 17:45 Real-ESRGAN / IQA increment

Added Real-ESRGAN formal SR postprocess registration, worker, status API, SR confirm/skip API, SR re-score/rollback logic, PyIQA status/query service, executor IQA fallback, and frontend readiness display. Current blockers: `realesrgan`, `basicsr`, and `pyiqa` are not installed in the pytorch env; `realesr-general-x4v3.pth` is missing.

## 2026-08-02 19:24 Dependency install

Installed `basicsr`, `realesrgan`, and `pyiqa` into the configured pytorch environment. Added worker compatibility shims for newer torchvision and missing local `realesrgan.version`. Real-ESRGAN x2 CUDA fp16 health check passed; PyIQA status is ready.

## 2026-08-02 IQA scope adjustment

Default IQA execution was narrowed to MUSIQ and CLIP-IQA. TOPIQ-NR, NIQE, and BRISQUE remain documented as disabled metrics for the current environment and no longer create repeated task-time errors.
# 2026-08-07 - Local context retrieval and prompt registry

- Added an allowlisted local knowledge base under `apps/api/knowledge`.
- Added `ContextRetrievalService` for bounded local context retrieval.
- Added `TaskRecord.retrieved_context` so V2 tasks can record retrieved context.
- Added report output for retrieved local context.
- Added prompt metadata helpers with SHA-256 hashes in `visionrestore.ai.prompts`.
- Kept retrieved context local-only by default; it is not sent to external multimodal providers.
- Added tests for context retrieval and safe prompt-context policy.

# 2026-08-08 - Region constraint monitor

- Added `RegionConstraint` schemas for object/ROI constraints such as protecting a streetlight from overexposure.
- Added `RegionConstraintService` to accept explicit bbox constraints, consume validated multimodal region constraints, and fall back to local bright-region detection when the user explicitly asks to protect a light source.
- `MultiCandidateExecutor` now records ROI overexposure and luminance checks for each real candidate output.
- `CandidateEvaluator` rejects hard ROI failures; the historical soft user-match contribution is now represented by the V2.2 `ConstraintScore` layer.
- `EnhancementAgentV2` now records `task.region_constraints` and performs one bounded quality retry from the original input when the first pass ran a small candidate set and all completed outputs violate hard region constraints.
- Updated multimodal prompt schema so region bounding boxes are advisory only and never direct pixel-edit authority.
- Updated the web UI so users can drag-select ROI constraints on the input image, send those constraints through task parameters, view overlays on original/result images, and inspect ROI pass/fail status in the Agent panel and candidate cards.

# 2026-08-20 - LangGraph V2.1 standardization

- Added a typed LangGraph state, explicit main graph, candidate execution subgraph, and interruptible postprocess subgraph.
- Made LangGraph the default V2 multi-candidate orchestrator while preserving legacy single-candidate and legacy V2 rollback paths.
- Added persistent SQLite checkpoints keyed by task ID.
- Added typed tool definitions, schemas, registry, router, normalized tool results, and tool timing metadata.
- Implemented candidate `Send` fan-out with reducers that only merge candidate results, workflow events, and messages.
- Preserved original-input-only execution, planning-score/final-score separation, local final authority, bounded ROI retry, user confirmation, re-score, and rollback.
- Added structured workflow events, graph run records, TaskRecord messages, pending confirmation state, and a read-only workflow observation endpoint.
- Routed postprocess decisions through `Command(resume=...)` for LangGraph tasks and added per-task resume locking and operation validation.
- Added graph integration tests for fan-out, reducer behavior, original input invariants, interrupt/resume idempotency, and tool schema validation.
- Updated stale LPDM tests to the active NAFNet recommendation and LPDM runtime-safety policy.
- Detailed design and migration notes are in `docs/langgraph_refactor_report.md`.

# 2026-08-24 - Config-driven family-internal checkpoint selection

- Added `CheckpointSelector` after model-family planning without changing the LangGraph topology.
- Every selected enhancement family now ranks its own healthy checkpoints from `config/planning_rules.yaml` instead of always loading the default weight.
- Internal matching uses ImageAnalyzer degradation signals, user intent/priority, validated scene evidence, and hardware gates.
- Added `checkpoint_score`, selection mode, ranked family checkpoints, and explainable evidence to candidate planning records and the frontend candidate card.
- Preserved LocalScore/LLMScore 50/50 family planning, direct LLM checkpoint prohibition, manual override priority, original-input-only execution, and FinalScore isolation.
- Reused the selector for the two multi-checkpoint postprocess families: NAFNet and Real-ESRGAN. Scale compatibility is a hard gate for Real-ESRGAN; Zero-DCE and LPDM remain single-checkpoint families.
