# Changelog

All notable changes to **VisionRestore Agent** are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/) and the
project adheres to [Semantic Versioning](https://semver.org/).

---

## [V2.3] — 2026-09

### Highlights
- **Lightweight knowledge enhancement** — bounded `[-5, +5]` adjustment from
  allowlisted local knowledge, with `knowledge_evidence` for every adjustment.
- **Provider-agnostic multimodal AI** — five slots (`disabled`, `openai`,
  `openai_compatible`, `anthropic`, `gemini`); `openai_compatible` accepts any
  OpenAI-style `/chat/completions` endpoint.
- **NAFNet postprocess (denoise)** — real inference for SIDD width32 / width64.
- **Real-ESRGAN postprocess (super-resolution)** — real inference for v2 (anime
  / photo) and v4 (photo) variants.
- **System Settings page** — live hardware / CUDA / PyTorch / IntentParser /
  LLM status; per-provider configuration; full keys never returned to UI.
- **Model Center redesign** — 10 worker modules and 24 checkpoints visualized
  in a single dashboard.
- **GitHub social preview card** — 1280×640 branded image at
  `docs/social-preview.png`.

### Workers
- Real inference: **DarkIR** (4 weights), **HVI-CIDNet** (4 weights),
  **FLOL** (2 weights), **LPDM** (1 weight, limited self-checks).
- Postprocess: **NAFNet**, **Real-ESRGAN** wired through
  `PostprocessController` with interrupt / resume and rollback.
- Manual baseline: **Zero-DCE** remains manual-only.

### Workflow
- 22-state V2 state machine documented in `docs/agent_workflow.md`.
- Region-constraint monitor for streetlight / face / specified object
  protection.
- Quality retry once if all completed candidates violate hard region
  constraints.

### Tests
- `tests/test_region_constraints.py`, `tests/test_postprocess_controller.py`,
  `tests/test_planning_knowledge.py`, `tests/test_ai_providers.py` added.
- 25+ test files, 10+ pass for full real-inference chain.

---

## [V2.2] — 2026-08

### Highlights
- **Hierarchical routing** — deterministic two-level routing (architecture
  then checkpoint) driven by `config/routing_rules.yaml`.
- **Residual degradation analyzer** — diagnoses whether denoise / super-
  resolution is worth running.
- **Candidate scoring** — `CandidateEvaluator` produces `final_score` from
  real outputs only.
- **Data-flow diagram** — `docs/diagrams/visionrestore_v22_dataflow.png`.
- **IQA integration** — pluggable no-reference IQA providers.

### Workflow
- 16+ state V2 state machine, persistent SQLite checkpoints.
- `current_v2_call_chain.md` and `candidate_scoring.md` introduced.

---

## [V2.1] — 2026-07

### Highlights
- **LangGraph refactor** — orchestration moved to LangGraph with persistent
  checkpoints and structured workflow events.
- **Multi-candidate fan-out / reduce** — `MultiCandidateExecutor`,
  `CandidatePlanner`, `CandidateRanker`, `ResultSelector`.
- **Provider-based multimodal AI** — initial pluggable providers.

---

## [V2.0] — 2026-06

### Highlights
- Initial runnable V2 release: FastAPI backend, React/Vite frontend, SQLite
  metadata store, model adapter contract.
- Single-candidate compatibility mode kept as `single_candidate` for
  migration.

---

## [V1.x] — legacy

- Single-model CLI, single-candidate scoring, no postprocess. Retained in
  branches for reference only.

[Unreleased]: https://github.com/JSXXQ/VisionRestore-Agent/compare/master...HEAD
[V2.3]: https://github.com/JSXXQ/VisionRestore-Agent/releases/tag/v2.3
[V2.2]: https://github.com/JSXXQ/VisionRestore-Agent/releases/tag/v2.2
[V2.1]: https://github.com/JSXXQ/VisionRestore-Agent/releases/tag/v2.1
[V2.0]: https://github.com/JSXXQ/VisionRestore-Agent/releases/tag/v2.0
