# Lightweight Knowledge-Augmented Planning

VisionRestore Agent includes a bounded local context retrieval layer and a deterministic planning adapter. This is intentionally lighter than a conventional embedding/vector-database RAG stack.

## Scope

This is not an unrestricted file RAG system. It reads only allowlisted project knowledge files under:

`apps/api/knowledge/`

Current knowledge groups:

- `model_roles.json`
- `workflow_guardrails.json`
- `postprocess_policy.json`
- `eval_history.json`

## Runtime Position

The V2 task flow retrieves context after:

1. local image analysis,
2. user intent parsing,
3. hardware inspection,
4. model registry snapshot.

The retrieved context is stored on `TaskRecord.retrieved_context`, written to the local database as `retrieved_context`, and included in generated reports.

The active flow is:

`query signals -> allowlisted lexical retrieval -> RetrievedContext validation -> PlanningKnowledgeAdapter -> CandidatePlanner`

`RetrievedContext` records `item_id`, `source`, `tags`, retrieval score, and matched query terms. `PlanningKnowledgeAdapter` only accepts `model_roles` and model-specific `eval_history` as planning-score sources. Workflow guardrails and postprocess policy remain available for tracing/reporting but do not change enhancement candidate scores.

## Planning Contract

The adapter can only adjust a candidate that already exists in the locally validated planner pool. It cannot introduce a model, select an unavailable checkpoint, or override manual selection.

The score formula is:

Knowledge does not contribute an independent planning score. Model-role definitions are supplied to the configured LLM as the scoring standard, then CandidatePlanner combines normalized LocalScore and validated LLMScore 50/50. Invalid or unavailable external scoring falls back to LocalScore.

## Safety Boundary

Only allowlisted `model_roles` definitions are sent to the configured external provider, and only when external analysis is enabled. Workflow, evaluation-history, and postprocess context remain local. Knowledge cannot add models, select checkpoints, or affect FinalScore.

External multimodal AI remains advisory. It cannot:

- directly select the final model,
- bypass ModelRegistry,
- bypass HardwareInspector,
- override hard output rejection,
- override real-output final scoring,
- read secrets or arbitrary local files.

Retrieved content is not interpreted as executable instructions. The planning adapter consumes validated source, tags, checkpoint relevance, and matched decision terms; it does not execute content or load paths.

## Prompt Registry

Prompt files remain in:

`apps/api/prompts/`

`visionrestore.ai.prompts` now exposes prompt metadata with SHA-256 hashes so reports and debugging can identify which prompt files were active.

## Why No Full RAG Stack

The current knowledge set is small, structured, and domain-specific. Embeddings, a vector database, document chunking, and a second generation call would add operational complexity without improving the core image-restoration decision enough to justify it. The lightweight design keeps retrieval deterministic, inspectable, offline, and testable.

## Future Extension

If external-context prompting is enabled later, it should require an explicit setting and must send only a compact, redacted context payload. It must never include API keys, `.env`, absolute checkpoint paths, or arbitrary local file contents.
