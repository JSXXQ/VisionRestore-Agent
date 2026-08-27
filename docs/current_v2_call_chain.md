# Current V2 Call Chain

## Before this fix

`POST /api/v2/tasks` entered `TaskService.create`, then submitted `EnhancementAgent.run`.
That agent still used the legacy path:

1. `ImageAnalyzer`
2. `IntentParser`
3. optional multimodal analysis
4. `HierarchicalRouter.route`
5. `route.selected_model / route.selected_checkpoint`
6. single primary inference, with at most one fallback
7. `QualityEvaluator`
8. direct `completed`

Observed defects:

- `CandidatePlanner` existed but was not the V2 task entry point.
- `MultiCandidateExecutor` existed but was not used by `/api/v2/tasks`.
- `CandidateRanker` existed but did not decide the final V2 result.
- Worker models could receive output paths outside the task directory, causing `Worker output_path must stay inside its task directory`.
- The UI still presented the process as one selected model route.
- Postprocess recommendation could appear after completion, but completion happened too early.

## Legacy V2 chain after the multi-candidate fix

`EnhancementAgentV2` remains available through `parameters.workflow_engine = "legacy"`. It is no longer the default V2 orchestrator.

`POST /api/v2/tasks`
-> `TaskService.create`
-> `EnhancementAgentV2.run`
-> `ImageAnalyzer`
-> `IntentParser`
-> optional `MultimodalAnalysisProvider` through validated provider flow
-> `HardwareInspector`
-> `ModelRegistry.list`
-> `CandidatePlanner.plan`
-> `MultiCandidateExecutor.execute`
-> optional ROI constraint evaluation for each real candidate output
-> optional one-pass quality retry from original input when all first-pass outputs violate hard ROI constraints
-> `CandidateEvaluator` and `CandidateRanker` through `ResultSelector`
-> `ResidualDegradationAnalyzer`
-> `awaiting_denoise_confirmation` or `awaiting_sr_confirmation` or `completed`
-> `PostprocessController` for user-confirmed postprocess
-> re-score and keep or rollback

`HierarchicalRouter` remains available for legacy compatibility and for future local scoring extraction, but V2 no longer treats its selected model as the final execution decision.

## LangGraph V2.2 active chain (2026-08-22)

Default multi-candidate tasks now use `LangGraphEnhancementAgentV2`.

`POST /api/v2/tasks`
-> `TaskService.create`
-> `LangGraphEnhancementAgentV2.run`
-> explicit analysis / intent / runtime / knowledge / advisory nodes
-> allowlisted local retrieval
-> model-role Knowledge is provided to the configured LLM as a scoring reference standard
-> `CandidatePlanner`
-> LangGraph `Send` fan-out to one candidate subgraph per candidate
-> candidate reducer aggregation
-> optional bounded ROI retry from original input
-> `ResultSelector`
-> residual diagnosis
-> optional postprocess subgraph `interrupt`
-> API decision calls `Command(resume=...)` on the same thread
-> postprocess re-score / rollback
-> finalize and report

`CandidatePlanner` combines LocalScore and validated LLMScore 50/50. If external advice is unavailable it falls back to LocalScore. Knowledge does not add a separate score and hardware/model readiness remains a gate. After a model family enters the execution plan, the local config-driven `CheckpointSelector` ranks only healthy allowlisted checkpoints in that family using image degradation, intent, validated scene evidence, priority, and hardware constraints. Its family-local `checkpoint_score` does not alter PlanningScore or FinalScore.

The same selector is reused after conditional postprocess routing. NAFNet ranks `sidd_width32` against `sidd_width64`; Real-ESRGAN ranks only checkpoints compatible with the confirmed x2/x4 scale. Zero-DCE and LPDM each expose only one checkpoint, so no artificial comparison is created for them.

Compatibility:

- `parameters.task_mode = "single_candidate"` -> `EnhancementAgent`
- `parameters.workflow_engine = "legacy"` -> `EnhancementAgentV2`
- default multi-candidate -> LangGraph V2.2

## Postprocess update

After best enhancement selection, optional denoise recommends NAFNet. LPDM remains blocked in the current Windows/PyTorch full-size postprocess path because it is runtime-unstable. Optional SR confirms Real-ESRGAN, executes from the current best result, re-scores, and rolls back if degraded. Candidate execution records real PyIQA metrics when available or an explicit local fallback when unavailable.
