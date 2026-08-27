# VisionRestore Agent V2 Workflow

V2 is a multi-candidate image restoration agent. Its default behavior is not to choose one model before inference. It plans a small set of complementary enhancement candidates, runs them independently on the same original image, scores the real outputs, and only then selects the best result.

## Default V2 flow

1. Analyze original image with `ImageAnalyzer`.
2. Parse user intent with `IntentParser`.
3. Run optional multimodal analysis before enhancement.
4. Inspect hardware and model readiness.
5. Build a `CandidateExecutionPlan` with `CandidatePlanner`.
6. Execute 1-3 candidates with `MultiCandidateExecutor`.
7. Score each real output with `CandidateEvaluator`.
8. If region constraints are present, evaluate those ROIs on each real output and reject hard failures.
9. If the first pass only ran a small candidate set and all completed outputs violate hard region constraints, run one bounded quality retry from the original input.
10. Rank candidates with `CandidateRanker` and select through `ResultSelector`.
11. Diagnose residual degradation with `ResidualDegradationAnalyzer`.
12. Route conditionally: residual noise -> NAFNet confirmation; insufficient resolution/detail -> Real-ESRGAN confirmation; otherwise finalize.
13. Execute only the confirmed postprocess with `PostprocessController` through the LangGraph conditional edge.
14. Re-score postprocess output and keep it only if it improves or preserves the score.
15. Roll back automatically if postprocess makes the result worse.
16. Finalize after no postprocess is needed, skipped, unavailable, or completed.

## Region constraint monitor

V2 can monitor local object/region constraints such as "do not overexpose the streetlight." Region constraints may come from request parameters, validated multimodal JSON, or a local bright-region fallback detector when the user explicitly asks to protect a light source.

Region constraints are evaluated after real model output. Hard failures are rejected by `CandidateEvaluator`; soft constraints contribute to `ConstraintScore`. If the initial pass ran fewer than three candidates and every completed output violates hard ROI constraints, the agent performs one quality retry from the original input. This preserves the no-serial-enhancement rule.

## Legacy mode

The old selected-model flow is still retained as `single_candidate` compatibility mode. It is not the V2 default.

## Task states

The V2 state machine includes:

`queued`, `analyzing_input`, `parsing_intent`, `running_multimodal_analysis`, `inspecting_hardware`, `planning_candidates`, `running_candidates`, `evaluating_candidates`, `ranking_candidates`, `selecting_best_candidate`, `diagnosing_residual_degradation`, `awaiting_denoise_confirmation`, `denoising`, `evaluating_denoise`, `awaiting_sr_confirmation`, `super_resolving`, `evaluating_super_resolution`, `rolling_back`, `finalizing`, `completed`, `failed`, `cancelled`.

A task is not considered complete immediately after enhancement if denoise or super-resolution confirmation is needed.
