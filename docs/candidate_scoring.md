# Candidate Scoring

Planning scores and final scores are separate.

- `planning_score`: used before inference to decide which candidates to execute.
- `final_score`: used after inference to choose the best real output.

`planning_score` combines normalized LocalScore and validated LLMScore 50/50. Knowledge supplies the LLM model-definition reference and is not an independent score. It cannot rescue an invalid output, override an ROI hard failure, or replace post-inference image evaluation.

## Execution record

Each candidate records model id, checkpoint id, status, mock flag, input/output hashes, checkpoint hash, adapter class, worker Python, device, precision, runtime, peak memory, image sizes, warnings, and errors.

## Hard validity checks

Candidates can be eliminated for missing output, decode failure, size mismatch, invalid numeric output, near-black output, severe overexposure, severe color cast, or weak structure preservation.

Region constraints add local hard checks when present. For example, a user request such as "protect the streetlight from overexposure" can create an ROI constraint. A candidate that passes whole-image checks can still be eliminated if the protected region violates a hard local overexposure threshold.

## FinalScore layers

The post-inference formula is:

`FinalScore = ImageQualityScore + RestorationScore + ConstraintScore + StabilityScore` (weighted by priority).

- `ImageQualityScore`: normalized MUSIQ/CLIP-IQA blended with the local perceptual score; when unavailable, an explicit local fallback is used.
- `RestorationScore`: target-aware brightness recovery, residual noise control, edge preservation, and detail recovery.
- `ConstraintScore`: ROI constraint score when enabled; otherwise global output validity.
- `StabilityScore`: near-black, overexposure, color cast, artifact, and numeric-integrity checks.

The local fallback is model-agnostic. It evaluates target-aware brightness,
scene-adaptive color naturalness, highlight safety, artifact control, effective
sharpness, and structure. When both MUSIQ and CLIP-IQA are available and each
clears the configured perceptual-quality threshold,
scene-adaptive color naturalness tolerates reasonable saturation recovery
relative to the low-light input while retaining an absolute guard and the
severe-cast hard-validity boundary. This avoids misclassifying a naturally green,
blue, or warm scene as an artificial color cast. Without complete perceptual IQA,
the evaluator retains the conservative absolute-color fallback used by the
validated offline scoring baseline. The
effective-sharpness curve distinguishes useful
restored detail from both over-smoothing and excessive high-frequency output.
All thresholds and weights are configured in `config/scoring_rules.yaml`; no
model family receives a name-based reward or penalty.

Runtime, memory, LocalScore, LLMScore, PlanningScore, and the family-local CheckpointScore remain observable but do not contribute to `final_score`. CheckpointScore only ranks healthy weights inside one already selected model family and is never compared across model families.

When ROI constraints are enabled, the scorer also records per-region overexposed ratio, p95 luminance, before/after mean luminance, pass/fail reasons, and an aggregate region constraint score.

## IQA layer

`ImageQualityScore` accepts normalized TOPIQ-NR, MUSIQ, CLIP-IQA, NIQE, and BRISQUE values with configuration-backed metric weights. Missing IQA never blocks a task; the report records the explicit local perceptual fallback source.

## Selection

`ResultSelector` uses `CandidateRanker` results. The selected best candidate is the real output with the best post-inference score, not the highest planning score.

## IQA update

Candidate scoring reads normalized PyIQA results when available. If PyIQA is unavailable, scoring falls back to local perceptual metrics and records the IQA unavailable state explicitly.

## IQA default metric policy

Candidate scoring now uses only available normalized MUSIQ and CLIP-IQA values by default. Disabled IQA metrics are reported in `/api/v2/iqa/status` but are not executed during normal task scoring.
