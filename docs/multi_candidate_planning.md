# Multi-Candidate Planning

`CandidatePlanner` is the V2 routing center. It decides which enhancement candidates are worth executing. Its score is a planning score, not a final image quality score.

## Formal enhancement experts

The automatic V2 enhancement pool contains:

- `retinexformer`: stable real low-light baseline and non-uniform illumination restoration.
- `darkir`: low-light noise and blur joint restoration.
- `hvi_cidnet`: color and luminance restoration.
- `flol`: fast, high-resolution, lower-VRAM enhancement.
- `sci`: lightweight fallback.

`NAFNet` is retained as `experimental/manual_only` postprocess denoising and does not compete with the enhancement experts.
`Zero-DCE` remains a legacy/manual baseline and is not part of the formal V2 automatic candidate pool.

## Candidate count

- Speed priority: 1 candidate, usually `FLOL`, fallback `SCI`.
- Balanced priority: up to 2 complementary candidates.
- Quality priority: up to 3 complementary candidates.

The planner must not run every installed model. Installed and healthy only means callable, not automatically selected.

## Score composition

V2.3 exposes three score concepts only:

- `local_score`: normalized 0-100 local model/input match. Internally it comes from the configured model prior and ImageAnalyzer degradation match.
- `llm_score`: normalized 0-100 external semantic model-fit score. The LLM must use allowlisted `model_roles` Knowledge as its capability standard.
- `final_score`: post-inference real-output quality score. It is never mixed with either planning score.

When validated external semantic scoring is available:

`planning_score = 0.5 * local_score + 0.5 * llm_score`

When the LLM is disabled, unavailable, low-confidence, malformed, or rejected by local validation:

`planning_score = local_score`

Knowledge is not an independent score. Hardware readiness, ModelRegistry availability, checkpoint health, and `auto_route` are gates rather than score adjustments. Persisted compatibility fields `knowledge_adjustment` and `hardware_adjustment` remain zero.

After family planning, automatic mode runs a second local-only step inside every selected model family. `CheckpointSelector` ranks that family's healthy allowlisted checkpoints from configuration using ImageAnalyzer degradation signals, user intent/priority, validated scene evidence, and hardware gates. The resulting `checkpoint_score` is only an internal weight-matching score: it is not added to family `planning_score` and never participates in `final_score`.

This internal-selection policy also covers multi-checkpoint postprocess families without moving them into the main enhancement candidate pool. NAFNet and Real-ESRGAN invoke `CheckpointSelector` only after their existing conditional route is chosen. Operation and scale compatibility are hard context gates; they are not quality bonuses.

The LLM still does not select checkpoints directly. Validated semantic scene evidence may be used as one bounded input to local checkpoint matching, but external `checkpoint_candidates` are ignored. Manual checkpoint selection retains priority after local existence validation.

All candidates use `input_policy = original_input_only`.
No enhancement candidate consumes another enhancement candidate's output.
