# Hierarchical Routing

VisionRestore Agent uses deterministic two-level routing.

## Level 1: architecture

- Quality + CUDA + sufficient GPU memory routes to Retinexformer.
- Balanced + sufficient GPU memory defaults to Retinexformer.
- Speed, CPU-only, or low-memory intent routes to SCI.
- Zero-DCE is manual/comparison/final fallback only, not the first automatic choice.

## Level 2: checkpoints

Retinexformer checkpoint order comes from `config/routing_rules.yaml`:

- Unknown scene: LOL-v2-real, NTIRE, SDSD-outdoor, SDSD-indoor.
- Indoor scene: SDSD-indoor, LOL-v2-real, NTIRE, SDSD-outdoor.
- Outdoor scene: SDSD-outdoor, LOL-v2-real, NTIRE, SDSD-indoor.

SCI weight routing uses a vote from mean luminance, median luminance, 25% percentile, dark-pixel ratio, and dynamic range. It does not infer indoor/outdoor semantics from brightness.

Automatic mode runs at most one main candidate plus one fallback. If the main result passes quality checks, fallback is skipped.
