# Evaluation

No GT image is accepted in the first version, so VisionRestore Agent does not compute PSNR, SSIM, or LPIPS.

The no-reference score combines shadow recovery, highlight protection, color stability, noise control, sharpness preservation, structure preservation, runtime, and peak memory. Weights are in `config/routing_rules.yaml` and differ for quality, balanced, and speed priorities.

The UI and reports display: "No-reference metrics are auxiliary and cannot replace human judgment."
