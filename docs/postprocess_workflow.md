# Postprocess Workflow

The best enhancement result is not automatically final if residual degradation remains.

## Residual diagnosis

`ResidualDegradationAnalyzer` checks residual noise, color noise risk, blur/clarity risk, overprocessing risk, and resolution needs. DarkIR outputs use a higher denoise trigger threshold because DarkIR already performs joint noise/blur restoration.

## Denoising

Default denoise policy is interactive.

- Preferred automatic recommendation: `LPDM / lpdm_lol`.
- `NAFNet / sidd_width32` and `NAFNet / sidd_width64` are retained as manual/experimental denoising options.
- NAFNet is not an enhancement candidate and is not a super-resolution model.

After denoise, the result is re-scored against the pre-denoise best enhancement result. If the score decreases, the task automatically rolls back to the previous best result and records the before/after scores.

## Super-resolution

The flow reserves `awaiting_sr_confirmation` and super-resolution evaluation states. If MambaIR realSR is not configured, the controller records that SR is unavailable and completes without fabricating a result.

## Conditional routing

The graph does not execute a fixed NAFNet -> Real-ESRGAN chain.

- Residual noise -> pause/confirm NAFNet.
- Insufficient resolution or low detail recovery -> pause/confirm Real-ESRGAN.
- Quality already satisfies the configured thresholds -> finalize directly.

If both denoise and SR are recommended, denoise is considered first and SR is only entered through the LangGraph conditional edge after the denoise decision/result. If SR is not recommended, denoise completion or skip routes directly to finalization.

Every executed postprocess output is evaluated by the same four-layer `CandidateEvaluator` used for enhancement candidates, using the original input as the evaluation reference. A lower `final_score` or hard validity failure triggers automatic rollback. Interrupt/resume remains checkpointed by task id.

After the conditional route chooses a postprocess model family, checkpoint selection is performed inside that family. NAFNet chooses between width32 and width64 from residual degradation, user priority, and hardware evidence. Real-ESRGAN first enforces the confirmed x2/x4 scale as a hard compatibility gate and then ranks only compatible checkpoints. This `checkpoint_score` is recorded for explanation but remains isolated from the postprocess `final_score`.

If a confirmed postprocess model is unavailable or execution fails, the graph
keeps the pre-postprocess best artifact and advances to the next conditional
step (SR confirmation or finalization). It does not repeatedly request the same
failed operation.

## Real-ESRGAN SR update

Super-resolution now uses Real-ESRGAN as the formal postprocess model. MambaIR realSR is disabled. SR requires user confirmation, re-scoring, and rollback on quality degradation.
