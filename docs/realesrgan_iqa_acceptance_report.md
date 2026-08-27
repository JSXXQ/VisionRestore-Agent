# Real-ESRGAN and IQA Acceptance Report

Updated: 2026-08-02 17:45

## Implemented

- Real-ESRGAN registered as `super_resolution` postprocess model.
- Real-ESRGAN worker added at `workers/realesrgan_worker.py`.
- Real-ESRGAN status API added.
- Super-resolution confirm/skip APIs added.
- SR execution performs size check, output pixel guard, post-SR quality re-score, keep-or-rollback decision, and final task completion.
- MambaIR realSR remains disabled as a formal SR path.
- IQA status service added for `pyiqa`.
- Candidate IQA query API added.
- MultiCandidateExecutor now records real IQA values when available or explicit unavailable fallback when not available.
- Frontend postprocess card now displays Real-ESRGAN readiness and IQA readiness.

## Current local verification

- Real-ESRGAN source path exists.
- Installed SR weights: 2.
- Missing expected SR weight: `realesr-general-x4v3.pth`.
- `torch`: available in pytorch env.
- `cv2`: available in pytorch env.
- `realesrgan`: not installed in pytorch env.
- `basicsr`: not installed in pytorch env.
- `pyiqa`: not installed in pytorch env.

## Result

The V2 workflow is wired for formal Real-ESRGAN and real PyIQA execution, but current local dependencies prevent actual SR/IQA inference. The system now reports these blockers explicitly and does not create mock SR outputs or fake IQA scores.

## Next installation commands

Install only if you want to run Real-ESRGAN/PyIQA in the existing pytorch env:

```powershell
E:/anconda/envs/pytorch/python.exe -m pip install basicsr realesrgan pyiqa
```

After installation, run:

```powershell
.venv/Scripts/python.exe -m pytest -q
```

Then run model health check:

```http
POST /api/v2/models/realesrgan/health-check
```

## Dependency installation update - 2026-08-02 19:24

Installed into `E:/anconda/envs/pytorch/python.exe` user site:

- `basicsr==1.4.2`
- `realesrgan==0.3.0`
- `pyiqa==0.1.16`

Compatibility fix applied:

- `workers/realesrgan_worker.py` now provides a small compatibility shim for BasicSR 1.4.x with newer torchvision where `torchvision.transforms.functional_tensor` was removed.
- The worker also provides a local `realesrgan.version` shim for local source snapshots that do not include `version.py`.

Verification:

- Real-ESRGAN x2 health check: passed.
- Health input size: `16x16`.
- Health output size: `32x32`.
- Device: `cuda:0`.
- Precision: `fp16`.
- Peak memory: about `63.86 MB`.
- IQA status: `pyiqa is ready`.
- Backend tests: `58 passed`.
- Frontend build: passed.

Still missing:

- `realesr-general-x4v3.pth` is not present locally. x2 and x4 plus weights are present.

## V2 workflow validation - 2026-08-02 19:53

Task: `3fbfed98-7844-4b49-9d27-10e325ff7a5c`

Input: `data/cache/test_images/noisy_low_light.png`

Observed workflow:

- V2 task generated 3 independent enhancement candidates.
- Candidates used original input only.
- Real models completed; no mock output was used.
- Candidate IQA executed with enabled metrics only: MUSIQ and CLIP-IQA.
- Best enhancement selected by final score: `hvi_cidnet:sice`, final score `72.89`.
- Residual analyzer recommended Real-ESRGAN x2 SR.
- Real-ESRGAN x2 executed successfully from the best enhancement result.
- SR output size matched expected x2 size: `768x512`.
- SR score changed from `72.89` to `75.39`.
- SR result was adopted, no rollback.
- Final task status: `completed`.

IQA policy update:

- Default IQA metrics are now MUSIQ and CLIP-IQA.
- TOPIQ-NR, NIQE, and BRISQUE are disabled for the current local workflow and no longer run during default scoring.
