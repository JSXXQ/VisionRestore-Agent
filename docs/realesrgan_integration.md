# Real-ESRGAN Integration

Updated: 2026-08-23

## Role

Real-ESRGAN is integrated only as a formal super-resolution postprocess model. It is not an enhancement expert and must not enter CandidatePlanner enhancement planning.

Enhancement experts remain: Retinexformer, DarkIR, HVI-CIDNet, FLOL, SCI.

## Local paths

- Source: `third_party/realesrgan`
- Weights: `weights/realesrgan`
- Worker: `workers/realesrgan_worker.py`
- Python: `E:/anconda/envs/pytorch/python.exe`

## Checkpoints

Configured:

- `realesrgan_x2plus`: `RealESRGAN_x2plus.pth`, default x2, user-confirmed postprocess.
- `realesrgan_x4plus`: `RealESRGAN_x4plus.pth`, manual/advanced x4 only.
- `realesr_general_x4v3`: `realesr-general-x4v3.pth`, installed and verified as a manual lightweight x4 option.

## Runtime policy

- PyTorch backend is primary.
- NCNN-Vulkan is reported only if `REALESRGAN_NCNN_EXECUTABLE` exists.
- Formal mode forbids mock output.
- x2 is the default confirmed SR path.
- x4 is never automatic; it requires explicit advanced selection.
- Output pixel count is capped by `postprocess_rules.yaml`.

## API

- `GET /api/v2/models/realesrgan/status`
- `POST /api/v2/tasks/{task_id}/super-resolution/confirm`
- `POST /api/v2/tasks/{task_id}/super-resolution/skip`

## Current environment status

The configured Conda environment cannot reliably resolve its user-site package
directory because of the current Windows username encoding. BasicSR 1.4.2 is
therefore isolated under `third_party/basicsr_runtime/`, and the Real-ESRGAN
worker adds that local dependency directory before importing the model.

## Current verified runtime

Real-ESRGAN x2 and `realesr_general_x4v3` health checks pass on CUDA fp16. The
x4v3 check produced a 64x64 result from a 16x16 input with the expected
`SRVGGNetCompact` architecture and 4x native scale. The worker includes
compatibility shims for BasicSR with newer torchvision and local source
snapshots missing `realesrgan.version`.
