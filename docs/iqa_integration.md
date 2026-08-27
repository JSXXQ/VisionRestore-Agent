# IQA Integration

Updated: 2026-08-02 17:45

## Goal

The V2 evaluator now has a real IQA layer based on `pyiqa`. Supported metrics are:

- TOPIQ-NR: `topiq_nr`
- MUSIQ: `musiq`
- CLIP-IQA: `clipiqa`
- NIQE: `niqe`
- BRISQUE: `brisque`

## API

- `GET /api/v2/iqa/status`
- `GET /api/v2/tasks/{task_id}/candidates/{candidate_id}/iqa`

## Execution policy

- If `pyiqa` is available, each completed candidate can receive raw IQA values, normalized IQA values, and per-metric error records.
- If `pyiqa` is unavailable, the system records `iqa_detail.status = unavailable` and uses local technical metrics only.
- Missing IQA never blocks a task.
- Missing IQA never produces fabricated TOPIQ/MUSIQ/CLIP-IQA/NIQE/BRISQUE values.

## Current environment status

`pyiqa` is not installed in `E:/anconda/envs/pytorch/python.exe`. The current status is therefore `unavailable`, with fallback `local_technical_score_only`.

## Scoring use

Candidate scoring reads `metrics.iqa_normalized` first. Raw NIQE/BRISQUE values are not treated as direct high-is-good scores.

## Current verified runtime

`pyiqa` imports successfully in the configured pytorch environment and `/api/v2/iqa/status` reports ready. Individual metrics may still download or initialize their own pretrained assets on first use.

## Current default metric policy

The default task IQA set is now limited to MUSIQ and CLIP-IQA. TOPIQ-NR, NIQE, and BRISQUE are disabled in the default workflow because they are unstable in the current local environment and are not required for the next stage.
