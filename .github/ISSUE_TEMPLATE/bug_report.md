---
name: Bug report
about: Create a report to help us improve
title: "[Bug] "
labels: ["bug", "needs-triage"]
assignees: []
---

## Summary

<!-- One sentence describing the bug -->

## Environment

| Field | Value |
|---|---|
| OS | <!-- e.g. Windows 11 23H2 --> |
| Python | <!-- `python --version` --> |
| PyTorch | <!-- `python -c "import torch;print(torch.__version__)"` --> |
| CUDA | <!-- `nvidia-smi` --> |
| GPU | <!-- e.g. RTX 3060 12GB --> |
| Node.js | <!-- `node -v` --> |
| Branch / commit | <!-- e.g. master @ 038e7bf --> |
| VisionRestore Agent version | <!-- e.g. V2.3 --> |

## Steps to reproduce

1.
2.
3.

## Expected behavior

<!-- What you expected to happen -->

## Actual behavior

<!-- What actually happened. Paste full error traceback below. -->

```
<paste full traceback here>
```

## Logs / screenshots

<!-- If applicable, drag-drop screenshots or attach task_id / task log path. -->
<!-- Task logs: apps/api/storage/logs/  -->

## Check environment output

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\check_environment.ps1
```

<!-- Paste the output above (omit anything sensitive). -->

## Possible cause

<!-- Optional: your guess about the root cause. -->
