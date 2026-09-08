---
name: Pull request
about: Contribute code, docs, or models to VisionRestore Agent
title: "[PR] "
---

## What does this change?

<!-- One paragraph. -->

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Documentation
- [ ] New model worker
- [ ] Performance / refactor
- [ ] Test only

## How was it tested?

- [ ] `./.venv/Scripts/python.exe -m pytest` passes
- [ ] `npm.cmd run build --prefix apps\web` passes
- [ ] Manual smoke test against real weights
- [ ] New tests added

### Test output

```
<paste pytest summary>
```

## Checklist

- [ ] My code follows the project's existing style
- [ ] I have read [docs/architecture.md](docs/architecture.md) and [docs/agent_workflow.md](docs/agent_workflow.md)
- [ ] For new model workers: I followed the contract in [docs/model_adapter.md](docs/model_adapter.md)
- [ ] I have NOT introduced any new direct dependencies on cloud services
- [ ] I have NOT included any model weights, API keys, or `.local.yaml` in this PR
- [ ] I have updated relevant docs / README if behavior changed

## Related

<!-- Closes #issue, fixes #issue, depends on #issue, related #issue -->

## Screenshots / logs

<!-- For UI changes: before / after screenshots. For backend: relevant log lines. -->
