# Artifact Lineage

Every v2 task has a stable artifact directory layout under `data/tasks/<task_id>/`.

Implemented foundation:

```text
data/tasks/<task_id>/
├─ input/
├─ previews/
├─ candidates/
├─ selected/
├─ postprocess/
├─ final/
├─ reports/
└─ logs/
```

`ArtifactLineageService` creates the layout and exposes current lineage through `/api/v2/tasks/{task_id}/artifacts`.

Current limitation:

- Existing v1 output files are still stored in `data/outputs` and reports in `data/reports`. The lineage service records references now; moving/copying every artifact into the full v2 directory structure will happen with the v2 executor integration.
