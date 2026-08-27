# Runtime data

This directory is intentionally empty in Git.

VisionRestore Agent creates uploads, task artifacts, LangGraph/SQLite checkpoints,
evaluation outputs, model caches, reports, and TensorRT export products here at
runtime. These files may contain user images, local paths, large binaries, or
machine-specific state and must not be committed.

The verified model and deployment summaries that are suitable for publication
belong under `docs/`; raw runtime artifacts remain local under `data/`.
