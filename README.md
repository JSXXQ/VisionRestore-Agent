# VisionRestore Agent

Local RGB still-image low-light enhancement Agent with FastAPI at `http://127.0.0.1:8000`, React/Vite, deterministic hierarchical routing, and real local CUDA inference through user-provided Retinexformer, SCI, and Zero-DCE source/weights.

No cloud API key is required for the default workflow. Inference uses the local PyTorch environment configured in `config/models.local.yaml`; `IntentParser` uses local rules, and LLM routing stays disabled unless explicitly enabled by future configuration.

See [README_zh.md](README_zh.md) for setup, model paths, routing, API, CLI, tests, and known limitations.
