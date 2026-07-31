# VisionRestore Agent

Local RGB still-image low-light enhancement Agent with FastAPI at `http://127.0.0.1:8000`, React/Vite, deterministic hierarchical routing, and real local CUDA inference through user-provided Retinexformer, SCI, and Zero-DCE source/weights.

No cloud API key is required for the default workflow. Inference uses the local PyTorch environment configured in `config/models.local.yaml`; `IntentParser` uses local rules.

Optional multimodal analysis is provider-based, not OpenAI-only. The backend includes `disabled`, `openai`, `openai_compatible`, `anthropic`, and `gemini` provider slots. `openai_compatible` can point at any vendor that supports an OpenAI-style `/chat/completions` endpoint. The default is disabled, and provider output is advisory only.

See [README_zh.md](README_zh.md) for setup, model paths, routing, API, CLI, tests, and known limitations.
