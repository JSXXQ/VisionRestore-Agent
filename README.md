# VisionRestore Agent

Local RGB still-image low-light enhancement Agent with FastAPI at `http://127.0.0.1:8000`, React/Vite, deterministic hierarchical routing, and real local CUDA inference through user-provided Retinexformer, SCI, and Zero-DCE source/weights.

Local model sources use `third_party/<model>/`; canonical runtime checkpoints use `weights/<model>/`. See [docs/model_file_layout.md](docs/model_file_layout.md).

The default V2 multi-candidate workflow is orchestrated by LangGraph with persistent checkpoints, explicit candidate fan-out/reduction, structured workflow events, and interrupt/resume for user-confirmed postprocessing. Legacy single-candidate and legacy V2 paths remain available as explicit compatibility modes. See [docs/langgraph_refactor_report.md](docs/langgraph_refactor_report.md).

No cloud API key is required for the default workflow. Inference uses the local PyTorch environment configured in `config/models.local.yaml`; `IntentParser` uses local rules.

Optional multimodal analysis is provider-based, not OpenAI-only. The backend includes `disabled`, `openai`, `openai_compatible`, `anthropic`, and `gemini` provider slots. `openai_compatible` can point at any vendor that supports an OpenAI-style `/chat/completions` endpoint. The default is disabled, and provider output is advisory only.

See [README_zh.md](README_zh.md) for setup, model paths, routing, API, CLI, tests, and known limitations.
