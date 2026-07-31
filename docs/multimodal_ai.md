# Multimodal AI Providers

Multimodal AI is optional. The local enhancement workflow must continue to work when every cloud provider is disabled or unconfigured.

## Providers

- `disabled`: pure local rules, no external call.
- `openai`: OpenAI chat completions endpoint.
- `openai_compatible`: any vendor that supports an OpenAI-style `/chat/completions` endpoint and bearer API key.
- `anthropic`: reserved configuration slot; not marked healthy until implemented and tested.
- `gemini`: reserved configuration slot; not marked healthy until implemented and tested.

## Role

Providers may only produce semantic analysis and routing advice:

- scene and subscene
- main subjects
- important light sources
- highlight-risk regions
- interpreted user intent
- candidate model/checkpoint suggestions
- human-readable reasoning

They must not execute commands, access arbitrary local files, load PyTorch models, choose absolute checkpoint paths, bypass `ModelRegistry`, or replace `QualityEvaluator`.

## Validation

Provider output is parsed as JSON and validated by `MultimodalAnalysisResult`.

Allowed model IDs:

- `retinexformer`
- `sci`
- `zero_dce`

Allowed checkpoints:

- Retinexformer: `lol_v2_real`, `sdsd_indoor`, `sdsd_outdoor`, `ntire`
- SCI: `easy`, `medium`, `difficult`
- Zero-DCE: `epoch99`

Invalid provider output is rejected and falls back to local rules when `MULTIMODAL_FALLBACK_TO_LOCAL=true`.

## Privacy

Default settings do not send text or images to any external API:

```env
MULTIMODAL_ANALYSIS_ENABLED=false
MULTIMODAL_PROVIDER=disabled
MULTIMODAL_SEND_IMAGE=false
```

When multimodal image mode is enabled, the backend sends only a resized JPEG preview with metadata removed. The original image is always used for local enhancement and is not uploaded by default.
