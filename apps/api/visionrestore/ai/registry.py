from visionrestore.ai.providers import (
    DisabledAnalysisProvider,
    MultimodalAnalysisProvider,
    OpenAICompatibleAnalysisProvider,
    UnimplementedConfiguredProvider,
)
from visionrestore.core.config import get_settings


class ProviderRegistry:
    def __init__(self):
        settings = get_settings()
        self.providers: dict[str, MultimodalAnalysisProvider] = {
            "disabled": DisabledAnalysisProvider(settings),
            "openai": OpenAICompatibleAnalysisProvider(
                settings,
                provider_id="openai",
                display_name="OpenAI",
                key_attr="openai_api_key",
                model_attr="openai_model",
                base_url_attr="openai_base_url",
                default_base_url="https://api.openai.com/v1",
            ),
            "openai_compatible": OpenAICompatibleAnalysisProvider(
                settings,
                provider_id="openai_compatible",
                display_name="OpenAI-compatible",
                key_attr="openai_compatible_api_key",
                model_attr="openai_compatible_model",
                base_url_attr="openai_compatible_base_url",
                default_base_url="",
            ),
            "anthropic": UnimplementedConfiguredProvider(settings, "anthropic", "Anthropic Claude", "anthropic_api_key", "anthropic_model"),
            "gemini": UnimplementedConfiguredProvider(settings, "gemini", "Google Gemini", "gemini_api_key", "gemini_model"),
        }

    def list_statuses(self) -> list[dict]:
        return [provider.status().model_dump() for provider in self.providers.values()]

    def get(self, provider_id: str | None = None) -> MultimodalAnalysisProvider:
        settings = get_settings()
        selected = provider_id or settings.multimodal_provider or "disabled"
        if not settings.multimodal_analysis_enabled:
            selected = "disabled"
        return self.providers.get(selected) or self.providers["disabled"]

    def get_by_id(self, provider_id: str) -> MultimodalAnalysisProvider:
        if provider_id not in self.providers:
            raise KeyError(provider_id)
        return self.providers[provider_id]
