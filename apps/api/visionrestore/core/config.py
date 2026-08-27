from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[4]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_prefix="", extra="ignore")

    host: str = Field("127.0.0.1", alias="VISIONRESTORE_HOST")
    port: int = Field(8000, alias="VISIONRESTORE_PORT")
    max_upload_mb: int = Field(50, alias="MAX_UPLOAD_MB")
    max_image_pixels: int = Field(50_000_000, alias="MAX_IMAGE_PIXELS")
    api_auth_enabled: bool = Field(False, alias="API_AUTH_ENABLED")
    api_token: str = Field("", alias="API_TOKEN")
    allowed_origins: str = Field("http://localhost:5173,http://127.0.0.1:5173", alias="ALLOWED_ORIGINS")
    default_device: str = Field("auto", alias="DEFAULT_DEVICE")
    default_precision: str = Field("fp32", alias="DEFAULT_PRECISION")
    allow_mock_model: bool = Field(False, alias="ALLOW_MOCK_MODEL")
    allow_mock_models: bool = Field(False, alias="ALLOW_MOCK_MODELS")
    multimodal_analysis_enabled: bool = Field(False, alias="MULTIMODAL_ANALYSIS_ENABLED")
    multimodal_provider: str = Field("disabled", alias="MULTIMODAL_PROVIDER")
    multimodal_send_image: bool = Field(False, alias="MULTIMODAL_SEND_IMAGE")
    multimodal_send_metrics: bool = Field(True, alias="MULTIMODAL_SEND_METRICS")
    multimodal_image_max_edge: int = Field(1024, alias="MULTIMODAL_IMAGE_MAX_EDGE")
    multimodal_image_quality: int = Field(88, alias="MULTIMODAL_IMAGE_QUALITY")
    multimodal_timeout_seconds: int = Field(30, alias="MULTIMODAL_TIMEOUT_SECONDS")
    multimodal_max_retries: int = Field(1, alias="MULTIMODAL_MAX_RETRIES")
    multimodal_fallback_to_local: bool = Field(True, alias="MULTIMODAL_FALLBACK_TO_LOCAL")
    multimodal_remove_metadata: bool = Field(True, alias="MULTIMODAL_REMOVE_METADATA")
    context_retrieval_enabled: bool = Field(True, alias="CONTEXT_RETRIEVAL_ENABLED")
    context_retrieval_max_items: int = Field(5, alias="CONTEXT_RETRIEVAL_MAX_ITEMS")
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    openai_model: str = Field("", alias="OPENAI_MODEL")
    openai_base_url: str = Field("", alias="OPENAI_BASE_URL")
    openai_compatible_api_key: str = Field("", alias="OPENAI_COMPATIBLE_API_KEY")
    openai_compatible_model: str = Field("", alias="OPENAI_COMPATIBLE_MODEL")
    openai_compatible_base_url: str = Field("", alias="OPENAI_COMPATIBLE_BASE_URL")
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field("", alias="ANTHROPIC_MODEL")
    gemini_api_key: str = Field("", alias="GEMINI_API_KEY")
    gemini_model: str = Field("", alias="GEMINI_MODEL")

    @property
    def mock_models_enabled(self) -> bool:
        return self.allow_mock_model or self.allow_mock_models

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def report_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def weights_dir(self) -> Path:
        return PROJECT_ROOT / "weights"

    @property
    def third_party_dir(self) -> Path:
        return PROJECT_ROOT / "third_party"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.data_dir / 'visionrestore.sqlite'}"

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.allowed_origins.split(",") if item.strip()]

@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    for p in [settings.upload_dir, settings.output_dir, settings.report_dir, settings.cache_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return settings
