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
    allow_mock_models: bool = Field(False, alias="ALLOW_MOCK_MODELS")

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
