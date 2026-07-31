from typing import Literal

from pydantic import BaseModel, Field, model_validator

AnalysisMode = Literal["local", "text_only", "multimodal"]
ProviderId = Literal["disabled", "openai", "openai_compatible", "anthropic", "gemini"]
Scene = Literal["indoor", "outdoor", "mixed", "synthetic", "unknown"]
ModelId = Literal["retinexformer", "sci", "zero_dce"]

CHECKPOINTS_BY_MODEL = {
    "retinexformer": {"lol_v2_real", "sdsd_indoor", "sdsd_outdoor", "ntire"},
    "sci": {"easy", "medium", "difficult"},
    "zero_dce": {"epoch99"},
}


class ProviderHealth(BaseModel):
    provider_id: str
    implemented: bool
    configured: bool
    healthy: bool
    supports_image: bool = False
    current_model: str | None = None
    last_error: str | None = None
    error_code: str | None = None


class ProviderStatus(ProviderHealth):
    display_name: str
    description: str = ""


class ModelSuggestion(BaseModel):
    model_id: ModelId
    score: float = 0
    reason: str = ""


class CheckpointSuggestion(BaseModel):
    model_id: ModelId
    checkpoint_id: str
    score: float = 0
    reason: str = ""

    @model_validator(mode="after")
    def validate_checkpoint(self):
        allowed = CHECKPOINTS_BY_MODEL[self.model_id]
        if self.checkpoint_id not in allowed:
            raise ValueError(f"checkpoint_id {self.checkpoint_id} is not valid for {self.model_id}")
        return self


class MultimodalAnalysisResult(BaseModel):
    provider: str = "disabled"
    model: str = "local_rules"
    scene: Scene = "unknown"
    subscene: str = ""
    scene_confidence: float = 0
    main_subjects: list[str] = Field(default_factory=list)
    important_light_sources: list[str] = Field(default_factory=list)
    critical_regions: list[str] = Field(default_factory=list)
    interpreted_intent: list[str] = Field(default_factory=list)
    model_candidates: list[ModelSuggestion] = Field(default_factory=list)
    checkpoint_candidates: list[CheckpointSuggestion] = Field(default_factory=list)
    reasoning_summary: str = ""
    warnings: list[str] = Field(default_factory=list)
    confidence: float = 0
    fallback_used: bool = False
    failure_reason: str | None = None
    analysis_mode: AnalysisMode = "local"
    sent_image: bool = False
    runtime_ms: int = 0


class AIAnalyzeRequest(BaseModel):
    image_id: str
    user_request: str = ""
    analysis_mode: AnalysisMode = "local"
    provider_id: str | None = None


class AISettingsPublic(BaseModel):
    enabled: bool
    provider: str
    send_image: bool
    send_metrics: bool
    image_max_edge: int
    image_quality: int
    timeout_seconds: int
    max_retries: int
    fallback_to_local: bool
    remove_metadata: bool
    providers: dict[str, dict]


class AISettingsUpdate(BaseModel):
    enabled: bool | None = None
    provider: str | None = None
    send_image: bool | None = None
    send_metrics: bool | None = None
    image_max_edge: int | None = None
    image_quality: int | None = None
    timeout_seconds: int | None = None
    max_retries: int | None = None
    fallback_to_local: bool | None = None
    remove_metadata: bool | None = None


class AIProviderConfigUpdate(BaseModel):
    enabled: bool | None = None
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    clear_api_key: bool = False
    send_image: bool | None = None
    send_metrics: bool | None = None
    fallback_to_local: bool | None = None
