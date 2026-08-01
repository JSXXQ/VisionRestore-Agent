from pydantic import BaseModel, Field


class PostprocessRecommendation(BaseModel):
    denoise_recommended: bool = False
    denoise_confidence: float = 0
    denoise_reason: list[str] = Field(default_factory=list)
    denoise_risk: list[str] = Field(default_factory=list)
    preferred_denoiser: str = "lpdm"

    super_resolution_recommended: bool = False
    sr_confidence: float = 0
    sr_reason: list[str] = Field(default_factory=list)
    sr_risk: list[str] = Field(default_factory=list)
    preferred_sr_model: str = "mambair_real_sr"
    preferred_scale: int = 2

    policy: str = "interactive"
    final_authority: str = "user_or_postprocess_controller"

class PostprocessDecision(BaseModel):
    operation: str
    decision: str
    model_id: str | None = None
    scale: int | None = None
    parameters: dict = Field(default_factory=dict)


class PostprocessResult(BaseModel):
    task_id: str
    operation: str
    decision: str
    accepted: bool
    executed: bool = False
    next_status: str | None = None
    model_id: str | None = None
    message: str
    rollback_performed: bool = False
    metadata: dict = Field(default_factory=dict)
