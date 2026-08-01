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
