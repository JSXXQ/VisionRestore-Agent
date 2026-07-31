from pydantic import BaseModel, Field

class UserPreferences(BaseModel):
    preserve_color: bool = False
    protect_highlights: bool = False
    recover_shadows: bool = False
    reduce_noise: bool = False
    natural_result: bool = False
    strong_enhancement: bool = False
    low_memory: bool = False
    allow_long_runtime: bool = False

class UserIntent(BaseModel):
    priority: str = "balanced"
    scene: str = "unknown"
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    manual_model: str | None = None
    manual_weight: str | None = None
    comparison_mode: bool = False
    raw_text: str = ""
    evidence: list[str] = Field(default_factory=list)
