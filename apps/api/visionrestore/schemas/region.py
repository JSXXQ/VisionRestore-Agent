from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


RegionConstraintType = Literal["avoid_overexposure", "preserve_detail", "reduce_noise"]
RegionConstraintSource = Literal["user", "multimodal", "local_highlight_detector", "api"]
RegionConstraintPriority = Literal["hard", "soft"]


class RegionConstraint(BaseModel):
    target: str
    constraint_type: RegionConstraintType = "avoid_overexposure"
    bbox: list[int] = Field(default_factory=list)
    confidence: float = 0
    priority: RegionConstraintPriority = "hard"
    source: RegionConstraintSource = "user"
    max_overexposed_ratio: float = 0.08
    max_luminance_p95: float = 248
    reason: str = ""

    @field_validator("target")
    @classmethod
    def normalize_target(cls, value: str) -> str:
        return (value or "").strip().lower()

    @field_validator("bbox")
    @classmethod
    def validate_bbox_shape(cls, value: list[int]) -> list[int]:
        if value and len(value) != 4:
            raise ValueError("bbox must contain [x1, y1, x2, y2]")
        return [int(item) for item in value]

    @model_validator(mode="after")
    def validate_thresholds(self):
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.max_overexposed_ratio = max(0.0, min(1.0, float(self.max_overexposed_ratio)))
        self.max_luminance_p95 = max(0.0, min(255.0, float(self.max_luminance_p95)))
        return self


class RegionConstraintEvaluation(BaseModel):
    target: str
    constraint_type: RegionConstraintType
    bbox: list[int]
    priority: RegionConstraintPriority
    passed: bool
    score: float
    overexposed_ratio_before: float = 0
    overexposed_ratio_after: float = 0
    luminance_p95_before: float = 0
    luminance_p95_after: float = 0
    mean_luminance_before: float = 0
    mean_luminance_after: float = 0
    reasons: list[str] = Field(default_factory=list)
