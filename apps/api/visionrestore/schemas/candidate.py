from pydantic import BaseModel, Field


class CandidatePlanItem(BaseModel):
    candidate_id: str
    model_id: str
    checkpoint_id: str
    role: str = "enhancement"
    reason: str
    estimated_cost: dict = Field(default_factory=dict)
    input_policy: str = "original_input_only"


class CandidateExecutionPlan(BaseModel):
    mode: str
    priority: str
    max_candidates: int
    input_image_id: str
    candidates: list[CandidatePlanItem] = Field(default_factory=list)
    rejected: list[dict] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
