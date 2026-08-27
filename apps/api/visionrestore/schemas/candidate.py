from pydantic import BaseModel, Field, field_validator


class PlanningKnowledgeEvidence(BaseModel):
    item_id: str
    source: str
    title: str
    matched_terms: list[str] = Field(default_factory=list)
    adjustment: float = 0
    reason: str = ""


class PlanningScoreEvidence(BaseModel):
    source: str
    signal: str
    observed: dict = Field(default_factory=dict)
    severity: float | None = None
    contribution: float = 0
    reason: str = ""


class CandidatePlanItem(BaseModel):
    candidate_id: str
    model_id: str
    checkpoint_id: str
    # Family-local checkpoint matching. This score only chooses a weight
    # inside model_id; it is excluded from planning_score and final_score.
    checkpoint_score: float = 0
    checkpoint_selection_mode: str = "local_checkpoint_match"
    checkpoint_evidence: list[PlanningScoreEvidence] = Field(default_factory=list)
    checkpoint_candidates: list[dict] = Field(default_factory=list)
    role: str = "enhancement"
    model_prior_score: float = 0
    input_match_score: float = 0
    planning_evidence: list[PlanningScoreEvidence] = Field(default_factory=list)
    # Compatibility projection for older clients. It is always
    # the normalized local model/input match score in V2.2.
    local_score: float = 0
    llm_score: float | None = None
    local_weight: float = 1.0
    llm_weight: float = 0.0
    planning_mode: str = "local_fallback"
    # Retained for API compatibility. It mirrors llm_score when external
    # advice is adopted and is not added separately to planning_score.
    ai_semantic_bonus: float = 0
    # Retained for persisted-task compatibility. Knowledge is an LLM
    # reference standard and no longer contributes an independent score.
    knowledge_adjustment: float = 0
    knowledge_evidence: list[PlanningKnowledgeEvidence] = Field(default_factory=list)
    # Hardware/model readiness is a gate, not a planning score component.
    hardware_adjustment: float = 0
    planning_score: float = 0
    reason: list[str] = Field(default_factory=list)
    estimated_cost: dict = Field(default_factory=dict)
    input_policy: str = "original_input_only"

    @field_validator("reason", mode="before")
    @classmethod
    def normalize_reason(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value


class CandidateExecutionPlan(BaseModel):
    task_mode: str = "multi_candidate"
    mode: str
    priority: str
    max_candidates: int
    input_image_id: str
    candidates: list[CandidatePlanItem] = Field(default_factory=list)
    rejected: list[dict] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
