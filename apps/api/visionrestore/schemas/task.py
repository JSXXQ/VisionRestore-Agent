from typing import Literal
from pydantic import BaseModel, Field
from .image import ImageAnalysisResult

TaskStatus = Literal[
    "queued", "analyzing", "planning", "loading_model", "running",
    "evaluating", "retrying", "completed", "failed", "cancelled"
]
TaskMode = Literal["auto", "manual", "compare"]
Priority = Literal["quality", "balanced", "speed"]

class EnhancementPlan(BaseModel):
    input_id: str
    user_goal: str = ""
    mode: TaskMode = "auto"
    priority: Priority = "balanced"
    selected_model: str
    selection_reason: str
    parameters: dict = Field(default_factory=dict)
    preprocessing_steps: list[str] = Field(default_factory=list)
    postprocessing_steps: list[str] = Field(default_factory=list)
    fallback_models: list[str] = Field(default_factory=list)
    evaluation_strategy: str = "no-reference heuristic score"
    max_retries: int = 2
    expected_memory_mb: float = 0
    expected_runtime_ms: int = 0
    expected_risks: list[str] = Field(default_factory=list)

class TaskCreate(BaseModel):
    image_id: str
    user_goal: str = ""
    mode: TaskMode = "auto"
    priority: Priority = "balanced"
    model_id: str | None = None
    models: list[str] | None = None
    parameters: dict = Field(default_factory=dict)

class RetryRecord(BaseModel):
    previous_model: str
    previous_parameters: dict
    previous_evaluation: dict
    retry_reason: str
    adjustment: dict
    expected_improvement: str

class CandidateResult(BaseModel):
    model_id: str
    output_file_id: str | None = None
    output_url: str | None = None
    status: str
    score: float = 0
    metrics: dict = Field(default_factory=dict)
    parameters: dict = Field(default_factory=dict)
    runtime_ms: int = 0
    peak_memory_mb: float = 0
    error: str | None = None

class TaskRecord(BaseModel):
    task_id: str
    status: TaskStatus
    image_id: str
    user_goal: str = ""
    mode: TaskMode
    priority: Priority
    progress: float = 0
    analysis: ImageAnalysisResult | None = None
    plan: EnhancementPlan | None = None
    logs: list[str] = Field(default_factory=list)
    retries: list[RetryRecord] = Field(default_factory=list)
    candidates: list[CandidateResult] = Field(default_factory=list)
    best_result: CandidateResult | None = None
    report_file_id: str | None = None
    error: str | None = None
    created_at: str
    completed_at: str | None = None
