from typing import Literal
from pydantic import BaseModel, Field
from .ai import AnalysisMode, MultimodalAnalysisResult
from .image import ImageAnalysisResult
from .intent import UserIntent

TaskStatus = Literal[
    "queued", "analyzing", "parsing_intent", "inspecting_hardware", "routing_model",
    "routing_checkpoint", "loading_model", "running", "evaluating", "fallback_running",
    "selecting_result", "completed", "failed", "cancelled"
]
TaskMode = Literal["auto", "manual", "compare"]
Priority = Literal["quality", "balanced", "speed"]

class EnhancementPlan(BaseModel):
    input_id: str
    user_goal: str = ""
    mode: TaskMode = "auto"
    priority: Priority = "balanced"
    selected_model: str
    selected_checkpoint: str | None = None
    selection_reason: str
    parameters: dict = Field(default_factory=dict)
    preprocessing_steps: list[str] = Field(default_factory=list)
    postprocessing_steps: list[str] = Field(default_factory=list)
    fallback_models: list[str] = Field(default_factory=list)
    fallback_checkpoints: list[str] = Field(default_factory=list)
    evaluation_strategy: str = "no-reference heuristic score"
    max_retries: int = 1
    expected_memory_mb: float = 0
    expected_runtime_ms: int = 0
    expected_risks: list[str] = Field(default_factory=list)

class TaskCreate(BaseModel):
    image_id: str
    user_goal: str = ""
    mode: TaskMode = "auto"
    priority: Priority = "balanced"
    analysis_mode: AnalysisMode = "local"
    model_id: str | None = None
    checkpoint_id: str | None = None
    weight_id: str | None = None
    models: list[str] | None = None
    parameters: dict = Field(default_factory=dict)

class RetryRecord(BaseModel):
    previous_model: str
    previous_checkpoint: str | None = None
    previous_parameters: dict
    previous_evaluation: dict
    retry_reason: str
    fallback_model: str | None = None
    fallback_checkpoint: str | None = None
    adjustment: dict = Field(default_factory=dict)
    expected_improvement: str = ""

class CandidateResult(BaseModel):
    model_id: str
    checkpoint_id: str | None = None
    output_file_id: str | None = None
    output_url: str | None = None
    status: str
    score: float = 0
    metrics: dict = Field(default_factory=dict)
    parameters: dict = Field(default_factory=dict)
    runtime_ms: int = 0
    peak_memory_mb: float = 0
    is_mock: bool = False
    adapter_class: str | None = None
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    device: str | None = None
    precision: str | None = None
    input_sha256: str | None = None
    output_sha256: str | None = None
    error: str | None = None

class TaskRecord(BaseModel):
    task_id: str
    status: TaskStatus
    image_id: str
    user_goal: str = ""
    mode: TaskMode
    priority: Priority
    analysis_mode: AnalysisMode = "local"
    progress: float = 0
    user_intent: UserIntent | None = None
    ai_analysis: MultimodalAnalysisResult | None = None
    hardware_info: dict | None = None
    model_candidates: list[dict] = Field(default_factory=list)
    checkpoint_candidates: list[dict] = Field(default_factory=list)
    analysis: ImageAnalysisResult | None = None
    plan: EnhancementPlan | None = None
    logs: list[str] = Field(default_factory=list)
    retries: list[RetryRecord] = Field(default_factory=list)
    candidates: list[CandidateResult] = Field(default_factory=list)
    best_result: CandidateResult | None = None
    report_file_id: str | None = None
    final_recommendation: str | None = None
    error: str | None = None
    created_at: str
    completed_at: str | None = None
