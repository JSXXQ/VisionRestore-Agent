from typing import Any, Literal

from pydantic import BaseModel, Field

WorkerOperation = Literal["health_check", "enhance", "denoise", "super_resolve"]


class WorkerRequest(BaseModel):
    request_id: str
    operation: WorkerOperation
    input_paths: list[str] = Field(default_factory=list)
    output_path: str
    checkpoint_id: str = ""
    device: str = "cuda:0"
    precision: str = "fp16"
    parameters: dict[str, Any] = Field(default_factory=dict)


class WorkerResponse(BaseModel):
    success: bool
    model_id: str
    checkpoint_id: str = ""
    is_mock: bool = False
    output_path: str | None = None
    runtime_ms: int = 0
    peak_memory_mb: float = 0
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
