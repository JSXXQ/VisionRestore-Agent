from pydantic import BaseModel, Field


class AnalyzeImageInput(BaseModel):
    image_path: str


class InspectHardwareInput(BaseModel):
    include_models: bool = True


class RetrieveContextInput(BaseModel):
    user_request: str = ""
    image_metrics: dict = Field(default_factory=dict)
    available_models: list[dict] = Field(default_factory=list)
    hardware_summary: dict = Field(default_factory=dict)


class ExecuteCandidateInput(BaseModel):
    task_id: str
    candidate: dict
    input_path: str
    priority: str = "balanced"
    task_dir: str
    region_constraints: list[dict] = Field(default_factory=list)
