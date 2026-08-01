from pydantic import BaseModel, Field


class ArtifactRecord(BaseModel):
    role: str
    file_id: str | None = None
    path: str | None = None
    url: str | None = None
    model_id: str | None = None
    checkpoint_id: str | None = None
    sha256: str | None = None
    metadata: dict = Field(default_factory=dict)


class ArtifactLineage(BaseModel):
    task_id: str
    task_dir: str
    input: ArtifactRecord | None = None
    candidates: list[ArtifactRecord] = Field(default_factory=list)
    selected: ArtifactRecord | None = None
    postprocess: list[ArtifactRecord] = Field(default_factory=list)
    final: ArtifactRecord | None = None
    reports: list[ArtifactRecord] = Field(default_factory=list)
    logs_dir: str | None = None
