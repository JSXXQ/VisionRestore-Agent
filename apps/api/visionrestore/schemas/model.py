from pydantic import BaseModel, Field

class ModelStatus(BaseModel):
    model_id: str
    display_name: str
    description: str
    repository_url: str
    license_name: str
    supported_devices: list[str]
    supported_precisions: list[str]
    installed: bool
    available: bool
    loaded: bool = False
    weight_path: str | None = None
    source_path: str | None = None
    status_message: str
    install_hint: str
    capabilities: dict = Field(default_factory=dict)
