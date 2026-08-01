from pydantic import BaseModel, Field


class ResultSelection(BaseModel):
    selected: dict | None = None
    second_best: dict | None = None
    successful: list[dict] = Field(default_factory=list)
    failed: list[dict] = Field(default_factory=list)
    close_competition: bool = False
    message: str = ""
    reason: str = ""
