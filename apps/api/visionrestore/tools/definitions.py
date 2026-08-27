from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field


class ToolContext(BaseModel):
    task_id: str
    run_id: str
    step_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    call_id: str
    tool_name: str
    tool_version: str
    success: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: int = 0


ToolHandler = Callable[[BaseModel, ToolContext], Any]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: type[BaseModel]
    handler: ToolHandler
    version: str = "1.0"
    risk_level: str = "local_read"
    supports_cancel: bool = False
