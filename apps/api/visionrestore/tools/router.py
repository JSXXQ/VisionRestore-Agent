from __future__ import annotations

import time
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from visionrestore.tools.definitions import ToolContext, ToolResult
from visionrestore.tools.registry import ToolRegistry


class ToolRouter:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, name: str, payload: dict, context: ToolContext) -> ToolResult:
        definition = self.registry.get(name)
        call_id = str(uuid4())
        started = time.perf_counter()
        try:
            validated = definition.input_schema.model_validate(payload)
            raw = definition.handler(validated, context)
            if isinstance(raw, BaseModel):
                output = raw.model_dump(mode="json")
            elif isinstance(raw, dict):
                output = raw
            else:
                output = {"value": raw}
            return ToolResult(
                call_id=call_id,
                tool_name=name,
                tool_version=definition.version,
                success=True,
                output=output,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        except ValidationError as exc:
            return ToolResult(
                call_id=call_id,
                tool_name=name,
                tool_version=definition.version,
                success=False,
                error_code="TOOL_INPUT_INVALID",
                error_message=str(exc),
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:
            return ToolResult(
                call_id=call_id,
                tool_name=name,
                tool_version=definition.version,
                success=False,
                error_code=f"TOOL_{name.upper()}_FAILED",
                error_message=str(exc),
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
