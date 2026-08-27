from __future__ import annotations

from visionrestore.tools.definitions import ToolDefinition


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ValueError(f"tool already registered: {definition.name}")
        self._tools[definition.name] = definition

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def list_definitions(self) -> list[dict]:
        return [
            {
                "name": item.name,
                "description": item.description,
                "version": item.version,
                "risk_level": item.risk_level,
                "supports_cancel": item.supports_cancel,
                "input_schema": item.input_schema.model_json_schema(),
            }
            for item in self._tools.values()
        ]
