from pydantic import BaseModel

from visionrestore.tools import ToolContext, ToolDefinition, ToolRegistry, ToolRouter


class EchoInput(BaseModel):
    value: int


def test_tool_router_validates_schema_and_records_metadata():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="test tool",
            input_schema=EchoInput,
            handler=lambda payload, _context: {"value": payload.value},
        )
    )
    router = ToolRouter(registry)
    context = ToolContext(task_id="task", run_id="run", step_id="step")

    success = router.execute("echo", {"value": 3}, context)
    invalid = router.execute("echo", {"value": "not-an-integer"}, context)

    assert success.success is True
    assert success.output == {"value": 3}
    assert success.call_id
    assert invalid.success is False
    assert invalid.error_code == "TOOL_INPUT_INVALID"
