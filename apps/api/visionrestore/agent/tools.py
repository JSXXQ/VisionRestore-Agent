class ToolRegistry:
    def __init__(self, **tools):
        self.tools = tools

    def call(self, name: str, *args, **kwargs):
        if name not in self.tools:
            raise KeyError(f"未注册 Agent 工具: {name}")
        return self.tools[name](*args, **kwargs)

    def list_tools(self) -> list[str]:
        return sorted(self.tools)
