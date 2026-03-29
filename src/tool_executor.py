from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tool import Tool


@dataclass
class ToolResult:
    tool_call_id: str
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None


class ToolExecutor:
    def __init__(self, tools: List[Tool] = None):
        self._tools: Dict[str, Tool] = {}
        if tools:
            self.register_tools(tools)

    def register_tool(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_tools(self, tools: List[Tool]) -> None:
        for tool in tools:
            self.register_tool(tool)
