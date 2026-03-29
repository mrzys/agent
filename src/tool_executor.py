import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tool import Tool
from llm_client import ToolCall


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

    def get_schema(self) -> List[Dict[str, Any]]:
        return [tool.to_openai_format() for tool in self._tools.values()]

    def execute(self, tool_call_id: str, tool_name: str, arguments: str) -> ToolResult:
        if tool_name not in self._tools:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found",
            )

        tool = self._tools[tool_name]
        try:
            args = json.loads(arguments) if arguments else {}
            result = tool.execute(**args)
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                success=True,
                result=result,
            )
        except json.JSONDecodeError as e:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Invalid JSON arguments: {e}",
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
            )

    def execute_batch(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        return [self.execute(tc.id, tc.name, tc.arguments) for tc in tool_calls]
