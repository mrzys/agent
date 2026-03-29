from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class LLMResponse:
    content: str
    tool_calls: List[ToolCall]
    created: int


class LLMClient:
    def __init__(self, model: str, tools_schema: List[Dict[str, Any]] = None):
        self.model = model
        self.tools_schema = tools_schema if tools_schema else []

    def _collect_tool_calls(
        self, buffer: Dict[int, Dict[str, str]], deltas: List[Any]
    ) -> None:
        for tc_delta in deltas:
            idx = tc_delta.index
            if idx not in buffer:
                buffer[idx] = {"id": tc_delta.id, "name": "", "arguments": ""}
            if tc_delta.function and tc_delta.function.name:
                buffer[idx]["name"] += tc_delta.function.name
            if tc_delta.function and tc_delta.function.arguments:
                buffer[idx]["arguments"] += tc_delta.function.arguments

    def _build_tool_calls(self, buffer: Dict[int, Dict[str, str]]) -> List[ToolCall]:
        return [
            ToolCall(
                id=buffer[idx]["id"],
                name=buffer[idx]["name"],
                arguments=buffer[idx]["arguments"],
            )
            for idx in sorted(buffer.keys())
        ]
