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
