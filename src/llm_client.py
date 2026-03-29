from dataclasses import dataclass
from time import time
from typing import Any, Dict, List

from litellm import completion


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

    def stream(
        self, messages: List[Dict[str, Any]], with_tools: bool = True
    ) -> LLMResponse:
        response = completion(
            model=self.model,
            messages=messages,
            stream=True,
            tools=self.tools_schema if with_tools and self.tools_schema else None,
        )

        created = response.created or int(time())
        content = ""
        tool_calls_buffer: Dict[int, Dict[str, str]] = {}

        for chunk in response:
            delta = chunk.choices[0].delta
            content += delta.content or ""
            if delta.tool_calls:
                self._collect_tool_calls(tool_calls_buffer, delta.tool_calls)

        tool_calls = self._build_tool_calls(tool_calls_buffer)

        return LLMResponse(content=content, tool_calls=tool_calls, created=created)
