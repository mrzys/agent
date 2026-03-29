import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import ToolCall, LLMResponse, LLMClient


class TestToolCall(unittest.TestCase):
    def test_tool_call_creation(self):
        tool_call = ToolCall(
            id="call-123", name="get_weather", arguments='{"location": "Boston"}'
        )

        self.assertEqual(tool_call.id, "call-123")
        self.assertEqual(tool_call.name, "get_weather")
        self.assertEqual(tool_call.arguments, '{"location": "Boston"}')

    def test_tool_call_to_dict(self):
        tool_call = ToolCall(
            id="call-456", name="search", arguments='{"query": "python"}'
        )

        result = tool_call.to_dict()

        self.assertEqual(result["id"], "call-456")
        self.assertEqual(result["type"], "function")
        self.assertEqual(result["function"]["name"], "search")
        self.assertEqual(result["function"]["arguments"], '{"query": "python"}')


class TestLLMResponse(unittest.TestCase):
    def test_llm_response_creation(self):
        tool_calls = [
            ToolCall(id="call-1", name="func1", arguments="{}"),
            ToolCall(id="call-2", name="func2", arguments='{"x": 1}'),
        ]

        response = LLMResponse(
            content="Hello world", tool_calls=tool_calls, created=1234567890
        )

        self.assertEqual(response.content, "Hello world")
        self.assertEqual(len(response.tool_calls), 2)
        self.assertEqual(response.tool_calls[0].name, "func1")
        self.assertEqual(response.tool_calls[1].name, "func2")
        self.assertEqual(response.created, 1234567890)

    def test_llm_response_empty_tool_calls(self):
        response = LLMResponse(content="No tools", tool_calls=[], created=9999999999)

        self.assertEqual(response.content, "No tools")
        self.assertEqual(response.tool_calls, [])
        self.assertEqual(response.created, 9999999999)


class TestLLMClient(unittest.TestCase):
    def test_llm_client_init_with_model(self):
        client = LLMClient(model="gpt-4")

        self.assertEqual(client.model, "gpt-4")
        self.assertEqual(client.tools_schema, [])

    def test_llm_client_init_with_tools_schema(self):
        tools = [{"type": "function", "function": {"name": "get_weather"}}]

        client = LLMClient(model="gpt-3.5-turbo", tools_schema=tools)

        self.assertEqual(client.model, "gpt-3.5-turbo")
        self.assertEqual(client.tools_schema, tools)

    def test_llm_client_init_none_tools_schema(self):
        client = LLMClient(model="gpt-4", tools_schema=None)

        self.assertEqual(client.model, "gpt-4")
        self.assertEqual(client.tools_schema, [])


if __name__ == "__main__":
    unittest.main()
