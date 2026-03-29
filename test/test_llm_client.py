import unittest
from pathlib import Path
import sys
from unittest import mock

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


class MockToolCallDelta:
    def __init__(self, index, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = MockFunction(name, arguments)


class MockFunction:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class TestLLMClientToolCallCollection(unittest.TestCase):
    def test_collect_single_tool_call(self):
        client = LLMClient(model="gpt-4")
        buffer = {}
        deltas = [
            MockToolCallDelta(
                index=0, id="call-123", name="get_weather", arguments='{"loc'
            )
        ]

        client._collect_tool_calls(buffer, deltas)

        self.assertEqual(buffer[0]["id"], "call-123")
        self.assertEqual(buffer[0]["name"], "get_weather")
        self.assertEqual(buffer[0]["arguments"], '{"loc')

    def test_collect_multiple_tool_calls(self):
        client = LLMClient(model="gpt-4")
        buffer = {}
        deltas = [
            MockToolCallDelta(index=0, id="call-1", name="func_a", arguments="{}"),
            MockToolCallDelta(index=1, id="call-2", name="func_b", arguments='{"x":1}'),
        ]

        client._collect_tool_calls(buffer, deltas)

        self.assertEqual(len(buffer), 2)
        self.assertEqual(buffer[0]["name"], "func_a")
        self.assertEqual(buffer[1]["name"], "func_b")

    def test_accumulate_arguments_across_chunks(self):
        client = LLMClient(model="gpt-4")
        buffer = {}

        deltas1 = [
            MockToolCallDelta(index=0, id="call-abc", name="get_", arguments='{"ci')
        ]
        deltas2 = [MockToolCallDelta(index=0, name="weather", arguments='ty": "Bos')]
        deltas3 = [MockToolCallDelta(index=0, arguments='ton"}')]

        client._collect_tool_calls(buffer, deltas1)
        client._collect_tool_calls(buffer, deltas2)
        client._collect_tool_calls(buffer, deltas3)

        self.assertEqual(buffer[0]["id"], "call-abc")
        self.assertEqual(buffer[0]["name"], "get_weather")
        self.assertEqual(buffer[0]["arguments"], '{"city": "Boston"}')

    def test_build_tool_calls_from_buffer(self):
        client = LLMClient(model="gpt-4")
        buffer = {
            0: {"id": "call-1", "name": "search", "arguments": '{"query": "test"}'}
        }

        result = client._build_tool_calls(buffer)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "call-1")
        self.assertEqual(result[0].name, "search")
        self.assertEqual(result[0].arguments, '{"query": "test"}')

    def test_build_tool_calls_sorted_by_index(self):
        client = LLMClient(model="gpt-4")
        buffer = {
            2: {"id": "call-3", "name": "third", "arguments": "{}"},
            0: {"id": "call-1", "name": "first", "arguments": "{}"},
            1: {"id": "call-2", "name": "second", "arguments": "{}"},
        }

        result = client._build_tool_calls(buffer)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].name, "first")
        self.assertEqual(result[1].name, "second")
        self.assertEqual(result[2].name, "third")

    def test_build_tool_calls_empty_buffer(self):
        client = LLMClient(model="gpt-4")
        buffer = {}

        result = client._build_tool_calls(buffer)

        self.assertEqual(result, [])

    def test_collect_tool_calls_empty_deltas(self):
        client = LLMClient(model="gpt-4")
        buffer = {}
        deltas = []

        client._collect_tool_calls(buffer, deltas)

        self.assertEqual(buffer, {})


class MockDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class MockChoice:
    def __init__(self, delta):
        self.delta = delta


class MockChunk:
    def __init__(self, delta):
        self.choices = [MockChoice(delta)]


class MockStreamResponse:
    def __init__(self, chunks, created=None):
        self.chunks = chunks
        self.created = created

    def __iter__(self):
        return iter(self.chunks)


class TestLLMClientStream(unittest.TestCase):
    def test_stream_returns_llm_response(self):
        client = LLMClient(model="gpt-4")
        messages = [{"role": "user", "content": "Hello"}]

        mock_chunk = MockChunk(MockDelta(content="Hi there!"))
        mock_response = MockStreamResponse([mock_chunk], created=1234567890)

        with mock.patch("llm_client.completion", return_value=mock_response):
            result = client.stream(messages)

        self.assertIsInstance(result, LLMResponse)
        self.assertEqual(result.content, "Hi there!")
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.created, 1234567890)

    def test_stream_accumulates_content(self):
        client = LLMClient(model="gpt-4")
        messages = [{"role": "user", "content": "Hello"}]

        chunks = [
            MockChunk(MockDelta(content="Hello ")),
            MockChunk(MockDelta(content="world")),
            MockChunk(MockDelta(content="!")),
        ]
        mock_response = MockStreamResponse(chunks, created=9999999999)

        with mock.patch("llm_client.completion", return_value=mock_response):
            result = client.stream(messages)

        self.assertEqual(result.content, "Hello world!")


def test_stream_with_tool_calls(self):
    client = LLMClient(model="gpt-4")
    messages = [{"role": "user", "content": "What's the weather?"}]

    mock_tool_call_delta = MockToolCallDelta(
        index=0, id="call-123", name="get_weather", arguments='{"city": "Boston"}'
    )
    chunks = [
        MockChunk(MockDelta(content="Let me check ")),
        MockChunk(MockDelta(content="the weather.", tool_calls=[mock_tool_call_delta])),
    ]
    mock_response = MockStreamResponse(chunks, created=1111111111)

    with mock.patch("llm_client.completion", return_value=mock_response):
        result = client.stream(messages)

    self.assertEqual(result.content, "Let me check the weather.")
    self.assertEqual(len(result.tool_calls), 1)
    self.assertEqual(result.tool_calls[0].id, "call-123")
    self.assertEqual(result.tool_calls[0].name, "get_weather")
    self.assertEqual(result.tool_calls[0].arguments, '{"city": "Boston"}')


if __name__ == "__main__":
    unittest.main()
