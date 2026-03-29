import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tool_executor import ToolResult, ToolExecutor
from tool import Tool
from pydantic import BaseModel, Field
from llm_client import ToolCall


class DummyParams(BaseModel):
    value: str = Field(..., description="A test value")


def dummy_function(value: str) -> str:
    return f"Processed: {value}"


class TestToolResult(unittest.TestCase):
    def test_tool_result_success(self):
        result = ToolResult(
            tool_call_id="call_123",
            tool_name="test_tool",
            success=True,
            result="Success output",
        )

        self.assertEqual(result.tool_call_id, "call_123")
        self.assertEqual(result.tool_name, "test_tool")
        self.assertTrue(result.success)
        self.assertEqual(result.result, "Success output")
        self.assertIsNone(result.error)

    def test_tool_result_failure(self):
        result = ToolResult(
            tool_call_id="call_456",
            tool_name="test_tool",
            success=False,
            result=None,
            error="Something went wrong",
        )

        self.assertEqual(result.tool_call_id, "call_456")
        self.assertEqual(result.tool_name, "test_tool")
        self.assertFalse(result.success)
        self.assertIsNone(result.result)
        self.assertEqual(result.error, "Something went wrong")


class TestToolExecutor(unittest.TestCase):
    def test_tool_executor_init_empty(self):
        executor = ToolExecutor()

        self.assertEqual(len(executor._tools), 0)
        self.assertIsInstance(executor._tools, dict)

    def test_tool_executor_init_with_tools(self):
        tool1 = Tool(
            name="tool1",
            description="First tool",
            parameters=DummyParams,
            func=dummy_function,
        )
        tool2 = Tool(
            name="tool2",
            description="Second tool",
            parameters=DummyParams,
            func=dummy_function,
        )

        executor = ToolExecutor(tools=[tool1, tool2])

        self.assertEqual(len(executor._tools), 2)
        self.assertIn("tool1", executor._tools)
        self.assertIn("tool2", executor._tools)
        self.assertEqual(executor._tools["tool1"], tool1)
        self.assertEqual(executor._tools["tool2"], tool2)

    def test_register_tool(self):
        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters=DummyParams,
            func=dummy_function,
        )

        executor = ToolExecutor()
        executor.register_tool(tool)

        self.assertEqual(len(executor._tools), 1)
        self.assertIn("test_tool", executor._tools)
        self.assertEqual(executor._tools["test_tool"], tool)

    def test_register_tools(self):
        tool1 = Tool(
            name="tool1",
            description="First tool",
            parameters=DummyParams,
            func=dummy_function,
        )
        tool2 = Tool(
            name="tool2",
            description="Second tool",
            parameters=DummyParams,
            func=dummy_function,
        )

        executor = ToolExecutor()
        executor.register_tools([tool1, tool2])

        self.assertEqual(len(executor._tools), 2)
        self.assertIn("tool1", executor._tools)
        self.assertIn("tool2", executor._tools)

    def test_get_schema_empty(self):
        executor = ToolExecutor()

        schema = executor.get_schema()

        self.assertEqual(schema, [])

    def test_get_schema_with_tools(self):
        tool1 = Tool(
            name="tool1",
            description="First tool",
            parameters=DummyParams,
            func=dummy_function,
        )
        tool2 = Tool(
            name="tool2",
            description="Second tool",
            parameters=DummyParams,
            func=dummy_function,
        )

        executor = ToolExecutor(tools=[tool1, tool2])
        schema = executor.get_schema()

        self.assertEqual(len(schema), 2)
        schema_names = [s["function"]["name"] for s in schema]
        self.assertIn("tool1", schema_names)
        self.assertIn("tool2", schema_names)

    def test_execute_success(self):
        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters=DummyParams,
            func=dummy_function,
        )

        executor = ToolExecutor(tools=[tool])
        result = executor.execute("call_123", "test_tool", '{"value": "hello"}')

        self.assertEqual(result.tool_call_id, "call_123")
        self.assertEqual(result.tool_name, "test_tool")
        self.assertTrue(result.success)
        self.assertEqual(result.result, "Processed: hello")
        self.assertIsNone(result.error)

    def test_execute_tool_not_found(self):
        executor = ToolExecutor()
        result = executor.execute("call_123", "unknown_tool", "{}")

        self.assertEqual(result.tool_call_id, "call_123")
        self.assertEqual(result.tool_name, "unknown_tool")
        self.assertFalse(result.success)
        self.assertIsNone(result.result)
        self.assertEqual(result.error, "Tool 'unknown_tool' not found")

    def test_execute_invalid_json(self):
        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters=DummyParams,
            func=dummy_function,
        )

        executor = ToolExecutor(tools=[tool])
        result = executor.execute("call_123", "test_tool", "not valid json")

        self.assertEqual(result.tool_call_id, "call_123")
        self.assertEqual(result.tool_name, "test_tool")
        self.assertFalse(result.success)
        self.assertIsNone(result.result)
        self.assertIn("Invalid JSON arguments", result.error)

    def test_execute_batch_empty(self):
        executor = ToolExecutor()
        results = executor.execute_batch([])

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
