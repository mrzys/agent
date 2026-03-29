import unittest
from pathlib import Path
import sys
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agent import Agent
from src.session import Session
from src.llm_client import LLMClient, ToolCall
from src.tool_executor import ToolExecutor
from src.message import Role
from src.tool import Tool
from pydantic import BaseModel, Field


class DummyParams(BaseModel):
    value: str = Field(..., description="A test value")


def dummy_function(value: str) -> str:
    return f"Processed: {value}"


class TestAgentInit(unittest.TestCase):
    def test_agent_init_creates_session(self):
        agent = Agent(
            name="test-agent",
            model="gpt-4",
            system_prompt="You are a test assistant.",
        )

        self.assertIsNotNone(agent.session_id)
        self.assertIsInstance(agent.session, Session)
        self.assertEqual(len(agent.session.message), 1)
        self.assertEqual(agent.session.message[0].role, Role.SYSTEM)
        self.assertEqual(agent.session.message[0].content, "You are a test assistant.")

    def test_agent_has_tool_executor(self):
        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters=DummyParams,
            func=dummy_function,
        )

        agent = Agent(
            name="test-agent",
            model="gpt-4",
            system_prompt="You are a test assistant.",
            tools=[tool],
        )

        self.assertIsInstance(agent._tool_executor, ToolExecutor)
        self.assertIn("test_tool", agent._tool_executor._tools)

    def test_agent_has_llm_client(self):
        agent = Agent(
            name="test-agent",
            model="gpt-4",
            system_prompt="You are a test assistant.",
        )

        self.assertIsInstance(agent._llm_client, LLMClient)
        self.assertEqual(agent._llm_client.model, "gpt-4")

    def test_agent_init_with_session_id(self):
        agent = Agent(
            name="test-agent",
            model="gpt-4",
            system_prompt="You are a test assistant.",
            session_id="test-session-123",
        )

        self.assertEqual(agent.session_id, "test-session-123")
        self.assertIsInstance(agent.session, Session)

    def test_agent_init_with_tools(self):
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

        agent = Agent(
            name="test-agent",
            model="gpt-4",
            system_prompt="You are a test assistant.",
            tools=[tool1, tool2],
        )

        self.assertIn("tool1", agent._tool_executor._tools)
        self.assertIn("tool2", agent._tool_executor._tools)
        self.assertEqual(len(agent._tool_executor._tools), 2)


class TestAgentChat(unittest.TestCase):
    def test_chat_adds_user_message(self):
        agent = Agent(
            name="test-agent",
            model="gpt-4",
            system_prompt="You are a test assistant.",
        )

        mock_response = mock.MagicMock()
        mock_response.content = "Hello! How can I help you?"
        mock_response.tool_calls = []
        mock_response.created = 1234567890

        with mock.patch.object(agent._llm_client, "stream", return_value=mock_response):
            result = agent.chat("Hello")

        self.assertEqual(result, "Hello! How can I help you?")
        self.assertEqual(len(agent.session.message), 3)
        self.assertEqual(agent.session.message[1].role, Role.USER)
        self.assertEqual(agent.session.message[1].content, "Hello")
        self.assertEqual(agent.session.message[2].role, Role.ASSISTANT)


class TestAgentThink(unittest.TestCase):
    def test_think_without_user_input(self):
        agent = Agent(
            name="test-agent",
            model="gpt-4",
            system_prompt="You are a test assistant.",
        )

        mock_response = mock.MagicMock()
        mock_response.content = "Thinking..."
        mock_response.tool_calls = []
        mock_response.created = 1234567890

        with mock.patch.object(agent._llm_client, "stream", return_value=mock_response):
            result = agent.think()

        self.assertEqual(result, "Thinking...")
        self.assertEqual(len(agent.session.message), 2)
        self.assertEqual(agent.session.message[1].role, Role.ASSISTANT)

    def test_think_with_tool_calls(self):
        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters=DummyParams,
            func=dummy_function,
        )

        agent = Agent(
            name="test-agent",
            model="gpt-4",
            system_prompt="You are a test assistant.",
            tools=[tool],
        )

        tool_call = ToolCall(
            id="call-123", name="test_tool", arguments='{"value": "hello"}'
        )

        first_response = mock.MagicMock()
        first_response.content = "Let me use the tool."
        first_response.tool_calls = [tool_call]
        first_response.created = 1234567890

        second_response = mock.MagicMock()
        second_response.content = "Final response."
        second_response.tool_calls = []
        second_response.created = 1234567891

        with mock.patch.object(
            agent._llm_client,
            "stream",
            side_effect=[first_response, second_response],
        ):
            result = agent.think()

        self.assertEqual(result, "Final response.")
        self.assertEqual(agent.session.message[1].role, Role.ASSISTANT)
        self.assertEqual(agent.session.message[2].role, Role.TOOL)
        self.assertEqual(agent.session.message[3].role, Role.ASSISTANT)

    def test_max_think_iterations(self):
        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters=DummyParams,
            func=dummy_function,
        )

        agent = Agent(
            name="test-agent",
            model="gpt-4",
            system_prompt="You are a test assistant.",
            tools=[tool],
            max_think_iterations=2,
        )

        tool_call = ToolCall(
            id="call-123", name="test_tool", arguments='{"value": "hello"}'
        )

        mock_response = mock.MagicMock()
        mock_response.content = "Using tool."
        mock_response.tool_calls = [tool_call]
        mock_response.created = 1234567890

        with mock.patch.object(agent._llm_client, "stream", return_value=mock_response):
            result = agent.think()

        self.assertIn("max iterations", result.lower())


if __name__ == "__main__":
    unittest.main()
