import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from message import Message, Role, ToolCall, ToolCallFunction
from session import Session
from storage import FileStorage


class TestFileStorage(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)

    def tearDown(self):
        """Clean up the temporary directory after each test"""
        os.chdir(self.original_dir)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_creates_sessions_directory(self):
        """Test that FileStorage creates .sessions directory if it doesn't exist"""
        session = Session("test-session-1")
        storage = FileStorage(session.session_id)

        self.assertTrue(os.path.exists(".sessions"))
        self.assertTrue(os.path.isdir(".sessions"))

        storage.close()

    def test_init_creates_file(self):
        """Test that FileStorage creates a file for the session"""
        session = Session("test-session-2")
        storage = FileStorage(session.session_id)

        expected_file = f".sessions/{session.session_id}.json"
        self.assertTrue(os.path.exists(expected_file))

        storage.close()

    def test_save_single_message(self):
        """Test saving a single message"""
        session = Session("test-session-3")
        storage = FileStorage(session.session_id)

        message = Message(role=Role.USER, content="Hello, World!", timestamp=1700000000)
        storage.save(message)

        expected_file = f".sessions/{session.session_id}.json"
        with open(expected_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 1)
        saved_data = json.loads(lines[0].strip())
        self.assertEqual(saved_data["role"], "user")
        self.assertEqual(saved_data["content"], "Hello, World!")

        storage.close()

    def test_save_multiple_messages(self):
        """Test saving multiple messages"""
        session = Session("test-session-4")
        storage = FileStorage(session.session_id)

        message1 = Message(
            role=Role.USER, content="First message", timestamp=1700000001
        )
        message2 = Message(
            role=Role.ASSISTANT, content="Second message", timestamp=1700000002
        )
        message3 = Message(
            role=Role.SYSTEM, content="Third message", timestamp=1700000003
        )

        storage.save(message1)
        storage.save(message2)
        storage.save(message3)

        expected_file = f".sessions/{session.session_id}.json"
        with open(expected_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 3)

        storage.close()

    def test_read_single_message(self):
        """Test reading a single message"""
        session = Session("test-session-5")
        storage = FileStorage(session.session_id)

        message = Message(role=Role.USER, content="Test message", timestamp=1700000000)
        storage.save(message)

        messages = list(storage.read())

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, Role.USER)
        self.assertEqual(messages[0].content, "Test message")

        storage.close()

    def test_read_multiple_messages(self):
        """Test reading multiple messages"""
        session = Session("test-session-6")
        storage = FileStorage(session.session_id)

        message1 = Message(role=Role.USER, content="Message 1", timestamp=1700000001)
        message2 = Message(
            role=Role.ASSISTANT, content="Message 2", timestamp=1700000002
        )
        message3 = Message(role=Role.TOOL, content="Message 3", timestamp=1700000003)

        storage.save(message1)
        storage.save(message2)
        storage.save(message3)

        messages = list(storage.read())

        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0].role, Role.USER)
        self.assertEqual(messages[0].content, "Message 1")
        self.assertEqual(messages[1].role, Role.ASSISTANT)
        self.assertEqual(messages[1].content, "Message 2")
        self.assertEqual(messages[2].role, Role.TOOL)
        self.assertEqual(messages[2].content, "Message 3")

        storage.close()

    def test_read_empty_file(self):
        """Test reading from an empty file"""
        session = Session("test-session-7")
        storage = FileStorage(session.session_id)

        messages = list(storage.read())

        self.assertEqual(len(messages), 0)

        storage.close()

    def test_read_with_blank_lines(self):
        """Test reading from a file with blank lines"""
        session = Session("test-session-8")
        storage = FileStorage(session.session_id)

        # Manually write with blank lines
        with open(f".sessions/{session.session_id}.json", "w", encoding="utf-8") as f:
            f.write('{"role": "user", "content": "Test", "timestamp": 1700000000}\n')
            f.write("\n")  # Blank line
            f.write(
                '{"role": "assistant", "content": "Response", "timestamp": 1700000001}\n'
            )

        # Reopen storage to read from beginning
        storage.file.close()
        storage.file = open(
            f".sessions/{session.session_id}.json", "a+", encoding="utf-8"
        )

        messages = list(storage.read())

        self.assertEqual(len(messages), 2)

        storage.close()

    def test_close_closes_file(self):
        """Test that close properly closes the file"""
        session = Session("test-session-9")
        storage = FileStorage(session.session_id)

        storage.close()

        # Should raise ValueError if trying to write to closed file
        with self.assertRaises(ValueError):
            storage.file.write("test")

    def test_message_with_tool_call(self):
        """Test saving and reading message with tool call"""
        session = Session("test-session-10")
        storage = FileStorage(session.session_id)

        tool_call = ToolCall(
            id="call-123",
            type="function",
            function=ToolCallFunction(
                name="test_function", arguments='{"arg": "value"}'
            ),
        )
        message = Message(
            role=Role.ASSISTANT,
            content="Using tool",
            tool_calls=[tool_call],
            timestamp=1700000000,
        )

        storage.save(message)

        messages = list(storage.read())

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].tool_calls[0].id, "call-123")
        self.assertEqual(messages[0].tool_calls[0].function.name, "test_function")

        storage.close()

    def test_reuse_session_id(self):
        """Test that same session ID reuses the same file"""
        session = Session("test-session-11")

        storage1 = FileStorage(session.session_id)
        message1 = Message(role=Role.USER, content="First", timestamp=1700000001)
        storage1.save(message1)
        storage1.close()

        storage2 = FileStorage(session.session_id)
        message2 = Message(role=Role.USER, content="Second", timestamp=1700000002)
        storage2.save(message2)
        storage2.close()

        # Reopen to read all messages
        storage3 = FileStorage(session.session_id)
        messages = list(storage3.read())
        storage3.close()

        self.assertEqual(len(messages), 2)


if __name__ == "__main__":
    unittest.main()
