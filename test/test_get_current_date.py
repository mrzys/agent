import unittest
from datetime import datetime
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tool.get_current_date import get_current_date


class TestGetCurrentDate(unittest.TestCase):
    def test_get_current_date_returns_string(self):
        """Test that get_current_date returns a string"""
        result = get_current_date()
        self.assertIsInstance(result, str)

    def test_get_current_date_format(self):
        """Test that the returned date is in correct format"""
        result = get_current_date()
        # Should be in format: YYYY-MM-DD HH:MM:SS
        try:
            parsed = datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
            self.assertIsNotNone(parsed)
        except ValueError:
            self.fail(f"Date format is incorrect: {result}")

    def test_get_current_date_is_recent(self):
        """Test that the returned date is recent (within last minute)"""
        result = get_current_date()
        parsed = datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
        now = datetime.now()

        # Should be within last minute
        diff = (now - parsed).total_seconds()
        self.assertLessEqual(diff, 60, "Date should be recent")

    def test_tool_name(self):
        """Test tool has correct name"""
        self.assertEqual(get_current_date.name, "get_current_date")

    def test_tool_description(self):
        """Test tool has description"""
        self.assertIsNotNone(get_current_date.description)
        self.assertIn("current", get_current_date.description.lower())

    def test_tool_parameters(self):
        """Test tool has empty parameters (no arguments needed)"""
        self.assertIsNotNone(get_current_date.parameters)
        schema = get_current_date.parameters.model_json_schema()
        # Should have no required parameters
        self.assertEqual(schema.get("properties", {}), {})

    def test_tool_to_openai_json(self):
        """Test tool can export to OpenAI format"""
        schema = get_current_date.to_openai_format()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "get_current_date")
        self.assertIn("parameters", schema["function"])
        # Should have empty properties since no parameters
        self.assertEqual(schema["function"]["parameters"].get("properties", {}), {})

    def test_tool_is_callable(self):
        """Test that the tool can be called as a function"""
        # Tool decorator should make it callable
        result = get_current_date()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
