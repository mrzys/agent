import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tool.web_fetcher_v2 import web_fetcher_v2


class TestWebFetcherV2(unittest.TestCase):
    """Test web_fetcher_v2 tool"""

    def test_fetch_xinhuanet_xml(self):
        """Test fetching content from xinhuanet.com XML feed"""
        url = "https://www.xinhuanet.com/politics/news_politics.xml"
        result = web_fetcher_v2(url=url)

        # Check that no error occurred
        self.assertFalse(result.startswith("Error:"), f"Request failed: {result}")

        # Check that we got content
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 100, "Content should not be empty or too short")

        # Check that it's XML content
        self.assertIn(
            "<?xml",
            result.lower() or "<rss" in result.lower() or "<" in result,
            "Response should be XML format",
        )

        # Check common XML feed elements
        self.assertIn("<", result, "Should contain XML tags")

        print(f"Successfully fetched {len(result)} characters from {url}")
        print(f"First 500 chars:\n{result[:1000]}...")

    def test_tool_name(self):
        """Test tool has correct name"""
        self.assertEqual(web_fetcher_v2.name, "web_fetcher_v2")

    def test_tool_description(self):
        """Test tool has description"""
        self.assertIsNotNone(web_fetcher_v2.description)
        self.assertIn("fetch", web_fetcher_v2.description.lower())

    def test_tool_parameters(self):
        """Test tool has parameters model"""
        self.assertIsNotNone(web_fetcher_v2.parameters)
        schema = web_fetcher_v2.parameters.model_json_schema()
        self.assertIn("properties", schema)
        # Should have url, timeout, max_length parameters
        self.assertIn("url", schema["properties"])
        self.assertIn("timeout", schema["properties"])
        self.assertIn("max_length", schema["properties"])

    def test_tool_to_openai_json(self):
        """Test tool can export to OpenAI format"""
        schema = web_fetcher_v2.to_openai_format()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "web_fetcher_v2")
        self.assertIn("parameters", schema["function"])

    def test_invalid_url(self):
        """Test handling of invalid URL"""
        result = web_fetcher_v2(url="")
        self.assertIn("Error:", result)

    def test_max_length_truncation(self):
        """Test that max_length parameter truncates content"""
        url = "https://www.xinhuanet.com/politics/news_politics.xml"
        max_length = 500
        result = web_fetcher_v2(url=url, max_length=max_length)

        # Check for truncation message if content was truncated
        if len(result) > max_length:
            self.assertIn("[Content truncated", result)


if __name__ == "__main__":
    unittest.main()
