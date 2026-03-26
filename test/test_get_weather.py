import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tool.get_weather import get_weather, get_geo


class TestGetWeather(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.mock_api_key = "test_api_key_12345"
        self.mock_geo_response = {
            "status": "1",
            "info": "OK",
            "geocodes": [
                {
                    "formatted_address": "湖北省武汉市",
                    "country": "中国",
                    "province": "湖北省",
                    "citycode": "027",
                    "city": "武汉市",
                    "district": "",
                    "adcode": "420100",
                    "location": "114.305469,30.592739",
                    "level": "市",
                }
            ],
        }
        self.mock_weather_response = {
            "status": "1",
            "info": "OK",
            "infocode": "10000",
            "forecasts": [
                {
                    "city": "武汉市",
                    "adcode": "420100",
                    "province": "湖北",
                    "reporttime": "2024-01-15 10:30:00",
                    "casts": [
                        {
                            "date": "2024-01-15",
                            "week": "1",
                            "dayweather": "晴",
                            "nightweather": "多云",
                            "daytemp": "15",
                            "nighttemp": "5",
                        },
                        {
                            "date": "2024-01-16",
                            "week": "2",
                            "dayweather": "阴",
                            "nightweather": "小雨",
                            "daytemp": "12",
                            "nighttemp": "3",
                        },
                    ],
                }
            ],
        }

    @patch.dict(os.environ, {"AMAP_API_KEY": "test_api_key_12345"})
    @patch("tool.get_weather.httpx.Client")
    def test_get_geo_success(self, mock_client_class):
        """Test get_geo function with successful API response"""
        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = self.mock_geo_response
        mock_response.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        # Call function
        result = get_geo("武汉")

        # Assert
        self.assertEqual(result, "420100")
        mock_client.get.assert_called_once_with(
            "https://restapi.amap.com/v3/geocode/geo",
            params={"address": "武汉", "key": "test_api_key_12345"},
        )

    @patch.dict(os.environ, {"AMAP_API_KEY": "test_api_key_12345"})
    @patch("tool.get_weather.httpx.Client")
    def test_get_geo_empty_response(self, mock_client_class):
        """Test get_geo with empty geocodes response"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"geocodes": []}
        mock_response.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        with self.assertRaises(IndexError):
            get_geo("UnknownLocation")

    @patch.dict(os.environ, {"AMAP_API_KEY": "test_api_key_12345"})
    @patch("tool.get_weather.httpx.Client")
    def test_get_weather_success(self, mock_client_class):
        """Test get_weather function with successful API response"""
        # Setup mock to handle both API calls
        mock_client = MagicMock()

        # Create separate mock responses for geo and weather
        def mock_get(url, **kwargs):
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            if "geocode/geo" in url:
                mock_response.json.return_value = self.mock_geo_response
            elif "weatherInfo" in url:
                mock_response.json.return_value = self.mock_weather_response
            return mock_response

        mock_client.get = mock_get
        mock_client_class.return_value.__enter__.return_value = mock_client

        # Call function
        result = get_weather(location="武汉")

        # Assert
        self.assertIn("武汉的天气", result)
        self.assertIn("2024-01-15", result)
        self.assertIn("晴", result)
        self.assertIn("15°C", result)
        self.assertIn("多云", result)
        self.assertIn("5°C", result)

    @patch.dict(os.environ, {"AMAP_API_KEY": "test_api_key_12345"})
    @patch("tool.get_weather.get_geo")
    @patch("tool.get_weather.httpx.Client")
    def test_get_weather_multiple_days(self, mock_client_class, mock_get_geo):
        """Test get_weather returns multiple days forecast"""
        mock_get_geo.return_value = "420100"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = self.mock_weather_response
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        result = get_weather(location="武汉")

        # Check both days are in result
        self.assertIn("2024-01-15", result)
        self.assertIn("2024-01-16", result)

    @patch.dict(os.environ, {"AMAP_API_KEY": "test_api_key_12345"})
    @patch("tool.get_weather.httpx.Client")
    def test_get_weather_api_error(self, mock_client_class):
        """Test get_weather handles API error"""
        import httpx

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.HTTPError("API Error")
        mock_client_class.return_value.__enter__.return_value = mock_client

        with self.assertRaises(httpx.HTTPError):
            get_weather(location="武汉")

    @patch.dict(os.environ, {"AMAP_API_KEY": "test_api_key_12345"})
    def test_get_weather_tool_validation(self):
        """Test that get_weather tool validates parameters"""
        # Test calling without required parameter
        result = get_weather()

        # Should return error dict due to validation
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
        self.assertFalse(result.get("success", True))

    @patch.dict(os.environ, {"AMAP_API_KEY": "test_api_key_12345"})
    def test_get_weather_tool_with_valid_param(self):
        """Test get_weather tool with valid parameter"""
        with patch("tool.get_weather.httpx.Client") as mock_client_class:
            mock_client = MagicMock()

            def mock_get(url, **kwargs):
                mock_response = MagicMock()
                mock_response.raise_for_status = MagicMock()
                if "geocode/geo" in url:
                    mock_response.json.return_value = self.mock_geo_response
                elif "weatherInfo" in url:
                    mock_response.json.return_value = self.mock_weather_response
                return mock_response

            mock_client.get = mock_get
            mock_client_class.return_value.__enter__.return_value = mock_client

            # Test with valid location
            result = get_weather(location="北京")

            self.assertIsInstance(result, str)
            self.assertIn("北京", result)


class TestGetWeatherToolProperties(unittest.TestCase):
    """Test Tool wrapper properties"""

    def test_tool_name(self):
        """Test tool has correct name"""
        self.assertEqual(get_weather.name, "get_weather")

    def test_tool_description(self):
        """Test tool has description"""
        self.assertIsNotNone(get_weather.description)
        self.assertIn("weather", get_weather.description.lower())

    def test_tool_parameters(self):
        """Test tool has parameters model"""
        self.assertIsNotNone(get_weather.parameters)

    def test_tool_to_openai_json(self):
        """Test tool can export to OpenAI format"""
        schema = get_weather.to_openai_format()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "get_weather")
        self.assertIn("parameters", schema["function"])


if __name__ == "__main__":
    unittest.main()
