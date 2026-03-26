import os

import httpx
from . import tool
from pydantic import Field


def get_geo(location: str) -> str:
    """Get the geocode for a specific location."""
    api_key = os.getenv("AMAP_API_KEY")
    with httpx.Client() as client:
        response = client.get(
            "https://restapi.amap.com/v3/geocode/geo",
            params={"address": location, "key": api_key},
        )
        response.raise_for_status()
        resp_json = response.json()
        return resp_json["geocodes"][0]["adcode"]


@tool
def get_weather(
    location: str = Field(
        ..., description="The location for which to get weather information"
    ),
) -> str:
    """
    Get the weather for a specific location

    Args:
        location: The location for which to get weather information. example: "武汉"
    Returns:
        A string describing the weather for the specified location.
    """
    api_key = os.getenv("AMAP_API_KEY")
    with httpx.Client() as client:
        ad_code = get_geo(location)

        response = client.get(
            "https://restapi.amap.com/v3/weather/weatherInfo",
            params={"city": ad_code, "key": api_key, "extensions": "all"},
        )

        resp_json = response.json()

        result = f"{location}的天气：\n"
        print(resp_json)
        for weather_info in resp_json["forecasts"][0]["casts"]:
            result += f"{weather_info['date']}：白天，{weather_info['dayweather']}，夜晚，{weather_info['nightweather']}。最高温度{weather_info['daytemp']}°C，最低温度{weather_info['nighttemp']}°C\n"

    return result


if __name__ == "__main__":
    print(get_weather.to_openai_format())
