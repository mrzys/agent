import os
import httpx
from typing import Optional
from src.tool import tool
from pydantic import Field


@tool
def web_search(
    query: str = Field(..., description="The search query string. "),
    num_results: Optional[int] = Field(
        default=10,
        description="Number of search results to return (default: 10)",
    ),
):
    """
    Execute a professional search via Serper.dev.

    INSTRUCTIONS:
    1. Convert user intent into high-entropy keywords or Google Dorks.
    2. For technical/GitHub searches, ALWAYS use 'site:github.com'.
    3. Use double quotes for exact phrase matching (e.g., \"AI Agent\").
    4. For date-specific queries, use the 'after:YYYY-MM-DD' operator.
    5. STRIP OUT subjective or conversational words like 'worth researching', 'popular', 'best', 'about'.

    EXAMPLE: For 'GitHub projects in 2026', use: 'site:github.com \"AI Agent\" after:2025-12-31'.

    Args:
        query: The search query string
        num_results: Number of search results to return (default: 10)

    Returns:
        String containing formatted search results with titles, links, and snippets
    """
    api_key = os.environ.get("SERPER_API_KEY")

    if api_key is None:
        return "Error: SERPER_API_KEY environment variable not set. Please set the Serper API key in environment variables."

    if not isinstance(query, str) or len(query.strip()) == 0:
        return "Error: Query must be a non-empty string."

    if num_results is not None and (
        not isinstance(num_results, int) or num_results <= 0
    ):
        return "Error: num_results must be a positive integer."

    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "q": query,
    }

    if num_results:
        payload["num"] = min(num_results, 100)

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        return format_search_results(data)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return "Error: Invalid Serper API key. Please check your SERPER_API_KEY environment variable."
        elif e.response.status_code == 429:
            return (
                "Error: Rate limit exceeded. Please wait before making more requests."
            )
        else:
            return f"Error: HTTP {e.response.status_code} - {e.response.text}"
    except httpx.TimeoutException:
        return "Error: Request timed out. Please try again."
    except Exception as e:
        return f"Unexpected error occurred: {str(e)}"


def format_search_results(data: dict) -> str:
    """
    Format the search results from Serper API response.

    Args:
        data: JSON response from Serper API

    Returns:
        Formatted string with search results
    """
    if "organic" not in data or not data["organic"]:
        return "No search results found."

    results = data["organic"]
    formatted_lines = [f"Found {len(results)} results:"]

    for idx, result in enumerate(results, 1):
        title = result.get("title", "No title")
        link = result.get("link", "No link")
        snippet = result.get("snippet", "No description")

        formatted_lines.append(f"\n{idx}. {title}")
        formatted_lines.append(f"   URL: {link}")
        formatted_lines.append(f"   {snippet}")

    return "\n".join(formatted_lines)
