import httpx
import re
from typing import Optional
from src.tool import tool
from pydantic import Field


def remove_markdown_images(text: str) -> str:
    """Remove all markdown image links from text."""
    # Pattern matches ![alt text](image_url) or ![alt](url)
    pattern = r"!\[.*?\]\(.*?\)"
    return re.sub(pattern, "", text)


@tool
def web_fetcher(
    url: str = Field(..., description="The URL of the webpage to fetch content from"),
    timeout: Optional[int] = Field(
        default=30,
        description="Request timeout in seconds (default: 30)",
    ),
):
    """
    # Fetch and extract clean text content from a webpage using Jina AI Reader API.

    Args:
        url: The URL of the webpage to fetch content from
        timeout: Request timeout in seconds (default: 30)

    Returns:
        String containing the extracted text content from the webpage
    """
    if not isinstance(url, str) or len(url.strip()) == 0:
        return "Error: URL must be a non-empty string."

    if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
        return "Error: timeout must be a positive integer."

    jina_api_url = f"https://r.jina.ai/{url}"

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(jina_api_url)
            response.raise_for_status()
            return remove_markdown_images(response.text)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"Error: URL not found or invalid: {url}"
        elif e.response.status_code == 429:
            return (
                "Error: Rate limit exceeded. Please wait before making more requests."
            )
        else:
            return f"Error: HTTP {e.response.status_code} - {e.response.text}"
    except httpx.TimeoutException:
        return f"Error: Request timed out after {timeout} seconds. The URL might be unresponsive."
    except httpx.ConnectError:
        return "Error: Failed to connect to the URL. Please check your network connection or the URL."
    except Exception as e:
        return f"Unexpected error occurred: {str(e)}"
