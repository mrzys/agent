import httpx
from typing import Optional
from . import tool
from pydantic import Field
from trafilatura import extract
from trafilatura.metadata import extract_metadata
import chardet

import logging

looger = logging.getLogger("web_fetcher_v2")


@tool
def web_fetcher_v2(
    url: str = Field(..., description="The URL to fetch content from directly"),
    timeout: Optional[int] = Field(
        default=30,
        description="Request timeout in seconds (default: 30)",
    ),
    max_length: Optional[int] = Field(
        default=50000,
        description="Maximum length of content to return (default: 10000 characters)",
    ),
):
    """
    Fetch raw content directly from a URL without using Jina AI Reader.
    Returns the raw HTML/text response from the target URL.

    Args:
        url: The URL to fetch content from
        timeout: Request timeout in seconds (default: 30)
        max_length: Maximum length of content to return (default: 10000 characters)

    Returns:
        Raw content from the URL, truncated to max_length if necessary
    """
    if not isinstance(url, str) or len(url.strip()) == 0:
        return "Error: URL must be a non-empty string."

    if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
        return "Error: timeout must be a positive integer."

    if max_length is not None and (not isinstance(max_length, int) or max_length <= 0):
        return "Error: max_length must be a positive integer."

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()

            # Handle encoding detection to avoid garbled text
            content_bytes = response.content

            detected = chardet.detect(content_bytes)
            encoding = detected.get("encoding", "utf-8")
            try:
                html_text = content_bytes.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                html_text = content_bytes.decode("utf-8", errors="replace")

            # Extract metadata to get title
            meta = extract_metadata(html_text)
            title = meta.title
            # Get text content
            content = extract(
                html_text,
                include_comments=False,
                include_formatting=True,
                include_images=False,
            )

            # Truncate if too long
            if len(content) > max_length:
                content = (
                    content[:max_length]
                    + f"\n\n[Content truncated. Total length: {len(html_text)} characters]"
                )
            elif len(content) < 50:
                looger.warning("The response's content is less than 50 characters")
                return "This URL is not unavailable temporary."
            return f"Title: {title}\nURL: {url}\nText: {content}\n"

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"Error: URL not found (404): {url}"
        elif e.response.status_code == 403:
            return f"Error: Access forbidden (403). The URL may block automated requests: {url}"
        elif e.response.status_code == 429:
            return f"Error: Rate limit exceeded (429). Please wait before making more requests: {url}"
        else:
            return f"Error: HTTP {e.response.status_code} - {e.response.text}"
    except httpx.TimeoutException:
        return f"Error: Request timed out after {timeout} seconds. The URL might be unresponsive: {url}"
    except httpx.ConnectError:
        return f"Error: Failed to connect to the URL. Please check your network connection or the URL: {url}"
    except Exception as e:
        return f"Unexpected error occurred: {str(e)}"


if __name__ == "__main__":
    # Test the tool
    # print(web_fetcher_v2.to_openai_json())
    # Example usage:
    result = web_fetcher_v2(
        "https://www.news.cn/politics/leaders/20260326/3f5cfc907d884022ba4567b2151780bd/c.html"
    )
    print(result)
