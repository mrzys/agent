# Changes

## [2026-03-26|86f4bfb](https://github.com/mrzys/agent/commit/86f4bfb890785e8807e5cd71bea3c8d449d17f25)

### Refactored
- **`src/agent.py`**: Refactored the `Agent` class to move system prompt initialization from the constructor to dynamic building within the `chat()` method, allowing for more flexible, context-aware prompts.
- **`src/tool/web_fetcher_v2.py`**: Increased `max_length` from 10000 to 50000 characters to support larger webpage content extraction.

### Added
- **`PlanAgent`**: Introduced a new agent class in `src/agent.py` specialized in generating structured, executable XML plans based on user goals.
- **`plan.py`**: A new script demonstrating the usage of `PlanAgent` for research and planning tasks.
- **`src/prompt.py`**: Placeholder for future prompt construction logic.

### Removed
- **`research_agent.py`**: Cleaned up the project by removing the obsolete research agent script.
