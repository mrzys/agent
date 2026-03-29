# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- **LLMClient component** - Extracted LLM API handling into dedicated class
  - `ToolCall` dataclass with `to_dict()` method for OpenAI format
  - `LLMResponse` dataclass for structured responses
  - `stream()` method for streaming LLM responses with tool call buffering
  - `_collect_tool_calls()` and `_build_tool_calls()` helper methods

- **ToolExecutor component** - Extracted tool execution into dedicated class
  - `ToolResult` dataclass with success/error handling
  - `register_tool()` and `register_tools()` for tool registry
  - `get_schema()` for OpenAI-compatible tool schema
  - `execute()` with error handling (missing tool, invalid JSON, execution errors)
  - `execute_batch()` for multiple tool calls

- **Agent tests** - Comprehensive test suite for refactored components
  - `test/test_llm_client.py` - 17 tests for LLMClient
  - `test/test_tool_executor.py` - 12 tests for ToolExecutor
  - `test/test_agent.py` - 9 tests for Agent orchestration

### Changed

- **Agent refactored** - Reduced from ~190 lines to ~84 lines
  - Now uses `LLMClient` for streaming and tool call buffering
  - Now uses `ToolExecutor` for tool registry and execution
  - `session_id` generation moved to `Session` class
  - Added `session_id` property for backward compatibility
  - Removed `Mode` enum (unused)
  - Removed `_build_prompt` method (unused)

- **Session improved** - Better timestamp handling
  - Fixed `kwargs.get()` to `kwargs.pop()` to prevent duplicate timestamp

- **Message model** - Fixed typo
  - Renamed `timestap` to `timestamp` in usage

### Fixed

- Added null checks for streaming edge cases (empty choices, null delta)
- Fixed `response.created` fallback to use `is not None` check
- Added type hints for `deltas` parameter in `_collect_tool_calls()`
- Fixed `test_storage.py` to include required `timestamp` field in Message creation
- Fixed import in `test_storage.py` to use correct `ToolCall` from `message` module
- Fixed import paths in `tool_executor.py` to use absolute imports (`src.tool`, `src.llm_client`)
- Fixed session initialization to add system prompt when session is empty (not just for new sessions)
- Added detailed logging for tool execution errors with input arguments and exception details

### Refactored

- Moved `src/tool.py` to `src/tool/__init__.py` for better package structure
- Updated default port in `main.py` from 5000 to 5001
- Updated default model from `dashscope/qwen3-max` to `deepseek/deepseek-chat`

### Documentation

- Added design spec at `docs/superpowers/specs/2026-03-29-agent-refactor-design.md`
- Added implementation plan at `docs/superpowers/plans/2026-03-29-agent-refactor.md`

---

## Previous Releases

### [0.1.0] - Initial Structure

- Initial project structure with Agent, Session, Message, Tool, and Storage components
- Basic LLM integration with litellm
- Tool system with pydantic validation
- File-based session persistence