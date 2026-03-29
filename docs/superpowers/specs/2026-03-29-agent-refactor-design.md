# Agent Refactor Design

**Date**: 2026-03-29
**Status**: Approved

## Summary

Refactor `src/agent.py` to extract `LLMClient` and `ToolExecutor` collaborator classes, reducing Agent's responsibilities from 4 to 1 (orchestration).

## Problem

The current `Agent` class violates Single Responsibility Principle:

- LLM API calls and streaming (should be LLMClient)
- Tool call buffering and collection (should be LLMClient)
- Tool execution and error handling (should be ToolExecutor)
- Session management (stays in Agent)
- Orchestration (stays in Agent)

This makes the code:
- Hard to test (can't mock LLM or tools independently)
- Hard to extend (changes affect multiple concerns)
- Contains dead code (`Mode` enum, `_build_prompt` method)

## Solution

Extract two collaborator classes:

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                      Agent                          │
│  - Coordinates components                           │
│  - Manages think loop                               │
│  - Public API: chat(), think()                      │
└───────────────┬─────────────────┬───────────────────┘
                │                 │
        ┌───────▼───────┐ ┌───────▼────────┐
        │   LLMClient   │ │  ToolExecutor  │
        │ - stream()    │ │ - execute()    │
        │ - buffering   │ │ - registry     │
        └───────────────┘ └────────────────┘
                │
        ┌───────▼───────┐
        │    Session    │
        │ (unchanged)   │
        └───────────────┘
```

### File Structure

```
src/
├── agent.py          # REFACTORED - orchestration only (~60 lines)
├── llm_client.py     # NEW - LLMClient, ToolCall, LLMResponse
├── tool_executor.py  # NEW - ToolExecutor, ToolResult
├── session.py        # UNCHANGED
├── message.py        # MINOR - fix timestap typo
└── tool/__init__.py  # UNCHANGED
```

## Detailed Design

### LLMClient (`src/llm_client.py`)

**Responsibilities:**
- Call LLM API with streaming
- Buffer tool call deltas during streaming
- Return structured `LLMResponse`

**Data Classes:**

```python
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments}
        }

@dataclass
class LLMResponse:
    content: str
    tool_calls: List[ToolCall]
    created: int
```

**Class Interface:**

```python
class LLMClient:
    def __init__(self, model: str, tools_schema: List[Dict] = None):
        self.model = model
        self.tools_schema = tools_schema or []
    
    def stream(self, messages: List[Dict], with_tools: bool = True) -> LLMResponse:
        """Stream LLM response, collect content and tool calls."""
        response = completion(
            model=self.model,
            messages=messages,
            stream=True,
            tools=self.tools_schema if with_tools and self.tools_schema else None,
        )
        
        created = response.created or int(time())
        content = ""
        tool_calls_buffer = {}
        
        for chunk in response:
            delta = chunk.choices[0].delta
            content += delta.content or ""
            if delta.tool_calls:
                self._collect_tool_calls(tool_calls_buffer, delta.tool_calls)
        
        tool_calls = self._build_tool_calls(tool_calls_buffer)
        
        return LLMResponse(content=content, tool_calls=tool_calls, created=created)
    
    def _collect_tool_calls(self, buffer: Dict, deltas: List) -> None:
        """Accumulate tool call deltas into buffer."""
        for tc_delta in deltas:
            idx = tc_delta.index
            if idx not in buffer:
                buffer[idx] = {"id": tc_delta.id, "name": "", "arguments": ""}
            if tc_delta.function.name:
                buffer[idx]["name"] += tc_delta.function.name
            if tc_delta.function.arguments:
                buffer[idx]["arguments"] += tc_delta.function.arguments
    
    def _build_tool_calls(self, buffer: Dict) -> List[ToolCall]:
        """Convert buffer to final tool call list."""
        return [
            ToolCall(
                id=buffer[idx]["id"],
                name=buffer[idx]["name"],
                arguments=buffer[idx]["arguments"]
            )
            for idx in sorted(buffer.keys())
        ]
```

### ToolExecutor (`src/tool_executor.py`)

**Responsibilities:**
- Register and manage available tools
- Validate tool arguments
- Execute tools and handle errors gracefully
- Format results for session

**Data Class:**

```python
@dataclass
class ToolResult:
    tool_call_id: str
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
```

**Class Interface:**

```python
class ToolExecutor:
    def __init__(self, tools: List[Tool] = None):
        self._tools: Dict[str, Tool] = {}
        if tools:
            self.register_tools(tools)
    
    def register_tool(self, tool: Tool) -> None:
        """Register a single tool."""
        self._tools[tool.name] = tool
    
    def register_tools(self, tools: List[Tool]) -> None:
        """Register multiple tools."""
        for tool in tools:
            self.register_tool(tool)
    
    def get_schema(self) -> List[Dict]:
        """Return OpenAI-compatible tools schema."""
        return [tool.to_openai_format() for tool in self._tools.values()]
    
    def execute(self, tool_call_id: str, tool_name: str, arguments: str) -> ToolResult:
        """Execute a tool call with error handling."""
        if tool_name not in self._tools:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found"
            )
        
        tool = self._tools[tool_name]
        try:
            args = json.loads(arguments) if arguments else {}
            result = tool.execute(**args)
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                success=True,
                result=result
            )
        except json.JSONDecodeError as e:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Invalid JSON arguments: {e}"
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e)
            )
    
    def execute_batch(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute multiple tool calls."""
        return [self.execute(tc.id, tc.name, tc.arguments) for tc in tool_calls]
```

### Agent (`src/agent.py` - Refactored)

**Responsibilities:**
- Coordinate LLMClient, ToolExecutor, and Session
- Manage think loop with iteration limit
- Provide simple public API: `chat()` and `think()`

```python
class Agent:
    def __init__(
        self,
        name: str,
        model: str,
        system_prompt: str,
        session_id: str = None,
        tools: List[Tool] = None,
        max_think_iterations: int = 100,
    ):
        self.name = name
        self.session_id = session_id or str(uuid.uuid4())
        self.session = Session(session_id=self.session_id)
        
        if session_id is None:
            self.session.add_message(Role.SYSTEM, system_prompt)
        
        self._tool_executor = ToolExecutor(tools)
        self._llm_client = LLMClient(model, self._tool_executor.get_schema())
        self._max_think_iterations = max_think_iterations
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def chat(self, user_input: str) -> str:
        """Process user input and return response."""
        self.session.add_message(Role.USER, user_input)
        return self._think()
    
    def think(self) -> str:
        """Continue conversation without new input."""
        return self._think()
    
    def _think(self, iteration: int = 0) -> str:
        """Internal think loop."""
        messages = self.session.to_openai_format()
        response = self._llm_client.stream(messages)
        
        self.session.add_message(
            role=Role.ASSISTANT,
            content=response.content,
            timestamp=response.created,
            tool_calls=[tc.to_dict() for tc in response.tool_calls] if response.tool_calls else None,
        )
        
        if response.tool_calls:
            return self._handle_tool_calls(response.tool_calls, iteration)
        
        return response.content
    
    def _handle_tool_calls(self, tool_calls: List[ToolCall], iteration: int) -> str:
        """Execute tools and continue think loop."""
        results = self._tool_executor.execute_batch(tool_calls)
        
        for result in results:
            self.session.add_message(
                Role.TOOL,
                result.error if not result.success else str(result.result),
                name=result.tool_name,
                tool_call_id=result.tool_call_id,
            )
        
        if iteration >= self._max_think_iterations:
            self._logger.warning(f"Reached max think iterations ({self._max_think_iterations})")
            return f"[Warning: Reached max iterations ({self._max_think_iterations})]"
        
        return self._think(iteration + 1)
```

### Message (`src/message.py` - Minor Fix)

Fix typo: `timestap` → `timestamp`

## Error Handling

**Before:**
- Tool errors logged but not returned to LLM
- JSON parse errors silently became `{}`
- Missing tools logged but no feedback to LLM

**After:**
- All errors wrapped in `ToolResult` with `success=False`
- Error messages added to session so LLM can react
- Clear error types (tool not found, JSON error, execution error)

## Testing Strategy

**Unit Tests:**

1. `LLMClient` tests:
   - Mock `litellm.completion`, verify tool call buffering
   - Test streaming with and without tool calls
   - Test edge cases (empty response, multiple tool calls)

2. `ToolExecutor` tests:
   - Mock tools, test validation
   - Test error handling (missing tool, JSON error, execution error)
   - Test `execute_batch`

3. `Agent` tests:
   - Mock `LLMClient` and `ToolExecutor`
   - Test think loop iteration limit
   - Test error propagation to session

**Integration Tests:**
- Full conversation flow (optional, requires API key)

## Migration Path

1. Create `src/llm_client.py` with `LLMClient`, `ToolCall`, `LLMResponse`
2. Create `src/tool_executor.py` with `ToolExecutor`, `ToolResult`
3. Refactor `src/agent.py` to use new classes
4. Fix typo in `src/message.py`
5. Add/update tests
6. Remove dead code

## Removed Code

- `Mode` enum (never used)
- `_build_prompt` method (never called)
- All streaming logic (moved to `LLMClient`)
- All tool execution logic (moved to `ToolExecutor`)

## Metrics

| Metric | Before | After |
|--------|--------|-------|
| Agent.py lines | ~190 | ~60 |
| Classes per file | 2 in 1 file | 3 in 3 files |
| Agent responsibilities | 4 | 1 |
| Testability | Hard to mock | Easy to mock each component |
| Error handling | Logged only | Returned to LLM |