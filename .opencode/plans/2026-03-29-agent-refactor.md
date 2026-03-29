# Agent Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor agent.py to extract LLMClient and ToolExecutor collaborator classes, improving testability and reducing Agent responsibilities.

**Architecture:** Extract LLMClient (LLM API calls, streaming, tool call buffering) and ToolExecutor (tool registry, execution, error handling) into separate files. Agent becomes an orchestrator coordinating these components.

**Tech Stack:** Python 3.13, litellm, pydantic, unittest for testing

---

## File Structure

```
src/
├── llm_client.py      # NEW - LLMClient, ToolCall, LLMResponse dataclasses
├── tool_executor.py   # NEW - ToolExecutor, ToolResult dataclass
├── agent.py           # REFACTORED - orchestration only
├── message.py         # MINOR FIX - timestap → timestamp typo
├── session.py         # UNCHANGED
└── tool/__init__.py   # UNCHANGED

test/
├── test_llm_client.py # NEW
├── test_tool_executor.py # NEW
└── test_agent.py      # NEW
```

---

### Task 1: Create LLMClient with ToolCall and LLMResponse dataclasses

**Files:**
- Create: `src/llm_client.py`
- Test: `test/test_llm_client.py`

- [ ] **Step 1: Write the failing test for ToolCall dataclass**

```python
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import ToolCall


class TestToolCall(unittest.TestCase):
    def test_tool_call_creation(self):
        tc = ToolCall(id="call-123", name="get_weather", arguments='{"location": "北京"}')
        self.assertEqual(tc.id, "call-123")
        self.assertEqual(tc.name, "get_weather")
        self.assertEqual(tc.arguments, '{"location": "北京"}')
    
    def test_tool_call_to_dict(self):
        tc = ToolCall(id="call-123", name="get_weather", arguments='{"location": "北京"}')
        result = tc.to_dict()
        self.assertEqual(result["id"], "call-123")
        self.assertEqual(result["type"], "function")
        self.assertEqual(result["function"]["name"], "get_weather")
        self.assertEqual(result["function"]["arguments"], '{"location": "北京"}')


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_llm_client -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'llm_client'"

- [ ] **Step 3: Write ToolCall dataclass**

```python
from dataclasses import dataclass
from typing import Any, Dict, List


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_llm_client.TestToolCall -v`
Expected: PASS

- [ ] **Step 5: Write failing test for LLMResponse dataclass**

```python
from llm_client import LLMResponse


class TestLLMResponse(unittest.TestCase):
    def test_llm_response_creation(self):
        tc = ToolCall(id="call-123", name="get_weather", arguments='{"location": "北京"}')
        resp = LLMResponse(content="Hello", tool_calls=[tc], created=1700000000)
        self.assertEqual(resp.content, "Hello")
        self.assertEqual(resp.tool_calls, [tc])
        self.assertEqual(resp.created, 1700000000)
    
    def test_llm_response_empty_tool_calls(self):
        resp = LLMResponse(content="Hello", tool_calls=[], created=1700000000)
        self.assertEqual(resp.tool_calls, [])
```

Add to `test/test_llm_client.py` after `TestToolCall` class.

- [ ] **Step 6: Run test to verify it fails**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_llm_client.TestLLMResponse -v`
Expected: FAIL with "ImportError: cannot import name 'LLMResponse'"

- [ ] **Step 7: Write LLMResponse dataclass**

Add to `src/llm_client.py`:

```python
@dataclass
class LLMResponse:
    content: str
    tool_calls: List[ToolCall]
    created: int
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_llm_client.TestLLMResponse -v`
Expected: PASS

- [ ] **Step 9: Write failing test for LLMClient initialization**

```python
from llm_client import LLMClient


class TestLLMClientInit(unittest.TestCase):
    def test_llm_client_init_with_model(self):
        client = LLMClient(model="deepseek/deepseek-chat")
        self.assertEqual(client.model, "deepseek/deepseek-chat")
        self.assertEqual(client.tools_schema, [])
    
    def test_llm_client_init_with_tools_schema(self):
        schema = [{"type": "function", "function": {"name": "test"}}]
        client = LLMClient(model="deepseek/deepseek-chat", tools_schema=schema)
        self.assertEqual(client.tools_schema, schema)
```

Add to `test/test_llm_client.py`.

- [ ] **Step 10: Run test to verify it fails**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_llm_client.TestLLMClientInit -v`
Expected: FAIL with "ImportError: cannot import name 'LLMClient'"

- [ ] **Step 11: Write LLMClient class init**

Add to `src/llm_client.py`:

```python
from typing import Dict, List, Any


class LLMClient:
    def __init__(self, model: str, tools_schema: List[Dict[str, Any]] = None):
        self.model = model
        self.tools_schema = tools_schema if tools_schema else []
```

- [ ] **Step 12: Run test to verify it passes**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_llm_client.TestLLMClientInit -v`
Expected: PASS

- [ ] **Step 13: Commit ToolCall, LLMResponse, and LLMClient init**

```bash
cd /Users/zhangyongsheng/Github/agent && git add src/llm_client.py test/test_llm_client.py && git commit -m "feat: add LLMClient with ToolCall and LLMResponse dataclasses"
```

---

### Task 2: Implement LLMClient streaming methods

**Files:**
- Modify: `src/llm_client.py`
- Modify: `test/test_llm_client.py`

- [ ] **Step 1: Write failing test for _collect_tool_calls**

```python
from unittest.mock import MagicMock
from llm_client import LLMClient


class TestLLMClientCollectToolCalls(unittest.TestCase):
    def test_collect_single_tool_call(self):
        client = LLMClient(model="test")
        buffer = {}
        
        tc_delta = MagicMock()
        tc_delta.index = 0
        tc_delta.id = "call-123"
        tc_delta.function.name = "get_weather"
        tc_delta.function.arguments = '{"location"'
        
        client._collect_tool_calls(buffer, [tc_delta])
        
        self.assertIn(0, buffer)
        self.assertEqual(buffer[0]["id"], "call-123")
        self.assertEqual(buffer[0]["name"], "get_weather")
        self.assertEqual(buffer[0]["arguments"], '{"location"')
    
    def test_collect_multiple_tool_calls(self):
        client = LLMClient(model="test")
        buffer = {}
        
        tc1 = MagicMock()
        tc1.index = 0
        tc1.id = "call-1"
        tc1.function.name = "get_weather"
        tc1.function.arguments = '{"loc'
        
        tc2 = MagicMock()
        tc2.index = 1
        tc2.id = "call-2"
        tc2.function.name = "web_search"
        tc2.function.arguments = '{"query'
        
        client._collect_tool_calls(buffer, [tc1, tc2])
        
        self.assertEqual(len(buffer), 2)
        self.assertEqual(buffer[0]["name"], "get_weather")
        self.assertEqual(buffer[1]["name"], "web_search")
    
    def test_accumulate_arguments_across_chunks(self):
        client = LLMClient(model="test")
        buffer = {}
        
        tc1 = MagicMock()
        tc1.index = 0
        tc1.id = "call-123"
        tc1.function.name = "get"
        tc1.function.arguments = '{"loc'
        
        tc2 = MagicMock()
        tc2.index = 0
        tc2.id = None
        tc2.function.name = None
        tc2.function.arguments = 'ation": "北京"}'
        
        client._collect_tool_calls(buffer, [tc1])
        client._collect_tool_calls(buffer, [tc2])
        
        self.assertEqual(buffer[0]["arguments"], '{"location": "北京"}')
        self.assertEqual(buffer[0]["name"], "get")
```

Add to `test/test_llm_client.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_llm_client.TestLLMClientCollectToolCalls -v`
Expected: FAIL with "AttributeError: 'LLMClient' object has no attribute '_collect_tool_calls'"

- [ ] **Step 3: Implement _collect_tool_calls method**

Add to `src/llm_client.py` in LLMClient class:

```python
    def _collect_tool_calls(self, buffer: Dict[int, Dict[str, str]], deltas: List) -> None:
        for tc_delta in deltas:
            idx = tc_delta.index
            if idx not in buffer:
                buffer[idx] = {"id": tc_delta.id, "name": "", "arguments": ""}
            if tc_delta.function.name:
                buffer[idx]["name"] += tc_delta.function.name
            if tc_delta.function.arguments:
                buffer[idx]["arguments"] += tc_delta.function.arguments
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_llm_client.TestLLMClientCollectToolCalls -v`
Expected: PASS

- [ ] **Step 5: Write failing test for _build_tool_calls**

```python
class TestLLMClientBuildToolCalls(unittest.TestCase):
    def test_build_tool_calls_from_buffer(self):
        client = LLMClient(model="test")
        buffer = {
            0: {"id": "call-1", "name": "get_weather", "arguments": '{"location": "北京"}'},
            1: {"id": "call-2", "name": "web_search", "arguments": '{"query": "test"}'}
        }
        
        result = client._build_tool_calls(buffer)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].id, "call-1")
        self.assertEqual(result[0].name, "get_weather")
        self.assertEqual(result[1].id, "call-2")
    
    def test_build_tool_calls_sorted_by_index(self):
        client = LLMClient(model="test")
        buffer = {
            2: {"id": "call-2", "name": "tool2", "arguments": "{}"},
            0: {"id": "call-0", "name": "tool0", "arguments": "{}"},
            1: {"id": "call-1", "name": "tool1", "arguments": "{}"}
        }
        
        result = client._build_tool_calls(buffer)
        
        self.assertEqual(result[0].id, "call-0")
        self.assertEqual(result[1].id, "call-1")
        self.assertEqual(result[2].id, "call-2")
    
    def test_build_tool_calls_empty_buffer(self):
        client = LLMClient(model="test")
        result = client._build_tool_calls({})
        self.assertEqual(result, [])
```

Add to `test/test_llm_client.py`.

- [ ] **Step 6: Run test to verify it fails**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_llm_client.TestLLMClientBuildToolCalls -v`
Expected: FAIL with "AttributeError: 'LLMClient' object has no attribute '_build_tool_calls'"

- [ ] **Step 7: Implement _build_tool_calls method**

Add to `src/llm_client.py` in LLMClient class:

```python
    def _build_tool_calls(self, buffer: Dict[int, Dict[str, str]]) -> List[ToolCall]:
        return [
            ToolCall(
                id=buffer[idx]["id"],
                name=buffer[idx]["name"],
                arguments=buffer[idx]["arguments"]
            )
            for idx in sorted(buffer.keys())
        ]
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_llm_client.TestLLMClientBuildToolCalls -v`
Expected: PASS

- [ ] **Step 9: Commit LLMClient collection methods**

```bash
cd /Users/zhangyongsheng/Github/agent && git add src/llm_client.py test/test_llm_client.py && git commit -m "feat: add LLMClient tool call collection methods"
```

---

### Task 3: Implement LLMClient.stream() method

**Files:**
- Modify: `src/llm_client.py`
- Modify: `test/test_llm_client.py`

- [ ] **Step 1: Write failing test for stream() without tool calls**

```python
from unittest.mock import patch, MagicMock
from llm_client import LLMClient, LLMResponse


class TestLLMClientStream(unittest.TestCase):
    @patch("llm_client.completion")
    def test_stream_returns_llm_response(self, mock_completion):
        client = LLMClient(model="test-model")
        
        mock_response = MagicMock()
        mock_response.created = 1700000000
        mock_response.__iter__ = MagicMock(return_value=iter([
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello", tool_calls=None))])
        ]))
        mock_completion.return_value = mock_response
        
        messages = [{"role": "user", "content": "test"}]
        result = client.stream(messages)
        
        self.assertIsInstance(result, LLMResponse)
        self.assertEqual(result.content, "Hello")
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.created, 1700000000)
    
    @patch("llm_client.completion")
    def test_stream_accumulates_content(self, mock_completion):
        client = LLMClient(model="test-model")
        
        mock_response = MagicMock()
        mock_response.created = 1700000000
        mock_response.__iter__ = MagicMock(return_value=iter([
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Hel", tool_calls=None))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="lo", tool_calls=None))]),
        ]))
        mock_completion.return_value = mock_response
        
        messages = [{"role": "user", "content": "test"}]
        result = client.stream(messages)
        
        self.assertEqual(result.content, "Hello")
```

Add to `test/test_llm_client.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_llm_client.TestLLMClientStream.test_stream_returns_llm_response -v`
Expected: FAIL with "AttributeError: 'LLMClient' object has no attribute 'stream'"

- [ ] **Step 3: Implement stream() method**

Add imports and method to `src/llm_client.py`:

```python
from time import time
from litellm import completion

class LLMClient:
    ...existing init...
    
    def stream(self, messages: List[Dict[str, Any]], with_tools: bool = True) -> LLMResponse:
        response = completion(
            model=self.model,
            messages=messages,
            stream=True,
            tools=self.tools_schema if with_tools and self.tools_schema else None,
        )
        
        created = response.created or int(time())
        content = ""
        tool_calls_buffer: Dict[int, Dict[str, str]] = {}
        
        for chunk in response:
            delta = chunk.choices[0].delta
            content += delta.content or ""
            if delta.tool_calls:
                self._collect_tool_calls(tool_calls_buffer, delta.tool_calls)
        
        tool_calls = self._build_tool_calls(tool_calls_buffer)
        
        return LLMResponse(content=content, tool_calls=tool_calls, created=created)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_llm_client.TestLLMClientStream -v`
Expected: PASS

- [ ] **Step 5: Write failing test for stream() with tool calls**

```python
    @patch("llm_client.completion")
    def test_stream_with_tool_calls(self, mock_completion):
        client = LLMClient(model="test-model")
        
        tc_delta1 = MagicMock()
        tc_delta1.index = 0
        tc_delta1.id = "call-123"
        tc_delta1.function.name = "get_weather"
        tc_delta1.function.arguments = '{"location"'
        
        tc_delta2 = MagicMock()
        tc_delta2.index = 0
        tc_delta2.id = None
        tc_delta2.function.name = None
        tc_delta2.function.arguments = ': "北京"}'
        
        mock_response = MagicMock()
        mock_response.created = 1700000000
        mock_response.__iter__ = MagicMock(return_value=iter([
            MagicMock(choices=[MagicMock(delta=MagicMock(content="", tool_calls=[tc_delta1]))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="", tool_calls=[tc_delta2]))]),
        ]))
        mock_completion.return_value = mock_response
        
        messages = [{"role": "user", "content": "test"}]
        result = client.stream(messages)
        
        self.assertEqual(result.content, "")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].id, "call-123")
        self.assertEqual(result.tool_calls[0].name, "get_weather")
        self.assertEqual(result.tool_calls[0].arguments, '{"location": "北京"}')
```

Add to `TestLLMClientStream` class.

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_llm_client.TestLLMClientStream.test_stream_with_tool_calls -v`
Expected: PASS (stream() already handles tool calls via _collect_tool_calls)

- [ ] **Step 7: Commit LLMClient.stream() method**

```bash
cd /Users/zhangyongsheng/Github/agent && git add src/llm_client.py test/test_llm_client.py && git commit -m "feat: add LLMClient.stream() method with tool call handling"
```

---

### Task 4: Create ToolExecutor with ToolResult dataclass

**Files:**
- Create: `src/tool_executor.py`
- Create: `test/test_tool_executor.py`

- [ ] **Step 1: Write failing test for ToolResult dataclass**

```python
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tool_executor import ToolResult


class TestToolResult(unittest.TestCase):
    def test_tool_result_success(self):
        result = ToolResult(
            tool_call_id="call-123",
            tool_name="get_weather",
            success=True,
            result="晴天",
            error=None
        )
        self.assertEqual(result.tool_call_id, "call-123")
        self.assertEqual(result.tool_name, "get_weather")
        self.assertTrue(result.success)
        self.assertEqual(result.result, "晴天")
        self.assertIsNone(result.error)
    
    def test_tool_result_failure(self):
        result = ToolResult(
            tool_call_id="call-123",
            tool_name="unknown",
            success=False,
            result=None,
            error="Tool 'unknown' not found"
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Tool 'unknown' not found")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_tool_executor.TestToolResult -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'tool_executor'"

- [ ] **Step 3: Write ToolResult dataclass**

Create `src/tool_executor.py`:

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json

from src.tool import Tool


@dataclass
class ToolResult:
    tool_call_id: str
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_tool_executor.TestToolResult -v`
Expected: PASS

- [ ] **Step 5: Write failing test for ToolExecutor initialization**

```python
from tool_executor import ToolExecutor
from tool.get_weather import get_weather


class TestToolExecutorInit(unittest.TestCase):
    def test_tool_executor_init_empty(self):
        executor = ToolExecutor()
        self.assertEqual(executor._tools, {})
    
    def test_tool_executor_init_with_tools(self):
        executor = ToolExecutor(tools=[get_weather])
        self.assertIn("get_weather", executor._tools)
    
    def test_register_tool(self):
        executor = ToolExecutor()
        executor.register_tool(get_weather)
        self.assertIn("get_weather", executor._tools)
```

Add to `test/test_tool_executor.py`.

- [ ] **Step 6: Run test to verify it fails**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_tool_executor.TestToolExecutorInit -v`
Expected: FAIL with "ImportError: cannot import name 'ToolExecutor'"

- [ ] **Step 7: Write ToolExecutor class init and register methods**

Add to `src/tool_executor.py`:

```python
class ToolExecutor:
    def __init__(self, tools: List[Tool] = None):
        self._tools: Dict[str, Tool] = {}
        if tools:
            self.register_tools(tools)
    
    def register_tool(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
    
    def register_tools(self, tools: List[Tool]) -> None:
        for tool in tools:
            self.register_tool(tool)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_tool_executor.TestToolExecutorInit -v`
Expected: PASS

- [ ] **Step 9: Commit ToolResult and ToolExecutor init**

```bash
cd /Users/zhangyongsheng/Github/agent && git add src/tool_executor.py test/test_tool_executor.py && git commit -m "feat: add ToolExecutor with ToolResult dataclass"
```

---

### Task 5: Implement ToolExecutor.get_schema() and execute()

**Files:**
- Modify: `src/tool_executor.py`
- Modify: `test/test_tool_executor.py`

- [ ] **Step 1: Write failing test for get_schema()**

```python
class TestToolExecutorGetSchema(unittest.TestCase):
    def test_get_schema_empty(self):
        executor = ToolExecutor()
        schema = executor.get_schema()
        self.assertEqual(schema, [])
    
    def test_get_schema_with_tools(self):
        executor = ToolExecutor(tools=[get_weather])
        schema = executor.get_schema()
        self.assertEqual(len(schema), 1)
        self.assertEqual(schema[0]["type"], "function")
        self.assertEqual(schema[0]["function"]["name"], "get_weather")
```

Add to `test/test_tool_executor.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_tool_executor.TestToolExecutorGetSchema -v`
Expected: FAIL with "AttributeError: 'ToolExecutor' object has no attribute 'get_schema'"

- [ ] **Step 3: Implement get_schema() method**

Add to `src/tool_executor.py` in ToolExecutor class:

```python
    def get_schema(self) -> List[Dict[str, Any]]:
        return [tool.to_openai_format() for tool in self._tools.values()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_tool_executor.TestToolExecutorGetSchema -v`
Expected: PASS

- [ ] **Step 5: Write failing test for execute() success case**

```python
class TestToolExecutorExecute(unittest.TestCase):
    def test_execute_success(self):
        executor = ToolExecutor(tools=[get_weather])
        result = executor.execute(
            tool_call_id="call-123",
            tool_name="get_weather",
            arguments='{"location": "北京"}'
        )
        self.assertEqual(result.tool_call_id, "call-123")
        self.assertEqual(result.tool_name, "get_weather")
        self.assertTrue(result.success)
```

Add to `test/test_tool_executor.py`.

Note: This test will pass if get_weather is properly mocked or if API works. We'll focus on structure validation.

- [ ] **Step 6: Run test to verify it fails**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_tool_executor.TestToolExecutorExecute.test_execute_success -v`
Expected: FAIL with "AttributeError: 'ToolExecutor' object has no attribute 'execute'"

- [ ] **Step 7: Implement execute() method**

Add to `src/tool_executor.py` in ToolExecutor class:

```python
    def execute(self, tool_call_id: str, tool_name: str, arguments: str) -> ToolResult:
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
```

- [ ] **Step 8: Run test**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_tool_executor.TestToolExecutorExecute.test_execute_success -v`
Expected: PASS (or may need API key for get_weather)

- [ ] **Step 9: Write test for execute() error cases**

```python
    def test_execute_tool_not_found(self):
        executor = ToolExecutor()
        result = executor.execute(
            tool_call_id="call-123",
            tool_name="unknown_tool",
            arguments='{}'
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Tool 'unknown_tool' not found")
    
    def test_execute_invalid_json(self):
        executor = ToolExecutor(tools=[get_weather])
        result = executor.execute(
            tool_call_id="call-123",
            tool_name="get_weather",
            arguments='invalid json'
        )
        self.assertFalse(result.success)
        self.assertIn("Invalid JSON", result.error)
```

Add to `TestToolExecutorExecute` class.

- [ ] **Step 10: Run test to verify passes**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_tool_executor.TestToolExecutorExecute -v`
Expected: PASS

- [ ] **Step 11: Write test for execute_batch()**

```python
class TestToolExecutorExecuteBatch(unittest.TestCase):
    def test_execute_batch_empty(self):
        executor = ToolExecutor()
        results = executor.execute_batch([])
        self.assertEqual(results, [])
```

Add to `test/test_tool_executor.py`.

- [ ] **Step 12: Run test to verify it fails**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_tool_executor.TestToolExecutorExecuteBatch -v`
Expected: FAIL with "AttributeError: 'ToolExecutor' object has no attribute 'execute_batch'"

- [ ] **Step 13: Implement execute_batch() method**

Add to `src/tool_executor.py`:
Import ToolCall from llm_client:

```python
from llm_client import ToolCall
```

Add method:

```python
    def execute_batch(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        return [self.execute(tc.id, tc.name, tc.arguments) for tc in tool_calls]
```

- [ ] **Step 14: Run test to verify it passes**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_tool_executor.TestToolExecutorExecuteBatch -v`
Expected: PASS

- [ ] **Step 15: Commit ToolExecutor methods**

```bash
cd /Users/zhangyongsheng/Github/agent && git add src/tool_executor.py test/test_tool_executor.py && git commit -m "feat: add ToolExecutor.get_schema(), execute(), and execute_batch()"
```

---

### Task 6: Fix timestamp typo in message.py

**Files:**
- Modify: `src/message.py`
- Modify: `src/session.py` (update usage)

- [ ] **Step 1: Identify typo**

Current: `timestap` in Message class (line 31 of message.py)
Expected: `timestamp`

- [ ] **Step 2: Fix typo in message.py**

Change line 31:
```python
    timestamp: int
```

And update any usage of `timestap` in message.py if present.

- [ ] **Step 3: Update session.py usage**

In `src/session.py`, the `add_message` method uses `timestamp` correctly. Check if there are any `timestap` references and fix them.

- [ ] **Step 4: Commit typo fix**

```bash
cd /Users/zhangyongsheng/Github/agent && git add src/message.py src/session.py && git commit -m "fix: correct timestap typo to timestamp"
```

---

### Task 7: Refactor Agent to use LLMClient and ToolExecutor

**Files:**
- Modify: `src/agent.py`
- Create: `test/test_agent.py`

- [ ] **Step 1: Write failing test for refactored Agent init**

```python
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent import Agent
from tool.get_current_date import get_current_date


class TestAgentInit(unittest.TestCase):
    def test_agent_init_creates_session(self):
        agent = Agent(
            name="test",
            model="test-model",
            system_prompt="You are a test agent",
            tools=[get_current_date]
        )
        self.assertIsNotNone(agent.session_id)
        self.assertIsNotNone(agent.session)
    
    def test_agent_init_with_session_id(self):
        agent = Agent(
            name="test",
            model="test-model",
            system_prompt="You are a test agent",
            session_id="test-session-123"
        )
        self.assertEqual(agent.session_id, "test-session-123")
    
    def test_agent_has_tool_executor(self):
        agent = Agent(
            name="test",
            model="test-model",
            system_prompt="test",
            tools=[get_current_date]
        )
        self.assertIsNotNone(agent._tool_executor)
        self.assertIn("get_current_date", agent._tool_executor._tools)
    
    def test_agent_has_llm_client(self):
        agent = Agent(
            name="test",
            model="test-model",
            system_prompt="test"
        )
        self.assertIsNotNone(agent._llm_client)
        self.assertEqual(agent._llm_client.model, "test-model")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify current state**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_agent.TestAgentInit -v`
Expected: May PASS or FAIL depending on current Agent state. Tests will verify refactored structure.

- [ ] **Step 3: Refactor Agent.__init__()**

Replace current Agent class in `src/agent.py`:

```python
import uuid
import logging
from typing import List

from src.session import Session
from src.message import Role
from src.tool import Tool
from src.llm_client import LLMClient, ToolCall
from src.tool_executor import ToolExecutor


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
        self._logger = logging.getLogger(self.__class__.__name__)
        
        self.session_id = session_id or str(uuid.uuid4())
        self._logger.info(f"Initializing agent with session_id: {self.session_id}")
        
        self.session = Session(session_id=self.session_id)
        if session_id is None:
            self.session.add_message(Role.SYSTEM, system_prompt)
        
        self._tool_executor = ToolExecutor(tools)
        self._llm_client = LLMClient(model, self._tool_executor.get_schema())
        self._max_think_iterations = max_think_iterations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_agent.TestAgentInit -v`
Expected: PASS

- [ ] **Step 5: Write test for Agent.chat() and think()**

```python
from unittest.mock import patch, MagicMock
from llm_client import LLMResponse, ToolCall


class TestAgentChat(unittest.TestCase):
    @patch("agent.LLMClient")
    def test_chat_adds_user_message(self, mock_llm_client_class):
        mock_llm = MagicMock()
        mock_llm.stream.return_value = LLMResponse(content="Hello!", tool_calls=[], created=1700000000)
        mock_llm_client_class.return_value = mock_llm
        
        agent = Agent(
            name="test",
            model="test-model",
            system_prompt="test"
        )
        agent._llm_client = mock_llm
        
        response = agent.chat("Hi")
        
        self.assertEqual(response, "Hello!")
    
    @patch("agent.LLMClient")
    def test_think_without_user_input(self, mock_llm_client_class):
        mock_llm = MagicMock()
        mock_llm.stream.return_value = LLMResponse(content="Thinking...", tool_calls=[], created=1700000000)
        mock_llm_client_class.return_value = mock_llm
        
        agent = Agent(
            name="test",
            model="test-model",
            system_prompt="test"
        )
        agent._llm_client = mock_llm
        
        response = agent.think()
        
        self.assertEqual(response, "Thinking...")
```

Add to `test/test_agent.py`.

- [ ] **Step 6: Run test to verify it fails**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_agent.TestAgentChat -v`
Expected: May fail or pass based on current state

- [ ] **Step 7: Implement Agent.chat() and think()**

Add to Agent class in `src/agent.py`:

```python
    def chat(self, user_input: str) -> str:
        self.session.add_message(Role.USER, user_input)
        return self._think()
    
    def think(self) -> str:
        return self._think()
```

- [ ] **Step 8: Implement Agent._think()**

Add to Agent class:

```python
    def _think(self, iteration: int = 0) -> str:
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
```

- [ ] **Step 9: Implement Agent._handle_tool_calls()**

Add to Agent class:

```python
    def _handle_tool_calls(self, tool_calls: List[ToolCall], iteration: int) -> str:
        results = self._tool_executor.execute_batch(tool_calls)
        
        for result in results:
            content = result.error if not result.success else str(result.result)
            self.session.add_message(
                Role.TOOL,
                content,
                name=result.tool_name,
                tool_call_id=result.tool_call_id,
            )
        
        if iteration >= self._max_think_iterations:
            self._logger.warning(f"Reached max think iterations ({self._max_think_iterations})")
            return f"[Warning: Reached max iterations ({self._max_think_iterations})]"
        
        return self._think(iteration + 1)
```

- [ ] **Step 10: Run test to verify it passes**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_agent.TestAgentChat -v`
Expected: PASS

- [ ] **Step 11: Write test for think iteration limit**

```python
class TestAgentThinkLoop(unittest.TestCase):
    @patch("agent.LLMClient")
    def test_think_iteration_limit(self, mock_llm_client_class):
        mock_llm = MagicMock()
        tc = ToolCall(id="call-1", name="test_tool", arguments="{}")
        mock_llm.stream.return_value = LLMResponse(content="", tool_calls=[tc], created=1700000000)
        
        agent = Agent(
            name="test",
            model="test-model",
            system_prompt="test",
            max_think_iterations=2
        )
        agent._llm_client = mock_llm
        
        response = agent.think()
        
        self.assertIn("Warning", response)
        self.assertIn("max iterations", response)
```

Add to `test/test_agent.py`.

- [ ] **Step 12: Run test**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_agent.TestAgentThinkLoop -v`
Expected: PASS

- [ ] **Step 13: Remove dead code from agent.py**

Remove from `src/agent.py`:
- `Mode` enum class
- `_build_prompt` method
- Old `_process_streaming_response` method
- Old `_collect_tool_calls` method
- Old `_build_tool_calls` method (now in LLMClient)
- Old `_execute_tool_calls` method (now in ToolExecutor)

- [ ] **Step 14: Update imports in agent.py**

Final imports in `src/agent.py`:

```python
import uuid
import logging
from typing import List

from src.session import Session
from src.message import Role
from src.tool import Tool
from src.llm_client import LLMClient, ToolCall
from src.tool_executor import ToolExecutor
```

- [ ] **Step 15: Run all agent tests**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest test.test_agent -v`
Expected: PASS

- [ ] **Step 16: Commit refactored Agent**

```bash
cd /Users/zhangyongsheng/Github/agent && git add src/agent.py test/test_agent.py && git commit -m "refactor: extract LLMClient and ToolExecutor from Agent"
```

---

### Task 8: Run full test suite and verify integration

**Files:**
- All test files

- [ ] **Step 1: Run all tests**

Run: `cd /Users/zhangyongsheng/Github/agent && python -m unittest discover test -v`
Expected: All tests PASS

- [ ] **Step 2: Verify agent.py main block still works**

Check that the `if __name__ == "__main__"` block in agent.py works correctly with refactored code.

Update imports if needed:
```python
if __name__ == "__main__":
    from src.tool.get_current_date import get_current_date
    from src.tool.get_weather import get_weather
    from src.tool.web_search import web_search
    from src.tool.web_fetcher_v2 import web_fetcher_v2
    
    agent = Agent(
        name="news-assistant",
        model="deepseek/deepseek-chat",
        system_prompt="...",
        tools=[get_weather, get_current_date, web_fetcher_v2, web_search],
        max_think_iterations=10,
    )
    resp = agent.think()
    print(resp)
```

- [ ] **Step 3: Final commit**

```bash
cd /Users/zhangyongsheng/Github/agent && git add -A && git commit -m "refactor: complete agent refactoring with extracted collaborators"
```

---

## Self-Review Checklist

After writing this plan, verify:

1. **Spec coverage**: Each section of the spec has a corresponding task
2. **Placeholder scan**: No TBD, TODO, or vague instructions
3. **Type consistency**: Method names and signatures match across tasks

Run after implementation:
- `python -m unittest discover test -v` - all tests pass
- Agent main block works without errors
- LLMClient handles streaming correctly
- ToolExecutor handles errors and returns to session
- Agent orchestrates components correctly