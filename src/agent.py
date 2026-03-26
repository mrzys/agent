from enum import Enum
import json
from time import time
from typing import Any, Dict, List, Union
import uuid
import logging

from litellm import completion
from litellm.types.utils import ChatCompletionDeltaToolCall

from src.session import Session
from src.message import Role
from src.tool import Tool

logging.basicConfig(level=logging.INFO)


class Mode(str, Enum):
    PLAN = "plan"
    EXECUTE = "execute"


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
        self._logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        if session_id is None:
            self.session_id = str(uuid.uuid4())
        else:
            self.session_id = session_id
        self._logger.info(f"Initializing agent with session_id: {self.session_id}")
        self.session = Session(session_id=self.session_id)
        if session_id is None:
            self.session.add_message(Role.SYSTEM, system_prompt)

        self.tools = {tool.name: tool for tool in tools} if tools else {}
        self.tools_schema = [tool.to_openai_format() for tool in self.tools.values()]
        self.max_think_iterations = max_think_iterations
        self._mode = Mode.PLAN

    def chat(self, user_input: str) -> str:
        """Process user input and handle tool calls if needed."""
        self.session.add_message(role=Role.USER, content=user_input)
        messages = self.session.to_openai_format()
        return self._process_streaming_response(messages, with_tools=True)

    def think(self, think_count: int = 0) -> str:
        """Continue conversation without new user input."""
        messages = self.session.to_openai_format()
        self._logger.info(f"{self.model} thinking...")
        response = self._process_streaming_response(
            messages, with_tools=True, think_count=think_count
        )
        self._logger.info(f"{self.model} thinks completedly")
        return response

    def _process_streaming_response(
        self, messages: List[Dict[str, Any]], with_tools: bool, think_count: int = 0
    ) -> str:
        """
        Process streaming response and handle tool calls.

        Args:
            messages: Messages in OpenAI format
            with_tools: Whether to enable tool calling
            think_count: Current think iteration count

        Returns:
            The final response text
        """
        response = completion(
            model=self.model,
            messages=messages,
            stream=True,
            tools=self.tools_schema if with_tools and self.tools else None,
        )

        timestap = response.created or int(time())
        full_response = ""
        tool_calls_buffer = {}

        for chunk in response:
            delta = chunk.choices[0].delta
            full_response += delta.content or ""

            if delta.tool_calls:
                self._collect_tool_calls(tool_calls_buffer, delta.tool_calls)

        # Build and add assistant message
        tool_calls = self._build_tool_calls(tool_calls_buffer)
        self.session.add_message(
            role=Role.ASSISTANT,
            content=full_response if full_response else "",
            timestap=timestap,
            tool_calls=tool_calls if tool_calls else None,
        )

        # Handle tool calls if any
        if tool_calls:
            tool_results = self._execute_tool_calls(tool_calls)
            if think_count >= self.max_think_iterations:
                self._logger.warning(
                    f"Reached max think iterations ({self.max_think_iterations})"
                )
                return (
                    full_response
                    + f"\n[Warning: Reached maximum think iterations limit ({self.max_think_iterations})]"
                )
            if tool_results:
                return self.think(think_count + 1)

        return full_response

    def _collect_tool_calls(
        self,
        tool_calls_buffer: Dict[str, Any],
        tool_calls: List[Union[ChatCompletionDeltaToolCall, Any]],
    ):
        """Collect tool calls from streaming chunks into buffer."""
        for tc_delta in tool_calls:
            idx = tc_delta.index
            if idx not in tool_calls_buffer:
                tool_calls_buffer[idx] = {
                    "id": tc_delta.id,
                    "name": "",
                    "arguments": "",
                    "tool_call_id": tc_delta.id,
                }
            if tc_delta.function.name:
                tool_calls_buffer[idx]["name"] += tc_delta.function.name
            if tc_delta.function.arguments:
                tool_calls_buffer[idx]["arguments"] += tc_delta.function.arguments

    def _build_tool_calls(
        self, tool_calls_buffer: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build final tool calls list from buffer."""
        tool_calls = []
        for idx in sorted(tool_calls_buffer.keys()):
            tc = tool_calls_buffer[idx]
            tool_calls.append(
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
            )
        return tool_calls

    def _execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls and add results to session."""
        results = []
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            tool_args_str = tc["function"]["arguments"]

            try:
                tool_args = json.loads(tool_args_str)
            except json.JSONDecodeError:
                tool_args = {}

            if tool_name in self.tools:
                tool = self.tools[tool_name]
                tool_result = tool.execute(**tool_args)

                self.session.add_message(
                    Role.TOOL,
                    str(tool_result),
                    name=tool_name,
                    tool_call_id=tc["id"],
                )
                results.append({"tool_call_id": tc["id"], "result": tool_result})
            else:
                self._logger.error(f"Tool '{tool_name}' not found.")

        return results

    def _build_prompt(self, prompt_templete: str, **kwargs) -> str:
        return prompt_templete.format(**kwargs)


if __name__ == "__main__":
    from src.tool.get_current_date import get_current_date
    from src.tool.get_weather import get_weather
    from src.tool.web_search import web_search
    from src.tool.web_fetcher_v2 import web_fetcher_v2

    agent = Agent(
        name="news-assistant",
        model="deepseek/deepseek-chat",
        system_prompt="""
你是一个新闻助手。

比必须按照以下流程来工作：
1. Thought（思考）
2. Action（选择一个行动）
3. Observation（等待系统返回结果）

你必须严格按照以下格式输出：

Thought: 你的思考过程
Action: 你决定要调用的tool，或者下一步要执行的动作

你可以从以下**新闻源**获取新闻：

- [Hacker News RSS](https://news.ycombinator.com/rss)
- [The Verge](https://www.theverge.com/rss/index.xml)
- [时政频道|中新网](https://www.chinanews.com.cn/rss/china.xml)
- [国际新闻|中新网](https://www.chinanews.com.cn/rss/world.xml)

也可以调用web_search工具来搜索新闻。

用户期望关注国家大事、最新的AI相关的进展、时政要闻等方面的内容。

# 工作流
1. 从**新闻源**或者通过web_search工具来获取新闻
3. 对新闻进行分类汇总，选择最值得关注的10-15条新闻
4. 每条新闻需要有标题、链接、一句话概括和你选择它的理由
5. 最终结果以markdown的形式返回

# hits
* 当用户的提问涉及到时间和日期的时候，需要确认一下当前的时间和日期


""",
        session_id="bb914239-57ac-4afd-a4d1-148c86b427a2",
        tools=[get_weather, get_current_date, web_fetcher_v2, web_search],
        max_think_iterations=10,
    )
    resp = agent.think(think_count=10)
    print(resp)
