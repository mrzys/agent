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

        self.session = Session(session_id=session_id)
        self._logger.info(
            f"Initializing agent with session_id: {self.session.session_id}"
        )

        if session_id is None:
            self.session.add_message(Role.SYSTEM, system_prompt)

        self._tool_executor = ToolExecutor(tools)
        self._llm_client = LLMClient(model, self._tool_executor.get_schema())
        self._max_think_iterations = max_think_iterations

    @property
    def session_id(self) -> str:
        return self.session.session_id

    def chat(self, user_input: str) -> str:
        self.session.add_message(Role.USER, user_input)
        return self._think()

    def think(self) -> str:
        return self._think()

    def _think(self, iteration: int = 0) -> str:
        messages = self.session.to_openai_format()
        response = self._llm_client.stream(messages)

        self.session.add_message(
            role=Role.ASSISTANT,
            content=response.content,
            timestamp=response.created,
            tool_calls=[tc.to_dict() for tc in response.tool_calls]
            if response.tool_calls
            else None,
        )

        if response.tool_calls:
            return self._handle_tool_calls(response.tool_calls, iteration)

        return response.content

    def _handle_tool_calls(self, tool_calls: List[ToolCall], iteration: int) -> str:
        results = self._tool_executor.execute_batch(tool_calls)

        for result in results:
            content = result.error or str(result.result)
            self.session.add_message(
                Role.TOOL,
                content,
                name=result.tool_name,
                tool_call_id=result.tool_call_id,
            )

        if iteration >= self._max_think_iterations:
            self._logger.warning(
                f"Reached max think iterations ({self._max_think_iterations})"
            )
            return f"[Warning: Reached max iterations ({self._max_think_iterations})]"

        return self._think(iteration + 1)


if __name__ == "__main__":
    from src.tool.get_current_date import get_current_date
    from src.tool.get_weather import get_weather
    from src.tool.web_search import web_search
    from src.tool.web_fetcher_v2 import web_fetcher_v2

    logging.basicConfig(level=logging.INFO)

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
    resp = agent.think()
    print(resp)
