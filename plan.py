from src.agent import PlanAgent
from src.tool.get_current_date import get_current_date
from src.tool.web_search import web_search
from src.tool.web_fetcher_v2 import web_fetcher_v2

if __name__ == "__main__":
    agent = PlanAgent(
        name="research-assistant",
        model="deepseek/deepseek-chat",
        system_prompt="""
""",
        tools=[get_current_date, web_fetcher_v2, web_search],
    )

    resp = agent.chat("周杰伦最新的专辑如何评价")
    print(resp)
