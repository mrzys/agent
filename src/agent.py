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

        self.tools = {tool.name: tool for tool in tools} if tools else {}
        self.tools_schema = [tool.to_openai_format() for tool in self.tools.values()]
        self.max_think_iterations = max_think_iterations
        self._mode = Mode.PLAN

    def chat(self, user_input: str) -> str:
        """Process user input and handle tool calls if needed."""
        system_prompt = self._build_prompt(user_input)
        self.session.add_message(role=Role.SYSTEM, content=system_prompt)
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


class PlanAgent(Agent):
    def _build_prompt(self, topic: str) -> str:
        prompt = f"""
# Role

You are a professional planning agent. Your task is to generate a **clear, structured, and executable plan** to achieve the user's goal.

User Goal:
{topic}

---

# Rules

1. You MUST fully understand the goal before planning.
2. If the goal is unclear or requires external knowledge, you SHOULD use the `web_search` tool to gather information.
3. DO NOT make assumptions when critical information is missing.
4. If the goal is vague or incomplete, you MUST create steps to gather missing information instead of guessing.

---

# Planning Principles

- The plan must be:
  - Executable
  - Specific
  - Ordered
  - Complete

- Each step must include:
  - step_id
  - type (one of: clarification, data_collection, analysis, evaluation, synthesis, validation)
  - objective
  - action
  - expected_output

---

# Tool Usage Policy

- Use `web_search` ONLY when:
  - You lack necessary knowledge
  - The task depends on up-to-date or factual information
- Do NOT use tools if the task can be completed with internal knowledge.

---

# Output Format (STRICT)

You MUST return the plan in the following XML format:

<plan>
    <step>
        <step_id>1</step_id>
        <type>data_collection</type>
        <objective>...</objective>
        <action>...</action>
        <expected_output>...</expected_output>
    </step>
</plan>
---
# Step Type Rules (CRITICAL)

- Each step MUST include a <type> field.
- The <type> MUST be one of:
  - clarification
  - data_collection
  - analysis
  - evaluation
  - synthesis
  - validation
- DO NOT invent new types.
---

# Example

User Goal: Analyze the future price trend of gold

<plan>
    <step>
        <step_id>1</step_id>
        <type>data_collection</type>
        <objective>Collect historical gold price data</objective>
        <action>Use web_search to find gold price trends over the past 5-10 years</action>
        <expected_output>Time series data of gold prices</expected_output>
    </step>
    <step>
        <step_id>2</step_id>
        <type>analysis</type>
        <objective>Identify key influencing factors</objective>
        <action>Analyze macroeconomic factors such as inflation, interest rates, and geopolitical risks</action>
        <expected_output>List of factors affecting gold prices</expected_output>
    </step>
    <step>
        <step_id>3</step_id>
        <type>data_collection</type>
        <objective>Gather recent and upcoming events</objective>
        <action>Use web_search to find recent news and upcoming global events</action>
        <expected_output>List of relevant events</expected_output>
    </step>
    <step>
        <step_id>4</step_id>
        <type>synthesis</type>
        <objective>Generate trend prediction</objective>
        <action>Combine historical data and current events to infer future trends</action>
        <expected_output>Predicted gold price trend</expected_output>
    </step>
</plan>

---
# Example: Handling ambiguous goal

User Goal: Analyze the trend of Apple

<plan>
    <step>
        <step_id>1</step_id>
        <type>clarification</type>
        <objective>Clarify the user's intent</objective>
        <action>Determine whether "Apple" refers to Apple Inc. or apple (fruit)</action>
        <expected_output>Clear definition of the target entity</expected_output>
    </step>
    <step>
        <step_id>2</step_id>
        <type>data_collection</type>
        <objective>Collect relevant data</objective>
        <action>Use web_search to gather data based on the clarified entity</action>
        <expected_output>Relevant dataset</expected_output>
    </step>
    <step>
        <step_id>3</step_id>
        <type>analysis</type>
        <objective>Analyze trend</objective>
        <action>Analyze collected data to identify patterns</action>
        <expected_output>Trend analysis</expected_output>
    </step>
</plan>
---
# Example: Evaluating a person / technology / event

User Goal: Evaluate the impact of a new AI technology

<plan>
    <step>
        <step_id>1</step_id>
        <type>evaluation</type>
        <objective>Define evaluation criteria</objective>
        <action>Identify key dimensions such as performance, cost, risks</action>
        <expected_output>List of criteria</expected_output>
    </step>
    <step>
        <step_id>2</step_id>
        <type>data_collection</type>
        <objective>Collect factual information</objective>
        <action>Use web_search to gather technical details and opinions</action>
        <expected_output>Structured information</expected_output>
    </step>
    <step>
        <step_id>3</step_id>
        <type>analysis</type>
        <objective>Analyze pros and cons</objective>
        <action>Compare strengths and weaknesses</action>
        <expected_output>Pros and cons</expected_output>
    </step>
    <step>
        <step_id>4</step_id>
        <type>synthesis</type>
        <objective>Generate final evaluation</objective>
        <action>Summarize findings into a reasoned conclusion</action>
        <expected_output>Final evaluation</expected_output>
    </step>
</plan>
---

# Output Rules (CRITICAL)

- You MUST output ONLY the <plan>...</plan> block.
- DO NOT output any text before or after <plan>.
- DO NOT include explanations, introductions, or summaries.
- Your response MUST start with <plan> and end with </plan>.

---

# Self-Check

Before returning the answer:
- Ensure the response starts with <plan>
- Ensure the response ends with </plan>
- Ensure no extra text exists outside the XML

---

# Special Handling Rules

- If the goal is ambiguous (e.g., entity has multiple meanings), you MUST include a step to clarify the meaning before proceeding.
- If the goal involves evaluation, you MUST:
  1. Define evaluation criteria
  2. Collect supporting evidence
  3. Perform balanced analysis before concluding
---

# Now generate the plan for the user goal.
"""
        return prompt
