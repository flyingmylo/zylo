import json
from abc import ABC, abstractmethod
from typing import Any

from src.llm.base import LLMProvider, LLMResponse
from src.state import WritingState


class BaseAgent(ABC):
    """所有自主 Agent 的基类"""

    def __init__(
        self,
        name: str,
        llm: LLMProvider,
        system_prompt: str,
        tools: list[Any] | None = None,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools or []
        self._tool_map = {}
        for t in self.tools:
            if hasattr(t, "schema") and "function" in t.schema:
                fn_name = t.schema["function"]["name"]
                self._tool_map[fn_name] = t

    @abstractmethod
    async def run(self, state: WritingState) -> WritingState:
        """子类具体执行流程与状态更新"""
        ...

    def _get_tool_schemas(self) -> list[dict[str, Any]] | None:
        if not self.tools:
            return None
        return [t.schema for t in self.tools if hasattr(t, "schema")]

    async def _execute_tool(self, name: str, args_json: str) -> str:
        """执行单个工具并返回文本结果"""
        tool = self._tool_map.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found."
        try:
            kwargs = json.loads(args_json) if args_json else {}
            if hasattr(tool, "execute"):
                res = await tool.execute(**kwargs)
            elif hasattr(tool, "search"):
                res = await tool.search(**kwargs)
            else:
                return f"Error: Tool '{name}' has no execution method."
            return json.dumps(res, ensure_ascii=False)
        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"

    async def _chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        state: WritingState,
        max_tool_iters: int = 5,
        temperature: float = 0.7,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        """统一的 Function Calling 工具调用循环"""
        schemas = self._get_tool_schemas()
        current_messages = list(messages)
        final_resp = None

        for _ in range(max_tool_iters):
            resp = await self.llm.chat(
                messages=current_messages,
                tools=schemas,
                temperature=temperature,
                response_format=response_format,
            )
            final_resp = resp

            # 累计 Token 消耗
            if resp.usage:
                for k, v in resp.usage.items():
                    state.token_usage[k] = state.token_usage.get(k, 0) + v

            # 如果没有工具调用，直接退出
            if not resp.tool_calls:
                break

            # 处理工具调用
            current_messages.append(
                {
                    "role": "assistant",
                    "content": resp.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in resp.tool_calls
                    ],
                }
            )

            for tc in resp.tool_calls:
                tool_output = await self._execute_tool(tc.name, tc.arguments)
                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_output,
                    }
                )

        return final_resp
