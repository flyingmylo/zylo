import copy
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Protocol

from src.llm.base import LLMProvider, LLMResponse
from src.state import WritingState
from src.tools.base import Tool


class AgentKnowledgeBase(Protocol):
    """Agent 视角的知识库契约：只声明写作流程实际依赖的能力。

    具体的 KnowledgeBase 结构化满足本协议；测试替身同样只需结构满足
    即可注入，单元测试因此不必拉起 chromadb 等重型依赖。
    """

    collection_name: str

    def add_documents(self, documents: list[dict[str, str]]) -> None: ...

    def retrieve(
        self,
        query_zh: str,
        query_en: str = "",
        top_k: int = 4,
        candidate_pool: int = 6,
    ) -> list[dict[str, str]]: ...

    def count(self) -> int: ...


class AgentSearchTool(Protocol):
    """Researcher 视角的搜索契约：确定性检索流程只用到关键词与条数。

    max_results 声明为仅关键字参数，以兼容实现方各自的位置参数排布
    （如 SearchTool 在 query 之后还有 search_depth）。
    """

    async def search(
        self, query: str, *, max_results: int = 5
    ) -> list[dict[str, Any]]: ...


class BaseAgent(ABC):
    """所有自主 Agent 的基类"""

    def __init__(
        self,
        name: str,
        llm: LLMProvider,
        system_prompt: str,
        tools: list[Tool] | None = None,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.logger = logging.getLogger(f"src.agents.{name}")

        # 以 LLM 侧的函数名为键建立分发表。schema 在建表时深拷贝快照，
        # 使「向模型宣告的工具」与「能调度的实现」不会因事后修改而分裂。
        self._tool_map: dict[str, Tool] = {}
        self._tool_schemas: dict[str, dict[str, Any]] = {}
        for t in tools or []:
            schema = self._validate_tool(t)
            if schema is None:
                continue
            fn_name = schema["function"]["name"]
            if fn_name in self._tool_map:
                self.logger.warning("工具名 %r 重复注册，后者将覆盖前者", fn_name)
            self._tool_map[fn_name] = t
            self._tool_schemas[fn_name] = copy.deepcopy(schema)

    def _validate_tool(self, tool: Any) -> dict[str, Any] | None:
        """校验工具是否满足 Tool 协议且带有可用的 schema，不合法时记 warning 并返回 None。

        isinstance 对 runtime_checkable 协议只做成员存在性检查，且各 Python 版本
        对数据成员的检查强度不一，因此 schema 结构校验才是权威判断。
        """
        if not isinstance(tool, Tool):
            self.logger.warning(
                "工具 %r 不满足 Tool 协议（需要 schema 属性与 execute 方法），已忽略", tool
            )
            return None
        schema = tool.schema
        fn = schema.get("function") if isinstance(schema, dict) else None
        if not isinstance(fn, dict) or not fn.get("name"):
            self.logger.warning("工具 %r 的 schema 缺少 function.name，已忽略", tool)
            return None
        return schema

    @abstractmethod
    async def run(self, state: WritingState) -> WritingState:
        """子类具体执行流程与状态更新"""
        ...

    @staticmethod
    def _parse_llm_json(raw: str) -> Any:
        """从 LLM 输出中安全解析 JSON 载荷（对象或数组），供各 Agent 结构化输出环节复用。

        解析分两级：剥掉 markdown 代码围栏后直接解析；失败再用正则截取最外层
        {...} / [...] 块重试（模型偶发在 JSON 前后夹带解释文字）。
        所有解析异常在本函数内吞掉并返回 None，由调用方决定兜底值——
        容错路径自身绝不允许再抛异常（回归：except 块内的二次 json.loads
        曾把整条写作流程直接打崩）。
        """
        text = raw.strip()
        if text.startswith("```json"):
            text = text[7:].rsplit("```", 1)[0].strip()
        elif text.startswith("```"):
            text = text[3:].rsplit("```", 1)[0].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for pattern in (r"\{.*\}", r"\[.*\]"):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    continue
        return None

    def _get_tool_schemas(self) -> list[dict[str, Any]] | None:
        """返回注册时快照的 JSON Schema；无工具时返回 None 而非空列表。"""
        if not self._tool_schemas:
            return None
        return list(self._tool_schemas.values())

    async def _chat(
        self,
        messages: list[dict[str, Any]],
        state: WritingState,
        temperature: float = 0.7,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """单次 LLM 调用，并统一累计 Token 消耗。

        所有 Agent 的 LLM 交互都应经过此处，否则 state.token_usage 会漏统计。
        """
        resp = await self.llm.chat(
            messages=messages,
            tools=tools,
            temperature=temperature,
        )
        if resp.usage:
            for k, v in resp.usage.items():
                state.token_usage[k] = state.token_usage.get(k, 0) + v
        return resp

    async def _execute_tool(self, name: str, args_json: str) -> str:
        """执行单个工具并返回文本结果。

        异常在此处被吞掉并转成错误文本回灌给模型，让它有机会自行修正参数重试；
        完整 traceback 保留在日志中。
        """
        tool = self._tool_map.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found."
        try:
            kwargs = json.loads(args_json) if args_json else {}
            res = await tool.execute(**kwargs)
            return json.dumps(res, ensure_ascii=False)
        except Exception as e:
            self.logger.exception("工具 '%s' 执行失败", name)
            return f"Error executing tool '{name}': {type(e).__name__}：{e!s}"

    async def _chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        state: WritingState,
        max_tool_iters: int = 5,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """统一的 Function Calling 工具调用循环；未配置工具时退化为一次普通调用。"""
        schemas = self._get_tool_schemas()
        current_messages = list(messages)
        final_resp: LLMResponse | None = None

        for _ in range(max_tool_iters):
            resp = await self._chat(
                current_messages,
                state,
                temperature=temperature,
                tools=schemas,
            )
            final_resp = resp

            # 如果没有工具调用，直接退出
            if not resp.tool_calls:
                break

            # assistant 消息必须与后续 tool 消息按 id 严格配对，否则下一轮请求会被拒
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

        if final_resp is None:
            raise RuntimeError("工具调用循环未获得任何 LLM 响应。")

        # 循环只在模型不再请求工具时 break，因此走到这里仍带着 tool_calls，
        # 就意味着已经跑满 max_tool_iters 上限
        if final_resp.tool_calls:
            self.logger.warning(
                "工具调用循环达到上限 %d 轮仍未收敛，已丢弃最后 %d 个未执行的工具调用",
                max_tool_iters,
                len(final_resp.tool_calls),
            )
        return final_resp
