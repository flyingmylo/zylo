import os
from typing import Any

from tavily import TavilyClient


class SearchTool:
    """
    Tavily 联网搜索封装
    专为 AI Agent 设计，返回已清洗后的结构化网页摘要与原始正文

    实现 src.tools.base.Tool 协议，可注册给 BaseAgent 供 LLM 调用
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self.client = TavilyClient(api_key=self.api_key) if self.api_key else None
        self.schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "通过互联网搜索技术概念、最新动态、学术观点与实证数据",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词或短语，支持中文或英文",
                        },
                        "search_depth": {
                            "type": "string",
                            "enum": ["basic", "advanced"],
                            "description": "搜索深度，默认为 basic",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Tool 协议入口：供 LLM Function Calling 调用。

        只接受 Schema 中声明过的参数，未声明的键（如 max_results）一律忽略，
        避免模型借此影响检索次数与配额消耗。
        """
        query = kwargs.get("query")
        if not query:
            raise ValueError("缺少必需参数 query")
        depth = kwargs.get("search_depth", "basic")
        if depth not in ("basic", "advanced"):
            depth = "basic"
        return await self.search(query=query, search_depth=depth)

    async def search(
        self, query: str, search_depth: str = "basic", max_results: int = 5
    ) -> list[dict[str, Any]]:
        """执行搜索并返回 [{title, url, content}, ...]"""
        if not self.client:
            return [
                {
                    "title": "Mock Search",
                    "url": "https://example.com",
                    "content": f"未配置 TAVILY_API_KEY，跳过在线搜索: {query}",
                }
            ]

        try:
            res = self.client.search(
                query=query, search_depth=search_depth, max_results=max_results
            )
            results = []
            for r in res.get("results", []):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                    }
                )
            return results
        except Exception as e:
            return [
                {
                    "title": "Search Error",
                    "url": "",
                    "content": f"搜索执行出错: {str(e)}",
                }
            ]
