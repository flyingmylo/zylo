from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    """LLM 可调用工具的契约。

    一个工具需要提供 LLM 侧可见的名字与参数 Schema（schema），
    以及统一的异步执行入口（execute）。工具类无需继承任何基类，
    只需满足本协议——BaseAgent 在注册时用 isinstance 做运行时校验。

    注意：runtime_checkable 只校验成员存在性，不校验类型签名，
    因此 schema 的结构仍由 BaseAgent 单独校验。
    """

    schema: dict[str, Any]

    async def execute(self, **kwargs: Any) -> Any:
        """按 Schema 声明的参数执行工具，返回可 JSON 序列化的结果。"""
        ...
