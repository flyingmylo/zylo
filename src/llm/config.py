import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


def _get_model_from_env() -> str:
    """模型只从环境变量读取，不提供任何默认值。

    未设置时返回空字符串，由调用方（CLI）给出明确报错，
    避免悄悄用某个硬编码模型去请求一个并不存在的端点。
    """
    m = os.getenv("LLM_MODEL", "").strip()
    return "" if m.lower() in ("", "none") else m


def _get_default_rerank() -> str:
    r = os.getenv("ENABLE_RERANK", "auto").strip().lower()
    if r in ("true", "1", "yes", "on"):
        return "true"
    if r in ("false", "0", "no", "off"):
        return "false"
    return "auto"


class LLMConfig(BaseModel):
    api_key: str = Field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    base_url: str | None = Field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", None)
    )
    model: str = Field(default_factory=_get_model_from_env)
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7"))
    )
    tavily_api_key: str = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    enable_rerank: str = Field(default_factory=_get_default_rerank)
