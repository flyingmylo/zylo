import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


def _get_default_model() -> str:
    m = os.getenv("LLM_MODEL")
    if m and m.strip() and m.strip().lower() != "none":
        return m.strip()
    b = os.getenv("LLM_BASE_URL", "")
    if "deepseek" in b.lower():
        return "deepseek-v4-pro"
    return "gpt-4o"


class LLMConfig(BaseModel):
    api_key: str = Field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    base_url: str | None = Field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", None)
    )
    model: str = Field(default_factory=_get_default_model)
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7"))
    )
    tavily_api_key: str = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
