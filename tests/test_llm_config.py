import pytest

from src.llm.config import LLMConfig
from src.llm.openai_provider import OpenAICompatibleProvider


def test_provider_requires_explicit_model():
    with pytest.raises(ValueError, match="不提供默认模型"):
        OpenAICompatibleProvider(api_key="test-key", model="")


def test_config_model_is_empty_when_env_unset(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert LLMConfig().model == ""


def test_config_treats_none_as_unset(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "none")
    assert LLMConfig().model == ""


def test_config_reads_model_from_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "  my-model  ")
    assert LLMConfig().model == "my-model"
