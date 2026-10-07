from src.events import (
    EVENT_SCHEMA_VERSION,
    EventStatus,
    RunEvent,
    SpanKind,
    sanitize_payload,
)


def _event(**overrides) -> RunEvent:
    base = {
        "sequence": 1,
        "run_id": "abc123",
        "kind": SpanKind.AGENT,
        "name": "researcher",
        "status": EventStatus.COMPLETED,
        "span_id": "s1",
    }
    base.update(overrides)
    return RunEvent(**base)


def test_event_defaults_and_version():
    event = _event()

    assert event.schema_version == EVENT_SCHEMA_VERSION
    assert event.parent_id is None
    assert event.payload == {}
    assert event.created_at.tzinfo is not None


def test_event_json_roundtrip():
    event = _event(payload={"total_tokens": 330})

    restored = RunEvent.from_json(event.to_json())

    assert restored == event


def test_sanitize_replaces_sensitive_values():
    payload = {
        "api_key": "sk-real-secret",
        "Authorization": "Bearer xxx",
        "config": {"password": "hunter2", "model": "deepseek-chat"},
    }

    cleaned = sanitize_payload(payload)

    assert cleaned["api_key"] == "[REDACTED]"
    assert cleaned["Authorization"] == "[REDACTED]"
    assert cleaned["config"]["password"] == "[REDACTED]"
    assert cleaned["config"]["model"] == "deepseek-chat"


def test_sanitize_preserves_token_statistics():
    """token_usage 等统计键不得被子串式误伤（这是精确键名匹配的原因）。"""
    payload = {"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    cleaned = sanitize_payload(payload)

    assert cleaned == payload


def test_sanitize_omits_content_fields_with_length_hint():
    payload = {"prompt": "写一篇长文……", "nested": [{"text": "正文片段"}]}

    cleaned = sanitize_payload(payload)

    assert cleaned["prompt"].startswith("[OMITTED")
    assert cleaned["nested"][0]["text"].startswith("[OMITTED")


def test_sanitize_truncates_long_strings_everywhere():
    long_text = "x" * 2000

    cleaned = sanitize_payload({"error": long_text, "items": [long_text]})

    assert cleaned["error"].endswith("[truncated 2000 chars]")
    assert cleaned["items"][0].endswith("[truncated 2000 chars]")
    assert len(cleaned["error"]) < 600
