import json

from app.agents.provider import (FailedProvider, OfflineProvider, OpenAICompatibleProvider, get_provider,
                                 narrate_checked)
from app.env import load_env, parse


def test_env_parser_tolerates_spaces_quotes_comments():
    d = parse("# c\nLLM_BASE_URL = https://x/v1\nLLM_MODEL = \"qwen\"\nexport LLM_API_KEY=abc # note\n")
    assert d == {"LLM_BASE_URL": "https://x/v1", "LLM_MODEL": "qwen", "LLM_API_KEY": "abc"}


def test_env_does_not_override_shell(tmp_path, monkeypatch):
    p = tmp_path / ".env"; p.write_text("LLM_MODEL = fromfile\n")
    monkeypatch.setenv("LLM_MODEL", "fromshell")
    load_env([p])
    import os
    assert os.environ["LLM_MODEL"] == "fromshell"


def test_auto_without_env_is_offline_and_missing_env_is_surfaced(monkeypatch):
    for k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert isinstance(get_provider("auto", 5), OfflineProvider)
    p = get_provider("openai_compatible", 5)
    assert isinstance(p, FailedProvider)
    r = narrate_checked(p, {}, set())
    assert r["text"] is None and "LLM_BASE_URL" in r["error"]


def test_openai_compatible_request_and_response():
    seen = {}

    def transport(url, body, headers):
        seen.update(url=url, body=json.loads(body), headers=headers)
        return {"choices": [{"message": {"content": "ok ev:atr:1"}}]}

    p = OpenAICompatibleProvider(5, "https://h/v1/chat/completions", "k", "m", transport=transport)
    assert p.chat("s", "u") == "ok ev:atr:1"
    assert seen["url"] == "https://h/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer k" and seen["body"]["model"] == "m"


def test_busy_router_is_retried_then_succeeds():
    from app.agents.provider import LLMError
    calls = []

    def transport(url, body, headers):
        calls.append(1)
        if len(calls) < 3:
            raise LLMError("HTTP 503 busy", 503)
        return {"choices": [{"message": {"content": "fine"}}]}

    p = OpenAICompatibleProvider(5, "https://h/v1", "k", "m", transport=transport, sleep=lambda s: None)
    assert p.chat("s", "u") == "fine" and len(calls) == 3


def test_auth_error_is_not_retried():
    import pytest
    from app.agents.provider import LLMError
    calls = []

    def transport(url, body, headers):
        calls.append(1)
        raise LLMError("HTTP 401 bad key", 401)

    p = OpenAICompatibleProvider(5, "https://h/v1", "k", "m", transport=transport, sleep=lambda s: None)
    with pytest.raises(LLMError):
        p.chat("s", "u")
    assert len(calls) == 1
