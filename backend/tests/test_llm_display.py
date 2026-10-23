import json
from fastapi.testclient import TestClient
from app.persistence.db import Store
from app.api.read_only import create_app
from app.agents.provider import OpenAICompatibleProvider


def test_holdout_explanation_readback(tmp_path, cfg):
    st = Store(tmp_path / "x.sqlite")
    st.create_experiment("e", "training", cfg.model_dump())
    c = TestClient(create_app(st.path))
    assert c.get("/api/experiments/e/final").json()["narration"] == []
    st.put("e", "holdout", "gene", {"narration": [{"text": "Saved LLM reply", "provider": "openai-compatible:test"}]})
    data = c.get("/api/experiments/e/final").json()
    assert data["narration"][0]["narration"][0]["text"] == "Saved LLM reply"


def test_reasoning_budget_preserves_json(monkeypatch):
    seen = []
    def transport(url, body, headers):
        seen.append(json.loads(body))
        return {"choices": [{"message": {"content": "OK"}}]}
    p = OpenAICompatibleProvider(1, "https://example.invalid/v1", "dummy", "test", transport=transport)
    record = {"risk": {"reason": "x" * 14000}}
    assert p.narrate(record) == "OK"
    assert seen[0]["max_tokens"] == 2048
    assert json.loads(seen[0]["messages"][1]["content"]) == record
    monkeypatch.setenv("LLM_MAX_TOKENS", "4096")
    p.narrate(record)
    assert seen[1]["max_tokens"] == 4096
