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


def test_analysis_api_llm_debate_parameters(tmp_path, monkeypatch):
    from unittest.mock import patch
    from app.features.pipeline import EvidenceBundle
    from app.agents.schemas import DiscussionReport, DiscussionTurn, Claim

    st = Store(tmp_path / "x.sqlite")
    c = TestClient(create_app(st.path))

    monkeypatch.setenv("LLM_BASE_URL", "http://mock")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_API_KEY", "k")
    # Mock live_market snapshot
    fake_snap = {
        "symbol": "BTCUSDT",
        "tf_minutes": 60,
        "server_time": 1000,
        "funding_rate": 0.0001,
        "candles": [
            [i * 60, 100 + i * 0.1, 105 + i * 0.1, 95 + i * 0.1, 102 + i * 0.1, 10, True] for i in range(150)
        ]
    }

    report = DiscussionReport(
        turns=[
            DiscussionTurn(role="bull", stage="opening", claims=[Claim(text="trend is up", evidence_id="ev:1", weight=0.6)], counterarguments=[], invalidation="drop", uncertainty=0.3),
            DiscussionTurn(role="bear", stage="cross_examination", claims=[Claim(text="resist ahead", evidence_id="ev:2", weight=0.5)], counterarguments=[], invalidation="break", uncertainty=0.4),
        ],
        score=0.1,
        recommendation="no_trade",
        judge_summary="Close debate on resistance"
    )

    with patch("app.api.live_market.snapshot", return_value=fake_snap), \
         patch("app.agents.provider.llm_status", return_value={"configured": True, "base_url": "http://mock", "model": "m"}), \
         patch("app.agents.discussion.discuss_with_llm", return_value=report) as mock_discuss:

        resp = c.get("/api/analysis/BTCUSDT?agents=llm&rounds=2&refresh=true")
        assert resp.status_code == 200
        data = resp.json()
        assert "agent_debate" in data
        assert data["agent_debate"]["status"] == "complete"
        assert data["agent_debate"]["rounds"] == 2
        assert len(data["agent_debate"]["turns"]) == 2
        assert data["agent_debate"]["judge_summary"] == "Close debate on resistance"
        assert mock_discuss.call_args[1]["rounds"] == 2
