import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.agents.narration import explain_experiment, explain_records
from app.agents.provider import FailedProvider, LLMError, OpenAICompatibleProvider, get_provider, llm_status
from app.genetic.population import initial
from app.live.runner import LiveTrader, pick_chromosome, resolve_experiment
from app.live.testnet import SymbolRules
from app.persistence.db import Store


def research(tmp_path, cfg, state="COMPLETED"):
    st = Store(tmp_path / "research.sqlite")
    st.create_experiment("exp_real", "test", cfg.model_dump())
    st.set_state("exp_real", state)
    return st


def chromosome():
    import random
    return initial(random.Random(42), 1)[0]


def record():
    return {"proposal": {"decision_id": "d1", "side": "no_trade", "evidence_ids": ["ev:atr:1"]},
            "risk": {"outcome": "DEFER", "reasons": ["insufficient evidence"]}}


def test_unknown_0001_explains_exact_ids(tmp_path, cfg):
    st = research(tmp_path, cfg)
    with pytest.raises(SystemExit, match="not sequence numbers like 0001"):
        resolve_experiment(st, "0001")


def test_latest_ignores_live_and_aborted(tmp_path, cfg):
    st = research(tmp_path, cfg)
    st.put("exp_real", "final", "ranking", {"ranking": []})
    st.create_experiment("live_new", "live", cfg.model_dump())
    st.set_state("live_new", "LIVE_DRY_RUN")
    assert resolve_experiment(st, "latest") == "exp_real"


def test_aborted_reason_is_reported(tmp_path, cfg):
    st = research(tmp_path, cfg, "ABORTED_INTEGRITY")
    st.append("exp_real", "state", {"to": "ABORTED_INTEGRITY", "reason": "Binance unavailable"})
    with pytest.raises(SystemExit, match="ABORTED_INTEGRITY; Binance unavailable"):
        pick_chromosome(st, "exp_real")


def test_empty_ranking_not_tradeable(tmp_path, cfg):
    st = research(tmp_path, cfg)
    st.put("exp_real", "final", "ranking", {"ranking": []})
    with pytest.raises(SystemExit, match="No QUALIFIED"):
        pick_chromosome(st, "exp_real")


def test_qualified_selection(tmp_path, cfg):
    st = research(tmp_path, cfg)
    c = chromosome()
    st.put("exp_real", "final", "ranking", {"ranking": [{"status": "QUALIFIED", "chromosome": c.to_dict()}]})
    assert pick_chromosome(st, "exp_real").id == c.id


def test_explicit_research_fallback_for_empty_archive(tmp_path, cfg):
    st = research(tmp_path, cfg)
    c = chromosome()
    st.put("exp_real", "final", "ranking", {"ranking": []})
    st.put("exp_real", "candidate", "1", {"status": "REJECTED", "chromosome": c.to_dict(),
                                          "oos": {"scorecard": {"net_pnl": -2}}})
    assert pick_chromosome(st, "exp_real", True).id == c.id


def test_failed_risk_cannot_be_fallback(tmp_path, cfg):
    st = research(tmp_path, cfg)
    st.put("exp_real", "final", "ranking", {"ranking": []})
    st.put("exp_real", "candidate", "1", {"status": "FAILED_RISK", "chromosome": chromosome().to_dict()})
    with pytest.raises(SystemExit, match="No QUALIFIED"):
        pick_chromosome(st, "exp_real", True)


def test_partial_key_is_not_silent_offline(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "dummy-test-key")
    p = get_provider("auto", 1)
    assert isinstance(p, FailedProvider)
    assert "LLM_BASE_URL" in p.reason
    assert llm_status()["configured"] is False


def test_configured_is_not_connected(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "dummy-model")
    monkeypatch.setenv("LLM_API_KEY", "dummy-key")
    status = llm_status()
    assert status["configured"] is True and status["connectivity"] == "untested"
    assert "dummy-key" not in json.dumps(status)


def test_provider_error_redacts_key():
    def transport(*args):
        raise LLMError("Unauthorized dummy-private-test-value", 401)
    p = OpenAICompatibleProvider(1, "https://example.invalid/v1", "dummy-private-test-value", "m", transport=transport)
    with pytest.raises(LLMError) as e:
        p.chat("s", "u")
    assert "dummy-private-test-value" not in str(e.value) and "[redacted]" in str(e.value)


def test_rejected_no_trade_gets_persisted_explanation(tmp_path, cfg, monkeypatch):
    st = research(tmp_path, cfg)
    st.put("exp_real", "candidate", "1", {"status": "REJECTED", "decisions": [record()],
                                           "oos": {"scorecard": {"net_pnl": -4}}})
    p = OpenAICompatibleProvider(1, "https://example.invalid/v1", "dummy", "m",
                                transport=lambda *a: {"choices": [{"message": {"content": "DEFER ev:atr:1"}}]})
    monkeypatch.setattr("app.agents.narration.get_provider", lambda *a: p)
    rows = explain_experiment(st, "exp_real", cfg)
    assert rows[0]["text"] == "DEFER ev:atr:1" and rows[0]["valid"] is True
    assert rows[0]["scope"] == "oos_research_not_qualified"
    assert st.artifact("exp_real", "final", "ranking") is None
    assert st.artifacts("exp_real", "narration")[0]["narration"][0]["decision_id"] == "d1"


def test_narrate_none_makes_no_call(tmp_path, cfg):
    st = research(tmp_path, cfg)
    cfg = cfg.model_copy(update={"llm": cfg.llm.model_copy(update={"narrate": "none"})})
    class NoCall:
        def narrate(self, r):
            pytest.fail("disabled narration must not call LLM")
    assert explain_records(st, "exp_real", cfg, [record()], "test", NoCall()) == []


def test_root_and_backend_launch_share_defaults():
    root = Path(__file__).parents[2]
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLM_")}
    env.pop("CGTRS_DB", None)
    results = []
    for path in (root, root / "backend"):
        e = dict(env)
        backend_p = str(root / "backend" if path == root else path)
        e["PYTHONPATH"] = f"{backend_p}:{e.get('PYTHONPATH', '')}".rstrip(":")
        res = subprocess.run([sys.executable, "-c", "from app.cli import DB; print(DB)"],
                             cwd=path, env=e, capture_output=True, text=True, check=True).stdout.strip()
        results.append(res)
    assert results[0] == results[1]
    assert results[0].endswith("backend/data/cgtrs.sqlite")


def test_cli_blocks_unqualified_order_mode_before_exchange():
    root = Path(__file__).parents[2]
    env_root = dict(os.environ)
    env_root["PYTHONPATH"] = f"{root / 'backend'}:{env_root.get('PYTHONPATH', '')}".rstrip(":")
    r = subprocess.run([sys.executable, "-m", "app.cli", "trade", "--experiment", "0001", "--allow-unqualified"],
                       cwd=root, env=env_root, capture_output=True, text=True)
    assert r.returncode == 2 and "requires --dry-run" in r.stderr


def test_dry_run_with_keys_never_reads_account_or_orders(tmp_path, cfg, monkeypatch):
    st = research(tmp_path, cfg)
    class PublicOnly:
        key = "present-but-must-not-be-used"
        base = "https://testnet.binancefuture.com"
        def rules(self, sym):
            return SymbolRules(step=.001, min_qty=.001, tick=.1, min_notional=5)
        def book(self, sym):
            return (100, 100.01)
        def __getattr__(self, name):
            pytest.fail(f"dry-run attempted private exchange operation {name}")
    trader = LiveTrader(cfg, chromosome(), st, PublicOnly(), str(tmp_path / "dry.json"), True,
                        str(tmp_path / "NO_KILL"))
    # Real pipeline executes; deterministic risk may approve, defer or block but never calls private endpoints.
    from app.data import ingest
    bars, _ = ingest.load(cfg.data)
    monkeypatch.setattr(trader, "bars", lambda: bars[-300:])
    out = trader.step()
    assert out in {"APPROVE", "RESIZE", "REJECT", "DEFER", "NO_TRADE", "DRY_APPROVED", "HALT"}
    assert not any(e["type"] == "order" for e in st.events(trader.eid))


def test_approved_dry_run_llm_http_and_api_readback(tmp_path, cfg, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from fastapi.testclient import TestClient
    from app.api.read_only import create_app
    from app.agents.schemas import TradeProposal, RiskDecision
    from app.data.ingest import synthetic_bars
    from dataclasses import replace
    import time
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, data["model"], self.headers.get("Authorization")))
            body = json.dumps({"choices": [{"message": {"content": "Approved dry-run ev:atr:1; no real orders."}}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.end_headers(); self.wfile.write(body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("LLM_API_KEY", "dummy-local-http-key")
    monkeypatch.setenv("LLM_MODEL", "test-http-model")
    cfg = cfg.model_copy(update={"llm": cfg.llm.model_copy(update={"provider": "auto"})})
    st = research(tmp_path, cfg)
    class PublicOnly:
        key = "present"
        base = "https://testnet.binancefuture.com"
        def rules(self, sym):
            return SymbolRules(step=.001, min_qty=.001, tick=.1, min_notional=5)
        def book(self, sym):
            return (30000, 30000.1)
        def __getattr__(self, name):
            pytest.fail(f"private operation in approved dry-run: {name}")
    try:
        live = LiveTrader(cfg, chromosome(), st, PublicOnly(), str(tmp_path / "dry.json"), True,
                          str(tmp_path / "NO_KILL"))
        bars = synthetic_bars(cfg.data)[-300:]
        bars = [replace(b, i=i, ts=int(time.time()) - (300-i)*3600) for i,b in enumerate(bars)]
        monkeypatch.setattr(live, "bars", lambda: bars)
        proposal = TradeProposal(decision_id="d-approved", chromosome_id=live.c.id, t=299, side="long",
                                 ref_price=30000, stop=29900, target=30300, qty=.01, evidence_ids=["ev:atr:1"],
                                 discussion_adoption="ADOPTED")
        monkeypatch.setattr("app.live.runner.trader.propose", lambda *a: proposal)
        monkeypatch.setattr("app.live.runner.gate.evaluate", lambda *a: RiskDecision(
            decision_id="d-approved", outcome="APPROVE", requested_qty=.01, approved_qty=.01,
            checks=[], reasons=["test deterministic approval"], risk_version="test"))
        assert live.step() == "DRY_APPROVED"
        assert requests == [("/v1/chat/completions", "test-http-model", "Bearer dummy-local-http-key")]
        client = TestClient(create_app(str(tmp_path / "research.sqlite")))
        final = client.get(f"/api/experiments/{live.eid}/final").json()
        assert final["narration"][0]["narration"][0]["text"] == "Approved dry-run ev:atr:1; no real orders."
        assert client.get("/api/llm").json()["connectivity"] == "untested"
        assert "dummy-local-http-key" not in json.dumps(final)
        assert not any(e["type"] == "order" for e in st.events(live.eid))
    finally:
        server.shutdown(); server.server_close(); thread.join()