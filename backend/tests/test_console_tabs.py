"""Training is visible while running; live + debate endpoints are read-only."""
from fastapi.testclient import TestClient

from app.api import analysis, live_market
from app.api.read_only import create_app
from app.orchestration.runner import run
from app.persistence.db import Store


def test_generation_events_carry_progress(small_cfg, tmp_path):
    store = Store(tmp_path / "t.sqlite")
    eid = run(small_cfg, store)
    gens = [e["payload"] for e in store.events(eid, 0, 2000) if e["type"] == "generation"]
    assert len(gens) == small_cfg.ga.generations * small_cfg.wfa.n_folds
    g = gens[0]
    assert g["population"] == small_cfg.ga.population
    assert 0 <= g["valid"] <= g["population"]
    assert g["worst"] <= g["mean"] <= g["best"]
    c = TestClient(create_app(str(tmp_path / "t.sqlite")))
    tr = c.get(f"/api/experiments/{eid}/training").json()
    assert len(tr["generations"]) == len(gens) and tr["states"]
    assert c.get(f"/api/experiments/{eid}/live").json() == []


def test_analysis_uses_closed_candles_only(monkeypatch, tmp_path):
    import math
    candles = [[1_700_000_000 + i * 3600, 100 + math.sin(i / 5) * 5, 106 + math.sin(i / 5) * 5,
                94 + math.sin(i / 5) * 5, 100 + math.sin(i / 5 + .3) * 5, 10.0, True] for i in range(150)]
    candles[-1][6] = False  # forming candle
    monkeypatch.setattr(live_market, "snapshot", lambda *a, **k: {"symbol": "BTCUSDT", "server_time": 0,
                                                                    "funding_rate": 0.0, "candles": candles})
    d = analysis.analyse("BTCUSDT", 60, 150)
    assert len(d["candles"]) == 149 and d["bar_ts"] == candles[-2][0]
    assert d["valid"] and d["debate"]["recommendation"] in ("long", "short", "no_trade")
    assert len(d["series"]["rsi14"]) == 149


def test_root_serves_new_console(tmp_path):
    Store(tmp_path / "c.sqlite")
    c = TestClient(create_app(str(tmp_path / "c.sqlite")))
    for path in ("/", "/console"):
        html = c.get(path).text
        assert "Agent Debate" in html and 'data-tab="training"' in html


def test_training_api_includes_arena_progress_and_results(tmp_path):
    store = Store(tmp_path / "arena.sqlite")
    eid = "arena_test"
    store.create_experiment(eid, "arena:BTCUSDT", {"ga": {"population": 8, "generations": 2}})
    store.put(eid, "arena", "plan", {"mode": "arena", "auto": True, "max_rounds": 5})
    store.append(eid, "state", {"to": "ARENA_EVOLUTION", "fold": 0})
    store.append(eid, "generation", {"fold": 0, "gen": 0, "best": 0.4, "mean": 0.1,
                                       "valid": 6, "population": 8, "generations": 2,
                                       "folds": 1, "failed_risk": 1, "best_id": "chr_a",
                                       "round": 2, "training_mode": "arena"})
    store.put(eid, "arena", "result", {"champion": {"id": "chr_a"}, "summary": "Champion crowned",
                                        "total_spawned": 8, "is_survivors": 3, "oos_qualifiers": 1,
                                        "combatants": []})
    client = TestClient(create_app(str(tmp_path / "arena.sqlite")))
    response = client.get(f"/api/experiments/{eid}/training")
    assert response.status_code == 200
    data = response.json()
    assert data["plan"]["mode"] == "arena" and data["plan"]["auto"] is True
    assert data["generations"][0]["round"] == 2
    assert data["arena_result"]["champion"]["id"] == "chr_a"


def test_api_experiments_categorizes_live_and_training(tmp_path):
    store = Store(tmp_path / "exp_cat.sqlite")
    # Training experiment
    store.create_experiment("train_1", "arena:BTCUSDT", {"mode": "arena"})
    # Live testnet experiment
    store.create_experiment("live_1", "testnet:BTCUSDT", {
        "chromosome": {
            "id": "chr_champion",
            "genes": {
                "trend": {"ema_fast": 20, "ema_slow": 50},
                "stop_target": {"stop_atr": 2.0, "rr": 1.5}
            }
        }
    })
    store.set_state("live_1", "LIVE_TESTNET")

    c = TestClient(create_app(str(tmp_path / "exp_cat.sqlite")))
    res = c.get("/api/experiments").json()
    assert len(res) == 2
    live_exp = next(x for x in res if x["id"] == "live_1")
    train_exp = next(x for x in res if x["id"] == "train_1")
    
    assert live_exp["category"] == "live"
    assert live_exp["chromosome_id"] == "chr_champion"
    assert train_exp["category"] == "training"
    assert train_exp["chromosome_id"] is None

    # Test chromosomes fallback for live session
    ch_res = c.get("/api/experiments/live_1/chromosomes").json()
    assert len(ch_res) == 1
    assert ch_res[0]["id"] == "chr_champion"
    assert ch_res[0]["body"]["genes"]["trend"]["ema_fast"] == 20


def test_root_console_has_separated_views(tmp_path):
    Store(tmp_path / "ui.sqlite")
    c = TestClient(create_app(str(tmp_path / "ui.sqlite")))
    html = c.get("/console").text
    assert 'data-tab="training"' in html
    assert 'data-tab="live"' in html
    assert 'id="trainExpSelect"' in html
    assert 'id="liveExpSelect"' in html
    assert 'id="trainHistoryTable"' in html
    assert 'Active Chromosome in Live Trading' in html
    assert 'id="hdrTraining"' in html
    assert 'id="hdrLive"' in html


def test_websocket_stream_disconnect(tmp_path):
    store = Store(tmp_path / "ws.sqlite")
    client = TestClient(create_app(str(tmp_path / "ws.sqlite")))
    with client.websocket_connect("/ws/live/BTCUSDT?tf=60") as ws:
        _ = ws.receive_json()
        # Normal client close
        ws.close()


def test_console_exposes_separate_optional_agent_debate_and_judge():
    from app.api.read_only import CONSOLE
    html = CONSOLE.read_text()
    assert 'id="agentDebate"' in html
    assert 'id="btnAgentDebate"' in html
    assert "Run bull vs bear agents" in html
    assert "display-only" in html
    assert 'agents=llm' in html
    assert 'c.judge?.verdict' in html
    assert 'id="chkAutoDebate"' in html
    assert 'id="btnQuickDebate"' in html
    assert 'id="debateStatusBadge"' in html
    assert 'debate-timeline' in html
    assert 'highlightEvidence' in html
