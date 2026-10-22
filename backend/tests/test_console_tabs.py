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
        assert "Debate &amp; indicators" in html and 'data-tab="training"' in html
