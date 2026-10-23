"""Tests for continuous learning, reflection memory, and online re-evolution."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.config import load_config
from app.data.ingest import Bar
from app.genetic.schema import Chromosome
from app.learning.online_evolution import evaluate_rolling_challenger
from app.learning.reflection import InhibitionVerdict, ReflectionMemory


def test_reflection_memory_inhibition_and_persistence(tmp_path: Path):
    mem = ReflectionMemory(veto_loss_rate=0.70, penalty_loss_rate=0.50, min_trades=2)
    sig = "breakout:long:trend_up:val_high"

    # 1. Unknown or insufficient trades -> ALLOW
    assert mem.check_inhibition(sig).action == "ALLOW"

    # 2. Record 1 loss -> still < min_trades (2) -> ALLOW
    mem.record_outcome(sig, net_pnl=-15.0, exit_reason="stop", t=100)
    assert mem.check_inhibition(sig).action == "ALLOW"

    # 3. Record 2nd loss -> 2/2 = 100% loss rate -> VETO
    mem.record_outcome(sig, net_pnl=-25.0, exit_reason="stop", t=101)
    v1 = mem.check_inhibition(sig)
    assert v1.inhibited is True
    assert v1.action == "VETO"
    assert v1.size_mult == 0.0

    # 4. Save and load persistence check
    save_file = tmp_path / "reflection.json"
    mem.save(save_file)
    loaded_mem = ReflectionMemory.load(save_file)
    v2 = loaded_mem.check_inhibition(sig)
    assert v2.action == "VETO"
    assert v2.size_mult == 0.0

    # 5. Recovery on winning trade
    loaded_mem.record_outcome(sig, net_pnl=50.0, exit_reason="target", t=102)
    # total: 3 trades, 2 losses -> 66.7% loss rate -> RESIZE_PENALTY (>= 50%, < 70%)
    v3 = loaded_mem.check_inhibition(sig)
    assert v3.inhibited is True
    assert v3.action == "RESIZE_PENALTY"
    assert v3.size_mult == 0.5


def test_online_evolution_challenger_evaluation():
    cfg = load_config(Path(__file__).parents[1] / "configs/default.json")
    
    # Create synthetic bars
    bars = []
    price = 100.0
    for i in range(120):
        # generate smooth walk
        price += 0.5 if i % 2 == 0 else -0.3
        bars.append(Bar(i=i, ts=1000 + i * 60, open=price - 0.2, high=price + 1.0, low=price - 1.0,
                        close=price, volume=1000.0, funding_rate=0.0001))

    # Base chromosome
    from app.genetic.schema import Chromosome
    incumbent = Chromosome(
        genes={
            "amt": {"mode": "poc_retest", "lookback": 20},
            "smc": {"fvg_threshold": 0.002, "swing_len": 5},
            "snr": {"cluster_tol": 0.005, "strength_min": 2},
            "trend": {"ema_fast": 9, "ema_slow": 21, "adx_min": 20.0},
            "entry": {"trigger": "trend_pullback", "confirm_bars": 2},
            "stop_target": {"atr_mult_stop": 1.5, "rr_ratio": 2.0, "max_hold_bars": 12},
            "sizing": {"risk_per_trade": 0.01, "vol_scale": 1.0},
            "cost": {"max_cost_to_reward": 0.35},
        },
        generation=1
    )

    eval_res = evaluate_rolling_challenger(incumbent, bars, cfg, min_improvement=0.05, pop_size=6, seed=1234)
    # Verify evaluation contract outputs
    assert eval_res.incumbent_id == incumbent.id
    assert isinstance(eval_res.won, bool)
    assert eval_res.reason != ""


def test_testnet_client_retries_transient_network_errors():
    from app.live.testnet import TestnetClient, ExchangeError
    calls = 0

    def mock_transport(method, url, headers):
        nonlocal calls
        calls += 1
        if calls < 3:
            # First two calls fail with 599 network timeout
            return 599, '{"error": "NetworkError", "detail": "Connection timed out"}'
        # Third call succeeds
        return 200, '{"result": "ok"}'

    client = TestnetClient(transport=mock_transport, max_retries=3, retry_backoff_s=0.01)
    res = client.request("GET", "/fapi/v1/ping")
    assert res == {"result": "ok"}
    assert calls == 3


def test_arena_train_and_combat_anti_overfitting():
    from app.learning.arena import train_and_combat
    cfg = load_config(Path(__file__).parents[1] / "configs/default.json")
    bars = []
    price = 100.0
    for i in range(360):
        price += 0.4 if i % 3 != 0 else -0.5
        bars.append(Bar(i=i, ts=1000 + i * 60, open=price - 0.2, high=price + 0.8,
                        low=price - 0.8, close=price, volume=1500.0, funding_rate=0.0001))

    res = train_and_combat(bars, cfg, pop_size=8, generations=2, elite_pool_size=4,
                           min_consistency=0.5, seed=999)
    assert res.total_spawned >= 8
    assert isinstance(res.combatants, list)
    assert res.summary != ""
    # Verify every combatant evaluated WFA consistency and neighbour retention
    for c in res.combatants:
        assert hasattr(c, "consistency")
        assert hasattr(c, "neighbour_retention")
        assert hasattr(c, "qualified")


def test_auto_train_until_survivor_and_analysis_reporting():
    from app.learning.arena import auto_train_until_survivor, analyze_arena_result
    cfg = load_config(Path(__file__).parents[1] / "configs/default.json")
    bars = []
    price = 100.0
    for i in range(360):
        price += 0.4 if i % 3 != 0 else -0.5
        bars.append(Bar(i=i, ts=1000 + i * 60, open=price - 0.2, high=price + 0.8,
                        low=price - 0.8, close=price, volume=1500.0, funding_rate=0.0001))

    res, rounds = auto_train_until_survivor(
        bars, cfg, pop_size=6, generations=2, elite_pool_size=3,
        min_consistency=0.5, seed=42, max_rounds=2
    )
    assert rounds >= 1
    assert res.total_spawned >= 6
    report = analyze_arena_result(res, cfg, round_num=rounds)
    assert "ARENA COMBAT & ANTI-OVERFITTING ANALYSIS REPORT" in report
    assert "Status:" in report


def test_arena_emits_generation_progress_events():
    from app.learning.arena import train_and_combat
    cfg = load_config(Path(__file__).parents[1] / "configs/default.json")
    bars = []
    price = 100.0
    for i in range(360):
        price += 0.4 if i % 3 != 0 else -0.5
        bars.append(Bar(i=i, ts=1000 + i * 60, open=price - 0.2, high=price + 0.8,
                        low=price - 0.8, close=price, volume=1500.0, funding_rate=0.0001))
    events = []
    train_and_combat(bars, cfg, pop_size=6, generations=2, elite_pool_size=3,
                     min_consistency=0.5, seed=2026,
                     progress=lambda event_type, payload: events.append((event_type, payload)))
    generations = [payload for event_type, payload in events if event_type == "generation"]
    assert len(generations) == 2
    assert [g["gen"] for g in generations] == [0, 1]
    assert all(g["training_mode"] == "arena" for g in generations)
    assert all(g["population"] == 6 for g in generations)
    assert any(payload.get("to") == "ARENA_COMBAT" for event_type, payload in events if event_type == "state")
