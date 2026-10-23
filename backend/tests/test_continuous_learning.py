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
    # The fixture cannot support the configured independent OOS windows and holdout.
    assert res.total_spawned == 0
    assert res.combatants == []
    assert "insufficient valid walk-forward history" in res.summary


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
    assert rounds == 2
    assert res.total_spawned == 0
    report = analyze_arena_result(res, cfg, round_num=rounds)
    assert "ARENA COMBAT & ANTI-OVERFITTING ANALYSIS REPORT" in report
    assert "insufficient valid walk-forward history" in report


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
    assert generations == []
    assert any("insufficient valid walk-forward history" in payload.get("reason", "")
               for event_type, payload in events if event_type == "state")


def test_arena_fails_closed_when_valid_wfa_windows_do_not_fit():
    """Short history must not be recycled as overlapping OOS and holdout data."""
    from app.learning.arena import train_and_combat
    cfg = load_config(Path(__file__).parents[1] / "configs/default.json")
    bars = []
    price = 100.0
    for i in range(360):
        price += 0.4 if i % 3 != 0 else -0.5
        bars.append(Bar(i=i, ts=1000 + i * 60, open=price - 0.2, high=price + 0.8,
                        low=price - 0.8, close=price, volume=1500.0, funding_rate=0.0001))

    result = train_and_combat(bars, cfg, pop_size=8, generations=2, seed=7)
    assert result.champion is None
    assert result.total_spawned == 0
    assert "insufficient valid walk-forward history" in result.summary



def test_arena_records_profitable_oos_fold_count():
    from app.learning.arena import ArenaCombatant
    from app.genetic.operators import random_chromosome
    import random

    record = ArenaCombatant(
        chromosome=random_chromosome(random.Random(9)),
        is_pnl=1.0, is_drawdown=0.01, oos_pnl=1.0, oos_drawdown=0.01,
        consistency=0.5, neighbour_retention=1.0, qualified=False,
        status_reason="too few profitable OOS folds (1 < 2)",
        positive_oos_folds=1, total_oos_folds=3,
    )
    serialized = record.to_dict()
    assert serialized["positive_oos_folds"] == 1
    assert serialized["total_oos_folds"] == 3


def test_arena_selection_fitness_penalizes_is_wfa_gap_and_complexity():
    from app.learning.arena import selection_fitness

    consistent, robust, gap = selection_fitness(10.0, [9.0, 8.0, 7.0], complexity=0.0)
    overfit, _, overfit_gap = selection_fitness(10.0, [3.0, 8.0, 7.0], complexity=0.0)
    simpler, _, _ = selection_fitness(10.0, [9.0, 8.0, 7.0], complexity=2.0)

    assert robust == 7.4  # weighted worst and median folds
    assert abs(gap - 2.6) < 1e-9
    assert overfit_gap > gap
    assert overfit < consistent
    assert simpler < consistent


def test_auto_train_stops_after_single_holdout_attempt(monkeypatch):
    from app.learning import arena

    calls = []
    result = arena.ArenaResult(
        champion=None, total_spawned=8, is_survivors=2, oos_qualifiers=1,
        combatants=[], summary="final holdout failed", holdout_attempted=True,
    )

    def fake_train(**kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(arena, "train_and_combat", fake_train)
    cfg = load_config(Path(__file__).parents[1] / "configs/default.json")
    returned, rounds = arena.auto_train_until_survivor(
        bars=[], cfg=cfg, pop_size=8, generations=4, max_rounds=10,
    )

    assert returned is result
    assert rounds == 1
    assert len(calls) == 1
