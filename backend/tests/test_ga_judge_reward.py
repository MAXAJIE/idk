import inspect
import random

from app.agents import judge
from app.agents.provider import OfflineProvider, Provider, narrate_checked
from app.agents.reward import reward
from app.genetic import operators, population
from app.genetic.schema import validate
from app.risk.scorecard import fitness, gates, scorecard


def test_mutation_always_changes_at_least_one_gene():
    rng = random.Random(0)
    for _ in range(200):
        c = operators.random_chromosome(rng)
        for rate in (0.0, 0.3):
            new, changed = operators.mutate(rng, c.genes, rate=rate)
            assert len(changed) >= 1
            diffs = [(f, g) for f in new for g in new[f] if new[f][g] != c.genes[f][g]]
            assert len(diffs) == len(changed)  # every recorded mutation is a real change


def test_mutation_is_independent_of_judge():
    for fn in (operators.mutate, operators.offspring, population.next_generation):
        assert "judge" not in inspect.signature(fn).parameters
        assert "judge" not in inspect.getsource(fn).lower()


def test_offspring_valid_and_lineage_recorded():
    rng = random.Random(1)
    a, b = operators.random_chromosome(rng), operators.random_chromosome(rng)
    for _ in range(50):
        ch = operators.offspring(rng, a, b, 1, 0.7, 0.2, "ga-ops-1")
        if ch:
            assert not validate(ch.genes)
            assert ch.parents and ch.changed and ch.operator.startswith("ga-ops-1")
            assert ch.genes["trend"]["ema_fast"] < ch.genes["trend"]["ema_slow"]


class GA:
    tournament_k, crossover_rate, mutation_rate, operator_version = 2, 0.7, 0.2, "t"


def test_failed_parents_excluded_and_population_restored():
    rng = random.Random(2)
    pop = population.initial(rng, 8)
    for c in pop[:5]:
        c.state = "FAILED_RISK"
    scored = [(c, float(i)) for i, c in enumerate(pop)]
    failed_ids = {c.id for c in pop[:5]}
    nxt, _ = population.next_generation(rng, scored, [], 8, 1, GA)
    assert len(nxt) == 8
    assert not any(p in failed_ids for c in nxt for p in c.parents)


def _trade(tid, pnl, reason="target", closed=1, did="d1"):
    return {"trade_id": tid, "decision_id": did, "net_pnl": pnl, "exit_reason": reason, "closed_t": closed}


def _packet(trades, ranges=None, decisions=None, n_oos=20):
    return {"trades": trades, "exit_range_atr": ranges or {}, "valid_evidence": {"ev:a:1"},
            "decisions": decisions or {"d1": {"evidence_ids": ["ev:a:1"]}}, "robustness": {"retention": 1.0},
            "oos_scorecard": {"n_trades": n_oos}}


def test_judge_nullifies_single_windfall_by_trade_id(cfg):
    trades = [_trade("small", 50, closed=1), _trade("lucky", 900, "time", closed=2), _trade("small2", 60, closed=3)]
    v = judge.review("c", _packet(trades, {"lucky": 5.0}), cfg.judge, cfg.gates, 10_000)
    assert v.nullified_trade_ids == ["lucky"]
    assert v.adjusted_net_pnl == 110
    assert any(f.startswith("LUCKY_WINDFALL:lucky") for f in v.flags)


def test_judge_flags_ungrounded_profit_and_rejects_systemic(cfg):
    trades = [_trade("u1", 100, did="dx"), _trade("u2", 100, did="dy", closed=2), _trade("ok", 300, closed=3)]
    dec = {"d1": {"evidence_ids": ["ev:a:1"]}, "dx": {"evidence_ids": ["ev:fake:9"]}, "dy": {"evidence_ids": []}}
    v = judge.review("c", _packet(trades, decisions=dec), cfg.judge, cfg.gates, 10_000)
    assert set(v.nullified_trade_ids) == {"u1", "u2"} and v.verdict == "REJECT"


def test_judge_never_removes_losses(cfg):
    trades = [_trade("l", -500, "time", closed=1), _trade("w", 600, closed=2)]
    v = judge.review("c", _packet(trades), cfg.judge, cfg.gates, 10_000)
    assert v.adjusted_net_pnl <= 100


def test_failed_gate_ranks_below_any_passing_candidate(cfg):
    good = {"failed_risk": False, "n_trades": 20, "max_drawdown": 0.05, "stop_compliance": 1.0,
            "cost_share_of_gross": 0.2, "net_return": 0.01}
    huge_but_risky = good | {"max_drawdown": 0.5, "net_return": 5.0}
    assert fitness(good, gates(good, cfg.gates)) > fitness(huge_but_risky, gates(huge_but_risky, cfg.gates))
    assert fitness(good | {"failed_risk": True}, []) == -1e9


def test_scorecard_reports_gross_and_net_separately():
    t = {"gross_mid_pnl": 100, "spread_cost": 5, "slippage_cost": 5, "fees": 10, "funding": 2, "net_pnl": 78,
         "stop_compliant": True, "bars_held": 3}
    sc = scorecard([t], [(0, 1e4), (1, 10078)], [(0, 0), (1, 0.5)], 1e4, False)
    assert sc["gross_pnl"] == 100 and sc["net_pnl"] == 78 and sc["cost_total"] == 20


BASE_T = {"trade_id": "t", "decision_id": "d", "net_pnl": 100.0, "spread_cost": 1, "slippage_cost": 1, "fees": 1,
          "stop_compliant": True}


def test_no_reward_unless_adopted(cfg):
    assert reward(BASE_T | {"adoption": "REJECTED"}, 1e4, cfg.reward) is None
    assert reward(BASE_T | {"adoption": "NOT_EVALUATED"}, 1e4, cfg.reward) is None
    full = reward(BASE_T | {"adoption": "ADOPTED"}, 1e4, cfg.reward)
    part = reward(BASE_T | {"adoption": "PARTIALLY_ADOPTED"}, 1e4, cfg.reward)
    assert part["credit"] == 0.5 and abs(part["reward"] - 0.5 * full["reward"]) < 1e-7


def test_reward_scale_free(cfg):
    a = reward(BASE_T | {"adoption": "ADOPTED"}, 1e4, cfg.reward)["reward"]
    b = reward({**BASE_T, "net_pnl": 1000.0, "spread_cost": 10, "slippage_cost": 10, "fees": 10,
                "adoption": "ADOPTED"}, 1e5, cfg.reward)["reward"]
    assert abs(a - b) < 1e-9


class Liar(Provider):
    name = "liar"

    def narrate(self, record):
        return "Strong buy per ev:whale_flow:123 and ev:atr:1."


class Broken(Provider):
    name = "broken"

    def narrate(self, record):
        raise TimeoutError("upstream timeout")


def test_narration_unsupported_ids_flagged_and_errors_surfaced():
    r = narrate_checked(Liar(), {}, {"ev:atr:1"})
    assert not r["valid"] and r["unsupported_ids"] == ["ev:whale_flow:123"]
    e = narrate_checked(Broken(), {}, set())
    assert e["text"] is None and "TimeoutError" in e["error"]
    rec = {"proposal": {"side": "long", "discussion_adoption": "ADOPTED", "evidence_ids": ["ev:atr:1"],
                        "rationale": "x"}, "risk": {"outcome": "APPROVE", "reasons": ["ok"]}}
    assert narrate_checked(OfflineProvider(), rec, {"ev:atr:1"})["valid"]


def test_llm_debate_is_two_turn_evidence_checked():
    import json
    from app.agents.discussion import discuss_with_llm
    from app.features.pipeline import EvidenceBundle

    eb = EvidenceBundle(t=1, ts=123, valid=True, items={
        "ev:regime:123": {"kind": "regime", "value": {"trend": "up"}},
        "ev:smc:123": {"kind": "smc", "value": {"bias": "bear"}},
    })

    class DebateProvider:
        def __init__(self):
            self.calls = []

        def chat(self, system, user, max_tokens=400):
            prompt = json.loads(user)
            self.calls.append(prompt)
            if prompt["role"] == "bull":
                return json.dumps({"claims": [{"text": "Regime trend is up", "evidence_id": "ev:regime:123", "weight": 0.5}],
                                   "counterarguments": ["Bearish structure is contrary evidence [ev:smc:123]"],
                                   "invalidation": "trend evidence changes", "uncertainty": 0.4})
            return json.dumps({"claims": [{"text": "Structure bias is bear", "evidence_id": "ev:smc:123", "weight": 0.6}],
                               "counterarguments": ["Bullish regime conflicts [ev:regime:123]"],
                               "invalidation": "structure evidence changes", "uncertainty": 0.3})

    provider = DebateProvider()
    report = discuss_with_llm(eb, provider)
    assert len(provider.calls) == 2
    assert provider.calls[1]["opposing_turn"]["role"] == "bull"
    assert [turn.role for turn in report.turns] == ["bull", "bear"]
    assert report.recommendation in ("long", "short", "no_trade")


def test_llm_debate_rejects_unknown_evidence_citation():
    import json
    from app.agents.discussion import discuss_with_llm
    from app.features.pipeline import EvidenceBundle

    eb = EvidenceBundle(t=1, ts=123, valid=True, items={
        "ev:regime:123": {"kind": "regime", "value": {"trend": "up"}},
    })

    class BadCitationProvider:
        def chat(self, system, user, max_tokens=400):
            return json.dumps({"claims": [{"text": "unsupported", "evidence_id": "ev:future:999", "weight": 1}],
                               "counterarguments": [], "invalidation": "unknown", "uncertainty": 0.5})

    import pytest
    with pytest.raises(ValueError, match="outside this candle"):
        discuss_with_llm(eb, BadCitationProvider())


def test_llm_debate_handles_unescaped_quotes_and_malformed_json():
    import json
    from app.agents.discussion import discuss_with_llm
    from app.features.pipeline import EvidenceBundle

    eb = EvidenceBundle(t=1, ts=123, valid=True, items={
        "ev:regime:123": {"kind": "regime", "value": {"trend": "up"}},
        "ev:smc:123": {"kind": "smc", "value": {"bias": "bear"}},
    })

    class MalformedJsonProvider:
        def chat(self, system, user, max_tokens=1500):
            prompt = json.loads(user)
            role = prompt["role"]
            if role == "bull":
                # Simulated markdown fences + unescaped quotes inside string
                return (
                    '```json\n{\n'
                    '  "claims": [\n'
                    '    {\n'
                    '      "text": "Regime shows \"bullish\" trend momentum [ev:regime:123]",\n'
                    '      "evidence_id": "ev:regime:123",\n'
                    '      "weight": 0.7\n'
                    '    }\n'
                    '  ],\n'
                    '  "counterarguments": [\n'
                    '    "Bearish SMC bias is \"contrary\" evidence [ev:smc:123]"\n'
                    '  ],\n'
                    '  "invalidation": "trend slope reverses below 0",\n'
                    '  "uncertainty": 0.3\n'
                    '}\n```'
                )
            # Trailing comma and missing comma between lines
            return (
                '{\n'
                '  "claims": [\n'
                '    {\n'
                '      "text": "SMC orderblock breakdown",\n'
                '      "evidence_id": "ev:smc:123",\n'
                '      "weight": 0.8\n'
                '    }\n'
                '  ],\n'
                '  "counterarguments": [\n'
                '    "Regime is still bullish [ev:regime:123]"\n'
                '  ],\n'
                '  "invalidation": "break of swing high",\n'
                '  "uncertainty": 0.25,\n'
                '}'
            )

    report = discuss_with_llm(eb, MalformedJsonProvider())
    assert len(report.turns) == 2
    assert report.turns[0].role == "bull"
    assert report.turns[0].claims[0].evidence_id == "ev:regime:123"
    assert report.turns[1].role == "bear"
    assert report.turns[1].claims[0].evidence_id == "ev:smc:123"
    assert report.recommendation in ("long", "short", "no_trade")


def test_llm_debate_full_two_round_cross_examination():
    import json
    from app.agents.discussion import discuss_with_llm
    from app.features.pipeline import EvidenceBundle

    eb = EvidenceBundle(t=1, ts=123, valid=True, items={
        "ev:regime:123": {"kind": "regime", "value": {"trend": "up"}},
        "ev:smc:123": {"kind": "smc", "value": {"bias": "bear"}},
    })

    class MultiTurnDebateProvider:
        def __init__(self):
            self.stages = []

        def chat(self, system, user, max_tokens=1500):
            p = json.loads(user)
            self.stages.append((p["role"], p["stage"]))
            if p["role"] == "bull":
                return json.dumps({
                    "claims": [{"text": f"Bull claim at {p['stage']}", "evidence_id": "ev:regime:123", "weight": 0.6}],
                    "counterarguments": ["Challenge to bear [ev:smc:123]"],
                    "invalidation": "loss of 64000", "uncertainty": 0.3
                })
            return json.dumps({
                "claims": [{"text": f"Bear counter at {p['stage']}", "evidence_id": "ev:smc:123", "weight": 0.5}],
                "counterarguments": ["Challenge to bull [ev:regime:123]"],
                "invalidation": "gain above 66000", "uncertainty": 0.35
            })

    provider = MultiTurnDebateProvider()
    report = discuss_with_llm(eb, provider, rounds=2)
    assert len(provider.stages) == 4
    assert [s[0] for s in provider.stages] == ["bull", "bear", "bull", "bear"]
    assert [s[1] for s in provider.stages] == ["opening", "cross_examination", "rebuttal", "closing"]
    assert len(report.turns) == 4
    assert "cross-examination" in report.judge_summary.lower()
    assert report.recommendation in ("long", "short", "no_trade")


def test_arena_evidence_ids_are_not_double_prefixed(cfg):
    from app.features.pipeline import EvidenceBundle
    from app.learning.arena import _valid_evidence_by_decision

    eb = EvidenceBundle(t=7, ts=1234, valid=True, items={
        "ev:atr:1234": {"kind": "atr", "value": {"atr14": 2.0}},
        "ev:snr:1234": {"kind": "snr", "value": {"support": None}},
    })
    decision = {"t": 7, "proposal": {"decision_id": "d7", "evidence_ids": ["ev:atr:1234", "ev:snr:1234"]}}

    valid = _valid_evidence_by_decision([eb] * 8, [decision])
    assert valid == {"d7": {"ev:atr:1234", "ev:snr:1234"}}

    trade = _trade("profit", 100, did="d7")
    packet = _packet([trade], decisions={"d7": decision["proposal"]})
    packet["valid_evidence"] = valid
    verdict = judge.review("c", packet, cfg.judge, cfg.gates, 10_000)
    assert verdict.nullified_trade_ids == []
    assert verdict.verdict != "REJECT"
