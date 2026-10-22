"""Judge: deterministic audit of a frozen IS+OOS packet. It cannot alter
mutation, risk thresholds or trade permissions; it can only remove profits
(trade-level) or candidates (chromosome-level)."""
from __future__ import annotations

from app.agents.schemas import JudgeVerdict
from app.config import GateConfig, JudgeConfig
from app.risk.scorecard import max_drawdown


def review(chromosome_id: str, packet: dict, cfg: JudgeConfig, gates: GateConfig, initial: float) -> JudgeVerdict:
    """packet keys: trades, exit_range_atr{trade_id}, decisions{decision_id: proposal},
    valid_evidence(set), robustness{retention}, oos_scorecard."""
    trades = packet["trades"]
    flags, concerns, nullified = [], [], []
    pos_total = sum(max(0.0, t["net_pnl"]) for t in trades)
    for t in trades:
        cited = packet["decisions"].get(t["decision_id"], {}).get("evidence_ids", [])
        if t["net_pnl"] > 0 and (not cited or any(e not in packet["valid_evidence"] for e in cited)):
            nullified.append(t["trade_id"])
            flags.append(f"UNGROUNDED_PROFIT:{t['trade_id']}")
            continue
        dominant = pos_total > 0 and t["net_pnl"] / pos_total >= cfg.windfall_dominance_frac
        extreme = packet["exit_range_atr"].get(t["trade_id"], 0.0) >= cfg.windfall_range_atr_mult
        off_plan = t["exit_reason"] not in ("target", "stop")
        if t["net_pnl"] > 0 and dominant and (extreme or off_plan):
            nullified.append(t["trade_id"])
            flags.append(f"LUCKY_WINDFALL:{t['trade_id']}")

    # post-audit ranking ledger: nullified profits -> 0; losses are never removed
    eq, curve = initial, [initial]
    for t in sorted(trades, key=lambda x: x["closed_t"]):
        eq += 0.0 if t["trade_id"] in nullified else t["net_pnl"]
        curve.append(eq)
    adj_net, adj_dd = eq - initial, max_drawdown(curve)

    rob = packet.get("robustness", {})
    if rob.get("retention", 1.0) < 0.5:
        concerns.append(f"fragile: neighbours retain {rob.get('retention')} of IS net")
    verdict, why = "RETAIN", "consistent with documented rules"
    if len(nullified) >= cfg.systemic_windfall_count:
        verdict, why = "REJECT", f"systemic: {len(nullified)} nullified trades"
    elif adj_dd > gates.max_drawdown_frac or adj_net / initial <= gates.min_net_return:
        verdict, why = "REJECT", "fails gates after windfall stripping"
    elif packet["oos_scorecard"]["n_trades"] < gates.min_trades:
        verdict, why = "DEFER", "insufficient OOS trades"
    elif concerns:
        verdict, why = "DEFER", "fragile parameter neighbourhood"
    return JudgeVerdict(chromosome_id=chromosome_id, verdict=verdict, nullified_trade_ids=nullified, flags=flags,
                        risk_concerns=concerns, evidence_refs=[t["trade_id"] for t in trades][:50],
                        adjusted_net_pnl=round(adj_net, 4), adjusted_max_drawdown=round(adj_dd, 6),
                        uncertainty=round(1 / (1 + len(trades) ** 0.5), 3), rationale=why)
