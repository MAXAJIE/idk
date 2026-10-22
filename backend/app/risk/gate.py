"""Authoritative deterministic pre-trade Risk Control gate.

Every cap can only lower the approved size; any hard failure blocks the order.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from app.agents.schemas import RiskCheck, RiskDecision, TradeProposal
from app.config import CostConfig, RiskConfig
from app.data.ingest import Bar
from app.simulator.costs import estimate_round_trip

MIN_NOTIONAL = 10.0


@dataclass(frozen=True)
class AccountState:
    equity: float
    open_positions: int
    halted: bool


@dataclass(frozen=True)
class MarketState:
    bar: Bar
    evidence_valid: bool
    staleness_bars: int


def evaluate(p: TradeProposal | dict, acct: AccountState, mkt: MarketState, risk: RiskConfig,
             costs: CostConfig) -> RiskDecision:
    if isinstance(p, dict):  # untrusted input path
        try:
            p = TradeProposal.model_validate(p)
        except Exception as e:  # noqa: BLE001
            return RiskDecision(decision_id=str(p.get("decision_id", "invalid")), outcome="REJECT",
                                requested_qty=0.0, approved_qty=0.0,
                                checks=[RiskCheck(name="schema", passed=False, detail=str(e)[:300])],
                                reasons=["malformed proposal"], risk_version=risk.version)
    checks: list[RiskCheck] = []

    def done(outcome, qty, reasons):
        return RiskDecision(decision_id=p.decision_id, outcome=outcome, requested_qty=p.qty, approved_qty=qty,
                            checks=checks, reasons=reasons, risk_version=risk.version)

    if p.side == "no_trade":
        return done("NO_TRADE", 0.0, ["no-trade is always permitted"])

    def hard(name, ok, value=None, limit=None, detail=""):
        checks.append(RiskCheck(name=name, passed=bool(ok), value=value, limit=limit, detail=detail))
        return ok

    if not hard("chromosome_active", not acct.halted, detail="halted chromosomes cannot trade"):
        return done("REJECT", 0.0, ["chromosome halted"])
    if not hard("data_fresh", mkt.evidence_valid and mkt.staleness_bars <= risk.max_data_staleness_bars,
                mkt.staleness_bars, risk.max_data_staleness_bars):
        return done("DEFER", 0.0, ["stale or invalid evidence"])
    if not hard("concurrent_positions", acct.open_positions < risk.max_concurrent_positions,
                acct.open_positions, risk.max_concurrent_positions):
        return done("REJECT", 0.0, ["concurrent position limit"])
    if not hard("equity_positive", acct.equity > 0, acct.equity, 0):
        return done("REJECT", 0.0, ["no equity"])

    side = 1 if p.side == "long" else -1
    px = p.ref_price
    stop_ok = (p.stop is not None and math.isfinite(p.stop) and p.stop > 0 and px > 0 and side * (px - p.stop) > 0)
    if not hard("stop_defined_valid", stop_ok, p.stop, f"{'below' if side > 0 else 'above'} {px}"):
        return done("REJECT", 0.0, ["missing or invalid stop"])
    target_ok = p.target is not None and math.isfinite(p.target) and side * (p.target - px) > 0
    if not hard("target_defined_valid", target_ok, p.target):
        return done("REJECT", 0.0, ["missing or invalid target"])
    if not hard("qty_positive", p.qty > 0, p.qty, ">0"):
        return done("REJECT", 0.0, ["non-positive quantity"])
    if not hard("spread", costs.half_spread_bps * 2 <= risk.max_spread_bps, costs.half_spread_bps * 2, risk.max_spread_bps):
        return done("DEFER", 0.0, ["spread too wide"])

    bar = mkt.bar
    stop_dist = abs(px - p.stop)
    caps = {
        "per_trade_risk": risk.max_risk_per_trade_frac * acct.equity / stop_dist,
        "leverage": risk.max_leverage * acct.equity / px,
        "liquidity": risk.max_bar_volume_frac * bar.volume,
        "stress_gap": risk.max_stress_loss_frac * acct.equity / (stop_dist * risk.stress_gap_mult),
        # isolated-margin approx: liq distance frac = equity/notional - mm >= k*stop_dist/px
        "liquidation_headroom": acct.equity / (risk.min_liq_distance_to_stop * stop_dist / px + risk.maintenance_margin_frac) / px,
    }
    qty = p.qty
    for name, cap in caps.items():
        hard(name, p.qty <= cap, round(p.qty, 8), round(cap, 8), "resized" if p.qty > cap else "")
        qty = min(qty, cap)
    reasons = [f"resized by {c.name}" for c in checks if not c.passed and c.detail == "resized"]

    if not hard("min_notional", qty * px >= MIN_NOTIONAL, round(qty * px, 4), MIN_NOTIONAL):
        return done("REJECT", 0.0, reasons + ["below minimum notional after caps"])

    rt = estimate_round_trip(costs, bar, px, qty)
    reward = qty * abs(p.target - px)
    ratio = rt / reward if reward > 0 else math.inf
    if not hard("cost_to_reward", ratio <= risk.max_cost_to_reward, round(ratio, 4), risk.max_cost_to_reward):
        return done("DEFER", 0.0, reasons + ["costs consume too much of the expected reward"])

    capped = any(cap < p.qty for cap in caps.values())
    qty = math.floor(qty * 1e8) / 1e8  # never round up
    return done("RESIZE" if capped else "APPROVE", qty, reasons or ["all checks passed"])
