"""Discussion reward attribution, logged separately from chromosome fitness.

Predeclared attribution: ADOPTED -> 1.0, PARTIALLY_ADOPTED -> 0.5, otherwise no
update. The fraction is fixed before outcomes are known.
"""
from __future__ import annotations

import math

from app.config import RewardConfig

CREDIT = {"ADOPTED": 1.0, "PARTIALLY_ADOPTED": 0.5}


def reward(trade: dict, equity_before: float, cfg: RewardConfig) -> dict | None:
    credit = CREDIT.get(trade["adoption"])
    if credit is None or equity_before <= 0:
        return None
    r = trade["net_pnl"] / equity_before  # scale-free
    util = math.log(max(1e-6, 1 + r))
    cost = (trade["spread_cost"] + trade["slippage_cost"] + trade["fees"]) / equity_before
    pen = cfg.downside_penalty * max(0.0, -r) ** 2 + cfg.cost_penalty * cost
    pen += cfg.violation_penalty * (0.0 if trade["stop_compliant"] else abs(r))
    return {"trade_id": trade["trade_id"], "decision_id": trade["decision_id"], "adoption": trade["adoption"],
            "credit": credit, "reward": round(credit * (util - pen), 8)}


def train_offline(records: list[dict], weights: dict[str, float], lr: float = 0.1) -> dict[str, float]:
    """Reproducible offline update of claim-kind weights from closed IS episodes.
    Never invoked during an OOS evaluation."""
    w = dict(weights)
    for rec in sorted(records, key=lambda x: x["trade_id"]):
        for k in rec.get("claim_kinds", []):
            w[k] = min(1.0, max(0.0, w.get(k, 0.5) + lr * rec["reward"]))
    return w
