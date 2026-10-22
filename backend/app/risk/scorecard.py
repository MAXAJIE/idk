"""Candidate risk scorecard (net of all costs) and hard qualification gates."""
from __future__ import annotations

import math

from app.config import GateConfig


def max_drawdown(curve: list[float]) -> float:
    peak, mdd = -math.inf, 0.0
    for v in curve:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, 1 - v / peak)
    return mdd


def scorecard(trades: list[dict], equity: list[tuple[int, float]], exposure: list[tuple[int, float]],
              initial: float, failed_risk: bool) -> dict:
    eq = [e for _, e in equity] or [initial]
    rets = [eq[k] / eq[k - 1] - 1 for k in range(1, len(eq)) if eq[k - 1] > 0]
    mean = sum(rets) / len(rets) if rets else 0.0
    sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / len(rets)) if rets else 0.0
    dsd = math.sqrt(sum(r * r for r in rets if r < 0) / len(rets)) if rets else 0.0
    gross = sum(t["gross_mid_pnl"] for t in trades)
    costs = sum(t["spread_cost"] + t["slippage_cost"] + t["fees"] for t in trades)
    net = sum(t["net_pnl"] for t in trades)
    pos_gross = sum(max(0.0, t["gross_mid_pnl"]) for t in trades)
    n = len(trades)
    xs = [x for _, x in exposure]
    return {
        "n_trades": n,
        "net_pnl": round(net, 4),
        "net_return": round(net / initial, 6),
        "gross_pnl": round(gross, 4),
        "cost_total": round(costs, 4),
        "funding_total": round(sum(t["funding"] for t in trades), 4),
        "cost_share_of_gross": round(costs / pos_gross, 4) if pos_gross > 0 else None,
        "max_drawdown": round(max_drawdown(eq), 6),
        "sharpe_bar": round(mean / sd, 6) if sd > 0 else 0.0,
        "sortino_bar": round(mean / dsd, 6) if dsd > 0 else 0.0,
        "win_rate": round(sum(t["net_pnl"] > 0 for t in trades) / n, 4) if n else None,
        "stop_compliance": round(sum(t["stop_compliant"] for t in trades) / n, 4) if n else None,
        "avg_bars_held": round(sum(t["bars_held"] for t in trades) / n, 2) if n else None,
        "avg_exposure": round(sum(xs) / len(xs), 4) if xs else 0.0,
        "max_exposure": round(max(xs), 4) if xs else 0.0,
        "time_in_market": round(sum(x > 0 for x in xs) / len(xs), 4) if xs else 0.0,
        "worst_trade": round(min((t["net_pnl"] for t in trades), default=0.0), 4),
        "failed_risk": failed_risk,
    }


def gates(sc: dict, g: GateConfig) -> list[dict]:
    """Ordered hard gates."""
    out = []

    def add(name, ok, value, limit):
        out.append({"gate": name, "passed": bool(ok), "value": value, "limit": limit})

    add("not_failed_risk", not sc["failed_risk"], sc["failed_risk"], False)
    add("min_trades", sc["n_trades"] >= g.min_trades, sc["n_trades"], g.min_trades)
    add("max_drawdown", sc["max_drawdown"] <= g.max_drawdown_frac, sc["max_drawdown"], g.max_drawdown_frac)
    c = sc["stop_compliance"]
    add("stop_compliance", c is not None and c >= g.min_stop_compliance, c, g.min_stop_compliance)
    cs = sc["cost_share_of_gross"]
    add("cost_share", cs is not None and cs <= g.max_cost_share_of_gross, cs, g.max_cost_share_of_gross)
    add("net_return", sc["net_return"] > g.min_net_return, sc["net_return"], g.min_net_return)
    return out


def fitness(sc: dict, gate_results: list[dict]) -> float:
    """Risk before return: each failed gate costs more than any achievable return."""
    if sc["failed_risk"]:
        return -1e9
    failed = sum(not x["passed"] for x in gate_results)
    return sc["net_return"] / max(sc["max_drawdown"], 0.01) - 100.0 * failed
