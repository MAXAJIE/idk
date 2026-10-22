"""Event-driven paper simulator for one chromosome over one segment.

Timing: decision at close of bar t using evidence(t) only; market entry fills at
open of t+1. Stops/targets are checked intrabar; if both are touched in one bar
the stop is assumed first (conservative). Gaps through a stop fill at the open.
Open positions at segment end close at the last close.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.agents import discussion as disc_mod
from app.agents import trader
from app.agents.reward import reward
from app.config import ExperimentConfig
from app.data.ingest import Bar
from app.features.pipeline import EvidenceBundle
from app.features.trend import ema
from app.genetic.schema import Chromosome
from app.risk import gate
from app.risk.scorecard import scorecard
from app.simulator.costs import market_fill
from app.simulator.ledger import Ledger, Position


@dataclass
class Context:
    cfg: ExperimentConfig
    bars: list[Bar]
    evidence: list[EvidenceBundle]
    _ema: dict[int, list[float]] = field(default_factory=dict)
    _disc: dict[int, object] = field(default_factory=dict)

    def ema(self, period: int) -> list[float]:
        if period not in self._ema:
            self._ema[period] = ema([b.close for b in self.bars], period)  # causal
        return self._ema[period]

    def discussion(self, t: int):
        if t not in self._disc:
            self._disc[t] = disc_mod.discuss(self.evidence[t])
        return self._disc[t]


def simulate(c: Chromosome, ctx: Context, start: int, end: int, record: bool = False) -> dict:
    cfg, bars = ctx.cfg, ctx.bars
    L = Ledger(cfg.risk.initial_equity)
    ef, es = ctx.ema(c.g("trend", "ema_fast")), ctx.ema(c.g("trend", "ema_slow"))
    # the chromosome's own cost tolerance may only tighten the global limit
    risk_cfg = cfg.risk.model_copy(update={"max_cost_to_reward": min(cfg.risk.max_cost_to_reward,
                                                                     c.g("cost", "max_cost_to_reward"))})
    max_hold = c.g("stop_target", "max_hold_bars")
    pending = None
    decisions, risk_counts, rewards, exit_range = [], {}, [], {}
    fail_event = None
    n_trade = 0

    def close(t, ref, reason, slip_mult=1.0):
        p = L.position
        fc = market_fill(cfg.costs, bars[t], ref, p.qty, -p.side, slip_mult)
        eq_before = L.initial + sum(x.net_pnl for x in L.trades)
        tr = L.close(t, ref, fc.price, fc.spread_cost, fc.slippage_cost, fc.fee, reason)
        atr = ctx.evidence[t].get("atr")
        exit_range[tr.trade_id] = (bars[t].high - bars[t].low) / atr["atr14"] if atr and atr["atr14"] else 0.0
        rw = reward(tr.to_dict(), eq_before, cfg.reward)
        if rw:
            rewards.append(rw)

    for t in range(start, end):
        b = bars[t]
        if pending is not None:
            prop, qty = pending
            pending = None
            side = 1 if prop.side == "long" else -1
            fc = market_fill(cfg.costs, b, b.open, qty, side)
            shift = fc.price - prop.ref_price  # keep stop/target distances relative to the fill
            n_trade += 1
            L.open(Position(f"tr:{c.id}:{n_trade}:{t}", side, qty, b.open, fc.price, prop.stop + shift,
                            prop.target + shift, t, qty * abs(prop.ref_price - prop.stop), fc.spread_cost,
                            fc.slippage_cost, fc.fee, prop.decision_id, prop.discussion_adoption))
        p = L.position
        if p:
            stop_hit = b.low <= p.stop if p.side > 0 else b.high >= p.stop
            tgt_hit = b.high >= p.target if p.side > 0 else b.low <= p.target
            if stop_hit:
                gap = b.open < p.stop if p.side > 0 else b.open > p.stop
                close(t, b.open if gap else p.stop, "stop")
            elif tgt_hit:
                close(t, p.target, "target")
            elif t - p.opened_t >= max_hold:
                close(t, b.close, "time")
        if L.position and (t - start) % cfg.costs.funding_interval_bars == 0:
            L.apply_funding(b.funding_rate, b.close)
        L.mark(t, b.close)
        if L.drawdown(b.close) >= cfg.risk.hard_drawdown_frac:
            if L.position:  # conservative adverse liquidation
                close(t, b.close, "risk_liquidation", slip_mult=3.0)
                L.equity_curve[-1] = (t, round(L.equity(b.close), 6))
            fail_event = {"t": t, "drawdown": round(1 - L.cash / L.peak, 6)}
            L.halted = True
            break
        if L.position or t >= end - 1:
            continue
        eb = ctx.evidence[t]
        d = ctx.discussion(t) if eb.valid else None
        prop = trader.propose(c, eb, d, ef[t], es[t], L.equity(b.close))
        if prop.side == "no_trade" and prop.discussion_adoption == "NOT_EVALUATED":
            continue
        rd = gate.evaluate(prop, gate.AccountState(L.equity(b.close), 0, L.halted),
                           gate.MarketState(b, eb.valid, 0), risk_cfg, cfg.costs)
        risk_counts[rd.outcome] = risk_counts.get(rd.outcome, 0) + 1
        if rd.outcome in ("APPROVE", "RESIZE"):
            pending = (prop, rd.approved_qty)
        if record:
            decisions.append({"t": t, "ts": b.ts, "proposal": prop.model_dump(), "risk": rd.model_dump(),
                              "discussion": d.model_dump() if d else None})
    else:
        if L.position:
            close(end - 1, bars[end - 1].close, "segment_end")
            L.equity_curve[-1] = (end - 1, round(L.equity(bars[end - 1].close), 6))

    trades = [t.to_dict() for t in L.trades]
    return {
        "chromosome_id": c.id, "start": start, "end": end, "trades": trades,
        "equity": L.equity_curve, "exposure": L.exposure_curve,
        "scorecard": scorecard(trades, L.equity_curve, L.exposure_curve, L.initial, fail_event is not None),
        "risk_counts": risk_counts, "decisions": decisions, "rewards": rewards,
        "exit_range_atr": exit_range, "failed_risk": fail_event, "reconcile_error": L.reconcile(),
    }
