"""Execution cost model. Every market fill pays half-spread + size/volatility
sensitive slippage + taker fee. Maker fills are never assumed."""
from __future__ import annotations

from dataclasses import dataclass

from app.config import CostConfig
from app.data.ingest import Bar


@dataclass(frozen=True)
class FillCost:
    price: float
    spread_cost: float
    slippage_cost: float
    fee: float


def slippage_bps(cfg: CostConfig, bar: Bar, notional: float) -> float:
    participation_pct = notional / max(bar.volume * bar.close, 1e-9) * 100
    range_bps = (bar.high - bar.low) / bar.close * 1e4
    return cfg.slippage_base_bps + cfg.slippage_impact_coef * participation_pct + cfg.slippage_vol_coef * range_bps


def market_fill(cfg: CostConfig, bar: Bar, ref_price: float, qty: float, side: int, slip_mult: float = 1.0) -> FillCost:
    """side=+1 buy, -1 sell; qty > 0."""
    spread = ref_price * cfg.half_spread_bps / 1e4
    slip = ref_price * slippage_bps(cfg, bar, ref_price * qty) * slip_mult / 1e4
    px = ref_price + side * (spread + slip)
    return FillCost(px, spread * qty, slip * qty, px * qty * cfg.taker_fee_bps / 1e4)


def estimate_round_trip(cfg: CostConfig, bar: Bar, price: float, qty: float) -> float:
    one = market_fill(cfg, bar, price, qty, 1)
    return 2 * (one.spread_cost + one.slippage_cost + one.fee)
