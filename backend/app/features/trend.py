"""Trend / momentum / volatility features (default algorithms, features-1)."""
from __future__ import annotations

from app.data.ingest import Bar


def ema(values: list[float], period: int) -> list[float]:
    """Causal EMA: out[k] depends only on values[:k+1]."""
    if not values:
        return []
    a = 2.0 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(out[-1] + a * (v - out[-1]))
    return out


def true_ranges(bars: list[Bar]) -> list[float]:
    out = []
    for k, b in enumerate(bars):
        pc = bars[k - 1].close if k else b.open
        out.append(max(b.high - b.low, abs(b.high - pc), abs(b.low - pc)))
    return out


def atr(bars: list[Bar], period: int = 14) -> float:
    tr = true_ranges(bars)[-period:]
    return sum(tr) / len(tr) if tr else 0.0


def rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) <= period:
        return 50.0
    gains = losses = 0.0
    for a, b in zip(closes[-period - 1:-1], closes[-period:]):
        d = b - a
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    if losses == 0:
        return 100.0
    return 100 - 100 / (1 + gains / losses)


def regime(bars: list[Bar], slope_lookback: int = 20) -> dict:
    closes = [b.close for b in bars]
    e = ema(closes, 50)
    a = atr(bars)
    if len(e) <= slope_lookback or a <= 0:
        return {"trend": "unknown", "slope_atr": 0.0, "vol_pct": 0.0}
    slope = (e[-1] - e[-1 - slope_lookback]) / a
    trend = "up" if slope > 1.0 else "down" if slope < -1.0 else "range"
    trs = true_ranges(bars)
    recent = sum(trs[-14:]) / 14
    rank = sum(1 for k in range(14, len(trs)) if sum(trs[k - 14:k]) / 14 <= recent) / max(1, len(trs) - 14)
    return {"trend": trend, "slope_atr": round(slope, 4), "vol_pct": round(rank, 4)}
