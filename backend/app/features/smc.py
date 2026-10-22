"""Smart Money Concepts: 3-bar fractal swings, BOS/CHoCH, 3-candle FVG, sweeps.

A fractal at bar k needs bar k+1, so it is only confirmed once k+1 is in the
window. Using unconfirmed swings would be look-ahead.
"""
from __future__ import annotations

from app.data.ingest import Bar


def swings(bars: list[Bar]) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    highs, lows = [], []
    for k in range(1, len(bars) - 1):
        if bars[k].high > bars[k - 1].high and bars[k].high > bars[k + 1].high:
            highs.append((k, bars[k].high))
        if bars[k].low < bars[k - 1].low and bars[k].low < bars[k + 1].low:
            lows.append((k, bars[k].low))
    return highs, lows


def structure(bars: list[Bar]) -> dict:
    highs, lows = swings(bars)
    if len(highs) < 2 or len(lows) < 2:
        return {"bias": "unknown", "event": "none", "last_swing_high": None, "last_swing_low": None, "sweep": "none"}
    hh = highs[-1][1] > highs[-2][1]
    hl = lows[-1][1] > lows[-2][1]
    prior = "bull" if hh and hl else "bear" if not hh and not hl else "mixed"
    last = bars[-1]
    c, sh, sl = last.close, highs[-1][1], lows[-1][1]
    event, bias = "none", prior
    if c > sh:
        event, bias = ("bos_up" if prior == "bull" else "choch_up"), "bull"
    elif c < sl:
        event, bias = ("bos_down" if prior == "bear" else "choch_down"), "bear"
    sweep = "none"
    if last.high > sh and c < sh:
        sweep = "buy_side_swept"
    elif last.low < sl and c > sl:
        sweep = "sell_side_swept"
    return {"bias": bias, "event": event, "last_swing_high": sh, "last_swing_low": sl, "sweep": sweep}


def fair_value_gaps(bars: list[Bar]) -> dict:
    """Most recent unmitigated bullish/bearish 3-candle gap."""
    bull = bear = None
    for k in range(2, len(bars)):
        if bars[k].low > bars[k - 2].high:
            bull = (k, bars[k - 2].high, bars[k].low)
        if bars[k].high < bars[k - 2].low:
            bear = (k, bars[k].high, bars[k - 2].low)

    def mitigated(g, bullish):
        if g is None:
            return True
        k, lo, hi = g
        later = bars[k + 1:]
        return any(b.low <= lo for b in later) if bullish else any(b.high >= hi for b in later)

    return {
        "bull_fvg": None if mitigated(bull, True) else {"lo": bull[1], "hi": bull[2]},
        "bear_fvg": None if mitigated(bear, False) else {"lo": bear[1], "hi": bear[2]},
    }
