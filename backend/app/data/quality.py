"""Data integrity checks. Fail closed: any error aborts the experiment."""
from __future__ import annotations

from app.data.ingest import Bar


def check(bars: list[Bar], timeframe_minutes: int, max_gap_bars: int) -> list[str]:
    errors: list[str] = []
    step = timeframe_minutes * 60
    for k, b in enumerate(bars):
        if min(b.open, b.high, b.low, b.close) <= 0:
            errors.append(f"{b.id}: non-positive price")
        if b.high < max(b.open, b.close) or b.low > min(b.open, b.close):
            errors.append(f"{b.id}: OHLC inconsistent")
        if b.volume < 0:
            errors.append(f"{b.id}: negative volume")
        if k:
            gap = b.ts - bars[k - 1].ts
            if gap <= 0:
                errors.append(f"{b.id}: non-monotonic timestamp")
            elif gap // step - 1 > max_gap_bars:
                errors.append(f"{b.id}: gap of {gap // step - 1} bars")
        if len(errors) >= 20:
            errors.append("... truncated")
            break
    return errors
