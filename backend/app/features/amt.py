"""Auction Market Theory: classical volume profile with 70% value area."""
from __future__ import annotations

from app.data.ingest import Bar


def volume_profile(bars: list[Bar], n_bins: int = 40, value_area: float = 0.70) -> dict:
    lo = min(b.low for b in bars)
    hi = max(b.high for b in bars)
    if hi <= lo:
        return {"poc": lo, "vah": hi, "val": lo, "location": "unknown"}
    width = (hi - lo) / n_bins
    vol = [0.0] * n_bins
    for b in bars:
        a = min(n_bins - 1, int((b.low - lo) / width))
        z = min(n_bins - 1, int((b.high - lo) / width))
        share = b.volume / (z - a + 1)
        for k in range(a, z + 1):
            vol[k] += share
    poc_i = max(range(n_bins), key=vol.__getitem__)
    target, acc, l, r = value_area * sum(vol), vol[poc_i], poc_i, poc_i
    while acc < target and (l > 0 or r < n_bins - 1):
        left = vol[l - 1] if l > 0 else -1.0
        right = vol[r + 1] if r < n_bins - 1 else -1.0
        if right >= left:
            r += 1
            acc += vol[r]
        else:
            l -= 1
            acc += vol[l]
    poc = lo + (poc_i + 0.5) * width
    val, vah = lo + l * width, lo + (r + 1) * width
    c = bars[-1].close
    loc = "above_va" if c > vah else "below_va" if c < val else "in_va"
    return {"poc": round(poc, 4), "vah": round(vah, 4), "val": round(val, 4), "location": loc}
