"""Support/resistance: horizontal clusters of confirmed swing points."""
from __future__ import annotations

from app.data.ingest import Bar
from app.features.smc import swings


def levels(bars: list[Bar], atr_value: float, tol_atr: float = 0.5) -> dict:
    highs, lows = swings(bars)
    pts = sorted(p for _, p in highs + lows)
    clusters: list[list[float]] = []
    tol = max(atr_value * tol_atr, 1e-9)
    for p in pts:
        if clusters and p - clusters[-1][-1] <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    lv = [{"price": round(sum(c) / len(c), 4), "touches": len(c)} for c in clusters]
    close = bars[-1].close
    sup = [x for x in lv if x["price"] < close]
    res = [x for x in lv if x["price"] > close]
    return {"support": max(sup, key=lambda x: x["price"]) if sup else None,
            "resistance": min(res, key=lambda x: x["price"]) if res else None, "n_levels": len(lv)}
