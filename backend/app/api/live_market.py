"""Live Binance USDT-M market snapshot (public endpoints, no keys).

Read-only: it only GETs public market data and caches it briefly so many
browser tabs never multiply upstream requests.
"""
from __future__ import annotations

import re
import threading
import time

from app.data import binance

SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,20}$")
TTL_S = 2.0
_cache: dict[tuple, tuple[float, dict]] = {}
_lock = threading.Lock()


def valid_symbol(s: str) -> str:
    s = (s or "").upper()
    if not SYMBOL_RE.match(s):
        raise ValueError("bad symbol")
    return s


def _build(symbol: str, tf: int, limit: int, base: str) -> dict:
    prem = binance._get(base, "/fapi/v1/premiumIndex", {"symbol": symbol}, timeout=8)
    tick = binance._get(base, "/fapi/v1/ticker/24hr", {"symbol": symbol}, timeout=8)
    rows = binance._get(base, "/fapi/v1/klines", {"symbol": symbol, "interval": binance.interval(tf),
                                                  "limit": limit}, timeout=8)
    now_ms = int(time.time() * 1000)
    candles = [[int(r[0]) // 1000, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]),
                int(r[6]) < now_ms] for r in rows]
    return {"symbol": symbol, "tf_minutes": tf, "server_time": now_ms / 1000,
            "mark": float(prem["markPrice"]), "index": float(prem["indexPrice"]),
            "funding_rate": float(prem["lastFundingRate"]), "next_funding": int(prem["nextFundingTime"]) / 1000,
            "last": float(tick["lastPrice"]), "change_pct": float(tick["priceChangePercent"]),
            "high_24h": float(tick["highPrice"]), "low_24h": float(tick["lowPrice"]),
            "quote_volume_24h": float(tick["quoteVolume"]),
            "candles": candles}  # [ts, o, h, l, c, v, closed]


def snapshot(symbol: str, tf: int = 60, limit: int = 120, base: str = binance.MAINNET_DATA) -> dict:
    symbol = valid_symbol(symbol)
    binance.interval(tf)
    limit = max(10, min(int(limit), 500))
    key = (symbol, tf, limit, base)
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < TTL_S:
            return hit[1]
    data = _build(symbol, tf, limit, base)
    with _lock:
        _cache[key] = (time.time(), data)
    return data
