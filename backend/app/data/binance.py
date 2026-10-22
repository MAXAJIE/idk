"""Binance USDT-M futures market data (public endpoints, no credentials).

Two backends:
  * "rest"  - direct HTTPS to fapi.binance.com (stdlib only, default)
  * "ccxt"  - optional; `pip install ccxt` (same data, different client)

Only CLOSED bars are returned. Funding rate per bar = most recent funding
settlement with fundingTime <= bar open time (point-in-time, forward-filled;
never back-filled from a future settlement).
"""
from __future__ import annotations

import bisect
import csv
import json
import time
import urllib.parse
import urllib.request

from app.data.ingest import Bar

MAINNET_DATA = "https://fapi.binance.com"
INTERVALS = {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 120: "2h", 240: "4h",
             360: "6h", 480: "8h", 720: "12h", 1440: "1d"}


def interval(minutes: int) -> str:
    if minutes not in INTERVALS:
        raise ValueError(f"unsupported timeframe {minutes}m; use one of {sorted(INTERVALS)}")
    return INTERVALS[minutes]


def _get(base: str, path: str, params: dict, timeout: float = 20.0):
    url = f"{base}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "cgtrs/2"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (fixed https hosts)
        return json.loads(r.read())


def parse_klines(rows: list[list], tf_minutes: int, now_ms: int) -> list[tuple]:
    """-> [(open_ts_s, o, h, l, c, v)], dropping any bar not yet closed."""
    out = []
    for r in rows:
        open_ms, close_ms = int(r[0]), int(r[6])
        if close_ms >= now_ms:
            continue
        out.append((open_ms // 1000, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])))
    return out


def attach_funding(klines: list[tuple], funding: list[tuple[int, float]]) -> list[Bar]:
    """funding: [(funding_time_s, rate)] sorted. Point-in-time forward fill."""
    times = [f[0] for f in funding]
    bars = []
    for i, (ts, o, h, l, c, v) in enumerate(klines):
        k = bisect.bisect_right(times, ts) - 1
        bars.append(Bar(i, ts, o, h, l, c, v, funding[k][1] if k >= 0 else 0.0))
    return bars


def _rest_klines(symbol: str, tf: int, n: int, base: str) -> list[list]:
    step_ms = tf * 60_000
    end = int(time.time() * 1000)
    start = end - (n + 2) * step_ms
    rows: list[list] = []
    while start < end:
        chunk = _get(base, "/fapi/v1/klines", {"symbol": symbol, "interval": interval(tf),
                                               "startTime": start, "limit": 1500})
        if not chunk:
            break
        rows.extend(chunk)
        start = int(chunk[-1][0]) + step_ms
        if len(chunk) < 1500:
            break
    seen, uniq = set(), []
    for r in rows:
        if r[0] not in seen:
            seen.add(r[0])
            uniq.append(r)
    return uniq


def _rest_funding(symbol: str, start_ms: int, base: str) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    s = start_ms - 8 * 3600_000  # include the settlement preceding the first bar
    while True:
        chunk = _get(base, "/fapi/v1/fundingRate", {"symbol": symbol, "startTime": s, "limit": 1000})
        if not chunk:
            break
        out.extend((int(x["fundingTime"]) // 1000, float(x["fundingRate"])) for x in chunk)
        if len(chunk) < 1000:
            break
        s = int(chunk[-1]["fundingTime"]) + 1
    return sorted(set(out))


def _ccxt_fetch(symbol: str, tf: int, n: int) -> tuple[list[list], list[tuple[int, float]]]:
    import ccxt  # optional dependency

    ex = ccxt.binanceusdm({"enableRateLimit": True})
    ex.load_markets()
    m = next(v for v in ex.markets.values() if v["id"] == symbol and v.get("linear"))
    step_ms = tf * 60_000
    since = int(time.time() * 1000) - (n + 2) * step_ms
    rows: list[list] = []
    while True:
        chunk = ex.fetch_ohlcv(m["symbol"], interval(tf), since=since, limit=1500)
        if not chunk:
            break
        # ccxt omits close time; synthesise it so parse_klines can drop the open bar
        rows.extend([c[0], c[1], c[2], c[3], c[4], c[5], c[0] + step_ms - 1] for c in chunk)
        since = chunk[-1][0] + step_ms
        if len(chunk) < 1500:
            break
    fr = ex.fetch_funding_rate_history(m["symbol"], since=rows[0][0] - 8 * 3600_000 if rows else None, limit=1000)
    return rows, sorted({(int(x["timestamp"]) // 1000, float(x["fundingRate"])) for x in fr})


def fetch(symbol: str, tf_minutes: int, n_bars: int, backend: str = "rest", base: str = MAINNET_DATA) -> list[Bar]:
    now_ms = int(time.time() * 1000)
    if backend == "ccxt":
        rows, funding = _ccxt_fetch(symbol, tf_minutes, n_bars)
    else:
        rows = _rest_klines(symbol, tf_minutes, n_bars, base)
        funding = _rest_funding(symbol, int(rows[0][0]) if rows else now_ms, base)
    kl = parse_klines(rows, tf_minutes, now_ms)[-n_bars:]
    return attach_funding(kl, funding)


def write_csv(bars: list[Bar], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume", "funding_rate"])
        for b in bars:
            w.writerow([b.ts, b.open, b.high, b.low, b.close, b.volume, b.funding_rate])
