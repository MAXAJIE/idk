"""Market data ingestion: deterministic synthetic generator and CSV loader.

CSV columns: ts (unix seconds), open, high, low, close, volume, funding_rate
"""
from __future__ import annotations

import csv
import hashlib
import math
import random
from dataclasses import dataclass, asdict

from app.config import DataConfig


@dataclass(frozen=True, slots=True)
class Bar:
    i: int
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    funding_rate: float

    @property
    def id(self) -> str:
        return f"bar:{self.ts}"


@dataclass(frozen=True)
class DatasetSnapshot:
    source: str
    exchange: str
    symbol: str
    timeframe_minutes: int
    start_ts: int
    end_ts: int
    n_bars: int
    checksum: str

    def to_dict(self) -> dict:
        return asdict(self)


def checksum(bars: list[Bar]) -> str:
    h = hashlib.sha256()
    for b in bars:
        h.update(f"{b.ts},{b.open:.8f},{b.high:.8f},{b.low:.8f},{b.close:.8f},{b.volume:.8f},{b.funding_rate:.10f}\n".encode())
    return h.hexdigest()[:16]


def synthetic_bars(cfg: DataConfig) -> list[Bar]:
    """Regime-switching random walk with occasional jumps. Research fixture only."""
    rng = random.Random(cfg.synthetic_seed)
    price, ts, step = 30000.0, 1_700_000_000, cfg.timeframe_minutes * 60
    drift, vol, regime_left = 0.0, 0.006, 0
    bars: list[Bar] = []
    for i in range(cfg.n_bars):
        if regime_left <= 0:
            drift = rng.choice([-0.0008, 0.0, 0.0, 0.0008])
            vol = rng.choice([0.004, 0.006, 0.01])
            regime_left = rng.randint(80, 300)
        regime_left -= 1
        ret = rng.gauss(drift, vol)
        if rng.random() < 0.004:
            ret += rng.choice([-1, 1]) * rng.uniform(0.03, 0.07)
        o = price
        c = max(1.0, o * math.exp(ret))
        h = max(o, c) * (1 + abs(rng.gauss(0, vol * 0.6)))
        l = min(o, c) * (1 - abs(rng.gauss(0, vol * 0.6)))
        v = 500 * (1 + abs(ret) / vol) * rng.uniform(0.6, 1.4)
        fr = 0.0001 + drift * 0.05 + rng.gauss(0, 0.00003)
        bars.append(Bar(i, ts + i * step, o, h, l, c, v, fr))
        price = c
    return bars


def csv_bars(path: str) -> list[Bar]:
    out: list[Bar] = []
    with open(path, newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            out.append(Bar(i, int(r["ts"]), float(r["open"]), float(r["high"]), float(r["low"]),
                           float(r["close"]), float(r["volume"]), float(r.get("funding_rate") or 0.0)))
    return out


def load(cfg: DataConfig) -> tuple[list[Bar], DatasetSnapshot]:
    if cfg.source == "binance":
        from app.data import binance
        bars = binance.fetch(cfg.symbol, cfg.timeframe_minutes, cfg.n_bars, cfg.backend)
        if bars:  # keep an immutable local copy of exactly what was fetched (reproducibility)
            import os
            d = os.environ.get("CGTRS_SNAPSHOT_DIR", "data/snapshots")
            os.makedirs(d, exist_ok=True)
            binance.write_csv(bars, os.path.join(d, f"{cfg.symbol}_{cfg.timeframe_minutes}m_{bars[0].ts}_{bars[-1].ts}.csv"))
    else:
        bars = synthetic_bars(cfg) if cfg.source == "synthetic" else csv_bars(cfg.csv_path or "")
    if not bars:
        raise ValueError("empty dataset")
    snap = DatasetSnapshot(cfg.source, cfg.exchange, cfg.symbol, cfg.timeframe_minutes,
                           bars[0].ts, bars[-1].ts, len(bars), checksum(bars))
    return bars, snap
