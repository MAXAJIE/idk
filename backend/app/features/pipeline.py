"""Point-in-time evidence bundle for decision index t. Each item has a stable
ID; agents may only cite IDs present in the bundle."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.data.point_in_time import PITView
from app.features import amt, smc, snr, trend

LOOKBACK = 200
MIN_BARS = 60


@dataclass
class EvidenceBundle:
    t: int
    ts: int
    valid: bool
    items: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def get(self, kind: str) -> dict | None:
        item = self.items.get(f"ev:{kind}:{self.ts}")
        return item["value"] if item else None

    def id(self, kind: str) -> str:
        return f"ev:{kind}:{self.ts}"


def build(view: PITView) -> EvidenceBundle:
    win = view.window(LOOKBACK)
    now = view.now
    eb = EvidenceBundle(view.t, now.ts, valid=len(win) >= MIN_BARS)
    if not eb.valid:
        eb.warnings.append(f"insufficient history: {len(win)} < {MIN_BARS}")
        return eb
    a = trend.atr(win)
    closes = [b.close for b in win]
    values = {
        "price": {"close": now.close, "high": now.high, "low": now.low, "volume": now.volume},
        "atr": {"atr14": round(a, 4), "range_atr": round((now.high - now.low) / a, 4) if a else 0.0},
        "rsi": {"rsi14": round(trend.rsi(closes), 2)},
        "regime": trend.regime(win),
        "amt": amt.volume_profile(win[-120:]),
        "smc": smc.structure(win),
        "fvg": smc.fair_value_gaps(win[-60:]),
        "snr": snr.levels(win, a),
        "funding": {"rate": now.funding_rate},
    }
    for kind, v in values.items():
        eb.items[eb.id(kind)] = {"kind": kind, "value": v}
    return eb
