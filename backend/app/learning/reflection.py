"""Reflective Experience Memory & Negative Constraint Inhibitor.

Learns from closed trades by tracking market state signatures (entry mode,
trend regime, auction value area location) against trade outcomes. When a
specific signature exhibits high recurring loss rates, the reflection inhibitor
defensively vetoes or downsizes subsequent trade proposals matching that
regime.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class ReflectionPattern:
    signature: str
    entry_mode: str
    side: str
    trend: str
    location: str
    total_trades: int = 0
    losses: int = 0
    wins: int = 0
    net_pnl: float = 0.0
    consecutive_losses: int = 0
    last_updated_t: int = 0

    @property
    def loss_rate(self) -> float:
        return self.losses / self.total_trades if self.total_trades > 0 else 0.0


@dataclass
class InhibitionVerdict:
    inhibited: bool
    action: Literal["ALLOW", "RESIZE_PENALTY", "VETO"]
    size_mult: float = 1.0
    signature: str | None = None
    reason: str = "allow"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReflectionMemory:
    def __init__(self, veto_loss_rate: float = 0.75, penalty_loss_rate: float = 0.60, min_trades: int = 2):
        self.veto_loss_rate = veto_loss_rate
        self.penalty_loss_rate = penalty_loss_rate
        self.min_trades = min_trades
        self.patterns: dict[str, ReflectionPattern] = {}

    @staticmethod
    def build_signature(entry_mode: str, side: str, trend: str, location: str) -> str:
        return f"{entry_mode}:{side}:{trend}:{location}"

    @classmethod
    def signature_from_context(cls, entry_mode: str, side: str, eb: Any) -> str:
        if eb is None or not getattr(eb, "valid", False):
            return cls.build_signature(entry_mode, side, "unknown", "unknown")
        regime = eb.get("regime") or {}
        amt = eb.get("amt") or {}
        trend = regime.get("trend", "unknown")
        loc = amt.get("location", "unknown")
        return cls.build_signature(entry_mode, side, trend, loc)

    def record_outcome(self, signature: str, net_pnl: float, exit_reason: str, t: int = 0) -> ReflectionPattern:
        parts = signature.split(":")
        entry_mode = parts[0] if len(parts) > 0 else "unknown"
        side = parts[1] if len(parts) > 1 else "unknown"
        trend = parts[2] if len(parts) > 2 else "unknown"
        loc = parts[3] if len(parts) > 3 else "unknown"

        if signature not in self.patterns:
            self.patterns[signature] = ReflectionPattern(
                signature=signature, entry_mode=entry_mode, side=side, trend=trend, location=loc
            )

        pat = self.patterns[signature]
        pat.total_trades += 1
        pat.net_pnl = round(pat.net_pnl + net_pnl, 4)
        pat.last_updated_t = t

        if net_pnl < 0:
            pat.losses += 1
            pat.consecutive_losses += 1
        else:
            pat.wins += 1
            pat.consecutive_losses = 0

        return pat

    def check_inhibition(self, signature: str) -> InhibitionVerdict:
        pat = self.patterns.get(signature)
        if not pat or pat.total_trades < self.min_trades:
            return InhibitionVerdict(inhibited=False, action="ALLOW", size_mult=1.0, signature=signature)

        rate = pat.loss_rate
        if rate >= self.veto_loss_rate or pat.consecutive_losses >= 3:
            return InhibitionVerdict(
                inhibited=True,
                action="VETO",
                size_mult=0.0,
                signature=signature,
                reason=f"recurring loss pattern (loss_rate={rate:.2f}, count={pat.losses}/{pat.total_trades})",
            )
        if rate >= self.penalty_loss_rate or pat.consecutive_losses >= 2:
            return InhibitionVerdict(
                inhibited=True,
                action="RESIZE_PENALTY",
                size_mult=0.5,
                signature=signature,
                reason=f"moderately adverse regime (loss_rate={rate:.2f}, count={pat.losses}/{pat.total_trades})",
            )

        return InhibitionVerdict(inhibited=False, action="ALLOW", size_mult=1.0, signature=signature)

    def to_dict(self) -> dict[str, Any]:
        return {
            "veto_loss_rate": self.veto_loss_rate,
            "penalty_loss_rate": self.penalty_loss_rate,
            "min_trades": self.min_trades,
            "patterns": {k: asdict(v) for k, v in self.patterns.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReflectionMemory:
        mem = cls(
            veto_loss_rate=data.get("veto_loss_rate", 0.75),
            penalty_loss_rate=data.get("penalty_loss_rate", 0.60),
            min_trades=data.get("min_trades", 2),
        )
        for k, v in data.get("patterns", {}).items():
            mem.patterns[k] = ReflectionPattern(**v)
        return mem

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ReflectionMemory:
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_dict(json.loads(p.read_text()))
