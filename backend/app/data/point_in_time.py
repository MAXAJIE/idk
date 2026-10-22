"""Point-in-time access: downstream code reads bars only through a PITView,
which refuses access beyond the decision index."""
from __future__ import annotations

from app.data.ingest import Bar


class LookAheadError(RuntimeError):
    pass


class PITView:
    __slots__ = ("_bars", "t")

    def __init__(self, bars: list[Bar], t: int):
        if not 0 <= t < len(bars):
            raise IndexError(t)
        self._bars = bars
        self.t = t

    def __getitem__(self, i: int) -> Bar:
        if i < 0 or i > self.t:
            raise LookAheadError(f"access bar {i} at decision index {self.t}")
        return self._bars[i]

    def window(self, n: int) -> list[Bar]:
        return self._bars[max(0, self.t - n + 1): self.t + 1]

    @property
    def now(self) -> Bar:
        return self._bars[self.t]
