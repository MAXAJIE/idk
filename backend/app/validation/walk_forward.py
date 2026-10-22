"""Rolling, purged, non-overlapping walk-forward folds plus an untouched holdout."""
from __future__ import annotations

from dataclasses import dataclass, asdict

from app.config import WFAConfig


@dataclass(frozen=True)
class Fold:
    k: int
    is_start: int
    is_end: int  # exclusive
    oos_start: int
    oos_end: int  # exclusive

    def to_dict(self):
        return asdict(self)


def make_folds(n_bars: int, warmup: int, w: WFAConfig) -> tuple[list[Fold], tuple[int, int]]:
    folds = []
    for k in range(w.n_folds):
        s = warmup + k * w.oos_bars
        is_end = s + w.is_bars
        oos_start = is_end + w.embargo_bars
        folds.append(Fold(k, s, is_end, oos_start, oos_start + w.oos_bars))
    h0 = folds[-1].oos_end + w.embargo_bars
    holdout = (h0, h0 + w.holdout_bars)
    if holdout[1] > n_bars:
        raise ValueError(f"dataset too short: need {holdout[1]} bars, have {n_bars}")
    validate(folds, holdout, w.embargo_bars)
    return folds, holdout


def validate(folds: list[Fold], holdout: tuple[int, int], embargo: int) -> None:
    for f in folds:
        if not (f.is_start < f.is_end and f.is_end + embargo <= f.oos_start < f.oos_end):
            raise AssertionError(f"bad fold {f}")
    for a, b in zip(folds, folds[1:]):
        if b.oos_start < a.oos_end:
            raise AssertionError("OOS windows overlap")
    if holdout[0] < folds[-1].oos_end + embargo:
        raise AssertionError("holdout not embargoed")
