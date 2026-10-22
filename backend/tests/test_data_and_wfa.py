import pytest

from app.data import ingest, quality
from app.data.ingest import Bar
from app.data.point_in_time import LookAheadError, PITView
from app.features import pipeline
from app.validation.walk_forward import make_folds


def test_pit_view_refuses_future(cfg):
    bars, _ = ingest.load(cfg.data)
    v = PITView(bars, 100)
    assert v[100] is bars[100]
    with pytest.raises(LookAheadError):
        v[101]


@pytest.mark.parametrize("t", [80, 250, 777, 1500])
def test_features_identical_with_future_removed(cfg, t):
    """No look-ahead: evidence at t is the same whether or not later bars exist."""
    bars, _ = ingest.load(cfg.data)
    full = pipeline.build(PITView(bars, t))
    truncated = pipeline.build(PITView(bars[: t + 1], t))
    assert full.items == truncated.items


def test_insufficient_history_is_marked_not_guessed(cfg):
    bars, _ = ingest.load(cfg.data)
    eb = pipeline.build(PITView(bars, 10))
    assert not eb.valid and eb.items == {}


def test_quality_fails_closed():
    good = Bar(0, 0, 10, 11, 9, 10, 1, 0)
    bad_ohlc = Bar(1, 3600, 10, 9, 8, 10, 1, 0)  # high < close
    gap = Bar(2, 3600 * 5, 10, 11, 9, 10, 1, 0)
    errs = quality.check([good, bad_ohlc, gap], 60, 0)
    assert any("OHLC" in e for e in errs) and any("gap" in e for e in errs)


def test_checksum_changes_with_data(cfg):
    bars, snap = ingest.load(cfg.data)
    assert ingest.checksum(bars[:-1]) != snap.checksum


def test_folds_non_overlapping_and_embargoed(cfg):
    folds, holdout = make_folds(cfg.data.n_bars, 200, cfg.wfa)
    e = cfg.wfa.embargo_bars
    for f in folds:
        assert f.oos_start - f.is_end >= e
    for a, b in zip(folds, folds[1:]):
        assert b.oos_start >= a.oos_end
        assert b.is_start > a.is_start
    assert holdout[0] - folds[-1].oos_end >= e


def test_folds_reject_short_dataset(cfg):
    with pytest.raises(ValueError):
        make_folds(1000, 200, cfg.wfa)
