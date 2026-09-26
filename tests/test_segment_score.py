"""Tests for the weighted change score between sample rows."""

from __future__ import annotations

import pytest

from super_processor.segments import SampleRow, change_score


def _row(**overrides: float) -> SampleRow:
    base = dict(
        time_s=0.0,
        luma_mean=128.0,
        luma_p05=40.0,
        luma_p95=200.0,
        clip_low=0.0,
        clip_high=0.0,
        rb_cast=0.0,
        variance=80.0,
        motion=0.0,
        subject_x=0.5,
    )
    base.update(overrides)
    return SampleRow(**base)


def test_identical_rows_score_zero() -> None:
    row = _row()
    assert change_score(row, _row(time_s=1.0)) == 0.0


def test_known_pair_has_a_fixed_score() -> None:
    before = _row()
    after = _row(
        time_s=1.0,
        luma_mean=128.0 + 25.5,
        rb_cast=0.5,
        variance=280.0,
        motion=25.5,
        subject_x=0.5 + 0.5,
    )
    # 25.5/255 + 1.2*0.5 + 0.8*(200/400) + 25.5/255 + 0.6*0.5
    expected = 0.1 + 0.6 + 0.4 + 0.1 + 0.3
    assert change_score(before, after) == pytest.approx(expected)
