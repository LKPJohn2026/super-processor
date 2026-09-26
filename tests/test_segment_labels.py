"""Tests for photographic labels and the key second."""

from __future__ import annotations

import pytest

from super_processor.segments import (
    SampleRow,
    SegmentError,
    context_label,
    keyframe_time,
    primary_problem,
)


def _row(**overrides: float) -> SampleRow:
    base: dict[str, float] = {
        "time_s": 0.0,
        "luma_mean": 128.0,
        "luma_p05": 40.0,
        "luma_p95": 200.0,
        "clip_low": 0.0,
        "clip_high": 0.0,
        "rb_cast": 0.0,
        "variance": 40.0,
        "motion": 0.0,
        "subject_x": 0.5,
        "upper_luma": 128.0,
    }
    base.update(overrides)
    return SampleRow(**base)


def test_low_light_is_the_primary_problem() -> None:
    rows = [_row(time_s=0.0, luma_mean=30.0), _row(time_s=1.0, luma_mean=40.0)]
    assert primary_problem(rows) == "low_light"
    assert context_label(rows) == "indoor"


def test_low_contrast_when_the_span_is_narrow() -> None:
    rows = [_row(luma_mean=120.0, luma_p05=110.0, luma_p95=130.0, upper_luma=120.0)]
    assert primary_problem(rows) == "low_contrast"
    assert context_label(rows) == "mixed"


def test_silhouette_when_the_subject_is_dark_against_a_bright_top() -> None:
    rows = [
        _row(
            luma_mean=110.0,
            luma_p05=10.0,
            luma_p95=220.0,
            upper_luma=220.0,
        )
    ]
    assert primary_problem(rows) == "silhouette"


def test_too_warm_noisy_shaky_and_off_center() -> None:
    assert primary_problem([_row(rb_cast=0.8, luma_mean=140.0, upper_luma=180.0)]) == (
        "too_warm"
    )
    assert primary_problem([_row(variance=500.0, luma_mean=140.0)]) == "noisy"
    assert primary_problem([_row(motion=40.0, luma_mean=140.0)]) == "shaky"
    assert primary_problem([_row(subject_x=0.95, luma_mean=140.0)]) == "off_center"


def test_outdoor_context_from_a_bright_upper_frame() -> None:
    rows = [_row(luma_mean=140.0, upper_luma=200.0)]
    assert context_label(rows) == "outdoor"


def test_keyframe_is_the_strongest_primary_second() -> None:
    rows = [
        _row(time_s=2.0, luma_mean=50.0),
        _row(time_s=8.0, luma_mean=20.0),
        _row(time_s=12.0, luma_mean=70.0),
    ]
    assert primary_problem(rows) == "low_light"
    assert keyframe_time(rows) == pytest.approx(8.0)


def test_empty_samples_are_refused() -> None:
    with pytest.raises(SegmentError, match="empty"):
        primary_problem([])
    with pytest.raises(SegmentError, match="empty"):
        context_label([])
    with pytest.raises(SegmentError, match="empty"):
        keyframe_time([])
