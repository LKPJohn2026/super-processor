"""Tests for preview windows clamped inside a trim."""

from __future__ import annotations

import pytest

from super_processor.preview import kept_range, preview_window
from super_processor.recipe import OpName
from super_processor.segments import TimelineSegment
from super_processor.treatments import Treatment, TreatmentError, TreatmentStep


def _segment(key: float) -> TimelineSegment:
    return TimelineSegment(0, 0.0, 60.0, "indoor", "noisy", key, 0)


def _trim(start: float, end: float) -> Treatment:
    return Treatment(
        "noisy.trimmed",
        "noisy",
        (TreatmentStep(OpName.TRIM, (("start_s", start), ("end_s", end))),),
    )


def test_window_is_three_seconds_around_the_key_frame() -> None:
    treatment = Treatment("noisy.light", "noisy", ())
    assert preview_window(_segment(10.0), treatment) == (8.5, 11.5)


def test_window_slides_to_stay_inside_the_segment() -> None:
    treatment = Treatment("noisy.light", "noisy", ())
    assert preview_window(_segment(0.0), treatment) == (0.0, 3.0)
    assert preview_window(_segment(60.0), treatment) == (57.0, 60.0)


def test_window_stays_inside_the_trim() -> None:
    treatment = _trim(20.0, 40.0)
    assert kept_range(_segment(10.0), treatment) == (20.0, 40.0)
    assert preview_window(_segment(10.0), treatment) == (20.0, 23.0)
    assert preview_window(_segment(30.0), treatment) == (28.5, 31.5)


def test_trim_shorter_than_five_seconds_is_refused() -> None:
    with pytest.raises(TreatmentError, match="5s"):
        preview_window(_segment(10.0), _trim(0.0, 4.0))
