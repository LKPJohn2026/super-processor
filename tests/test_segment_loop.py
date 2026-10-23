"""Segment loop: dense still sampling and the wizard's segment list."""

from __future__ import annotations

import math

import pytest

from super_processor.segment_loop import END_MARGIN_S, MAX_STILLS, sample_times


def test_sample_times_one_a_second_with_both_ends() -> None:
    times = sample_times(10.0)
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(10.0 - END_MARGIN_S)
    assert len(times) == 11
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert max(gaps) <= 1.0 + 1e-6


def test_sample_times_caps_long_clips_with_even_stride() -> None:
    times = sample_times(30 * 60.0)
    assert len(times) == MAX_STILLS
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(30 * 60.0 - END_MARGIN_S)
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert max(gaps) - min(gaps) < 0.01


def test_sample_times_short_clip_keeps_first_and_last() -> None:
    assert sample_times(0.5) == [0.0, 0.4]
    assert sample_times(0.05) == [0.0]


def test_sample_times_are_sorted_unique_milliseconds() -> None:
    times = sample_times(7.3337, per_s=3.0)
    assert times == sorted(set(times))
    assert all(round(t, 3) == t for t in times)


@pytest.mark.parametrize("duration", [0.0, -1.0, math.nan, math.inf])
def test_sample_times_refuses_no_duration(duration: float) -> None:
    with pytest.raises(ValueError):
        sample_times(duration)


def test_sample_times_refuses_bad_policy() -> None:
    with pytest.raises(ValueError):
        sample_times(10.0, per_s=0.0)
    with pytest.raises(ValueError):
        sample_times(10.0, cap=1)
