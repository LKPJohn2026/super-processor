"""Tests for summing kept duration into the size-cap floor."""

from __future__ import annotations

import pytest

from super_processor.preview import kept_duration, size_cap_bitrate_kbps
from super_processor.recipe import OpName
from super_processor.segments import TimelineSegment
from super_processor.treatments import Treatment, TreatmentError, TreatmentStep


def _segment(index: int) -> TimelineSegment:
    return TimelineSegment(
        index,
        float(index * 10),
        float((index + 1) * 10),
        "indoor",
        "noisy",
        float(index * 10 + 5),
        index,
    )


def test_trimmed_lengths_add_up_and_raise_the_size_cap_bitrate() -> None:
    full = Treatment("noisy.full", "noisy", ())
    trimmed = Treatment(
        "noisy.trimmed",
        "noisy",
        (TreatmentStep(OpName.TRIM, (("start_s", 10.0), ("end_s", 15.0))),),
    )
    segments = [_segment(0), _segment(1)]
    duration = kept_duration(segments, [full, trimmed])
    assert duration == 15.0
    # 1 MiB over the 15 s kept, not the 20 s the segments span.
    assert size_cap_bitrate_kbps(1.0, duration) == pytest.approx(8 * 1024 * 1024 / 15e3)
    assert size_cap_bitrate_kbps(1.0, duration) > size_cap_bitrate_kbps(1.0, 20.0)


def test_a_treatment_per_segment_is_required() -> None:
    full = Treatment("noisy.full", "noisy", ())
    with pytest.raises(TreatmentError, match="does not match the segments"):
        kept_duration([_segment(0), _segment(1)], [full])


def test_nothing_kept_has_no_size_cap_bitrate() -> None:
    with pytest.raises(TreatmentError, match="must be positive"):
        size_cap_bitrate_kbps(1.0, 0.0)
