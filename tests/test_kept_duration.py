"""Tests for summing kept duration into the size-cap floor."""

from __future__ import annotations

from super_processor.preview import kept_duration, size_cap_bitrate_kbps
from super_processor.recipe import OpName
from super_processor.segments import TimelineSegment
from super_processor.treatments import Treatment, TreatmentStep


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
    assert size_cap_bitrate_kbps(1.0, duration) > size_cap_bitrate_kbps(1.0, 20.0)
