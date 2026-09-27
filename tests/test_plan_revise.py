"""Tests for rebuilding plans around one named segment."""

from __future__ import annotations

import pytest

from super_processor.plans import revise_plans
from super_processor.segments import TimelineSegment
from super_processor.treatments import TreatmentError, treatments_for


def _segment(index: int, problem: str, group: int) -> TimelineSegment:
    return TimelineSegment(
        index=index,
        start_s=float(index * 10),
        end_s=float((index + 1) * 10),
        context="indoor",
        problem=problem,
        keyframe_s=float(index * 10),
        look_group=group,
    )


def test_only_the_named_segment_changes() -> None:
    segments = [_segment(0, "noisy", 0), _segment(1, "low_contrast", 1)]
    base = [treatments_for("noisy")[0], treatments_for("low_contrast")[0]]
    revised = revise_plans(segments, base, 1, "not-sharp")
    assert len(revised) == 1
    assert revised[0][0].treatment_id == base[0].treatment_id
    assert revised[0][1].treatment_id == "low_contrast.contrast_sharpen"


def test_same_look_group_neighbor_keeps_the_base_treatment() -> None:
    segments = [_segment(0, "noisy", 0), _segment(1, "noisy", 0)]
    base = [treatments_for("noisy")[1], treatments_for("noisy")[1]]
    revised = revise_plans(segments, base, 1, "too-much-denoise")
    assert len(revised) >= 2
    for plan in revised:
        assert plan[0].treatment_id == base[0].treatment_id
        assert plan[1].treatment_id != base[1].treatment_id


def test_segment_index_outside_the_timeline_is_refused() -> None:
    segments = [_segment(0, "noisy", 0)]
    base = [treatments_for("noisy")[0]]
    with pytest.raises(TreatmentError, match="outside"):
        revise_plans(segments, base, 3, "not-sharp")
