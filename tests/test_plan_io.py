"""Tests for plans.json round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_processor.plans import build_plans, load_plans, write_plans
from super_processor.segments import TimelineSegment
from super_processor.treatments import TreatmentError


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


def test_plans_round_trip(tmp_path: Path) -> None:
    segments = [_segment(0, "noisy", 0), _segment(1, "low_contrast", 1)]
    plans = build_plans(segments)
    assert len(plans) == 5
    write_plans(tmp_path, plans)
    loaded = load_plans(tmp_path)
    assert loaded == plans
    assert loaded[0].treatments()[0].treatment_id == loaded[0].treatment_ids[0]


def test_missing_plans_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TreatmentError, match="not found"):
        load_plans(tmp_path)
