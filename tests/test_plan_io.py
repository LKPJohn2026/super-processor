"""Tests for plans.json round-trip."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from super_processor.plans import (
    PLANS_SCHEMA_VERSION,
    TimelinePlan,
    assignment_score,
    build_plans,
    load_plans,
    write_plans,
)
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
    assert [plan.index for plan in loaded] == [0, 1, 2, 3, 4]
    # The stored score is the score of the stored treatments.
    for plan in loaded:
        assert plan.score == assignment_score(segments, plan.treatments())
    data = json.loads((tmp_path / "plans.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == PLANS_SCHEMA_VERSION
    # The write goes through a temporary file, which must not be left behind.
    assert not (tmp_path / "plans.tmp").exists()


def test_missing_plans_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TreatmentError, match="not found"):
        load_plans(tmp_path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "root must be an object"),
        ({"schema_version": 1}, "plans list is missing"),
        ({"plans": ["noisy.light"]}, "each plan must be an object"),
        (
            {"plans": [{"index": 0, "score": 1.0, "treatment_ids": "noisy.light"}]},
            "list of strings",
        ),
        (
            {"plans": [{"index": 0, "score": 1.0, "treatment_ids": [3]}]},
            "list of strings",
        ),
        (
            {"plans": [{"index": True, "score": 1.0, "treatment_ids": []}]},
            "index must be an integer",
        ),
        (
            {"plans": [{"index": 0, "score": "2", "treatment_ids": []}]},
            "score must be numeric",
        ),
        (
            {"plans": [{"index": 0, "score": False, "treatment_ids": []}]},
            "score must be numeric",
        ),
    ],
)
def test_a_damaged_plans_file_is_refused(
    tmp_path: Path, payload: Any, message: str
) -> None:
    (tmp_path / "plans.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TreatmentError, match=message):
        load_plans(tmp_path)


def test_an_integer_score_loads_as_a_float() -> None:
    plan = TimelinePlan.from_dict(
        {"index": 2, "score": 1, "treatment_ids": ["noisy.light"]}
    )
    assert plan == TimelinePlan(index=2, score=1.0, treatment_ids=("noisy.light",))
    assert isinstance(plan.score, float)
