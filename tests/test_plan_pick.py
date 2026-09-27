"""Tests for choosing five timeline plans."""

from __future__ import annotations

import pytest

from super_processor.plans import assignment_score, choose_plans
from super_processor.segments import TimelineSegment
from super_processor.treatments import Treatment, TreatmentError


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


def _ids(plan: list[Treatment]) -> tuple[str, ...]:
    return tuple(item.treatment_id for item in plan)


def test_first_plan_scores_best_and_the_next_changes_the_most_groups() -> None:
    segments = [_segment(0, "noisy", 0), _segment(1, "low_contrast", 1)]
    plans = choose_plans(segments)
    assert len(plans) == 5
    scores = [assignment_score(segments, plan) for plan in plans]
    assert scores[0] == max(scores)
    assert scores[0] == 2.0
    first = _ids(plans[0])
    second = _ids(plans[1])
    assert sum(left != right for left, right in zip(first, second, strict=True)) == 2
    assert len({_ids(plan) for plan in plans}) == 5


def test_one_look_group_shares_a_single_treatment() -> None:
    segments = [_segment(0, "low_light", 0), _segment(1, "low_light", 0)]
    plans = choose_plans(segments)
    assert len(plans) == 3
    for plan in plans:
        assert plan[0].treatment_id == plan[1].treatment_id


def test_a_large_timeline_still_returns_distinct_plans() -> None:
    segments = [_segment(index, "low_light", index) for index in range(8)]
    plans = choose_plans(segments)
    assert len(plans) == 5
    assert len({_ids(plan) for plan in plans}) == 5
    first = _ids(plans[0])
    second = _ids(plans[1])
    assert all(left != right for left, right in zip(first, second, strict=True))


def test_a_look_group_with_two_problems_is_refused() -> None:
    segments = [_segment(0, "low_light", 0), _segment(1, "noisy", 0)]
    with pytest.raises(TreatmentError, match="more than one problem"):
        choose_plans(segments)
