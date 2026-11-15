"""Tests for scoring one treatment assignment."""

from __future__ import annotations

import pytest

from super_processor.plans import assignment_score
from super_processor.recipe import OpName
from super_processor.segments import TimelineSegment
from super_processor.treatments import (
    Treatment,
    TreatmentError,
    TreatmentStep,
    treatments_for,
)


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


def test_matching_treatments_score_one_point_each() -> None:
    segments = [_segment(0, "low_light", 0), _segment(1, "noisy", 1)]
    chosen = [
        treatments_for("low_light")[0],
        treatments_for("noisy")[0],
    ]
    assert assignment_score(segments, chosen) == 2.0


def test_heavy_denoise_beside_sharpen_is_penalized() -> None:
    segments = [_segment(0, "noisy", 0), _segment(1, "low_contrast", 1)]
    chosen = [
        treatments_for("noisy")[1],
        next(
            item
            for item in treatments_for("low_contrast")
            if "sharpen" in item.treatment_id
        ),
    ]
    assert chosen[0].steps[0].as_dict()["strength"] >= 0.5
    assert assignment_score(segments, chosen) == 1.5
    # The clash counts whichever side the sharpen is on.
    flipped = [_segment(0, "low_contrast", 0), _segment(1, "noisy", 1)]
    assert assignment_score(flipped, chosen[::-1]) == 1.5
    # Light denoise beside sharpen is fine.
    gentle = [treatments_for("noisy")[0], chosen[1]]
    assert assignment_score(segments, gentle) == 2.0


def test_cooling_one_warm_neighbor_and_not_the_other_is_penalized() -> None:
    segments = [_segment(0, "too_warm", 0), _segment(1, "too_warm", 1)]
    cooled = treatments_for("too_warm")[0]
    untouched = Treatment(
        treatment_id="too_warm.untouched",
        problem="too_warm",
        steps=(
            TreatmentStep(OpName.CONTRAST, (("contrast", 1.1), ("brightness", 0.0))),
        ),
    )
    assert assignment_score(segments, [cooled, untouched]) == 1.5
    # Cooling both warm neighbors, even by different amounts, is consistent.
    mild = treatments_for("too_warm")[1]
    assert assignment_score(segments, [cooled, mild]) == 2.0


def test_a_look_group_must_share_one_treatment() -> None:
    segments = [_segment(0, "low_light", 0), _segment(1, "low_light", 0)]
    options = treatments_for("low_light")
    with pytest.raises(TreatmentError, match="look group"):
        assignment_score(segments, [options[0], options[1]])


def test_a_treatment_for_another_problem_scores_nothing() -> None:
    segments = [_segment(0, "noisy", 0)]
    assert assignment_score(segments, [treatments_for("low_light")[0]]) == 0.0
    with pytest.raises(TreatmentError, match="does not match"):
        assignment_score(segments, [])
