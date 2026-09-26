"""Scoring for one treatment assigned to every segment."""

from __future__ import annotations

from .recipe import OpName
from .segments import TimelineSegment
from .treatments import STRONG_DENOISE, Treatment, TreatmentError


def _has_op(treatment: Treatment, op: OpName) -> bool:
    return any(step.op is op for step in treatment.steps)


def _denoise_strength(treatment: Treatment) -> float:
    for step in treatment.steps:
        if step.op is OpName.DENOISE:
            return float(step.as_dict().get("strength", 0.0))
    return 0.0


def _cools_warmth(treatment: Treatment) -> bool:
    for step in treatment.steps:
        if step.op is OpName.WHITE_BALANCE:
            return float(step.as_dict().get("temperature", 6500.0)) > 6500.0
    return False


def _neighbors_clash(left: Treatment, right: Treatment) -> bool:
    """Heavy denoise beside sharpen, or a cool grade beside an untouched warm one."""
    left_denoise = _denoise_strength(left) >= STRONG_DENOISE
    right_denoise = _denoise_strength(right) >= STRONG_DENOISE
    if (left_denoise and _has_op(right, OpName.SHARPEN)) or (
        right_denoise and _has_op(left, OpName.SHARPEN)
    ):
        return True
    left_warm = left.problem == "too_warm"
    right_warm = right.problem == "too_warm"
    if left_warm and right_warm:
        return _cools_warmth(left) != _cools_warmth(right)
    return False


def assignment_score(segments: list[TimelineSegment], chosen: list[Treatment]) -> float:
    """Score one treatment per segment.

    Each segment whose treatment names its problem adds one point. A clashing
    neighbor pair subtracts half a point. Segments in one look group must
    share a treatment.
    """
    if len(segments) != len(chosen):
        raise TreatmentError("assignment length does not match the segments")
    by_group: dict[int, str] = {}
    for segment, treatment in zip(segments, chosen, strict=True):
        previous = by_group.get(segment.look_group)
        if previous is not None and previous != treatment.treatment_id:
            raise TreatmentError(
                f"look group {segment.look_group} must share one treatment"
            )
        by_group[segment.look_group] = treatment.treatment_id
    score = 0.0
    for segment, treatment in zip(segments, chosen, strict=True):
        if treatment.problem == segment.problem:
            score += 1.0
    for index in range(len(chosen) - 1):
        if _neighbors_clash(chosen[index], chosen[index + 1]):
            score -= 0.5
    return score
