"""Scoring and selection for timeline treatment plans."""

from __future__ import annotations

import itertools

from .recipe import OpName
from .segments import TimelineSegment
from .treatments import (
    STRONG_DENOISE,
    Treatment,
    TreatmentError,
    treatment_violation,
    treatments_for,
)

PLAN_COUNT = 5
_ENUM_LIMIT = 4096


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


def _look_groups(segments: list[TimelineSegment]) -> list[tuple[int, str]]:
    """Return look groups in timeline order, each with its shared problem."""
    groups: list[tuple[int, str]] = []
    seen: dict[int, str] = {}
    for segment in segments:
        problem = seen.get(segment.look_group)
        if problem is None:
            seen[segment.look_group] = segment.problem
            groups.append((segment.look_group, segment.problem))
        elif problem != segment.problem:
            raise TreatmentError(
                f"look group {segment.look_group} has more than one problem"
            )
    return groups


def _expand(
    segments: list[TimelineSegment],
    groups: list[tuple[int, str]],
    picks: tuple[Treatment, ...],
) -> list[Treatment]:
    by_group = {
        group_id: treatment
        for (group_id, _problem), treatment in zip(groups, picks, strict=True)
    }
    return [by_group[segment.look_group] for segment in segments]


def _hamming(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def _select_plans(
    scored: list[tuple[float, tuple[int, ...], tuple[Treatment, ...]]],
    limit: int,
) -> list[tuple[Treatment, ...]]:
    """Keep the best score, then the remaining plans that differ in the most groups."""
    unique: dict[tuple[int, ...], tuple[float, tuple[Treatment, ...]]] = {}
    for score, indexes, picks in scored:
        current = unique.get(indexes)
        if current is None or score > current[0]:
            unique[indexes] = (score, picks)
    remaining = [(score, indexes, picks) for indexes, (score, picks) in unique.items()]
    if not remaining:
        return []
    remaining.sort(key=lambda item: (-item[0], item[1]))
    chosen = [remaining.pop(0)]
    while remaining and len(chosen) < limit:

        def rank(
            item: tuple[float, tuple[int, ...], tuple[Treatment, ...]],
        ) -> tuple[int, float, tuple[int, ...]]:
            distance = min(_hamming(item[1], kept[1]) for kept in chosen)
            return (-distance, -item[0], item[1])

        best = min(remaining, key=rank)
        remaining.remove(best)
        chosen.append(best)
    return [picks for _score, _indexes, picks in chosen]


def _picks_for(
    options: list[tuple[Treatment, ...]],
    indexes: list[int],
) -> tuple[Treatment, ...]:
    return tuple(
        group_options[index]
        for group_options, index in zip(options, indexes, strict=True)
    )


def _score_indexes(
    segments: list[TimelineSegment],
    groups: list[tuple[int, str]],
    options: list[tuple[Treatment, ...]],
    indexes: list[int],
) -> tuple[float, tuple[int, ...], tuple[Treatment, ...]]:
    frozen = tuple(indexes)
    picks = _picks_for(options, indexes)
    return (
        assignment_score(segments, _expand(segments, groups, picks)),
        frozen,
        picks,
    )


def _bounded_candidates(
    segments: list[TimelineSegment],
    groups: list[tuple[int, str]],
    options: list[tuple[Treatment, ...]],
) -> list[tuple[float, tuple[int, ...], tuple[Treatment, ...]]]:
    """Climb to a strong assignment, then vary one group or one catalog slot."""
    indexes = [0] * len(options)
    improved = True
    while improved:
        improved = False
        for group_index, group_options in enumerate(options):
            current = _score_indexes(segments, groups, options, indexes)
            best_index = indexes[group_index]
            best_score = current[0]
            for option_index in range(len(group_options)):
                trial = list(indexes)
                trial[group_index] = option_index
                score = _score_indexes(segments, groups, options, trial)[0]
                if score > best_score:
                    best_score = score
                    best_index = option_index
            if best_index != indexes[group_index]:
                indexes[group_index] = best_index
                improved = True
    candidates = [_score_indexes(segments, groups, options, indexes)]
    for group_index, group_options in enumerate(options):
        for option_index in range(len(group_options)):
            trial = list(indexes)
            trial[group_index] = option_index
            candidates.append(_score_indexes(segments, groups, options, trial))
    widest = max(len(group_options) for group_options in options)
    for slot in range(widest):
        trial = [min(slot, len(group_options) - 1) for group_options in options]
        candidates.append(_score_indexes(segments, groups, options, trial))
    return candidates


def choose_plans(
    segments: list[TimelineSegment],
    *,
    limit: int = PLAN_COUNT,
) -> list[list[Treatment]]:
    """Assign one treatment per look group and keep up to five plans.

    The first plan is the highest score. Each later plan is the highest-scoring
    remaining assignment that changes the most look groups. Neighbors that share
    a look group share a treatment.
    """
    if limit < 1:
        raise TreatmentError("plan limit must be positive")
    if not segments:
        return []
    groups = _look_groups(segments)
    options: list[tuple[Treatment, ...]] = []
    for _group_id, problem in groups:
        legal = tuple(
            treatment
            for treatment in treatments_for(problem)
            if treatment_violation(treatment) is None
        )
        if not legal:
            raise TreatmentError(f"no legal treatments for {problem}")
        options.append(legal)
    product = 1
    for group_options in options:
        product *= len(group_options)
    if product <= _ENUM_LIMIT:
        scored = [
            _score_indexes(segments, groups, options, list(indexes))
            for indexes in itertools.product(
                *(range(len(group_options)) for group_options in options)
            )
        ]
    else:
        scored = _bounded_candidates(segments, groups, options)
    return [_expand(segments, groups, picks) for picks in _select_plans(scored, limit)]
