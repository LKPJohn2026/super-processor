"""Scoring and selection for timeline treatment plans."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

from .recipe import OpName
from .segments import TimelineSegment
from .treatments import (
    STRONG_DENOISE,
    Treatment,
    TreatmentError,
    filter_treatments,
    treatment_by_id,
    treatment_violation,
    treatments_for,
)

PLANS_FILE_NAME = "plans.json"
PLANS_SCHEMA_VERSION = 1

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


@dataclass(frozen=True, slots=True)
class TimelinePlan:
    """One saved assignment: a score and one treatment id per segment."""

    index: int
    score: float
    treatment_ids: tuple[str, ...]

    def treatments(self) -> list[Treatment]:
        return [treatment_by_id(treatment_id) for treatment_id in self.treatment_ids]

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "score": self.score,
            "treatment_ids": list(self.treatment_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TimelinePlan:
        raw_ids = data.get("treatment_ids")
        if not isinstance(raw_ids, list) or not all(
            isinstance(item, str) for item in raw_ids
        ):
            raise TreatmentError("treatment_ids must be a list of strings")
        index = data.get("index")
        score = data.get("score")
        if isinstance(index, bool) or not isinstance(index, int):
            raise TreatmentError("plan index must be an integer")
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise TreatmentError("plan score must be numeric")
        return cls(index=index, score=float(score), treatment_ids=tuple(raw_ids))


def revise_plans(
    segments: list[TimelineSegment],
    base: list[Treatment],
    segment_index: int,
    fault: str,
) -> list[list[Treatment]]:
    """Rebuild plans that differ only on one named segment.

    Other segments keep the base treatment, including neighbors that used to
    share its look group. The named segment uses the fault's filtered list.
    """
    if len(base) != len(segments):
        raise TreatmentError("assignment length does not match the segments")
    if segment_index < 0 or segment_index >= len(segments):
        raise TreatmentError("segment index is outside the timeline")
    options = filter_treatments(segments[segment_index].problem, fault)
    revised: list[list[Treatment]] = []
    for treatment in options[:PLAN_COUNT]:
        chosen = list(base)
        chosen[segment_index] = treatment
        revised.append(chosen)
    return revised


def build_plans(
    segments: list[TimelineSegment],
    *,
    limit: int = PLAN_COUNT,
) -> list[TimelinePlan]:
    """Choose plans and record each score beside its treatment ids."""
    chosen = choose_plans(segments, limit=limit)
    plans: list[TimelinePlan] = []
    for index, treatments in enumerate(chosen):
        plans.append(
            TimelinePlan(
                index=index,
                score=assignment_score(segments, treatments),
                treatment_ids=tuple(treatment.treatment_id for treatment in treatments),
            )
        )
    return plans


def plans_path(job_dir: Path) -> Path:
    """Return the on-disk plan list for a job."""
    return job_dir / PLANS_FILE_NAME


def write_plans(job_dir: Path, plans: list[TimelinePlan]) -> Path:
    """Atomically write the five timeline plans."""
    path = plans_path(job_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": PLANS_SCHEMA_VERSION,
        "plans": [plan.to_dict() for plan in plans],
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def load_plans(job_dir: Path) -> list[TimelinePlan]:
    """Load timeline plans from a job directory."""
    path = plans_path(job_dir)
    if not path.is_file():
        raise TreatmentError(f"plans not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TreatmentError("plans root must be an object")
    raw_plans = data.get("plans")
    if not isinstance(raw_plans, list):
        raise TreatmentError("plans list is missing")
    plans: list[TimelinePlan] = []
    for item in raw_plans:
        if not isinstance(item, dict):
            raise TreatmentError("each plan must be an object")
        plans.append(TimelinePlan.from_dict(item))
    return plans
