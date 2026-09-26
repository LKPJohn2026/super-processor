"""Timeline segmentation bounds for the review loops."""

from __future__ import annotations

import math

MIN_SEGMENT_S = 5.0
MAX_SEGMENT_S = 120.0
MAX_SEGMENTS = 15
MAX_DURATION_S = 30.0 * 60.0


class SegmentError(ValueError):
    """Raised when a timeline cannot be segmented inside the bounds."""


def legal_segment_counts(duration_s: float) -> tuple[int, int]:
    """Return the inclusive ``(fewest, most)`` segment counts for ``duration_s``.

    A segment is 5–120 seconds, and a timeline holds at most 15 of them.
    Durations longer than 30 minutes, or too short to hold one segment, are
    refused.
    """
    if duration_s <= 0:
        raise SegmentError("duration must be positive")
    if duration_s > MAX_DURATION_S:
        raise SegmentError(
            f"duration {duration_s:.1f}s exceeds the {MAX_DURATION_S:.0f}s limit"
        )
    fewest = math.ceil(duration_s / MAX_SEGMENT_S)
    most = min(MAX_SEGMENTS, math.floor(duration_s / MIN_SEGMENT_S))
    if fewest > most or most < 1:
        raise SegmentError(
            f"duration {duration_s:.1f}s cannot be split into "
            f"{MIN_SEGMENT_S:.0f}–{MAX_SEGMENT_S:.0f}s segments"
        )
    return fewest, most


def _equal_pieces(start: float, end: float) -> list[tuple[float, float]]:
    """Cut ``[start, end]`` into equal pieces of at most ``MAX_SEGMENT_S``."""
    length = end - start
    count = max(1, math.ceil(length / MAX_SEGMENT_S - 1e-9))
    step = length / count
    pieces: list[tuple[float, float]] = []
    cursor = start
    for index in range(count):
        nxt = end if index == count - 1 else cursor + step
        pieces.append((cursor, nxt))
        cursor = nxt
    return pieces


def _strongest_interior_cut(
    start: float,
    end: float,
    scores: list[float],
) -> float | None:
    """Return the integer second with the strongest positive change score."""
    best_time: float | None = None
    best_score = 0.0
    time_s = int(math.floor(start)) + 1
    while time_s < end:
        index = time_s - 1
        if 0 <= index < len(scores) and scores[index] > best_score:
            best_score = scores[index]
            best_time = float(time_s)
        time_s += 1
    return best_time


def split_oversized(
    intervals: list[tuple[float, float]],
    scores: list[float] | None = None,
) -> list[tuple[float, float]]:
    """Break every interval longer than ``MAX_SEGMENT_S``.

    A flat interval (no positive interior score) is divided into equal pieces.
    Otherwise the cut falls on the strongest interior score, and both sides are
    repaired the same way.
    """
    repaired: list[tuple[float, float]] = []
    for start, end in intervals:
        if end < start:
            raise SegmentError("interval end is before its start")
        length = end - start
        if length <= MAX_SEGMENT_S + 1e-9:
            repaired.append((start, end))
            continue
        cut = None if scores is None else _strongest_interior_cut(start, end, scores)
        if cut is None:
            repaired.extend(_equal_pieces(start, end))
            continue
        repaired.extend(split_oversized([(start, cut), (cut, end)], scores))
    return repaired


def _boundary_score(time_s: float, scores: list[float] | None) -> float:
    if not scores:
        return 0.0
    index = int(round(time_s)) - 1
    if index < 0 or index >= len(scores):
        return 0.0
    return scores[index]


def _merge_pair(
    intervals: list[tuple[float, float]],
    index: int,
) -> list[tuple[float, float]]:
    start = intervals[index][0]
    end = intervals[index + 1][1]
    return [*intervals[:index], (start, end), *intervals[index + 2 :]]


def _merge_shorter_than_min(
    intervals: list[tuple[float, float]],
    scores: list[float] | None,
) -> list[tuple[float, float]]:
    """Absorb pieces shorter than ``MIN_SEGMENT_S`` into a neighbor."""
    pieces = list(intervals)
    guard = 0
    while guard < len(pieces) + 2:
        guard += 1
        short_index = next(
            (
                index
                for index, (start, end) in enumerate(pieces)
                if end - start < MIN_SEGMENT_S - 1e-6
            ),
            None,
        )
        if short_index is None or len(pieces) == 1:
            return pieces
        left = short_index - 1 if short_index > 0 else None
        right = short_index if short_index < len(pieces) - 1 else None
        if left is None and right is None:
            return pieces
        if left is None:
            assert right is not None
            pieces = _merge_pair(pieces, right)
            continue
        if right is None:
            pieces = _merge_pair(pieces, left)
            continue
        left_score = _boundary_score(pieces[short_index][0], scores)
        right_score = _boundary_score(pieces[short_index][1], scores)
        # Equal scores absorb into the previous piece, which keeps a short tail stable.
        pieces = _merge_pair(pieces, left if left_score <= right_score else right)
    return pieces


def merge_short_and_cap(
    intervals: list[tuple[float, float]],
    scores: list[float] | None = None,
) -> list[tuple[float, float]]:
    """Absorb sub-5s pieces, then merge down to ``MAX_SEGMENTS``.

    The count cap merges the weakest boundary first and skips a merge that
    would exceed ``MAX_SEGMENT_S``. A merge of a short piece that overshoots
    the cap is split again so the result still respects both bounds.
    """
    pieces = _merge_shorter_than_min(list(intervals), scores)
    pieces = split_oversized(pieces, scores)
    while len(pieces) > MAX_SEGMENTS:
        candidates: list[tuple[float, int]] = []
        for index in range(len(pieces) - 1):
            combined = pieces[index + 1][1] - pieces[index][0]
            if combined > MAX_SEGMENT_S + 1e-6:
                continue
            boundary = pieces[index][1]
            candidates.append((_boundary_score(boundary, scores), index))
        if not candidates:
            raise SegmentError(
                f"cannot merge {len(pieces)} segments without exceeding "
                f"{MAX_SEGMENT_S:.0f}s"
            )
        _, index = min(candidates, key=lambda item: (item[0], item[1]))
        pieces = _merge_pair(pieces, index)
    return pieces
