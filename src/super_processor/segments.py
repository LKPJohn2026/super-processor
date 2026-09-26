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
