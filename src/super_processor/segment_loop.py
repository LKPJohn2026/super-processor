"""Segment loop: dense stills for Gemini to cut the video into segments.

No video leaves the machine. FFmpeg saves a grid of small stills, about one
a second and never more than :data:`MAX_STILLS`, always including the first
and the last frame. Gemini proposes segment boundaries from those stills and
their times; the editor approves them before anything is planned.
"""

from __future__ import annotations

import math

MAX_STILLS = 120
STILLS_PER_S = 1.0
# Seeking to the exact duration finds no frame; the last still is taken this
# far before the end.
END_MARGIN_S = 0.1


def sample_times(
    duration_s: float, *, per_s: float = STILLS_PER_S, cap: int = MAX_STILLS
) -> list[float]:
    """Evenly spaced still times from the first frame to the last.

    About ``per_s`` stills a second; a long clip keeps the even spacing and
    stops at ``cap`` stills. Times are rounded to the millisecond, which is
    also how the still files are named.
    """
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("the video has no duration to sample")
    if per_s <= 0 or cap < 2:
        raise ValueError("sampling needs a positive rate and room for two stills")
    last = max(0.0, duration_s - END_MARGIN_S)
    if last == 0.0:
        return [0.0]
    # Round up so stills are never further apart than 1 / per_s.
    count = min(cap, math.ceil(last * per_s) + 1)
    times = [round(last * i / (count - 1), 3) for i in range(count)]
    return sorted(set(times))
