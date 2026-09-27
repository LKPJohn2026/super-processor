"""Short preview windows taken from inside a segment's kept trim."""

from __future__ import annotations

from .recipe import OpName
from .segments import MIN_SEGMENT_S, TimelineSegment
from .treatments import Treatment, TreatmentError

PREVIEW_SPAN_S = 3.0


def kept_range(segment: TimelineSegment, treatment: Treatment) -> tuple[float, float]:
    """Return the in and out points a treatment keeps inside the segment."""
    start = segment.start_s
    end = segment.end_s
    for step in treatment.steps:
        if step.op is not OpName.TRIM:
            continue
        params = step.as_dict()
        start = max(start, float(params.get("start_s", start)))
        end = min(end, float(params.get("end_s", end)))
    if end - start < MIN_SEGMENT_S:
        raise TreatmentError("trim leaves less than 5s inside the segment")
    return start, end


def preview_window(
    segment: TimelineSegment,
    treatment: Treatment,
    *,
    span_s: float = PREVIEW_SPAN_S,
) -> tuple[float, float]:
    """Return about three seconds around the key frame, inside the kept trim."""
    if span_s <= 0:
        raise TreatmentError("preview span must be positive")
    kept_start, kept_end = kept_range(segment, treatment)
    key = min(max(segment.keyframe_s, kept_start), kept_end)
    start = key - span_s / 2.0
    end = key + span_s / 2.0
    if start < kept_start:
        end += kept_start - start
        start = kept_start
    if end > kept_end:
        start -= end - kept_end
        end = kept_end
    start = max(start, kept_start)
    end = min(end, kept_end)
    return start, end
