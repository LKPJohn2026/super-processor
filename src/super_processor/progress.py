"""How far a wizard job is, and how long its running step has left.

The wizard runs in workers separated by the editor's reviews: segmenting,
then each segment's previews, then the render. :func:`overall_pct` places a
worker's own percentage in a fixed band of the whole job, so the overall
number only moves forward. :class:`StepClock` turns a worker's fractions
into a time-left estimate from how fast it is actually going.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bands of the whole job, in percent. Segmenting is short; the segment
# previews are a GPU pass over a few seconds of each segment; the render
# restores every frame and dominates on any real video.
SEGMENTING_BAND = (0.0, 8.0)
PREVIEWS_BAND = (8.0, 38.0)
RENDER_BAND = (38.0, 100.0)
# How quickly the recent rate forgets older steps, in seconds of history.
_RATE_MEMORY_S = 60.0
# Below this much measured progress the rate is too noisy to show.
_MIN_MEASURED = 0.005
_MIN_MEASURED_S = 3.0


def overall_pct(
    phase: str,
    pct: float,
    *,
    segment_index: int = 0,
    segment_count: int = 1,
    revising: bool = False,
) -> int:
    """The whole job's percentage for a worker at ``pct`` of its own step.

    A render after the result (a revise) is a job of its own: its overall
    percentage is its own.
    """
    done = min(max(pct, 0.0), 100.0) / 100.0
    if phase == "rendering":
        if revising:
            return round(done * 100.0)
        low, high = RENDER_BAND
    elif phase == "analyzing":
        count = max(1, segment_count)
        index = min(max(segment_index, 0), count - 1)
        low, high = PREVIEWS_BAND
        width = (high - low) / count
        low, high = low + index * width, low + (index + 1) * width
    else:  # sampling, uploading, proposing
        low, high = SEGMENTING_BAND
    return round(low + (high - low) * done)


@dataclass
class StepClock:
    """Estimate one worker step's time left from its fractions over time.

    The first report is only a starting point: the time before it (loading
    FlashVSR, uploading) says nothing about how fast frames go. After that,
    the rate is a blend of the step's average rate and a recent one that
    follows changes (a heavier segment, the GPU throttling), so the estimate
    neither jumps on one slow chunk nor ignores a lasting change.
    """

    started_at: float
    _first: tuple[float, float] | None = None
    _last: tuple[float, float] | None = None
    _recent_rate: float | None = None

    def update(self, fraction: float, now: float) -> float | None:
        """Record ``fraction`` (0 to 1) done at ``now``; return seconds left."""
        fraction = min(max(fraction, 0.0), 1.0)
        if self._first is None:
            self._first = self._last = (now, fraction)
            return None
        assert self._last is not None
        last_t, last_f = self._last
        if fraction > last_f and now > last_t:
            rate = (fraction - last_f) / (now - last_t)
            if self._recent_rate is None:
                self._recent_rate = rate
            else:
                weight = min(1.0, (now - last_t) / _RATE_MEMORY_S)
                self._recent_rate += weight * (rate - self._recent_rate)
            self._last = (now, fraction)
        return self.time_left(now)

    def time_left(self, now: float) -> float | None:
        """Seconds left at ``now``, or None while there is too little to go on."""
        if self._first is None or self._last is None or self._recent_rate is None:
            return None
        first_t, first_f = self._first
        last_t, last_f = self._last
        measured = last_f - first_f
        if measured < _MIN_MEASURED or last_t - first_t < _MIN_MEASURED_S:
            return None
        average = measured / (last_t - first_t)
        rate = (average + self._recent_rate) / 2
        if rate <= 0:
            return None
        left = (1.0 - last_f) / rate
        # Time already spent since the last report counts against it.
        return max(0.0, left - (now - last_t))

    def elapsed(self, now: float) -> float:
        return max(0.0, now - self.started_at)
