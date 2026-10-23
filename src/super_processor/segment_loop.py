"""Segment loop: dense stills for Gemini to cut the video into segments.

No video leaves the machine. FFmpeg saves a grid of small stills, about one
a second and never more than :data:`MAX_STILLS`, always including the first
and the last frame. Gemini proposes segment boundaries from those stills and
their times; the editor approves them before anything is planned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .segments import STILLS_DIR_NAME
from .shots import ShotError, extract_still

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


class SegmentLoopError(RuntimeError):
    """Raised when the segment loop cannot sample or keep its segments."""


@dataclass(frozen=True, slots=True)
class SegmentStill:
    """One sampled still: its source time and its path inside the job."""

    t_s: float
    path: str


def still_name(t_s: float) -> str:
    """File name for a still at ``t_s``; the time is in the name, in ms."""
    return f"t_{round(t_s * 1000):08d}.jpg"


def extract_segment_stills(
    source: Path,
    job_dir: Path,
    times: list[float],
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> list[SegmentStill]:
    """Save one still per time under ``job_dir`` and return them in order.

    Stills from an earlier sampling are removed first. Stills from the older
    segment review (``seg_NN.ppm``) share the folder and are left alone.
    """
    folder = job_dir / STILLS_DIR_NAME
    if folder.is_symlink():
        raise SegmentLoopError(f"refusing {STILLS_DIR_NAME}: it is a symbolic link")
    folder.mkdir(parents=True, exist_ok=True)
    if not folder.resolve().is_relative_to(job_dir.resolve()):
        raise SegmentLoopError(f"{STILLS_DIR_NAME} is outside the job folder")
    for old in folder.glob("t_*.jpg"):
        old.unlink()
    stills: list[SegmentStill] = []
    for t_s in times:
        relative = f"{STILLS_DIR_NAME}/{still_name(t_s)}"
        try:
            extract_still(source, t_s, job_dir / relative, ffmpeg_bin=ffmpeg_bin)
        except ShotError as exc:
            raise SegmentLoopError(f"could not sample the video: {exc}") from exc
        stills.append(SegmentStill(t_s=t_s, path=relative))
    return stills
