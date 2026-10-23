"""Segment loop: dense stills for Gemini to cut the video into segments.

No video leaves the machine. FFmpeg saves a grid of small stills, about one
a second and never more than :data:`MAX_STILLS`, always including the first
and the last frame. Gemini proposes segment boundaries from those stills and
their times; the editor approves them before anything is planned.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .segments import SEGMENTS_FILE_NAME, STILLS_DIR_NAME, WIZARD_SEGMENTS_KIND
from .shots import ShotError, extract_still

MAX_STILLS = 120
STILLS_PER_S = 1.0
# Seeking to the exact duration finds no frame; the last still is taken this
# far before the end.
END_MARGIN_S = 0.1
# segments.json is shared with the older segment review, which writes
# schema 1 without a kind. Each side refuses the other's file.
SEGMENT_LIST_SCHEMA_VERSION = 2
SEGMENT_LIST_KIND = WIZARD_SEGMENTS_KIND


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


@dataclass(frozen=True, slots=True)
class LoopSegment:
    """One span of the video that is planned and previewed on its own."""

    start_s: float
    end_s: float
    label: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        start, end = float(self.start_s), float(self.end_s)
        if not (math.isfinite(start) and math.isfinite(end)):
            raise SegmentLoopError("segment times must be finite")
        if start < 0 or end <= start:
            raise SegmentLoopError("segment end must be after its start")
        object.__setattr__(self, "start_s", start)
        object.__setattr__(self, "end_s", end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "label": self.label,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Any) -> LoopSegment:
        if not isinstance(data, dict):
            raise SegmentLoopError("each segment must be an object")
        try:
            return cls(
                start_s=float(data["start_s"]),
                end_s=float(data["end_s"]),
                label=str(data.get("label", "")),
                note=str(data.get("note", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SegmentLoopError("a segment is missing start_s or end_s") from exc


@dataclass(slots=True)
class SegmentList:
    """The proposed segments, and whether the editor approved them."""

    segments: list[LoopSegment] = field(default_factory=list)
    approved: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SEGMENT_LIST_SCHEMA_VERSION,
            "kind": SEGMENT_LIST_KIND,
            "approved": self.approved,
            "error": self.error,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: Any) -> SegmentList:
        if not isinstance(data, dict) or data.get("kind") != SEGMENT_LIST_KIND:
            raise SegmentLoopError(
                f"{SEGMENTS_FILE_NAME} belongs to the older segment review; "
                "pick the file again in the wizard"
            )
        raw = data.get("segments")
        if not isinstance(raw, list):
            raise SegmentLoopError("the segment list is missing")
        error = data.get("error")
        return cls(
            segments=[LoopSegment.from_dict(item) for item in raw],
            approved=bool(data.get("approved", False)),
            error=error if isinstance(error, str) else None,
        )


def save_segment_list(job_dir: Path, segments: SegmentList) -> Path:
    """Write ``segments.json`` atomically."""
    path = job_dir / SEGMENTS_FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(segments.to_dict(), indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def load_segment_list(job_dir: Path) -> SegmentList | None:
    """Read ``segments.json``, or ``None`` when there is none yet."""
    path = job_dir / SEGMENTS_FILE_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SegmentLoopError(f"cannot read {SEGMENTS_FILE_NAME}: {exc}") from exc
    return SegmentList.from_dict(data)
