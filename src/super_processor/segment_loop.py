"""Segment loop: dense stills for Gemini to cut the video into segments.

No video leaves the machine. FFmpeg saves a grid of small stills, about one
a second and never more than :data:`MAX_STILLS`, always including the first
and the last frame. Gemini proposes segment boundaries from those stills and
their times; the editor approves them before anything is planned.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .segments import SEGMENTS_FILE_NAME, STILLS_DIR_NAME, WIZARD_SEGMENTS_KIND
from .shots import (
    CONTENTS,
    Shot,
    ShotError,
    ShotList,
    extract_still,
    measure_range,
    shot_still,
)

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


_STILL_NAME = re.compile(r"t_(\d{8})\.jpg")


def load_segment_stills(job_dir: Path) -> list[SegmentStill]:
    """The sampled stills already in the job, in time order."""
    folder = job_dir / STILLS_DIR_NAME
    if folder.is_symlink() or not folder.is_dir():
        return []
    stills: list[SegmentStill] = []
    for path in folder.iterdir():
        found = _STILL_NAME.fullmatch(path.name)
        if found and path.is_file() and not path.is_symlink():
            stills.append(
                SegmentStill(
                    t_s=int(found.group(1)) / 1000,
                    path=f"{STILLS_DIR_NAME}/{path.name}",
                )
            )
    return sorted(stills, key=lambda still: still.t_s)


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
    # Where restoration mistakes would be obvious (see :data:`.shots.CONTENTS`).
    contains: tuple[str, ...] = ()

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
            "contains": list(self.contains),
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
                contains=_contents(data.get("contains")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SegmentLoopError("a segment is missing start_s or end_s") from exc


@dataclass(slots=True)
class SegmentList:
    """The proposed segments, and whether the editor approved them."""

    segments: list[LoopSegment] = field(default_factory=list)
    approved: bool = False
    error: str | None = None
    # The editor's reason for sending the last proposal back.
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SEGMENT_LIST_SCHEMA_VERSION,
            "kind": SEGMENT_LIST_KIND,
            "approved": self.approved,
            "error": self.error,
            "note": self.note,
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
            note=str(data.get("note", "")),
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


# Gemini's boundaries come from stills about a second apart, so a sliver
# shorter than this is a rounding artefact, not a segment of its own.
MIN_LOOP_SEGMENT_S = 0.5


def _contents(raw: Any) -> tuple[str, ...]:
    """Known content names from a model reply, in order, without repeats."""
    if not isinstance(raw, list):
        return ()
    names = [str(item).strip().lower() for item in raw]
    return tuple(dict.fromkeys(name for name in names if name in CONTENTS))


_Span = tuple[float, float, str, tuple[str, ...]]


def _proposed_spans(raw: list[dict[str, Any]], duration_s: float) -> list[_Span]:
    """Gemini's spans clamped to the clip, in start order; broken ones dropped."""
    spans: list[_Span] = []
    for item in raw:
        try:
            start = min(max(float(item["start_s"]), 0.0), duration_s)
            end = min(max(float(item["end_s"]), 0.0), duration_s)
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
            continue
        label = " ".join(str(item.get("label", "")).split())[:80]
        spans.append((start, end, label, _contents(item.get("contains"))))
    return sorted(spans)


def repair_coverage(raw: list[dict[str, Any]], duration_s: float) -> list[LoopSegment]:
    """Turn Gemini's proposal into segments covering 0 to the end exactly once.

    Each segment starts where the one before it ends; the first starts at 0
    and the last ends at ``duration_s``. Nothing is shorter than
    :data:`MIN_LOOP_SEGMENT_S` unless the whole clip is.
    """
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise SegmentLoopError("the video has no duration to segment")
    spans = _proposed_spans(raw, duration_s)
    if not spans:
        return [LoopSegment(0.0, duration_s, label="whole video")]
    # A gap is split down the middle; on an overlap the later span's start
    # wins, since Gemini saw the change somewhere inside it. A span wholly
    # inside an earlier one adds no cut. A cut too close to the last one
    # would leave a sliver, so it is dropped and the newer span's label
    # takes over the segment it would have started.
    cuts: list[float] = []
    labels: list[tuple[str, tuple[str, ...]]] = [(spans[0][2], spans[0][3])]
    reach = spans[0][1]
    for start, end, label, contains in spans[1:]:
        if end <= reach:
            continue
        cut = (reach + start) / 2 if start > reach else start
        reach = end
        if cut > duration_s - MIN_LOOP_SEGMENT_S:
            break
        if cut - (cuts[-1] if cuts else 0.0) < MIN_LOOP_SEGMENT_S:
            # Keep what the sliver contained, so its faces stay protected.
            merged = tuple(dict.fromkeys((*labels[-1][1], *contains)))
            labels[-1] = (label, merged)
            continue
        cuts.append(cut)
        labels.append((label, contains))
    edges = [0.0, *cuts, duration_s]
    return [
        LoopSegment(round(start, 3), round(end, 3), label=label, contains=contains)
        for start, end, (label, contains) in zip(
            edges[:-1], edges[1:], labels, strict=True
        )
    ]


def shots_from_segments(
    source: Path,
    job_dir: Path,
    segments: list[LoopSegment],
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> ShotList:
    """Measure and still each approved segment, as the shot planner expects.

    The planner works on shots; in the segment loop each segment is one.
    """
    shots: list[Shot] = []
    for segment in segments:
        try:
            metrics = measure_range(
                source, segment.start_s, segment.end_s, ffmpeg_bin=ffmpeg_bin
            )
            still = shot_still(
                source, job_dir, segment.start_s, segment.end_s, ffmpeg_bin=ffmpeg_bin
            )
        except ShotError as exc:
            raise SegmentLoopError(f"could not measure a segment: {exc}") from exc
        shots.append(
            Shot(
                start_s=segment.start_s,
                end_s=segment.end_s,
                metrics=metrics,
                label=segment.label,
                issues=metrics.hints(),
                contains=segment.contains,
                still=still,
            )
        )
    return ShotList(shots=shots)
