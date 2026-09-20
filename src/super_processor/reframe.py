"""Vertical reframe path and size-cap feasibility for social export."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .probe import MediaFacts
from .validator import MIN_BITRATE_KBPS

REFRAME_FILE_NAME = "reframe_plan.json"
REFRAME_SCHEMA_VERSION = 1
# Offset from frame center (fraction of width) before we claim saliency tracking.
_SALIENCY_OFFSET_FRAC = 0.04


class ReframeError(ValueError):
    """Raised when a reframe/size-cap plan cannot be produced."""


@dataclass(slots=True)
class SizeCapAssessment:
    """Bitrate-floor check for a requested delivery size."""

    max_size_mb: float
    duration_s: float
    avg_kbps: float
    floor_kbps: float
    status: str  # ok | warn | infeasible
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SizeCapAssessment:
        return cls(
            max_size_mb=float(data["max_size_mb"]),
            duration_s=float(data["duration_s"]),
            avg_kbps=float(data["avg_kbps"]),
            floor_kbps=float(data["floor_kbps"]),
            status=str(data["status"]),
            message=str(data["message"]),
        )


@dataclass(slots=True)
class ReframePath:
    """9:16 crop path with optional classical saliency subject tracking."""

    mode: str
    input_width: int
    input_height: int
    output_width: int
    output_height: int
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    padding: float
    subject_strategy: str
    subject_cx: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReframePath:
        return cls(
            mode=str(data["mode"]),
            input_width=int(data["input_width"]),
            input_height=int(data["input_height"]),
            output_width=int(data["output_width"]),
            output_height=int(data["output_height"]),
            crop_x=int(data["crop_x"]),
            crop_y=int(data["crop_y"]),
            crop_w=int(data["crop_w"]),
            crop_h=int(data["crop_h"]),
            padding=float(data["padding"]),
            subject_strategy=str(data["subject_strategy"]),
            subject_cx=float(data.get("subject_cx", 0.5)),
        )


@dataclass(slots=True)
class SocialExportPlan:
    """Combined vertical reframe + size-cap assessment for a job."""

    schema_version: int
    source_path: str
    reframe: ReframePath
    size_cap: SizeCapAssessment | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_path": self.source_path,
            "reframe": self.reframe.to_dict(),
            "size_cap": None if self.size_cap is None else self.size_cap.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SocialExportPlan:
        size_raw = data.get("size_cap")
        return cls(
            schema_version=int(data.get("schema_version", REFRAME_SCHEMA_VERSION)),
            source_path=str(data["source_path"]),
            reframe=ReframePath.from_dict(dict(data["reframe"])),
            size_cap=(
                None
                if size_raw is None
                else SizeCapAssessment.from_dict(dict(size_raw))
            ),
        )


def assess_size_cap(
    *,
    max_size_mb: float,
    duration_s: float,
    floor_kbps: float = MIN_BITRATE_KBPS,
) -> SizeCapAssessment:
    """Classify a size cap as ok, warn, or infeasible against a bitrate floor."""
    if duration_s <= 0:
        raise ReframeError("duration must be positive for size-cap assessment")
    if max_size_mb <= 0:
        raise ReframeError("max_size_mb must be positive")
    budget_bits = max_size_mb * 1024 * 1024 * 8
    avg_kbps = (budget_bits / duration_s) / 1000.0
    if avg_kbps < floor_kbps:
        status = "infeasible"
        message = (
            f"max_size_mb={max_size_mb:g} implies ~{avg_kbps:.1f} kbps over "
            f"{duration_s:.1f}s; below floor {floor_kbps:.0f} kbps"
        )
    elif avg_kbps < floor_kbps * 1.5:
        status = "warn"
        message = (
            f"max_size_mb={max_size_mb:g} implies ~{avg_kbps:.1f} kbps; "
            "quality may be poor"
        )
    else:
        status = "ok"
        message = f"max_size_mb={max_size_mb:g} implies ~{avg_kbps:.1f} kbps"
    return SizeCapAssessment(
        max_size_mb=max_size_mb,
        duration_s=duration_s,
        avg_kbps=avg_kbps,
        floor_kbps=floor_kbps,
        status=status,
        message=message,
    )


def _even(value: int) -> int:
    return value - (value % 2)


def _crop_window_9_16(width: int, height: int) -> tuple[int, int]:
    """Return even crop_w/crop_h for a 9:16 window inside the source frame."""
    source_aspect = width / height
    target_aspect = 9 / 16
    if source_aspect > target_aspect:
        crop_h = height
        crop_w = int(round(height * target_aspect))
    else:
        crop_w = width
        crop_h = int(round(width / target_aspect))
    crop_w = _even(max(2, min(width, crop_w)))
    crop_h = _even(max(2, min(height, crop_h)))
    return crop_w, crop_h


def subject_x_from_gray(frame: bytes, width: int, height: int) -> float:
    """Estimate horizontal subject center as a fraction of frame width.

    Uses column edge energy (classical saliency proxy). Values are in [0, 1].
    """
    if width <= 1 or height <= 1 or len(frame) < width * height:
        return 0.5
    scores = [0.0] * width
    for y in range(height - 1):
        row = y * width
        nxt = (y + 1) * width
        for x in range(width):
            scores[x] += abs(frame[row + x] - frame[nxt + x])
    for y in range(height):
        row = y * width
        for x in range(width - 1):
            scores[x] += 0.5 * abs(frame[row + x] - frame[row + x + 1])
    total = sum(scores)
    if total <= 0:
        return 0.5
    com = sum(index * score for index, score in enumerate(scores)) / total
    return max(0.0, min(1.0, com / float(width - 1)))


def estimate_subject_x_frac(
    source: Path,
    facts: MediaFacts,
    *,
    sample_width: int = 160,
    sample_height: int = 90,
) -> float:
    """Average classical saliency across stratified sample windows."""
    # Local import avoids a circular dependency with estimators ↔ reframe.
    from .estimators import EstimatorError, choose_sample_windows, extract_gray_frame

    windows = choose_sample_windows(facts, count=3)
    fractions: list[float] = []
    for window in windows:
        try:
            frame = extract_gray_frame(
                source,
                at_s=window.start_s,
                width=sample_width,
                height=sample_height,
            )
        except EstimatorError:
            continue
        fractions.append(subject_x_from_gray(frame, sample_width, sample_height))
    if not fractions:
        return 0.5
    return sum(fractions) / len(fractions)


def compute_center_reframe(
    *,
    width: int,
    height: int,
    max_height: int = 1920,
    padding: float = 0.0,
    subject_cx: float = 0.5,
) -> ReframePath:
    """Compute a 9:16 crop anchored on ``subject_cx`` (0=left, 1=right)."""
    if width <= 0 or height <= 0:
        raise ReframeError("video dimensions must be positive")
    padding = max(0.0, min(0.25, padding))
    subject_cx = max(0.0, min(1.0, subject_cx))
    target_h = _even(max(2, int(max_height)))
    target_w = _even(max(2, int(round(target_h * 9 / 16))))

    crop_w, crop_h = _crop_window_9_16(width, height)
    ideal_x = int(round(subject_cx * width - crop_w / 2.0))
    crop_x = _even(max(0, min(width - crop_w, ideal_x)))
    crop_y = _even(max(0, (height - crop_h) // 2))

    strategy = (
        "saliency" if abs(subject_cx - 0.5) >= _SALIENCY_OFFSET_FRAC else "center"
    )
    return ReframePath(
        mode="vertical_9_16",
        input_width=width,
        input_height=height,
        output_width=target_w,
        output_height=target_h,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_w=crop_w,
        crop_h=crop_h,
        padding=padding,
        subject_strategy=strategy,
        subject_cx=round(subject_cx, 4),
    )


def plan_social_export(
    facts: MediaFacts,
    *,
    max_height: int = 1920,
    padding: float = 0.0,
    max_size_mb: float | None = None,
    source: Path | None = None,
) -> SocialExportPlan:
    """Build a vertical reframe path and optional size-cap assessment."""
    if not facts.has_video:
        raise ReframeError("social export planning requires a video stream")
    width = None
    height = None
    for stream in facts.streams:
        if stream.codec_type == "video" and stream.width and stream.height:
            width = stream.width
            height = stream.height
            break
    if width is None or height is None:
        raise ReframeError("media facts lack video width/height")

    subject_cx = 0.5
    media = source if source is not None else Path(facts.source_path)
    if media.is_file():
        try:
            subject_cx = estimate_subject_x_frac(media, facts)
        except Exception:
            subject_cx = 0.5

    reframe = compute_center_reframe(
        width=width,
        height=height,
        max_height=max_height,
        padding=padding,
        subject_cx=subject_cx,
    )
    size_cap = None
    if max_size_mb is not None:
        if facts.duration_s is None:
            raise ReframeError("size-cap planning requires media duration")
        size_cap = assess_size_cap(
            max_size_mb=max_size_mb,
            duration_s=facts.duration_s,
        )
    return SocialExportPlan(
        schema_version=REFRAME_SCHEMA_VERSION,
        source_path=facts.source_path,
        reframe=reframe,
        size_cap=size_cap,
    )


def reframe_plan_path(job_dir: Path) -> Path:
    """Return the on-disk social export plan path."""
    return job_dir / REFRAME_FILE_NAME


def write_reframe_plan(job_dir: Path, plan: SocialExportPlan) -> Path:
    """Atomically write a social export plan."""
    path = reframe_plan_path(job_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return path


def load_reframe_plan(job_dir: Path) -> SocialExportPlan:
    """Load a social export plan from a job directory."""
    path = reframe_plan_path(job_dir)
    if not path.is_file():
        raise ReframeError(f"reframe plan not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ReframeError("reframe plan root must be an object")
    return SocialExportPlan.from_dict(data)
