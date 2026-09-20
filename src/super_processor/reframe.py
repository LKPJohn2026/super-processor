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
    """Center-weighted 9:16 crop path (face/saliency deferred to later)."""

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReframePath:
        return cls(**{key: data[key] for key in cls.__dataclass_fields__})  # type: ignore[misc]


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


def compute_center_reframe(
    *,
    width: int,
    height: int,
    max_height: int = 1920,
    padding: float = 0.0,
) -> ReframePath:
    """Compute a center-weighted 9:16 crop, with optional pad inset."""
    if width <= 0 or height <= 0:
        raise ReframeError("video dimensions must be positive")
    padding = max(0.0, min(0.25, padding))
    target_h = _even(max(2, int(max_height)))
    target_w = _even(max(2, int(round(target_h * 9 / 16))))

    # Source crop window that matches 9:16, centered.
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
    crop_x = _even(max(0, (width - crop_w) // 2))
    crop_y = _even(max(0, (height - crop_h) // 2))

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
        subject_strategy="center",  # face/saliency arrives with later ML work
    )


def plan_social_export(
    facts: MediaFacts,
    *,
    max_height: int = 1920,
    padding: float = 0.0,
    max_size_mb: float | None = None,
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

    reframe = compute_center_reframe(
        width=width,
        height=height,
        max_height=max_height,
        padding=padding,
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
