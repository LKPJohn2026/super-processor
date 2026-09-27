"""Short preview windows taken from inside a segment's kept trim."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

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


def kept_duration(
    segments: list[TimelineSegment], treatments: list[Treatment]
) -> float:
    """Return the sum of the kept ranges, which the size-cap floor uses."""
    if len(segments) != len(treatments):
        raise TreatmentError("assignment length does not match the segments")
    total = 0.0
    for segment, treatment in zip(segments, treatments, strict=True):
        start, end = kept_range(segment, treatment)
        total += end - start
    return total


def size_cap_bitrate_kbps(max_size_mb: float, duration_s: float) -> float:
    """Return the average bitrate implied by a size cap over ``duration_s``."""
    if duration_s <= 0:
        raise TreatmentError("kept duration must be positive")
    budget_bits = max_size_mb * 1024 * 1024 * 8
    return (budget_bits / duration_s) / 1000.0


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


class PreviewError(RuntimeError):
    """Raised when a short preview clip cannot be encoded."""


def preview_cache_key(segment_index: int, treatment_id: str) -> str:
    """Cache identity for one segment and one treatment."""
    return f"{segment_index}:{treatment_id}"


def preview_output_path(job_dir: Path, segment_index: int, treatment_id: str) -> Path:
    """Return the cached preview path for a segment and treatment."""
    safe = treatment_id.replace("/", "_")
    return job_dir / "previews" / f"seg_{segment_index:02d}_{safe}.mp4"


def _step_filter(step_op: OpName, params: dict[str, float]) -> str | None:
    if step_op is OpName.WHITE_BALANCE:
        temperature = params.get("temperature", 6500.0)
        return f"colortemperature=temperature={temperature:g}:mix=1"
    if step_op is OpName.CONTRAST:
        contrast = params.get("contrast", 1.0)
        brightness = params.get("brightness", 0.0)
        gamma = params.get("gamma", 1.0)
        return f"eq=contrast={contrast:g}:brightness={brightness:g}:gamma={gamma:g}"
    if step_op is OpName.DENOISE:
        strength = params.get("strength", 0.0)
        spatial = max(0.0, min(8.0, strength * 6.0))
        temporal = max(0.0, min(6.0, strength * 4.5))
        return f"hqdn3d={spatial:g}:{spatial * 0.75:g}:{temporal:g}:{temporal * 0.75:g}"
    if step_op is OpName.SHARPEN:
        size = int(params.get("luma_size", 5.0))
        amount = params.get("luma_amount", 0.0)
        return f"unsharp={size}:{size}:{amount:g}"
    if step_op is OpName.STABILIZE:
        return "deshake"
    if step_op is OpName.REFRAME_VERTICAL:
        subject = min(1.0, max(0.0, params.get("subject_cx", 0.5)))
        return f"crop=iw*0.9:ih:iw*{subject:g}*0.1:0"
    return None


def preview_filters(treatment: Treatment) -> list[str]:
    """Return the filter graph for one treatment, without a shell string."""
    filters: list[str] = []
    for step in treatment.steps:
        if step.op is OpName.TRIM:
            continue
        fragment = _step_filter(step.op, step.as_dict())
        if fragment is not None:
            filters.append(fragment)
    return filters


def build_preview_argv(
    source: Path,
    segment: TimelineSegment,
    treatment: Treatment,
    dest: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    """Build the argv for one short graded preview."""
    start, end = preview_window(segment, treatment)
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{end - start:.3f}",
    ]
    filters = preview_filters(treatment)
    if filters:
        command.extend(["-vf", ",".join(filters)])
    command.extend(
        [
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(dest),
        ]
    )
    return command


@dataclass(frozen=True, slots=True)
class PreviewClip:
    """A graded preview and the trim it was cut from."""

    segment_index: int
    treatment_id: str
    in_s: float
    out_s: float
    path: Path
    cache_key: str
    cached: bool

    def format_line(self) -> str:
        return f"in {self.in_s:.1f} out {self.out_s:.1f}  {self.path}"


def encode_preview(
    job_dir: Path,
    source: Path,
    segment: TimelineSegment,
    treatment: Treatment,
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> PreviewClip:
    """Encode one preview, reusing the file when the cache key matches."""
    cache_key = preview_cache_key(segment.index, treatment.treatment_id)
    dest = preview_output_path(job_dir, segment.index, treatment.treatment_id)
    marker = dest.with_suffix(".key")
    in_s, out_s = preview_window(segment, treatment)
    if (
        dest.is_file()
        and marker.is_file()
        and marker.read_text(encoding="utf-8") == cache_key
    ):
        return PreviewClip(
            segment.index,
            treatment.treatment_id,
            in_s,
            out_s,
            dest,
            cache_key,
            True,
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = build_preview_argv(
        source, segment, treatment, dest, ffmpeg_bin=ffmpeg_bin
    )
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreviewError(f"failed to encode preview: {exc}") from exc
    if completed.returncode != 0 or not dest.is_file():
        detail = (completed.stderr or b"").decode("utf-8", errors="replace").strip()
        raise PreviewError(detail or "ffmpeg preview failed")
    marker.write_text(cache_key, encoding="utf-8")
    return PreviewClip(
        segment.index,
        treatment.treatment_id,
        in_s,
        out_s,
        dest,
        cache_key,
        False,
    )
