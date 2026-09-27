"""Per-segment encode argument lists and the concat list that joins them."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .doctor import check_ffmpeg_capability
from .encoders import ALLOWED_ENCODERS, SOFTWARE_ENCODER, encoder_rate_args
from .preview import kept_range, preview_filters
from .segments import TimelineSegment
from .treatments import Treatment


class RenderError(RuntimeError):
    """Raised when a chosen plan cannot be encoded."""


def build_segment_argv(
    source: Path,
    segment: TimelineSegment,
    treatment: Treatment,
    dest: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    include_audio: bool = True,
    encoder: str = SOFTWARE_ENCODER,
) -> list[str]:
    """Build one segment encode. ``-ss`` and ``-t`` cut every stream the same way."""
    start, end = kept_range(segment, treatment)
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
    command.extend(encoder_rate_args(encoder))
    if include_audio:
        command.extend(["-c:a", "aac"])
    else:
        command.append("-an")
    command.append(str(dest))
    return command


def write_concat_list(path: Path, parts: list[Path]) -> Path:
    """Write a concat-demuxer list. Paths are escaped as single-quoted file lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for part in parts:
        escaped = str(part.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_concat_argv(
    concat_list: Path,
    dest: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    """Build the argv that concatenates segment files without a shell string."""
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c",
        "copy",
        str(dest),
    ]


def _run_ffmpeg(command: list[str]) -> None:
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RenderError(f"failed to start ffmpeg: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RenderError(detail or "ffmpeg failed")


def ensure_encoder(encoder: str) -> None:
    """Fail before the first segment when the encoder is not in this ffmpeg."""
    if encoder not in ALLOWED_ENCODERS:
        raise RenderError(f"unsupported encoder {encoder}")
    result = check_ffmpeg_capability(
        f"encoder:{encoder}",
        kind="encoders",
        token=encoder,
        required=True,
        hint="install ffmpeg with this encoder or render with libx265",
    )
    if not result.ok:
        raise RenderError(result.detail)


def render_chosen_plan(
    job_dir: Path,
    source: Path,
    segments: list[TimelineSegment],
    treatments: list[Treatment],
    *,
    ffmpeg_bin: str = "ffmpeg",
    include_audio: bool = True,
    encoder: str = SOFTWARE_ENCODER,
) -> Path:
    """Encode each kept segment, then concatenate them into one file."""
    ensure_encoder(encoder)
    if len(segments) != len(treatments):
        raise RenderError("assignment length does not match the segments")
    parts: list[Path] = []
    for segment, treatment in zip(segments, treatments, strict=True):
        dest = job_dir / "segment_encodes" / f"seg_{segment.index:02d}.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        _run_ffmpeg(
            build_segment_argv(
                source,
                segment,
                treatment,
                dest,
                ffmpeg_bin=ffmpeg_bin,
                include_audio=include_audio,
                encoder=encoder,
            )
        )
        parts.append(dest)
    listing = write_concat_list(job_dir / "concat.txt", parts)
    final = job_dir / "output.mp4"
    _run_ffmpeg(build_concat_argv(listing, final, ffmpeg_bin=ffmpeg_bin))
    if not final.is_file():
        raise RenderError("concat produced no output")
    return final
