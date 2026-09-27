"""Per-segment encode argument lists and the concat list that joins them."""

from __future__ import annotations

from pathlib import Path

from .preview import kept_range, preview_filters
from .segments import TimelineSegment
from .treatments import Treatment


def build_segment_argv(
    source: Path,
    segment: TimelineSegment,
    treatment: Treatment,
    dest: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    include_audio: bool = True,
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
    command.extend(["-c:v", "libx265"])
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
