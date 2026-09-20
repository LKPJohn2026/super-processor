"""FFprobe-backed media facts for grounded planning."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from .jobs import JobError

MEDIA_FACTS_NAME = "media_facts.json"
MEDIA_FACTS_SCHEMA_VERSION = 1


class ProbeError(JobError):
    """Raised when ffprobe cannot produce usable media facts."""


@dataclass(slots=True)
class StreamFacts:
    """Selected facts for one media stream."""

    index: int
    codec_type: str
    codec_name: str | None = None
    width: int | None = None
    height: int | None = None
    pix_fmt: str | None = None
    avg_frame_rate: str | None = None
    r_frame_rate: str | None = None
    nb_frames: int | None = None
    duration_s: float | None = None
    bit_rate: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    channel_layout: str | None = None
    color_primaries: str | None = None
    color_transfer: str | None = None
    color_space: str | None = None
    color_range: str | None = None
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize stream facts."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamFacts:
        """Deserialize stream facts."""
        return cls(
            index=int(data["index"]),
            codec_type=str(data["codec_type"]),
            codec_name=_optional_str(data.get("codec_name")),
            width=_optional_int(data.get("width")),
            height=_optional_int(data.get("height")),
            pix_fmt=_optional_str(data.get("pix_fmt")),
            avg_frame_rate=_optional_str(data.get("avg_frame_rate")),
            r_frame_rate=_optional_str(data.get("r_frame_rate")),
            nb_frames=_optional_int(data.get("nb_frames")),
            duration_s=_optional_float(data.get("duration_s")),
            bit_rate=_optional_int(data.get("bit_rate")),
            sample_rate=_optional_int(data.get("sample_rate")),
            channels=_optional_int(data.get("channels")),
            channel_layout=_optional_str(data.get("channel_layout")),
            color_primaries=_optional_str(data.get("color_primaries")),
            color_transfer=_optional_str(data.get("color_transfer")),
            color_space=_optional_str(data.get("color_space")),
            color_range=_optional_str(data.get("color_range")),
            tags={
                str(key): str(value)
                for key, value in dict(data.get("tags") or {}).items()
            },
        )


@dataclass(slots=True)
class MediaFacts:
    """Versioned probe summary used by later planning stages."""

    schema_version: int
    source_path: str
    format_name: str | None
    format_long_name: str | None
    duration_s: float | None
    size_bytes: int | None
    bit_rate: int | None
    streams: list[StreamFacts]
    has_video: bool
    has_audio: bool
    is_vfr: bool | None
    rotate: int | None = None
    probe_tool: str = "ffprobe"

    def to_dict(self) -> dict[str, Any]:
        """Serialize media facts for JSON storage."""
        payload = asdict(self)
        payload["streams"] = [stream.to_dict() for stream in self.streams]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MediaFacts:
        """Deserialize media facts from JSON data."""
        streams = [
            StreamFacts.from_dict(item) for item in list(data.get("streams") or [])
        ]
        return cls(
            schema_version=int(data.get("schema_version", MEDIA_FACTS_SCHEMA_VERSION)),
            source_path=str(data["source_path"]),
            format_name=_optional_str(data.get("format_name")),
            format_long_name=_optional_str(data.get("format_long_name")),
            duration_s=_optional_float(data.get("duration_s")),
            size_bytes=_optional_int(data.get("size_bytes")),
            bit_rate=_optional_int(data.get("bit_rate")),
            streams=streams,
            has_video=bool(data.get("has_video")),
            has_audio=bool(data.get("has_audio")),
            is_vfr=_optional_bool(data.get("is_vfr")),
            rotate=_optional_int(data.get("rotate")),
            probe_tool=str(data.get("probe_tool", "ffprobe")),
        )

    def primary_video(self) -> StreamFacts | None:
        """Return the first video stream, if any."""
        for stream in self.streams:
            if stream.codec_type == "video":
                return stream
        return None


def find_ffprobe() -> str:
    """Return the absolute path to ffprobe or raise."""
    found = shutil.which("ffprobe")
    if found is None:
        raise ProbeError("ffprobe not found on PATH")
    return str(Path(found).resolve())


def run_ffprobe_json(source: Path, *, ffprobe: str | None = None) -> dict[str, Any]:
    """Run ffprobe and return its JSON object."""
    resolved = source.expanduser().resolve()
    if not resolved.is_file():
        raise ProbeError(f"source is not a readable file: {resolved}")

    executable = ffprobe or find_ffprobe()
    command = [
        executable,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(resolved),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProbeError(f"ffprobe failed to start: {exc}") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown error").strip()
        raise ProbeError(f"ffprobe exited {completed.returncode}: {detail}")

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ProbeError("ffprobe returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ProbeError("ffprobe JSON root must be an object")
    return payload


def parse_frame_rate(value: str | None) -> float | None:
    """Parse an FFmpeg rate string such as ``30000/1001``."""
    if value is None or value in {"", "0/0", "N/A"}:
        return None
    try:
        return float(Fraction(value))
    except (ZeroDivisionError, ValueError):
        return None


def detect_vfr(avg_frame_rate: str | None, r_frame_rate: str | None) -> bool | None:
    """Heuristically detect variable frame rate from rate strings."""
    avg = parse_frame_rate(avg_frame_rate)
    raw = parse_frame_rate(r_frame_rate)
    if avg is None or raw is None:
        return None
    if avg == 0 or raw == 0:
        return None
    return abs(avg - raw) > 0.05


def extract_rotate(tags: dict[str, str]) -> int | None:
    """Read a rotation tag when present."""
    for key in ("rotate", "Rotation"):
        if key in tags:
            try:
                return int(float(tags[key]))
            except ValueError:
                return None
    return None


def stream_from_ffprobe(raw: dict[str, Any]) -> StreamFacts:
    """Map one ffprobe stream object into StreamFacts."""
    tags = {str(key): str(value) for key, value in dict(raw.get("tags") or {}).items()}
    return StreamFacts(
        index=int(raw.get("index", 0)),
        codec_type=str(raw.get("codec_type", "unknown")),
        codec_name=_optional_str(raw.get("codec_name")),
        width=_optional_int(raw.get("width")),
        height=_optional_int(raw.get("height")),
        pix_fmt=_optional_str(raw.get("pix_fmt")),
        avg_frame_rate=_optional_str(raw.get("avg_frame_rate")),
        r_frame_rate=_optional_str(raw.get("r_frame_rate")),
        nb_frames=_optional_int(raw.get("nb_frames")),
        duration_s=_optional_float(raw.get("duration")),
        bit_rate=_optional_int(raw.get("bit_rate")),
        sample_rate=_optional_int(raw.get("sample_rate")),
        channels=_optional_int(raw.get("channels")),
        channel_layout=_optional_str(raw.get("channel_layout")),
        color_primaries=_optional_str(raw.get("color_primaries")),
        color_transfer=_optional_str(raw.get("color_transfer")),
        color_space=_optional_str(raw.get("color_space")),
        color_range=_optional_str(raw.get("color_range")),
        tags=tags,
    )


def media_facts_from_ffprobe(
    source: Path,
    payload: dict[str, Any],
) -> MediaFacts:
    """Build MediaFacts from an ffprobe JSON payload."""
    resolved = source.expanduser().resolve()
    fmt = dict(payload.get("format") or {})
    streams = [stream_from_ffprobe(item) for item in list(payload.get("streams") or [])]
    video = next((stream for stream in streams if stream.codec_type == "video"), None)
    rotate = extract_rotate(video.tags) if video is not None else None
    is_vfr = (
        detect_vfr(video.avg_frame_rate, video.r_frame_rate)
        if video is not None
        else None
    )
    return MediaFacts(
        schema_version=MEDIA_FACTS_SCHEMA_VERSION,
        source_path=str(resolved),
        format_name=_optional_str(fmt.get("format_name")),
        format_long_name=_optional_str(fmt.get("format_long_name")),
        duration_s=_optional_float(fmt.get("duration")),
        size_bytes=_optional_int(fmt.get("size")),
        bit_rate=_optional_int(fmt.get("bit_rate")),
        streams=streams,
        has_video=any(stream.codec_type == "video" for stream in streams),
        has_audio=any(stream.codec_type == "audio" for stream in streams),
        is_vfr=is_vfr,
        rotate=rotate,
    )


def probe_file(source: Path, *, ffprobe: str | None = None) -> MediaFacts:
    """Probe a local media file and return versioned media facts."""
    payload = run_ffprobe_json(source, ffprobe=ffprobe)
    return media_facts_from_ffprobe(source, payload)


def media_facts_path(job_dir: Path) -> Path:
    """Return the on-disk path for a job's media facts."""
    return job_dir / MEDIA_FACTS_NAME


def write_media_facts(job_dir: Path, facts: MediaFacts) -> Path:
    """Atomically write media facts into a job directory."""
    path = media_facts_path(job_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(facts.to_dict(), indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return path


def load_media_facts(job_dir: Path) -> MediaFacts:
    """Load media facts previously written for a job."""
    path = media_facts_path(job_dir)
    if not path.is_file():
        raise ProbeError(f"media facts not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ProbeError(f"corrupt media facts: {path}")
    return MediaFacts.from_dict(data)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "N/A":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "N/A":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)
