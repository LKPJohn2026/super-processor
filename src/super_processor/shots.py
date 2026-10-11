"""Loop 1: find the shots in a video, measure each one, and keep the list.

FFmpeg does the work in two kinds of pass:

1. One pass over a small copy of the picture with ``scdet`` finds the cuts.
2. One short pass per shot measures it at two frames a second:
   ``blockdetect`` (compression blocks), ``blurdetect`` (softness),
   ``signalstats`` (brightness and contrast), and the difference between
   each frame and a spatially denoised copy of it (noise). Running one pass
   per shot keeps filters that average over everything they have seen
   (``blockdetect`` does) from mixing one shot into the next.

The thresholds that turn numbers into issue hints are rough and only hints:
Gemini sees a still of each shot next to the numbers and gives the final
label, and the editor approves the list before anything renders.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .regions import Region, regions_from_dicts
from .upscale import UpscalePlan, UpscaleSpan

SHOTS_FILE = "shots.json"
SHOTS_SCHEMA_VERSION = 1
STILLS_DIR = "shot_stills"
# scdet score (0-100) above which a frame starts a new shot.
CUT_THRESHOLD = 10.0
# Shots shorter than this are folded into a neighbour: a flash or a one
# second insert is not worth its own settings.
MIN_SHOT_S = 1.0
# Upper bound on shots in one list, so the review screen and the Gemini
# request stay a sensible size. The shortest shots are merged first.
MAX_SHOTS = 40
MEASURE_FPS = 2
STILL_WIDTH = 480

# Problems a shot can have. Gemini may only use these names.
ISSUES = (
    "blocky",
    "noisy",
    "soft",
    "dark",
    "overexposed",
    "flat",
    "color_cast",
    "shaky",
)
# Things in the picture that restoration gets wrong most visibly. Later
# steps protect them.
CONTENTS = ("faces", "hands", "text", "fine_pattern")

# Rough hint thresholds; see the module docstring.
_BLOCKY_AT = 18.0
_NOISY_AT = 2.0
_SOFT_AT = 6.0
_DARK_BELOW = 0.25
_BRIGHT_ABOVE = 0.75
_FLAT_BELOW = 0.35

_METADATA_LINE = re.compile(r"\[Parsed_metadata_\d+ @ [^\]]+\]\s*(.*)$")
_PTS_TIME = re.compile(r"pts_time:([-\d.eE+]+)")


class ShotError(RuntimeError):
    """Raised when shots cannot be found, measured, or edited."""


@dataclass(frozen=True, slots=True)
class ShotMetrics:
    """Median measurements for one shot. Brightness and contrast are 0-1."""

    blockiness: float = 0.0
    blur: float = 0.0
    noise: float = 0.0
    brightness: float = 0.5
    contrast: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return {
            "blockiness": round(self.blockiness, 3),
            "blur": round(self.blur, 3),
            "noise": round(self.noise, 3),
            "brightness": round(self.brightness, 3),
            "contrast": round(self.contrast, 3),
        }

    @classmethod
    def from_dict(cls, data: Any) -> ShotMetrics:
        if not isinstance(data, dict):
            return cls()
        values: dict[str, float] = {}
        for name in ("blockiness", "blur", "noise", "brightness", "contrast"):
            raw = data.get(name)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                values[name] = float(raw)
        return cls(**values)

    def hints(self) -> tuple[str, ...]:
        """Issues the numbers alone suggest."""
        found: list[str] = []
        if self.blockiness >= _BLOCKY_AT:
            found.append("blocky")
        if self.noise >= _NOISY_AT:
            found.append("noisy")
        if self.blur >= _SOFT_AT:
            found.append("soft")
        if self.brightness < _DARK_BELOW:
            found.append("dark")
        if self.brightness > _BRIGHT_ABOVE:
            found.append("overexposed")
        if self.contrast < _FLAT_BELOW:
            found.append("flat")
        return tuple(found)


def merged_metrics(
    a: ShotMetrics, a_s: float, b: ShotMetrics, b_s: float
) -> ShotMetrics:
    """Duration-weighted mean of two shots' metrics."""
    total = a_s + b_s
    if total <= 0:
        return a

    def mix(x: float, y: float) -> float:
        return (x * a_s + y * b_s) / total

    return ShotMetrics(
        blockiness=mix(a.blockiness, b.blockiness),
        blur=mix(a.blur, b.blur),
        noise=mix(a.noise, b.noise),
        brightness=mix(a.brightness, b.brightness),
        contrast=mix(a.contrast, b.contrast),
    )


@dataclass(frozen=True, slots=True)
class Shot:
    """One shot: a time range, its measurements, and its label."""

    start_s: float
    end_s: float
    metrics: ShotMetrics = field(default_factory=ShotMetrics)
    label: str = ""
    issues: tuple[str, ...] = ()
    contains: tuple[str, ...] = ()
    still: str = ""
    # Protection boxes, placed while planning (see :mod:`.regions`).
    regions: tuple[Region, ...] = ()

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_s": round(self.start_s, 6),
            "end_s": round(self.end_s, 6),
            "metrics": self.metrics.to_dict(),
            "label": self.label,
            "issues": list(self.issues),
            "contains": list(self.contains),
            "still": self.still,
            "regions": [region.to_dict() for region in self.regions],
        }

    @classmethod
    def from_dict(cls, data: Any) -> Shot:
        if not isinstance(data, dict):
            raise ShotError("a shot must be an object")
        try:
            start = float(data["start_s"])
            end = float(data["end_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ShotError("a shot needs start_s and end_s") from exc
        return cls(
            start_s=start,
            end_s=end,
            metrics=ShotMetrics.from_dict(data.get("metrics")),
            label=str(data.get("label", "")),
            issues=_known(data.get("issues"), ISSUES),
            contains=_known(data.get("contains"), CONTENTS),
            still=str(data.get("still", "")),
            regions=regions_from_dicts(data.get("regions")),
        )


@dataclass(slots=True)
class ShotList:
    """The shots of one job, in time order, covering the whole clip."""

    shots: list[Shot]
    label_error: str | None = None
    schema_version: int = SHOTS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label_error": self.label_error,
            "shots": [shot.to_dict() for shot in self.shots],
        }

    @classmethod
    def from_dict(cls, data: Any) -> ShotList:
        if not isinstance(data, dict) or not isinstance(data.get("shots"), list):
            raise ShotError("shots.json must hold a list of shots")
        error = data.get("label_error")
        return cls(
            shots=[Shot.from_dict(item) for item in data["shots"]],
            label_error=str(error) if error else None,
        )

    def to_plan(self) -> UpscalePlan:
        """One upscale span per shot, at the first-pass scale and strength."""
        if not self.shots:
            raise ShotError("there are no shots to render")
        return UpscalePlan(
            spans=tuple(UpscaleSpan(shot.start_s, shot.end_s) for shot in self.shots)
        )


def _known(raw: Any, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    seen: list[str] = []
    for item in raw:
        name = str(item).strip().lower()
        if name in allowed and name not in seen:
            seen.append(name)
    return tuple(seen)


def shots_path(job_dir: Path) -> Path:
    return job_dir / SHOTS_FILE


def save_shots(job_dir: Path, shots: ShotList) -> Path:
    path = shots_path(job_dir)
    path.write_text(json.dumps(shots.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_shots(job_dir: Path) -> ShotList | None:
    path = shots_path(job_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ShotError(f"shots.json is not valid JSON: {exc}") from exc
    return ShotList.from_dict(data)


def _run(command: list[str]) -> str:
    """Run FFmpeg and return its log (stderr), where ``metadata`` prints."""
    try:
        completed = subprocess.run(command, check=False, capture_output=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ShotError(f"failed to start ffmpeg: {exc}") from exc
    log = (completed.stderr or b"").decode("utf-8", errors="replace")
    if completed.returncode != 0:
        tail = log.strip().splitlines()[-3:]
        raise ShotError("ffmpeg failed: " + " ".join(tail))
    return log


def parse_metadata_log(log: str) -> list[tuple[float, dict[str, float]]]:
    """Frames printed by ``metadata=mode=print``: (time, {key: value})."""
    frames: list[tuple[float, dict[str, float]]] = []
    for line in log.splitlines():
        match = _METADATA_LINE.search(line)
        if match is None:
            continue
        text = match.group(1).strip()
        time_match = _PTS_TIME.search(text)
        if time_match is not None:
            frames.append((float(time_match.group(1)), {}))
            continue
        if "=" not in text or not frames:
            continue
        key, _, raw = text.partition("=")
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            frames[-1][1][key.strip()] = value
    return frames


def detect_cuts(source: Path, *, ffmpeg_bin: str = "ffmpeg") -> list[float]:
    """Times where a new shot starts, ascending, excluding 0."""
    log = _run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "info",
            "-i",
            str(source),
            "-an",
            "-sn",
            "-vf",
            f"scale=320:-2,scdet=threshold={CUT_THRESHOLD:g},"
            "metadata=mode=select:key=lavfi.scd.time,metadata=mode=print",
            "-f",
            "null",
            "-",
        ]
    )
    times = sorted({round(t, 6) for t, values in parse_metadata_log(log) if values})
    return [t for t in times if t > 0]


def measure_range(
    source: Path, start_s: float, end_s: float, *, ffmpeg_bin: str = "ffmpeg"
) -> ShotMetrics:
    """Median measurements over ``start_s``-``end_s`` of ``source``."""
    duration = max(end_s - start_s, 0.05)
    log = _run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "info",
            "-ss",
            f"{start_s:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(source),
            "-an",
            "-sn",
            "-filter_complex",
            # The shot's first frame is always measured, so a shot shorter
            # than one sample period still gets numbers.
            f"[0:v]fps={MEASURE_FPS}:start_time=0:round=down,format=yuv420p,"
            "split=2[a][b];[b]hqdn3d=4:3:0:0[d];"
            "[a][d]psnr,blockdetect,blurdetect,signalstats,metadata=mode=print",
            "-f",
            "null",
            "-",
        ]
    )
    frames = [values for _time, values in parse_metadata_log(log) if values]
    if not frames:
        return ShotMetrics()

    def median(key: str, default: float) -> float:
        values = [frame[key] for frame in frames if key in frame]
        return statistics.median(values) if values else default

    low = median("lavfi.signalstats.YLOW", 16.0)
    high = median("lavfi.signalstats.YHIGH", 235.0)
    return ShotMetrics(
        blockiness=median("lavfi.block", 0.0),
        blur=median("lavfi.blur", 0.0),
        noise=math.sqrt(max(median("lavfi.psnr.mse.y", 0.0), 0.0)),
        brightness=median("lavfi.signalstats.YAVG", 128.0) / 255.0,
        contrast=max(0.0, high - low) / 255.0,
    )


def shot_ranges(
    cuts: list[float],
    duration_s: float,
    *,
    min_shot_s: float = MIN_SHOT_S,
    max_shots: int = MAX_SHOTS,
) -> list[tuple[float, float]]:
    """Turn cut times into ranges covering ``0``-``duration_s``.

    Ranges shorter than ``min_shot_s`` join the previous range (the next one
    for the first). Then the shortest range joins its shorter neighbour
    until at most ``max_shots`` remain.
    """
    if duration_s <= 0:
        raise ShotError("the video has no duration")
    edges = [0.0, *(t for t in cuts if 0.0 < t < duration_s), duration_s]
    ranges = [(a, b) for a, b in zip(edges, edges[1:], strict=False) if b > a]
    ranges = _fold_short(ranges, min_shot_s)
    while len(ranges) > max(1, max_shots):
        index = min(range(len(ranges)), key=lambda i: ranges[i][1] - ranges[i][0])
        ranges = _join_with_neighbour(ranges, index)
    return ranges


def _fold_short(
    ranges: list[tuple[float, float]], min_shot_s: float
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for start, end in ranges:
        if out and end - start < min_shot_s:
            out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    if len(out) > 1 and out[0][1] - out[0][0] < min_shot_s:
        out[1] = (out[0][0], out[1][1])
        out.pop(0)
    return out


def _join_with_neighbour(
    ranges: list[tuple[float, float]], index: int
) -> list[tuple[float, float]]:
    if len(ranges) < 2:
        return ranges
    if index == 0:
        other = 1
    elif index == len(ranges) - 1:
        other = index - 1
    else:
        before = ranges[index - 1][1] - ranges[index - 1][0]
        after = ranges[index + 1][1] - ranges[index + 1][0]
        other = index - 1 if before <= after else index + 1
    first, second = sorted((index, other))
    joined = (ranges[first][0], ranges[second][1])
    return [*ranges[:first], joined, *ranges[second + 1 :]]


def extract_still(
    source: Path,
    at_s: float,
    dest: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    width: int = STILL_WIDTH,
) -> Path:
    """Save one frame near ``at_s`` as a small JPEG."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{max(0.0, at_s):.6f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            f"scale={width}:-2",
            "-q:v",
            "4",
            str(dest),
        ]
    )
    if not dest.is_file():
        raise ShotError(f"could not save a still at {at_s:.2f}s")
    return dest


def _still_for(
    source: Path, job_dir: Path, start_s: float, end_s: float, *, ffmpeg_bin: str
) -> str:
    name = f"{STILLS_DIR}/{int(round(start_s * 1000)):09d}.jpg"
    extract_still(source, (start_s + end_s) / 2, job_dir / name, ffmpeg_bin=ffmpeg_bin)
    return name


def find_shots(
    source: Path,
    job_dir: Path,
    duration_s: float,
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> ShotList:
    """Detect, measure, and still every shot. Labels are only metric hints."""
    cuts = detect_cuts(source, ffmpeg_bin=ffmpeg_bin)
    shots: list[Shot] = []
    for start, end in shot_ranges(cuts, duration_s):
        metrics = measure_range(source, start, end, ffmpeg_bin=ffmpeg_bin)
        shots.append(
            Shot(
                start_s=start,
                end_s=end,
                metrics=metrics,
                issues=metrics.hints(),
                still=_still_for(source, job_dir, start, end, ffmpeg_bin=ffmpeg_bin),
            )
        )
    return ShotList(shots=shots)


def merge_with_next(shots: ShotList, index: int) -> ShotList:
    """Join shot ``index`` and the one after it."""
    if not 0 <= index < len(shots.shots) - 1:
        raise ShotError("there is no next shot to merge with")
    a, b = shots.shots[index], shots.shots[index + 1]
    merged = Shot(
        start_s=a.start_s,
        end_s=b.end_s,
        metrics=merged_metrics(a.metrics, a.duration_s, b.metrics, b.duration_s),
        label=a.label or b.label,
        issues=tuple(dict.fromkeys((*a.issues, *b.issues))),
        contains=tuple(dict.fromkeys((*a.contains, *b.contains))),
        still=a.still,
    )
    rest = [*shots.shots[:index], merged, *shots.shots[index + 2 :]]
    return ShotList(shots=rest, label_error=shots.label_error)


def split_at(
    shots: ShotList,
    index: int,
    at_s: float,
    *,
    source: Path,
    job_dir: Path,
    ffmpeg_bin: str = "ffmpeg",
) -> ShotList:
    """Cut shot ``index`` in two at ``at_s`` and measure both halves."""
    if not 0 <= index < len(shots.shots):
        raise ShotError("no such shot")
    shot = shots.shots[index]
    if not shot.start_s + MIN_SHOT_S / 2 <= at_s <= shot.end_s - MIN_SHOT_S / 2:
        raise ShotError(
            f"split inside the shot, at least {MIN_SHOT_S / 2:g}s from its edges "
            f"({shot.start_s:.2f}s to {shot.end_s:.2f}s)"
        )
    halves: list[Shot] = []
    for start, end in ((shot.start_s, at_s), (at_s, shot.end_s)):
        metrics = measure_range(source, start, end, ffmpeg_bin=ffmpeg_bin)
        halves.append(
            Shot(
                start_s=start,
                end_s=end,
                metrics=metrics,
                label=shot.label,
                issues=shot.issues,
                contains=shot.contains,
                still=_still_for(source, job_dir, start, end, ffmpeg_bin=ffmpeg_bin),
            )
        )
    rest = [*shots.shots[:index], *halves, *shots.shots[index + 1 :]]
    return ShotList(shots=rest, label_error=shots.label_error)


def apply_labels(shots: ShotList, labels: list[dict[str, Any]]) -> ShotList:
    """Merge Gemini's per-shot labels in by index. Unknown names are dropped."""
    updated = list(shots.shots)
    for item in labels:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        if not 0 <= index < len(updated):
            continue
        shot = updated[index]
        label = " ".join(str(item.get("label", "")).split())[:80]
        updated[index] = Shot(
            start_s=shot.start_s,
            end_s=shot.end_s,
            metrics=shot.metrics,
            label=label or shot.label,
            issues=(
                _known(item.get("issues"), ISSUES)
                if isinstance(item.get("issues"), list)
                else shot.issues
            ),
            contains=_known(item.get("contains"), CONTENTS),
            still=shot.still,
        )
    return ShotList(shots=updated, label_error=None)
