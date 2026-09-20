"""Engineering-owned FFmpeg argv templates for validated Recipe documents.

Paths are always separate argument-list elements. Nothing is interpolated into
a shell string.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .doctor import which
from .recipe import OpName, Recipe, RecipeOp, TargetMode

PREVIEW_FILE_NAME = "preview.mp4"
OUTPUT_FILE_NAME = "output.mp4"
TRANSFORMS_FILE_NAME = "transforms.trf"

_FILTER_CACHE: dict[str, bool] | None = None


class TemplateError(ValueError):
    """Raised when a recipe cannot be mapped to an FFmpeg plan."""


def ffmpeg_has_filter(name: str, *, ffmpeg_bin: str | None = None) -> bool:
    """Return whether the local ffmpeg build exposes a named filter."""
    global _FILTER_CACHE
    binary = ffmpeg_bin or which("ffmpeg") or "ffmpeg"
    if _FILTER_CACHE is None:
        _FILTER_CACHE = {}
        try:
            completed = subprocess.run(
                [binary, "-hide_banner", "-filters"],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            listing = completed.stdout or ""
        except (OSError, subprocess.SubprocessError):
            listing = ""
        for line in listing.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] not in _FILTER_CACHE:
                _FILTER_CACHE[parts[1]] = True
    return bool(_FILTER_CACHE.get(name))


@dataclass(slots=True)
class FFmpegStep:
    """One subprocess invocation expressed as an argv list."""

    name: str
    argv: list[str]
    writes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "argv": list(self.argv),
            "writes": list(self.writes),
        }


@dataclass(slots=True)
class FFmpegPlan:
    """Ordered FFmpeg steps that realize a recipe."""

    mode: str
    input_path: str
    output_path: str
    steps: list[FFmpegStep]
    filter_graph: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "filter_graph": self.filter_graph,
            "steps": [step.to_dict() for step in self.steps],
        }


def _ffmpeg_bin() -> str:
    path = which("ffmpeg")
    if path is None:
        raise TemplateError("ffmpeg not found on PATH")
    return path


def _op_map(recipe: Recipe) -> dict[OpName, RecipeOp]:
    return {op.op: op for op in recipe.enabled_ops()}


def _look_filters(ops: dict[OpName, RecipeOp]) -> list[str]:
    filters: list[str] = []

    wb = ops.get(OpName.WHITE_BALANCE)
    if wb is not None:
        temperature = float(wb.params.get("temperature", 6500.0))
        tint = float(wb.params.get("tint", 0.0))
        filters.append(f"colortemperature=temperature={temperature:g}:mix=1")
        if tint != 0.0:
            # Map tint into a mild green/magenta colorbalance nudge.
            amount = max(-1.0, min(1.0, tint / 100.0)) * 0.25
            filters.append(f"colorbalance=gm={amount:g}")

    contrast = ops.get(OpName.CONTRAST)
    if contrast is not None:
        c = float(contrast.params.get("contrast", 1.0))
        b = float(contrast.params.get("brightness", 0.0))
        g = float(contrast.params.get("gamma", 1.0))
        filters.append(f"eq=contrast={c:g}:brightness={b:g}:gamma={g:g}")

    denoise = ops.get(OpName.DENOISE)
    if denoise is not None:
        strength = float(denoise.params.get("strength", 0.0))
        spatial = max(0.0, min(8.0, strength * 6.0))
        temporal = max(0.0, min(6.0, strength * 4.5))
        filters.append(
            f"hqdn3d={spatial:g}:{spatial * 0.75:g}:{temporal:g}:{temporal * 0.75:g}"
        )

    return filters


def _stabilize_detect(
    ops: dict[OpName, RecipeOp],
    transforms: Path,
    *,
    ffmpeg_bin: str | None = None,
) -> str | None:
    stabilize = ops.get(OpName.STABILIZE)
    if stabilize is None:
        return None
    if not ffmpeg_has_filter("vidstabdetect", ffmpeg_bin=ffmpeg_bin):
        # Homebrew/Chocolatey builds often omit vid.stab; deshake is single-pass.
        return None
    shakiness = int(float(stabilize.params.get("shakiness", 5.0)))
    return (
        f"vidstabdetect=shakiness={shakiness}:accuracy=15:"
        f"result={_escape_filter_path(transforms)}"
    )


def _stabilize_transform(
    ops: dict[OpName, RecipeOp],
    transforms: Path,
    *,
    ffmpeg_bin: str | None = None,
) -> str | None:
    stabilize = ops.get(OpName.STABILIZE)
    if stabilize is None:
        return None
    if ffmpeg_has_filter("vidstabtransform", ffmpeg_bin=ffmpeg_bin):
        smoothing = int(float(stabilize.params.get("smoothing", 10.0)))
        max_crop = float(stabilize.params.get("max_crop_pct", 10.0))
        zoom = max(0.0, min(30.0, max_crop))
        return (
            f"vidstabtransform=input={_escape_filter_path(transforms)}:"
            f"smoothing={smoothing}:crop=black:zoom={zoom:g}:optzoom=0"
        )
    # Prefer deshake when vid.stab is missing or filter probing is unavailable
    # (cibuildwheel containers often have no ffmpeg on PATH during unit tests).
    return "deshake"


def _reframe_filter(ops: dict[OpName, RecipeOp], recipe: Recipe) -> str | None:
    reframe = ops.get(OpName.REFRAME_VERTICAL)
    if reframe is None and recipe.target.export.aspect != "9:16":
        return None
    if reframe is None:
        return None

    padding = float(reframe.params.get("padding", 0.0))
    max_height = recipe.target.export.max_height or 1920
    size_op = ops.get(OpName.ENCODE_HEVC_SIZE_CAP)
    if size_op is not None and "max_height" in size_op.params:
        max_height = int(float(size_op.params["max_height"]))

    target_h = int(max_height)
    target_w = int(round(target_h * 9 / 16))
    pad_px = int(round(min(target_w, target_h) * padding))
    inner_w = max(2, target_w - 2 * pad_px)
    inner_h = max(2, target_h - 2 * pad_px)
    # Keep even dimensions for yuv420p / libx265.
    inner_w -= inner_w % 2
    inner_h -= inner_h % 2
    target_w -= target_w % 2
    target_h -= target_h % 2

    crop_keys = ("crop_x", "crop_y", "crop_w", "crop_h")
    if all(key in reframe.params for key in crop_keys):
        crop_x = int(float(reframe.params["crop_x"]))
        crop_y = int(float(reframe.params["crop_y"]))
        crop_w = int(float(reframe.params["crop_w"]))
        crop_h = int(float(reframe.params["crop_h"]))
        crop_w -= crop_w % 2
        crop_h -= crop_h % 2
        crop_x -= crop_x % 2
        crop_y -= crop_y % 2
        return (
            f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
            f"scale={inner_w}:{inner_h},"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"
        )

    # Fallback: center-crop via scale+crop when no planned window is present.
    return (
        f"scale={inner_w}:{inner_h}:force_original_aspect_ratio=increase,"
        f"crop={inner_w}:{inner_h},"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"
    )


def _escape_filter_path(path: Path) -> str:
    """Escape a filesystem path for use inside an FFmpeg filtergraph."""
    text = path.resolve().as_posix()
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace(",", "\\,")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _join_filters(parts: list[str]) -> str:
    return ",".join(part for part in parts if part)


def _size_cap_bitrate_kbps(recipe: Recipe, duration_s: float | None) -> int | None:
    ops = _op_map(recipe)
    max_size_mb = recipe.target.export.max_size_mb
    size_op = ops.get(OpName.ENCODE_HEVC_SIZE_CAP)
    if size_op is not None and "max_size_mb" in size_op.params:
        max_size_mb = float(size_op.params["max_size_mb"])
    if max_size_mb is None:
        return None
    if duration_s is None or duration_s <= 0:
        raise TemplateError("size-cap encode requires a positive media duration")
    budget_bits = max_size_mb * 1024 * 1024 * 8
    # Reserve ~128 kbps for audio when re-encoding.
    video_bits = max(0.0, budget_bits - (128_000 * duration_s))
    kbps = int(video_bits / duration_s / 1000.0)
    return max(100, kbps)


def _encode_args(recipe: Recipe, duration_s: float | None) -> list[str]:
    args = [
        "-c:v",
        recipe.encode.video_codec,
        "-preset",
        recipe.encode.preset,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    bitrate = _size_cap_bitrate_kbps(recipe, duration_s)
    if bitrate is not None:
        args.extend(
            [
                "-b:v",
                f"{bitrate}k",
                "-maxrate",
                f"{bitrate}k",
                "-bufsize",
                f"{bitrate * 2}k",
            ]
        )
    else:
        args.extend(["-crf", str(recipe.encode.crf)])

    if recipe.encode.audio_action == "copy":
        args.extend(["-c:a", "copy"])
    else:
        # Default copy_or_aac: re-encode audio for filtergraph safety.
        args.extend(["-c:a", "aac", "-b:a", "128k"])
    return args


def build_ffmpeg_plan(
    recipe: Recipe,
    *,
    output_path: Path,
    work_dir: Path,
    duration_s: float | None = None,
    ffmpeg_bin: str | None = None,
) -> FFmpegPlan:
    """Translate a validated recipe into ordered FFmpeg argv steps."""
    binary = ffmpeg_bin or _ffmpeg_bin()
    input_path = Path(recipe.source_path).expanduser().resolve()
    if not input_path.is_file():
        raise TemplateError(f"source is not a readable file: {input_path}")

    output = output_path.expanduser().resolve()
    work_dir = work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    transforms = work_dir / TRANSFORMS_FILE_NAME

    mode = recipe.target.mode.value
    ops = _op_map(recipe)
    look = _look_filters(ops)
    detect = _stabilize_detect(ops, transforms, ffmpeg_bin=binary)
    transform = _stabilize_transform(ops, transforms, ffmpeg_bin=binary)
    reframe = _reframe_filter(ops, recipe)

    steps: list[FFmpegStep] = []

    preview_seek: list[str] = []
    if recipe.target.mode is TargetMode.PREVIEW:
        window = recipe.target.preview_window
        preview_seek = [
            "-ss",
            f"{window.start_s:g}",
            "-t",
            f"{window.duration_s:g}",
        ]

    if detect is not None:
        detect_graph = _join_filters([*look, detect])
        detect_argv = [
            binary,
            "-hide_banner",
            "-y",
            *preview_seek,
            "-i",
            str(input_path),
            "-vf",
            detect_graph,
            "-an",
            "-f",
            "null",
            "-",
        ]
        steps.append(
            FFmpegStep(
                name="stabilize_detect",
                argv=detect_argv,
                writes=[str(transforms)],
            )
        )

    encode_filters = list(look)
    if transform is not None:
        encode_filters.append(transform)
    if reframe is not None:
        encode_filters.append(reframe)
    filter_graph = _join_filters(encode_filters)

    encode_argv = [
        binary,
        "-hide_banner",
        "-y",
        "-progress",
        "pipe:1",
        "-nostats",
        *preview_seek,
        "-i",
        str(input_path),
    ]

    if filter_graph:
        encode_argv.extend(["-vf", filter_graph])
    encode_argv.extend(_encode_args(recipe, duration_s))
    # Placeholder output; worker replaces with a partial path.
    encode_argv.append(str(output))

    steps.append(
        FFmpegStep(
            name="encode",
            argv=encode_argv,
            writes=[str(output)],
        )
    )

    return FFmpegPlan(
        mode=mode,
        input_path=str(input_path),
        output_path=str(output),
        steps=steps,
        filter_graph=filter_graph,
    )


def default_output_path(job_dir: Path, recipe: Recipe) -> Path:
    """Return the conventional preview or final output path for a job."""
    if recipe.target.mode is TargetMode.PREVIEW:
        return job_dir / PREVIEW_FILE_NAME
    return job_dir / OUTPUT_FILE_NAME
