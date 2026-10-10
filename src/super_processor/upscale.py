"""Local FlashVSR restoration upscale.

FFmpeg probes, trims, concatenates, and copies audio. It does not choose the
look. Gemini may only return a time range plus ``scale`` and ``strength``.
The real engine refuses to run without CUDA; tests use :class:`FakeUpscaleEngine`.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

UPSCALE_PLAN_FILE = "upscale_plan.json"
CHUNK_SECONDS = 8.0
CHUNK_OVERLAP_S = 0.5
_SCALES = frozenset({2, 4})
_WEIGHT_FILES = (
    "diffusion_pytorch_model_streaming_dmd.safetensors",
    "LQ_proj_in.ckpt",
    "TCDecoder.ckpt",
)


class UpscaleError(RuntimeError):
    """Raised when a local upscale cannot run or a param patch is invalid."""


@dataclass(frozen=True, slots=True)
class UpscaleSpan:
    """One source-time span and the knobs used to restore it."""

    start_s: float
    end_s: float
    scale: int = 2
    strength: float = 0.5
    source: str = "original"

    def __post_init__(self) -> None:
        scale = _require_scale(self.scale)
        strength = _require_strength(self.strength)
        start = float(self.start_s)
        end = float(self.end_s)
        if start < 0 or end <= start:
            raise UpscaleError("span end must be greater than span start")
        if self.source != "original":
            raise UpscaleError("spans must be restored from the original source")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "strength", strength)
        object.__setattr__(self, "start_s", start)
        object.__setattr__(self, "end_s", end)

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "scale": self.scale,
            "strength": self.strength,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Any) -> UpscaleSpan:
        if not isinstance(data, dict):
            raise UpscaleError("upscale span must be an object")
        try:
            return cls(
                start_s=float(data["start_s"]),
                end_s=float(data["end_s"]),
                scale=int(data.get("scale", 2)),
                strength=float(data.get("strength", 0.5)),
                source=str(data.get("source", "original")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise UpscaleError("upscale span is missing start_s or end_s") from exc


@dataclass(frozen=True, slots=True)
class UpscalePlan:
    """Ordered spans covering a job. Chunk execution is reserved, not run."""

    spans: tuple[UpscaleSpan, ...]
    pending: UpscaleSpan | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "spans": [span.to_dict() for span in self.spans],
            "chunk_seconds": CHUNK_SECONDS,
            "chunk_overlap_s": CHUNK_OVERLAP_S,
        }
        if self.pending is not None:
            payload["pending"] = self.pending.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: Any) -> UpscalePlan:
        if not isinstance(data, dict):
            raise UpscaleError("upscale plan must be an object")
        raw_spans = data.get("spans")
        if not isinstance(raw_spans, list) or not raw_spans:
            raise UpscaleError("upscale plan needs at least one span")
        pending_raw = data.get("pending")
        pending = UpscaleSpan.from_dict(pending_raw) if pending_raw else None
        return cls(
            spans=tuple(UpscaleSpan.from_dict(item) for item in raw_spans),
            pending=pending,
        )


@dataclass(frozen=True, slots=True)
class UpscaleRequest:
    """One engine call. The caller has already trimmed ``source`` when needed."""

    source: Path
    output: Path
    scale: int = 2
    strength: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", _require_scale(self.scale))
        object.__setattr__(self, "strength", _require_strength(self.strength))


class UpscaleEngine(Protocol):
    """Restore and upscale a clip. Implementations must not invent a grade."""

    def upscale(self, request: UpscaleRequest) -> Path:
        """Write ``request.output`` and return that path."""


def default_span(duration_s: float) -> UpscaleSpan:
    """First pass: the whole clip at 2× and mid strength."""
    if duration_s <= 0:
        raise UpscaleError("media has no duration")
    return UpscaleSpan(0.0, duration_s, scale=2, strength=0.5)


def planned_chunks(
    duration_s: float, *, scale: int = 2, strength: float = 0.5
) -> list[UpscaleSpan]:
    """Span list for a later chunked runner (about 8s pieces, 0.5s overlap)."""
    if duration_s <= 0:
        raise UpscaleError("media has no duration")
    if duration_s <= CHUNK_SECONDS:
        return [UpscaleSpan(0.0, duration_s, scale=scale, strength=strength)]
    spans: list[UpscaleSpan] = []
    cursor = 0.0
    step = CHUNK_SECONDS - CHUNK_OVERLAP_S
    while cursor < duration_s - 0.05:
        end = min(duration_s, cursor + CHUNK_SECONDS)
        spans.append(UpscaleSpan(cursor, end, scale=scale, strength=strength))
        if end >= duration_s - 0.05:
            break
        cursor += step
    return spans


def flashvsr_knobs(strength: float) -> tuple[float, int]:
    """Map strength to FlashVSR stability knobs. Lower strength invents less."""
    value = _require_strength(strength)
    sparse_ratio = 2.0 if value < 0.75 else 1.5
    local_range = 11 if value < 0.55 else 9
    return sparse_ratio, local_range


def models_dir() -> Path:
    """Gitignored weights root. Override with ``SUPER_PROCESSOR_MODELS``."""
    raw = os.environ.get("SUPER_PROCESSOR_MODELS", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.cwd() / ".dogfood" / "models"


def flashvsr_home() -> Path:
    """FlashVSR checkout. Override with ``FLASHVSR_HOME``."""
    raw = os.environ.get("FLASHVSR_HOME", "").strip()
    if raw:
        return Path(raw).expanduser()
    return models_dir() / "FlashVSR"


def flashvsr_weights_dir(home: Path | None = None) -> Path:
    root = home or flashvsr_home()
    return root / "examples" / "WanVSR" / "FlashVSR-v1.1"


def resolve_flashvsr_weights(home: Path | None = None) -> Path:
    """Require the v1.1 tiny weight trio or raise a wizard-facing error."""
    folder = flashvsr_weights_dir(home)
    missing = [name for name in _WEIGHT_FILES if not (folder / name).is_file()]
    if missing:
        listed = ", ".join(missing)
        raise UpscaleError(
            "FlashVSR v1.1 weights were not found "
            f"({listed}) under {folder}. Clone "
            "https://github.com/OpenImagingLab/FlashVSR into "
            f"{flashvsr_home()} and download JunhaoZhuang/FlashVSR-v1.1 into "
            "examples/WanVSR/FlashVSR-v1.1. CUDA is required; there is no "
            "FFmpeg grade fallback."
        )
    return folder


def cuda_available() -> bool:
    """True when PyTorch can see an NVIDIA GPU."""
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return False
    cuda = getattr(torch, "cuda", None)
    available = getattr(cuda, "is_available", None)
    return bool(available and available())


def save_upscale_plan(job_dir: Path, plan: UpscalePlan) -> Path:
    path = job_dir / UPSCALE_PLAN_FILE
    path.write_text(json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_upscale_plan(job_dir: Path) -> UpscalePlan | None:
    path = job_dir / UPSCALE_PLAN_FILE
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return UpscalePlan.from_dict(data)


def clamp_span(span: UpscaleSpan, duration_s: float) -> UpscaleSpan:
    """Keep a Gemini patch inside the file."""
    start = min(max(0.0, span.start_s), max(0.0, duration_s - 0.1))
    end = min(max(start + 0.1, span.end_s), duration_s)
    return UpscaleSpan(start, end, scale=span.scale, strength=span.strength)


def replace_overlapping(
    plan: UpscalePlan, patch: UpscaleSpan, duration_s: float
) -> UpscalePlan:
    """Store a range revise. A scale change replaces the whole timeline."""
    patch = clamp_span(patch, duration_s)
    scales = {span.scale for span in plan.spans}
    covers_all = patch.start_s <= 0.05 and patch.end_s >= duration_s - 0.05
    if covers_all or patch.scale not in scales:
        return UpscalePlan(spans=(patch,), pending=patch)
    kept = tuple(
        span
        for span in plan.spans
        if span.end_s <= patch.start_s + 0.05 or span.start_s >= patch.end_s - 0.05
    )
    return UpscalePlan(spans=(*kept, patch), pending=patch)


class FakeUpscaleEngine:
    """Test double that scales with FFmpeg so splices have frames."""

    def __init__(self, *, ffmpeg_bin: str = "ffmpeg") -> None:
        self.ffmpeg_bin = ffmpeg_bin

    def upscale(self, request: UpscaleRequest) -> Path:
        request.output.parent.mkdir(parents=True, exist_ok=True)
        _run_ffmpeg(
            [
                self.ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(request.source),
                "-vf",
                f"scale=iw*{request.scale}:ih*{request.scale}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(request.output),
            ]
        )
        if not request.output.is_file():
            raise UpscaleError("fake upscale produced no output")
        return request.output


class FlashVsrEngine:
    """Temporal restoration via the FlashVSR v1.1 tiny pipeline."""

    def upscale(self, request: UpscaleRequest) -> Path:
        if not cuda_available():
            raise UpscaleError(
                "FlashVSR needs an NVIDIA GPU with CUDA. Refusing to fall "
                "back to an FFmpeg grade. Install the gpu extra and a driver."
            )
        home = flashvsr_home()
        infer = home / "examples" / "WanVSR" / "infer_flashvsr_v1.1_tiny.py"
        if not infer.is_file():
            raise UpscaleError(
                "FlashVSR checkout is missing "
                f"{infer}. Clone OpenImagingLab/FlashVSR into {home}."
            )
        resolve_flashvsr_weights(home)
        return _run_flashvsr(request, home=home, infer=infer)


def _run_flashvsr(request: UpscaleRequest, *, home: Path, infer: Path) -> Path:
    """Load the upstream tiny script and run one file. Weights stay relative."""
    wan = infer.parent
    sparse_ratio, local_range = flashvsr_knobs(request.strength)
    spec_name = "flashvsr_infer_tiny"
    loader = importlib.util.spec_from_file_location(spec_name, infer)
    if loader is None or loader.loader is None:
        raise UpscaleError(f"could not load {infer}")
    module = importlib.util.module_from_spec(loader)
    previous = os.getcwd()
    try:
        os.chdir(wan)
        loader.loader.exec_module(module)
        pipe = module.init_pipeline()
        low, height, width, frames, fps = module.prepare_input_tensor(
            str(request.source),
            scale=float(request.scale),
            dtype=module.torch.bfloat16,
            device="cuda",
        )
        video = pipe(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=0,
            LQ_video=low,
            num_frames=frames,
            height=height,
            width=width,
            is_full_block=False,
            if_buffer=True,
            topk_ratio=sparse_ratio * 768 * 1280 / (height * width),
            kv_ratio=3.0,
            local_range=local_range,
            color_fix=True,
        )
        pictures = module.tensor2video(video)
        request.output.parent.mkdir(parents=True, exist_ok=True)
        module.save_video(pictures, str(request.output), fps=fps, quality=6)
    except UpscaleError:
        raise
    except Exception as exc:
        raise UpscaleError(f"FlashVSR failed: {exc}") from exc
    finally:
        os.chdir(previous)
    if not request.output.is_file():
        raise UpscaleError("FlashVSR produced no output")
    return request.output


def trim_source(
    source: Path,
    dest: Path,
    start_s: float,
    end_s: float,
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    """Cut a source range. Re-encode so the splice does not depend on keyframes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_s:.3f}",
            "-to",
            f"{end_s:.3f}",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(dest),
        ]
    )
    return dest


def apply_range_revise(
    source: Path,
    current: Path,
    span: UpscaleSpan,
    output: Path,
    *,
    engine: UpscaleEngine,
    ffmpeg_bin: str = "ffmpeg",
    duration_s: float,
) -> Path:
    """Re-upscale ``span`` from the original source and splice it into ``current``."""
    work = output.parent / "range_work"
    work.mkdir(parents=True, exist_ok=True)
    covers = span.start_s <= 0.05 and span.end_s >= duration_s - 0.05
    if covers or not current.is_file():
        engine.upscale(
            UpscaleRequest(source, output, scale=span.scale, strength=span.strength)
        )
        mux_source_audio(output, source, ffmpeg_bin=ffmpeg_bin)
        return output
    clip = trim_source(
        source, work / "range.mp4", span.start_s, span.end_s, ffmpeg_bin=ffmpeg_bin
    )
    middle = engine.upscale(
        UpscaleRequest(
            clip,
            work / "range_up.mp4",
            scale=span.scale,
            strength=span.strength,
        )
    )
    parts: list[Path] = []
    if span.start_s > 0.05:
        head = work / "head.mp4"
        trim_source(current, head, 0.0, span.start_s, ffmpeg_bin=ffmpeg_bin)
        parts.append(head)
    parts.append(middle)
    if span.end_s < duration_s - 0.05:
        tail = work / "tail.mp4"
        trim_source(current, tail, span.end_s, duration_s, ffmpeg_bin=ffmpeg_bin)
        parts.append(tail)
    silent = work / "spliced.mp4"
    _concat(parts, silent, ffmpeg_bin=ffmpeg_bin)
    silent.replace(output)
    mux_source_audio(output, source, ffmpeg_bin=ffmpeg_bin)
    return output


def mux_source_audio(
    video: Path,
    source: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    """Copy the source audio onto an upscaled picture. No audio is fine."""
    mixed = video.with_suffix(".mux.mp4")
    completed = subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(mixed),
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0 or not mixed.is_file():
        return video
    mixed.replace(video)
    return video


def _concat(parts: list[Path], dest: Path, *, ffmpeg_bin: str) -> None:
    from .render import build_concat_argv, write_concat_list

    listing = write_concat_list(dest.parent / "splice.txt", parts)
    _run_ffmpeg(build_concat_argv(listing, dest, ffmpeg_bin=ffmpeg_bin))


def _run_ffmpeg(command: list[str]) -> None:
    try:
        completed = subprocess.run(command, check=False, capture_output=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise UpscaleError(f"failed to start ffmpeg: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or b"").decode("utf-8", errors="replace").strip()
        raise UpscaleError(detail or "ffmpeg failed")


def _require_scale(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise UpscaleError("scale must be 2 or 4")
    if value not in _SCALES:
        raise UpscaleError("scale must be 2 or 4")
    return value


def _require_strength(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UpscaleError("strength must be a number from 0.0 to 1.0")
    strength = float(value)
    if strength < 0.0 or strength > 1.0:
        raise UpscaleError("strength must be between 0.0 and 1.0")
    return strength
