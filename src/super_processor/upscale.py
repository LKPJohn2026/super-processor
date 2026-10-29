"""Local FlashVSR restoration upscale.

FFmpeg probes, trims, concatenates, and copies audio. Each span (a shot) also
carries a :class:`~super_processor.look.ShotLook`: bounded FFmpeg clean-up
before FlashVSR and finishing after it, always in that order. Gemini may only
return a time range plus ``scale`` and ``strength``.
Shapes, colour, and layout always come from a plain lanczos upscale of the
same (cleaned) frames; ``strength`` is how much of FlashVSR's fine detail is
added on top, so 0.0 invents nothing and 1.0 adds all of it. Inside protected
regions (faces, hands, text) the share is lower. FlashVSR itself always runs
at its upstream settings.
The real engine refuses to run without CUDA; tests use :class:`FakeUpscaleEngine`.

Work clips handed to the engine are encoded losslessly. The engine output is
encoded once into the delivery settings, with a keyframe every second. A range
revise snaps outward to those keyframes and stream-copies the head and tail, so
the parts of the video a note did not name are never encoded again.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from .look import LookError, ShotLook
from .regions import Region, regions_from_dicts, strength_expr

UPSCALE_PLAN_FILE = "upscale_plan.json"
CHUNK_SECONDS = 8.0
CHUNK_OVERLAP_S = 0.5
DELIVERY_VERSION = 1
DELIVERY_CRF = 16
KEYFRAME_INTERVAL_S = 1.0
_KEY_EPSILON_S = 0.02
# FlashVSR works in fixed-size frame blocks and may pad or drop a few frames
# at the end of a clip. Up to this many (or 2% of the clip) are conformed back
# to the source count; more than that means the clip is misaligned.
FRAME_SLACK = 8
# Shapes, colour, and layout come from the plain upscale of the source; only
# detail finer than this (a Gaussian sigma, in source pixels) comes from
# FlashVSR. FlashVSR can sharpen what is there but cannot move, reshape, or
# recolour it.
STRUCTURE_SIGMA = 1.5
# 1 is "do not upscale": the clip keeps its size, goes through clean-up and
# finishing, and never reaches FlashVSR (see :func:`restore_clip`).
_SCALES = frozenset({1, 2, 4})
# The upstream script is imported with a chdir into its folder, which is
# process-wide. Only one FlashVSR load or inference may run at a time.
_FLASHVSR_LOCK = threading.Lock()
_WEIGHT_FILES = (
    "diffusion_pytorch_model_streaming_dmd.safetensors",
    "LQ_proj_in.ckpt",
    "TCDecoder.ckpt",
)


ProgressFn = Callable[[float], None]


def _part_of(
    on_progress: ProgressFn | None, start: float, share: float
) -> ProgressFn | None:
    """Map a part's own 0-1 progress onto ``start`` to ``start + share``."""
    if on_progress is None:
        return None
    report = on_progress

    def part(done: float) -> None:
        report(start + share * min(max(done, 0.0), 1.0))

    return part


class UpscaleError(RuntimeError):
    """Raised when a local upscale cannot run or a param patch is invalid."""


def require_supported_timing(source: Path, *, ffmpeg_bin: str = "ffmpeg") -> None:
    """Preflight original media before GPU work or replacing any output."""
    from .probe import ProbeError, require_cfr_timing

    try:
        require_cfr_timing(source, ffprobe=_ffprobe_for(ffmpeg_bin))
    except ProbeError as exc:
        raise UpscaleError(str(exc)) from exc


class GpuOutOfMemory(UpscaleError):
    """FlashVSR ran out of GPU memory; a shorter piece may still fit."""


# FlashVSR's working memory grows with the frames in one call: at 2x of
# 854x480, 16 frames peak at 15.5 GiB and 24 at 16.9 GiB, against 3.4 GiB of
# weights. When a clip does not fit (another program holds GPU memory), it is
# retried in pieces of half the length, down to this many frames; finished
# pieces go to disk, so memory follows the piece, not the clip.
FLASHVSR_MIN_CHUNK_FRAMES = 17


def _flashvsr_failure(exc: BaseException) -> UpscaleError:
    """The error to raise for a failed FlashVSR call, holding no GPU memory.

    The traceback keeps every frame's locals alive, and with them the input
    and output tensors and the pipeline. Releasing the GPU while it exists
    frees nothing, and an error raised ``from`` it keeps it after that: a
    failed preview held 13 GiB until the server restarted. So the frames are
    cleared and the new error does not chain to it.
    """
    import traceback

    # The system-RAM allocator refuses past its limit with std::bad_alloc.
    out_of_memory = type(exc).__name__ in ("OutOfMemoryError", "MemoryError") or any(
        sign in str(exc).lower() for sign in ("out of memory", "bad_alloc")
    )
    text = str(exc).splitlines()[0] if str(exc) else repr(exc)
    message = str(exc) if isinstance(exc, UpscaleError) else f"FlashVSR failed: {text}"
    traceback.clear_frames(exc.__traceback__)
    exc.__traceback__ = None
    if out_of_memory and not isinstance(exc, GpuOutOfMemory):
        return GpuOutOfMemory(message)
    return exc if isinstance(exc, UpscaleError) else UpscaleError(message)


@dataclass(frozen=True, slots=True)
class UpscaleSpan:
    """One source-time span and the knobs used to restore it."""

    start_s: float
    end_s: float
    scale: int = 2
    strength: float = 0.5
    source: str = "original"
    look: ShotLook = field(default_factory=ShotLook)
    regions: tuple[Region, ...] = ()
    # False: skip FlashVSR and only resize this span (see :func:`restore_clip`).
    use_upscale: bool = True

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "scale": self.scale,
            "strength": self.strength,
            "source": self.source,
            "look": self.look.to_dict(),
            "regions": [region.to_dict() for region in self.regions],
            "use_upscale": self.use_upscale,
        }

    @classmethod
    def from_dict(cls, data: Any) -> UpscaleSpan:
        if not isinstance(data, dict):
            raise UpscaleError("upscale span must be an object")
        try:
            look = ShotLook.from_dict(data.get("look"))
        except LookError as exc:
            raise UpscaleError(f"invalid shot look: {exc}") from exc
        try:
            return cls(
                start_s=float(data["start_s"]),
                end_s=float(data["end_s"]),
                scale=int(data.get("scale", 2)),
                strength=float(data.get("strength", 0.5)),
                source=str(data.get("source", "original")),
                look=look,
                regions=regions_from_dicts(data.get("regions")),
                use_upscale=data.get("use_upscale") is not False,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise UpscaleError("upscale span is missing start_s or end_s") from exc


@dataclass(frozen=True, slots=True)
class UpscalePlan:
    """Ordered spans covering a job. Long clips run as overlapping chunks."""

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
    look: ShotLook = field(default_factory=ShotLook)
    regions: tuple[Region, ...] = ()
    # Called with the share of this clip done (0 to 1) as an engine works
    # through it; FlashVSR reports once per chunk.
    on_progress: ProgressFn | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", _require_scale(self.scale))
        object.__setattr__(self, "strength", _require_strength(self.strength))


class UpscaleEngine(Protocol):
    """Restore and upscale a clip.

    Implementations apply ``request.look.clean_filter()`` to the frames they
    read and nothing else; finishing happens in the delivery encode.
    """

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
    """About 8s pieces with 0.5s overlap, used when a clip is longer."""
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


# Upstream FlashVSR v1.1 tiny defaults. These trade attention cost against
# quality; they are not a restoration strength, so they stay fixed.
FLASHVSR_SPARSE_RATIO = 2.0
FLASHVSR_LOCAL_RANGE = 11


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
    return replace(span, start_s=start, end_s=end)


def span_at(plan: UpscalePlan, time_s: float) -> UpscaleSpan | None:
    """The latest span covering ``time_s``."""
    for span in reversed(plan.spans):
        if span.start_s - 0.05 <= time_s < span.end_s:
            return span
    return None


def look_at(plan: UpscalePlan, time_s: float) -> ShotLook:
    """The look of the latest span covering ``time_s``, or a neutral one."""
    span = span_at(plan, time_s)
    return span.look if span is not None else ShotLook()


def _covers_all(span: UpscaleSpan, duration_s: float) -> bool:
    return span.start_s <= 0.05 and span.end_s >= duration_s - 0.05


def replace_overlapping(
    plan: UpscalePlan,
    patch: UpscaleSpan,
    duration_s: float,
    *,
    preserve_shot_settings: bool = False,
) -> UpscalePlan:
    """Store a range revise. A scale change re-renders the whole timeline.

    The patch trims the spans it overlaps instead of dropping them, so every
    other shot keeps its strength and look. One output file has one
    resolution, so a new scale cannot be spliced into part of it: every span
    takes the new scale, and the pending span covers the whole file, which
    :func:`render_plan` reads as "render every shot again".

    Result notes set ``preserve_shot_settings`` to retune only scale and
    strength. Split those patches at shot boundaries; a multi-shot revision
    uses the full-render path because a range splice accepts only one recipe.
    """
    patch = clamp_span(patch, duration_s)
    if _covers_all(patch, duration_s) and not preserve_shot_settings:
        full = replace(patch, start_s=0.0, end_s=duration_s)
        return UpscalePlan(spans=(full,), pending=full)
    spans = plan.spans
    rescale = patch.scale not in {span.scale for span in spans}
    if rescale:
        spans = tuple(replace(span, scale=patch.scale) for span in spans)
    kept: list[UpscaleSpan] = []
    patches: list[UpscaleSpan] = []
    for span in spans:
        if span.end_s <= patch.start_s + 0.05 or span.start_s >= patch.end_s - 0.05:
            kept.append(span)
            continue
        if preserve_shot_settings:
            patches.append(
                replace(
                    span,
                    start_s=max(span.start_s, patch.start_s),
                    end_s=min(span.end_s, patch.end_s),
                    scale=patch.scale,
                    strength=patch.strength,
                )
            )
        if patch.start_s - span.start_s > 0.05:
            kept.append(replace(span, end_s=patch.start_s))
        if span.end_s - patch.end_s > 0.05:
            kept.append(replace(span, start_s=patch.end_s))
    if not preserve_shot_settings:
        patches = [patch]
    if not patches:
        raise UpscaleError("the revised range does not overlap the plan")
    pending = patches[-1]
    if rescale or len(patches) > 1:
        pending = replace(patch, start_s=0.0, end_s=duration_s)
    # The latest patches stay last; the wizard reads the prior knobs there.
    return UpscalePlan(spans=(*kept, *patches), pending=pending)


class FakeUpscaleEngine:
    """Test double that scales with FFmpeg so splices have frames."""

    def __init__(self, *, ffmpeg_bin: str = "ffmpeg") -> None:
        self.ffmpeg_bin = ffmpeg_bin

    def upscale(self, request: UpscaleRequest) -> Path:
        require_supported_timing(request.source, ffmpeg_bin=self.ffmpeg_bin)
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
                _join_filters(
                    request.look.clean_filter(),
                    f"scale=iw*{request.scale}:ih*{request.scale}",
                ),
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


def _flashvsr_ready() -> tuple[Path, Path]:
    """The FlashVSR checkout and its tiny script, or a wizard-facing error."""
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
    return home, infer


def plain_upscale(request: UpscaleRequest, *, ffmpeg_bin: str = "ffmpeg") -> Path:
    """Resize ``request.source`` by its scale with lanczos, without FlashVSR."""
    require_supported_timing(request.source, ffmpeg_bin=ffmpeg_bin)
    request.output.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(request.source),
            "-vf",
            f"scale=iw*{request.scale}:ih*{request.scale}:flags=lanczos",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(request.output),
        ]
    )
    if not request.output.is_file():
        raise UpscaleError("the plain upscale produced no output")
    return request.output


def restore_clip(
    engine: UpscaleEngine,
    request: UpscaleRequest,
    *,
    use_upscale: bool,
    ffmpeg_bin: str = "ffmpeg",
) -> tuple[Path, float]:
    """The restored clip and the strength to blend it at.

    Without FlashVSR (``use_upscale`` off, or scale 1) the clip is only
    resized, or at scale 1 only re-encoded. Blended at strength 0 the
    delivery encode keeps just its lanczos base with the look applied, so a
    pass-through span matches its neighbours' size and frame timing.
    """
    require_supported_timing(request.source, ffmpeg_bin=ffmpeg_bin)
    if use_upscale and request.scale > 1:
        return engine.upscale(request), request.strength
    return plain_upscale(request, ffmpeg_bin=ffmpeg_bin), 0.0


def upscale_all(engine: UpscaleEngine, requests: list[UpscaleRequest]) -> list[Path]:
    """Run several short requests, loading the model once when the engine can."""
    for request in requests:
        require_supported_timing(request.source)
    many = getattr(engine, "upscale_many", None)
    if callable(many):
        return list(many(requests))
    return [engine.upscale(request) for request in requests]


class FlashVsrEngine:
    """Temporal restoration via the FlashVSR v1.1 tiny pipeline."""

    def __init__(self, *, chunk_s: float = CHUNK_SECONDS) -> None:
        # Longest piece to try first; the system-RAM child starts at the
        # shortest, since its memory has no GPU-sized ceiling of its own.
        self._chunk_s = chunk_s

    def upscale_many(self, requests: list[UpscaleRequest]) -> list[Path]:
        """Restore several short clips (previews) with one model load."""
        if not requests:
            return []
        for request in requests:
            require_supported_timing(request.source)
        home, infer = _flashvsr_ready()
        from .probe import ProbeError, probe_file

        ready: list[UpscaleRequest] = []
        for request in requests:
            request = replace(
                request,
                source=request.source.resolve(),
                output=request.output.resolve(),
            )
            try:
                duration = float(probe_file(request.source).duration_s or 0.0)
            except ProbeError:
                duration = 0.0
            if duration > CHUNK_SECONDS + 0.05:
                raise UpscaleError(
                    f"{request.source.name} is {duration:.1f}s; batched clips "
                    f"must be at most {CHUNK_SECONDS:g}s"
                )
            clean = request.look.clean_filter()
            if clean:
                cleaned = trim_source(
                    request.source,
                    request.output.with_name(request.output.stem + ".clean.mp4"),
                    0.0,
                    duration,
                    video_filter=clean,
                )
                request = replace(request, source=cleaned)
            ready.append(request)
        failed: UpscaleError | None = None
        with _FLASHVSR_LOCK:
            try:
                module, pipe, previous, path_added = _load_flashvsr(home, infer)
            except Exception as exc:  # noqa: BLE001 - becomes an UpscaleError
                failure = _flashvsr_failure(exc)
                del exc
                _release_gpu()
                raise failure from None
            try:
                for request in ready:
                    _infer_loaded(
                        module,
                        pipe,
                        request.source,
                        request.output,
                        scale=request.scale,
                    )
            except Exception as exc:  # noqa: BLE001 - becomes an UpscaleError
                failed = _flashvsr_failure(exc)
                del exc
            finally:
                _release_gpu(pipe)
                module = None
                pipe = None
                _release_gpu()
                _unload_flashvsr(previous, path_added)
        if isinstance(failed, GpuOutOfMemory):
            # One at a time, each in pieces as short as it needs.
            return [self.upscale(request) for request in requests]
        if failed is not None:
            raise failed from None
        missing = [r.output.name for r in ready if not r.output.is_file()]
        if missing:
            raise UpscaleError(f"FlashVSR produced no output for {missing[0]}")
        return [request.output for request in ready]

    def upscale(self, request: UpscaleRequest) -> Path:
        require_supported_timing(request.source)
        home, infer = _flashvsr_ready()
        from .probe import ProbeError, probe_file

        # The loader chdirs into the FlashVSR checkout; relative job paths
        # would then point inside it.
        request = replace(
            request,
            source=request.source.resolve(),
            output=request.output.resolve(),
        )
        try:
            duration = float(probe_file(request.source).duration_s or 0.0)
        except ProbeError:
            duration = 0.0
        chunk_s = self._chunk_s
        with _FLASHVSR_LOCK:
            while True:
                try:
                    return self._upscale_once(
                        request,
                        home=home,
                        infer=infer,
                        duration=duration,
                        chunk_s=chunk_s,
                    )
                except GpuOutOfMemory as exc:
                    fps = _frame_rate(request.source)
                    shortest = FLASHVSR_MIN_CHUNK_FRAMES / fps
                    if min(chunk_s, duration) > shortest + 0.5 / fps:
                        chunk_s = max(shortest, min(chunk_s, duration) / 2)
                        continue
                    reason = str(exc)
                    del exc
                    return _last_resort(request, reason, shortest)

    def _upscale_once(
        self,
        request: UpscaleRequest,
        *,
        home: Path,
        infer: Path,
        duration: float,
        chunk_s: float,
    ) -> Path:
        if duration > chunk_s + 0.05:
            return _run_flashvsr_chunked(
                request, home=home, infer=infer, chunk_s=chunk_s
            )
        clean = request.look.clean_filter()
        if clean:
            cleaned = trim_source(
                request.source,
                request.output.with_name(request.output.stem + ".clean.mp4"),
                0.0,
                duration,
                video_filter=clean,
            )
            request = replace(request, source=cleaned)
        return _run_flashvsr(request, home=home, infer=infer)


def _last_resort(request: UpscaleRequest, reason: str, shortest_s: float) -> Path:
    """Restore a clip whose shortest piece did not fit, using system RAM too.

    Runs in a child with managed GPU memory (see
    :mod:`super_processor.managed_memory`); inside that child, or when it
    cannot run, the out-of-memory error is raised for the wizard.
    """
    from .managed_memory import in_managed_memory, upscale_in_managed_memory

    advice = (
        "Close other programs that use the GPU (games, other AI tools) and try again."
    )
    if in_managed_memory():
        raise GpuOutOfMemory(
            "FlashVSR ran out of GPU memory and system RAM even in the "
            f"shortest pieces. {advice} ({reason})"
        )
    try:
        return upscale_in_managed_memory(request, chunk_s=shortest_s)
    except GpuOutOfMemory as exc:
        # The child's own message already says GPU and RAM were both short.
        message = str(exc)
    except UpscaleError as exc:
        # No system RAM to fall back on here (no nvcc, a failed build or
        # child): still an out-of-memory error, with what stopped the fallback.
        message = (
            "FlashVSR ran out of GPU memory even in the shortest pieces, and "
            f"could not use system RAM instead ({exc}). {advice} ({reason})"
        )
    raise GpuOutOfMemory(message) from None


def _frame_rate(video: Path) -> float:
    """Frames per second of ``video``; 24 when it cannot be read."""
    try:
        return float(Fraction(video_stream_facts(video)[1]))
    except (UpscaleError, ValueError, ZeroDivisionError):
        return 24.0


def _load_flashvsr(home: Path, infer: Path) -> tuple[Any, Any, str, list[str]]:
    """Import the tiny script with WanVSR and the repo root on ``sys.path``."""
    wan = infer.parent
    loader = importlib.util.spec_from_file_location("flashvsr_infer_tiny", infer)
    if loader is None or loader.loader is None:
        raise UpscaleError(f"could not load {infer}")
    module = importlib.util.module_from_spec(loader)
    previous = os.getcwd()
    path_added = [entry for entry in (str(wan), str(home)) if entry not in sys.path]
    sys.path[:0] = path_added
    # Lets a later 2GB block reuse the reserved cache instead of failing
    # while several gigabytes sit fragmented. Must be set before CUDA starts.
    # The managed-memory child brings its own allocator, which ignores this.
    if os.environ.get("SUPER_PROCESSOR_MANAGED_MEMORY") != "1":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.chdir(wan)
    try:
        loader.loader.exec_module(module)
        pipe = module.init_pipeline()
    except Exception:
        os.chdir(previous)
        for entry in path_added:
            with contextlib.suppress(ValueError):
                sys.path.remove(entry)
        raise
    return module, pipe, previous, path_added


def _unload_flashvsr(previous: str, path_added: list[str]) -> None:
    os.chdir(previous)
    for entry in path_added:
        with contextlib.suppress(ValueError):
            sys.path.remove(entry)


def _clear_pipeline_state(pipe: Any) -> None:
    """Drop what a loaded pipeline keeps from the clip it just restored.

    FlashVSR clears its input-projection cache and decoder memory at the
    start of a call, not the end, so they stayed on the GPU while the next
    clip was loaded next to them: a 24-frame clip that fits on its own ran
    out of memory right after a 16-frame one.
    """
    import gc

    dit = getattr(pipe, "dit", None)
    for holder, name in (
        (getattr(dit, "LQ_proj_in", None), "clear_cache"),
        (getattr(pipe, "TCDecoder", None), "clean_mem"),
    ):
        clear = getattr(holder, name, None)
        if callable(clear):
            with contextlib.suppress(Exception):
                clear()
    gc.collect()
    torch_mod = sys.modules.get("torch")
    cuda = getattr(torch_mod, "cuda", None) if torch_mod is not None else None
    if cuda is not None and getattr(cuda, "is_available", lambda: False)():
        cuda.empty_cache()


def _release_gpu(pipe: Any | None = None) -> None:
    """Move one pipeline off the GPU and return its cache to the driver.

    Deleting the name inside this function does not drop the caller's
    reference, so the caller sets its own variables to None and calls this
    again. The weights are a reference cycle with the VRAM wrappers, and
    PyTorch will not collect that cycle while it is trying to allocate.
    """
    import gc

    mover = getattr(pipe, "to", None)
    if mover is not None:
        with contextlib.suppress(Exception):
            mover("cpu")
    del pipe
    sys.modules.pop("flashvsr_infer_tiny", None)
    gc.collect()
    _return_freed_heap()
    torch_mod = sys.modules.get("torch")
    cuda = getattr(torch_mod, "cuda", None) if torch_mod is not None else None
    if cuda is None or not getattr(cuda, "is_available", lambda: False)():
        return
    empty = getattr(cuda, "empty_cache", None)
    if empty is not None:
        empty()
    ipc = getattr(cuda, "ipc_collect", None)
    if ipc is not None:
        ipc()


def _return_freed_heap() -> None:
    """Hand the heap a FlashVSR call freed back to the system (glibc only).

    Loading the weights and decoding the frames go through several GiB of
    host memory. Freed, it stays in glibc's heap, split over one arena per
    worker thread, so the wizard kept about 10 GiB between previews. Asking
    glibc to trim takes a process from 5.1 to 2.5 GiB after each call; what
    is left is mostly CUDA's own host state. Elsewhere this does nothing.
    """
    import ctypes
    import ctypes.util

    name = ctypes.util.find_library("c")
    if not name or not sys.platform.startswith("linux"):
        return
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL(name).malloc_trim(0)


# FlashVSR works on whole 128-pixel blocks of the upscaled picture, and its
# input loader centre-crops anything past the last whole block.
FLASHVSR_BLOCK = 128


def flashvsr_padding(width: int, height: int, scale: int) -> tuple[int, int, int, int]:
    """Edge padding (left, top, right, bottom) so FlashVSR crops nothing.

    Without it, 854x480 at 2x would come back as the middle 1664x896 of
    1708x960: smaller than promised, and out of line with the full-frame
    base it is blended over. Padded to a multiple of ``128 / scale``, the
    upscaled clip is whole blocks; the padding is cropped off afterwards.
    Offsets stay even so 4:2:0 chroma lines up.
    """
    step = FLASHVSR_BLOCK // scale
    padded_w = -(-width // step) * step
    padded_h = -(-height // step) * step
    left = (padded_w - width) // 4 * 2
    top = (padded_h - height) // 4 * 2
    return left, top, padded_w - width - left, padded_h - height - top


def flashvsr_extra_frames(frames: int) -> int:
    """Copies of the last frame to add so FlashVSR returns every real frame.

    Its loader appends four copies of the last frame, keeps the largest
    8n + 1 frames that fit, and returns four fewer: up to three short of the
    clip, which the conform step used to fill by freezing the last frame.
    With ``frames + extra + 3`` a multiple of 8 it returns ``frames + extra``,
    and the conform step trims the copies off again.
    """
    return -(frames + 3) % 8


def _pad_for_flashvsr(
    source: Path, pad: tuple[int, int, int, int], extra_frames: int
) -> Path:
    """``source`` with its edge pixels repeated into the padding and its last
    frame repeated ``extra_frames`` times, losslessly."""
    left, top, right, bottom = pad
    padded = source.with_name(source.stem + ".pad.mp4")
    space = (
        f"pad=iw+{left + right}:ih+{top + bottom}:{left}:{top},"
        f"fillborders=left={left}:right={right}:top={top}:bottom={bottom}"
        ":mode=smear"
        if any(pad)
        else ""
    )
    time = f"tpad=stop_mode=clone:stop={extra_frames}" if extra_frames else ""
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            _join_filters(space, time),
            "-an",
            "-c:v",
            "libx264",
            "-qp",
            "0",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            str(padded),
        ]
    )
    return padded


def _crop_after_flashvsr(
    output: Path, pad: tuple[int, int, int, int], size: tuple[int, int], scale: int
) -> None:
    """Cut the scaled padding off FlashVSR's output, in place, losslessly."""
    left, top, _right, _bottom = pad
    width, height = size
    cropped = output.with_name(output.stem + ".crop.mp4")
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(output),
            "-vf",
            f"crop={width * scale}:{height * scale}:{left * scale}:{top * scale}",
            "-an",
            "-c:v",
            "libx264",
            "-qp",
            "0",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            str(cropped),
        ]
    )
    cropped.replace(output)


def _infer_loaded(
    module: Any,
    pipe: Any,
    source: Path,
    output: Path,
    *,
    scale: int,
) -> None:
    """Run one already-loaded pipeline on a short clip.

    The clip is padded first so FlashVSR keeps every pixel and every frame,
    and the output is cropped back to exactly ``scale`` times the clip (see
    :func:`flashvsr_padding` and :func:`flashvsr_extra_frames`).
    """
    size = _frame_size(source)
    pad = flashvsr_padding(*size, scale) if size else (0, 0, 0, 0)
    extra = flashvsr_extra_frames(video_stream_facts(source)[0])
    padded = any(pad) or extra > 0
    reads = _pad_for_flashvsr(source, pad, extra) if padded else source
    low, height, width, frames, fps = module.prepare_input_tensor(
        str(reads),
        scale=float(scale),
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
        topk_ratio=FLASHVSR_SPARSE_RATIO * 768 * 1280 / (height * width),
        kv_ratio=3.0,
        local_range=FLASHVSR_LOCAL_RANGE,
        color_fix=True,
    )
    pictures = module.tensor2video(video)
    del low, video
    _clear_pipeline_state(pipe)
    output.parent.mkdir(parents=True, exist_ok=True)
    module.save_video(pictures, str(output), fps=fps, quality=6)
    if size and any(pad):
        _crop_after_flashvsr(output, pad, size, scale)
    if padded:
        reads.unlink(missing_ok=True)
    conform_frames(output, source)


def video_stream_facts(video: Path, *, ffprobe_bin: str = "ffprobe") -> tuple[int, str]:
    """Frame count (packets, no decode) and ``r_frame_rate`` of the video."""
    try:
        completed = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets,r_frame_rate",
                "-of",
                "json",
                str(video),
            ],
            check=False,
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise UpscaleError(f"failed to start ffprobe: {exc}") from exc
    try:
        streams = json.loads(completed.stdout or b"{}").get("streams") or []
        stream = streams[0]
        return int(stream["nb_read_packets"]), str(stream["r_frame_rate"])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        detail = (completed.stderr or b"").decode("utf-8", errors="replace").strip()
        raise UpscaleError(
            f"could not count the frames of {video}: {detail or exc}"
        ) from exc


def conform_frames(
    restored: Path,
    source: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> Path:
    """Make ``restored`` hold exactly the source's frames at the source's rate.

    The keyframe splice and the chunk overlap both cut by time, so a restored
    clip that is a few frames short or long, or written at a rounded rate,
    would shift everything after it. A small difference is fixed losslessly at
    the end of the clip (trim, or repeat the last frame). A larger one raises
    before anything is spliced.
    """
    want, rate = video_stream_facts(source, ffprobe_bin=ffprobe_bin)
    have, have_rate = video_stream_facts(restored, ffprobe_bin=ffprobe_bin)
    if have == want and have_rate == rate:
        return restored
    if abs(have - want) > max(FRAME_SLACK, int(want * 0.02)):
        raise UpscaleError(
            f"FlashVSR returned {have} frames for a {want}-frame clip "
            f"({source.name}); refusing to splice a misaligned picture"
        )
    filters = [f"setpts=N/({rate})/TB"]
    if have < want:
        filters.append(f"tpad=stop_mode=clone:stop={want - have}")
    tmp = restored.with_name(restored.stem + ".conform" + restored.suffix)
    _run_ffmpeg(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(restored),
            "-vf",
            ",".join(filters),
            "-r",
            rate,
            "-frames:v",
            str(want),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-qp",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(tmp),
        ]
    )
    tmp.replace(restored)
    return restored


def _run_flashvsr(request: UpscaleRequest, *, home: Path, infer: Path) -> Path:
    """Load the upstream tiny script and run one file. Weights stay relative."""
    try:
        module, pipe, previous, path_added = _load_flashvsr(home, infer)
    except Exception as exc:  # noqa: BLE001 - becomes an UpscaleError
        failure = _flashvsr_failure(exc)
        del exc
        _release_gpu()
        raise failure from None
    failed: UpscaleError | None = None
    try:
        _infer_loaded(
            module,
            pipe,
            request.source,
            request.output,
            scale=request.scale,
        )
    except Exception as exc:  # noqa: BLE001 - becomes an UpscaleError
        failed = _flashvsr_failure(exc)
        del exc
    finally:
        _release_gpu(pipe)
        module = None
        pipe = None
        _release_gpu()
        _unload_flashvsr(previous, path_added)
    if failed is not None:
        raise failed from None
    if not request.output.is_file():
        raise UpscaleError("FlashVSR produced no output")
    return request.output


def chunk_frames(total: int, length: int, overlap: int) -> list[tuple[int, int]]:
    """``[first, stop)`` frame ranges of ``length`` overlapping by ``overlap``.

    Planned in whole frames so a cut never lands between two frames (at 25
    fps, 7.5 s is frame 187.5). The last range is a full ``length`` too,
    reaching further back into the one before: a short remainder can be
    below what FlashVSR accepts.
    """
    if total <= 0:
        raise UpscaleError("media has no frames")
    if length >= total:
        return [(0, total)]
    step = length - overlap
    ranges: list[tuple[int, int]] = []
    first = 0
    while first + length < total:
        ranges.append((first, first + length))
        first += step
    ranges.append((total - length, total))
    return ranges


def _run_flashvsr_chunked(
    request: UpscaleRequest,
    *,
    home: Path,
    infer: Path,
    chunk_s: float = CHUNK_SECONDS,
) -> Path:
    """Upscale overlapping pieces of ``chunk_s`` so the whole clip is not on the GPU.

    Each piece's restored picture goes to disk before the next one starts;
    the join, blend, and delivery encode then run once on the files.
    """
    total, rate = video_stream_facts(request.source)
    fps = float(Fraction(rate))
    length = max(FLASHVSR_MIN_CHUNK_FRAMES, round(chunk_s * fps))
    overlap = min(round(CHUNK_OVERLAP_S * fps), length // 4)
    chunks = chunk_frames(total, length, overlap)
    work = request.output.parent / "flash_chunks"
    work.mkdir(parents=True, exist_ok=True)
    try:
        module, pipe, previous, path_added = _load_flashvsr(home, infer)
    except Exception as exc:  # noqa: BLE001 - becomes an UpscaleError
        failure = _flashvsr_failure(exc)
        del exc
        _release_gpu()
        raise failure from None
    pieces: list[Path] = []
    failed: UpscaleError | None = None
    covered = 0
    try:
        for index, (first, stop) in enumerate(chunks):
            clip = trim_source(
                request.source,
                work / f"in_{index}.mp4",
                _cut_time(first, fps),
                _cut_time(stop, fps),
                video_filter=request.look.clean_filter(),
            )
            raw = work / f"up_{index}.mp4"
            _infer_loaded(module, pipe, clip, raw, scale=request.scale)
            # Every piece goes through the same lossless encode so the join
            # reads one set of codec parameters; the frames the previous
            # piece already covers are cut off here.
            kept = work / f"keep_{index}.mp4"
            skip = covered - first
            trim_source(raw, kept, _cut_time(skip, fps), _cut_time(stop - first, fps))
            covered = stop
            pieces.append(kept)
            if request.on_progress is not None:
                request.on_progress((index + 1) / len(chunks))
    except Exception as exc:  # noqa: BLE001 - becomes an UpscaleError
        failed = _flashvsr_failure(exc)
        del exc
    finally:
        _release_gpu(pipe)
        module = None
        pipe = None
        _release_gpu()
        _unload_flashvsr(previous, path_added)
    if failed is not None:
        raise failed from None
    # Join, blend, and encode once into the delivery settings, so the caller
    # does not encode this picture a second time.
    listing = _write_listing(work / "join.txt", pieces)
    encode_delivery(
        listing,
        request.output,
        concat=True,
        blend_with=request.source,
        strength=request.strength,
        look=request.look,
        regions=request.regions,
    )
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
    video_filter: str = "",
) -> Path:
    """Cut a range into a lossless work clip.

    Frame-accurate, so it re-encodes, but at ``-qp 0`` so the engine and the
    chunk join see the decoded pixels unchanged. Work clips are short-lived.
    ``video_filter`` (a look's clean step, or a timestamp reset) must not
    change the frame count.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    filters = ["-vf", video_filter] if video_filter else []
    _run_ffmpeg(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_s:.6f}",
            "-to",
            f"{end_s:.6f}",
            "-i",
            str(source),
            *filters,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-qp",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(dest),
        ]
    )
    return dest


def delivery_marker_path(video: Path) -> Path:
    return video.with_name(video.name + ".delivery.json")


def _delivery_marker() -> dict[str, Any]:
    return {
        "version": DELIVERY_VERSION,
        "bframes": 0,
        "codec": "libx264",
        "crf": DELIVERY_CRF,
        "keyframe_s": KEYFRAME_INTERVAL_S,
    }


def _read_marker(video: Path) -> dict[str, Any] | None:
    marker = delivery_marker_path(video)
    if not video.is_file() or not marker.is_file():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def is_delivery(video: Path) -> bool:
    """True when ``video`` was written by :func:`encode_delivery` (this version)."""
    data = _read_marker(video)
    if data is None:
        return False
    return all(data.get(key) == value for key, value in _delivery_marker().items())


def _write_marker(
    video: Path,
    *,
    blend: dict[str, Any] | None = None,
    look: ShotLook | None = None,
) -> None:
    data = _delivery_marker()
    if blend is not None:
        data["blend"] = blend
    if look is not None:
        data["look"] = look.to_dict()
    delivery_marker_path(video).write_text(json.dumps(data) + "\n", encoding="utf-8")


def _blend_record(strength: float, regions: tuple[Region, ...]) -> dict[str, Any]:
    """What a blended encode was made with, so an identical one can be skipped."""
    return {
        "strength": strength,
        "structure_sigma": STRUCTURE_SIGMA,
        "regions": [region.to_dict() for region in regions],
    }


def blend_filter(
    width: int,
    height: int,
    strength: float,
    look: ShotLook | None = None,
    *,
    regions: tuple[Region, ...] = (),
    sigma: float = 0.0,
) -> str:
    """Mix the restored picture (input 0) with a lanczos upscale of input 1.

    With ``sigma`` (in output pixels), the restored picture's coarse layer is
    swapped for the base's before the mix: the result is
    ``base + s * (detail(restored) - detail(base))``, where ``detail`` is a
    frame minus its Gaussian blur. At strength 0 that is the base; at 1 it is
    the base's shapes and colour with FlashVSR's fine detail. The strength
    ``s`` is per pixel: lower inside protected ``regions``.

    Every blend takes the restored picture's branch as its first (main) input
    with ``eof_action=pass``, so the frame count the keyframe splice relies
    on never changes. The base is padded by a second of its last frame so a
    few missing frames do not leave the tail unmixed. The base gets the look's
    clean step, as the engine input did, and the mix gets its finish step.
    """
    _require_strength(strength)
    look = look or ShotLook()
    share = strength_expr(strength, regions)
    base = _join_filters(
        "setpts=PTS-STARTPTS",
        look.clean_filter(),
        f"scale={width}:{height}:flags=lanczos,format=yuv420p",
        "tpad=stop_mode=clone:stop_duration=1",
    )
    finish = look.finish_filter()
    mixed = "[mix];[mix]" + finish + "[v]" if finish else "[v]"
    mix = f"blend=all_expr='B+(A-B)*({share})':eof_action=pass{mixed}"
    if sigma <= 0:
        return (
            "[0:v]setpts=PTS-STARTPTS,format=yuv420p[up];"
            f"[1:v]{base}[base];"
            f"[up][base]{mix}"
        )
    blur = f"gblur=sigma={sigma:.3f}"
    return (
        "[0:v]setpts=PTS-STARTPTS,format=yuv420p,split[up][upc];"
        f"[upc]{blur}[uplow];"
        f"[1:v]{base},split[basec][basemix];"
        f"[basec]{blur}[baselow];"
        # Restored detail, offset to mid grey so negative values survive.
        "[up][uplow]blend=all_expr='A-B+128':eof_action=pass[detail];"
        # The base's coarse layer with the restored detail on top.
        "[detail][baselow]blend=all_expr='B+A-128':eof_action=pass[rebuilt];"
        f"[rebuilt][basemix]{mix}"
    )


def _join_filters(*parts: str) -> str:
    return ",".join(part for part in parts if part)


def encode_delivery(
    source: Path,
    dest: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    concat: bool = False,
    blend_with: Path | None = None,
    strength: float = 1.0,
    look: ShotLook | None = None,
    regions: tuple[Region, ...] = (),
) -> Path:
    """Encode a picture once into the settings every splice part shares.

    A forced IDR frame every second, and no B-frames, give a range revise
    clean points to cut the current output by stream copy. ``concat`` reads
    ``source`` as a concat-demuxer listing. With ``blend_with``, the picture
    is mixed with a lanczos upscale of ``blend_with`` in the same encode,
    taking its coarse layer from that base (see :func:`blend_filter`). The
    look's finish step runs in this encode too.
    """
    look = look or ShotLook()
    record = _blend_record(strength, regions) if blend_with is not None else None
    finish = look.finish_filter()
    if not concat and is_delivery(source):
        done = _read_marker(source) or {}
        same_look = done.get("look", ShotLook().to_dict()) == look.to_dict()
        if done.get("blend") == record and same_look:
            if source != dest:
                source.replace(dest)
                delivery_marker_path(source).replace(delivery_marker_path(dest))
            return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    marker = delivery_marker_path(dest)
    marker.unlink(missing_ok=True)
    inputs = (
        ["-f", "concat", "-safe", "0", "-i", str(source)]
        if concat
        else [
            "-i",
            str(source),
        ]
    )
    picture = ["-map", "0:v:0"]
    if blend_with is not None:
        size = _frame_size(_first_part(source) if concat else source)
        if size is None:
            raise UpscaleError(f"could not read the frame size of {source}")
        base_size = _frame_size(blend_with)
        if base_size is None:
            raise UpscaleError(f"could not read the frame size of {blend_with}")
        sigma = STRUCTURE_SIGMA * size[0] / base_size[0]
        inputs.extend(["-i", str(blend_with)])
        picture = [
            "-filter_complex",
            blend_filter(
                size[0], size[1], strength, look, regions=regions, sigma=sigma
            ),
            "-map",
            "[v]",
        ]
    elif finish:
        picture = ["-filter_complex", f"[0:v]{finish}[v]", "-map", "[v]"]
    tmp = dest.with_name(dest.stem + ".encoding" + dest.suffix)
    _run_ffmpeg(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *inputs,
            *picture,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(DELIVERY_CRF),
            "-pix_fmt",
            "yuv420p",
            "-force_key_frames",
            f"expr:gte(t,n_forced*{KEYFRAME_INTERVAL_S:g})",
            "-forced-idr",
            "1",
            # No B-frames: presentation and decode order match, so a stream
            # copy cut at a keyframe takes exactly the frames before it.
            "-bf",
            "0",
            "-an",
            str(tmp),
        ]
    )
    tmp.replace(dest)
    _write_marker(dest, blend=record, look=None if look.is_neutral() else look)
    return dest


def _first_part(listing: Path) -> Path:
    """The first file named in a concat-demuxer listing."""
    for line in listing.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("file "):
            return Path(line[5:].strip().strip("'").replace("'\\''", "'"))
    raise UpscaleError(f"empty concat listing {listing}")


def keyframe_times(video: Path, *, ffprobe_bin: str = "ffprobe") -> list[float]:
    """Presentation times of the video keyframes, ascending."""
    try:
        completed = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-skip_frame",
                "nokey",
                "-show_entries",
                "frame=pts_time",
                "-of",
                "csv=p=0",
                str(video),
            ],
            check=False,
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise UpscaleError(f"failed to start ffprobe: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or b"").decode("utf-8", errors="replace").strip()
        raise UpscaleError(detail or "ffprobe failed")
    times: list[float] = []
    for line in completed.stdout.decode("utf-8", errors="replace").splitlines():
        value = line.strip().rstrip(",")
        if not value or value == "N/A":
            continue
        try:
            times.append(float(value))
        except ValueError:
            continue
    return sorted(set(times))


def snap_to_keyframes(
    keys: list[float], start_s: float, end_s: float, duration_s: float
) -> tuple[float, float | None]:
    """Widen ``start_s``/``end_s`` out to keyframes. ``None`` means to the end."""
    before = [key for key in keys if key <= start_s + _KEY_EPSILON_S]
    start = before[-1] if before else 0.0
    if start <= 0.05:
        start = 0.0
    after = [key for key in keys if key >= end_s - _KEY_EPSILON_S]
    end = after[0] if after else None
    if end is not None and end >= duration_s - 0.05:
        end = None
    return start, end


def _ffprobe_for(ffmpeg_bin: str) -> str:
    """The ffprobe that sits next to ``ffmpeg_bin``, else the one on PATH."""
    path = Path(ffmpeg_bin)
    if path.parent != Path("."):
        sibling = path.with_name(path.name.replace("ffmpeg", "ffprobe"))
        if sibling.is_file():
            return str(sibling)
    return "ffprobe"


def _copy_range(
    video: Path,
    dest: Path,
    start_s: float,
    end_s: float | None,
    *,
    ffmpeg_bin: str,
) -> Path:
    """Stream-copy a keyframe-aligned range. No pixels are re-encoded."""
    command = [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y"]
    if start_s > 0:
        command.extend(["-ss", f"{start_s:.6f}"])
    command.extend(["-i", str(video)])
    if end_s is not None:
        command.extend(["-t", f"{end_s - start_s:.6f}"])
    command.extend(
        [
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-an",
            "-avoid_negative_ts",
            "make_zero",
            str(dest),
        ]
    )
    _run_ffmpeg(command)
    return dest


def _frame_size(video: Path) -> tuple[int, int] | None:
    from .probe import ProbeError, probe_file

    try:
        stream = probe_file(video).primary_video()
    except ProbeError:
        return None
    if stream is None or not stream.width or not stream.height:
        return None
    return stream.width, stream.height


def _upscale_full(
    source: Path,
    output: Path,
    span: UpscaleSpan,
    work: Path,
    *,
    engine: UpscaleEngine,
    ffmpeg_bin: str,
    on_progress: ProgressFn | None = None,
) -> Path:
    raw, strength = restore_clip(
        engine,
        UpscaleRequest(
            source,
            work / "full_up.mp4",
            scale=span.scale,
            strength=span.strength,
            look=span.look,
            regions=span.regions,
            on_progress=on_progress,
        ),
        use_upscale=span.use_upscale,
        ffmpeg_bin=ffmpeg_bin,
    )
    encode_delivery(
        raw,
        output,
        ffmpeg_bin=ffmpeg_bin,
        blend_with=source,
        strength=strength,
        look=span.look,
        regions=span.regions,
    )
    mux_source_audio(output, source, ffmpeg_bin=ffmpeg_bin)
    return output


def apply_range_revise(
    source: Path,
    current: Path,
    span: UpscaleSpan,
    output: Path,
    *,
    engine: UpscaleEngine,
    ffmpeg_bin: str = "ffmpeg",
    duration_s: float,
    on_progress: ProgressFn | None = None,
) -> Path:
    """Re-upscale ``span`` from the original source and splice it into ``current``.

    The range widens out to the keyframes of ``current``. The head and tail are
    stream-copied, so repeated revises never re-encode the untouched parts.
    A ``current`` from before the delivery settings existed is encoded into
    them once first.
    """
    require_supported_timing(source, ffmpeg_bin=ffmpeg_bin)
    work = output.parent / "range_work"
    work.mkdir(parents=True, exist_ok=True)
    if _covers_all(span, duration_s) or not current.is_file():
        return _upscale_full(
            source,
            output,
            span,
            work,
            engine=engine,
            ffmpeg_bin=ffmpeg_bin,
            on_progress=on_progress,
        )
    require_supported_timing(current, ffmpeg_bin=ffmpeg_bin)
    if not is_delivery(current):
        current = encode_delivery(current, work / "current.mp4", ffmpeg_bin=ffmpeg_bin)
    keys = keyframe_times(current, ffprobe_bin=_ffprobe_for(ffmpeg_bin))
    start, end = snap_to_keyframes(keys, span.start_s, span.end_s, duration_s)
    if start == 0.0 and end is None:
        return _upscale_full(
            source,
            output,
            span,
            work,
            engine=engine,
            ffmpeg_bin=ffmpeg_bin,
            on_progress=on_progress,
        )
    clip = trim_source(
        source,
        work / "range.mp4",
        start,
        duration_s if end is None else end,
        ffmpeg_bin=ffmpeg_bin,
    )
    raw, strength = restore_clip(
        engine,
        UpscaleRequest(
            clip,
            work / "range_up.mp4",
            scale=span.scale,
            strength=span.strength,
            look=span.look,
            regions=span.regions,
            on_progress=on_progress,
        ),
        use_upscale=span.use_upscale,
        ffmpeg_bin=ffmpeg_bin,
    )
    middle = encode_delivery(
        raw,
        work / "range_delivery.mp4",
        ffmpeg_bin=ffmpeg_bin,
        blend_with=clip,
        strength=strength,
        look=span.look,
        regions=span.regions,
    )
    current_size = _frame_size(current)
    middle_size = _frame_size(middle)
    if current_size and middle_size and current_size != middle_size:
        raise UpscaleError(
            f"the revised range is {middle_size[0]}x{middle_size[1]} but the "
            f"output is {current_size[0]}x{current_size[1]}; a scale change "
            "must re-render the whole clip"
        )
    parts: list[Path] = []
    if start > 0:
        parts.append(
            _copy_range(current, work / "head.mp4", 0.0, start, ffmpeg_bin=ffmpeg_bin)
        )
    parts.append(middle)
    if end is not None:
        parts.append(
            _copy_range(current, work / "tail.mp4", end, None, ffmpeg_bin=ffmpeg_bin)
        )
    silent = work / "spliced.mp4"
    _concat(parts, silent, ffmpeg_bin=ffmpeg_bin)
    ffprobe_bin = _ffprobe_for(ffmpeg_bin)
    before, _rate = video_stream_facts(current, ffprobe_bin=ffprobe_bin)
    after, _rate = video_stream_facts(silent, ffprobe_bin=ffprobe_bin)
    if after != before:
        raise UpscaleError(
            f"the revised output would have {after} frames instead of {before}; "
            "the previous output is left unchanged"
        )
    silent.replace(output)
    _write_marker(output)
    mux_source_audio(output, source, ffmpeg_bin=ffmpeg_bin)
    return output


def render_plan(
    source: Path,
    output: Path,
    plan: UpscalePlan,
    *,
    engine: UpscaleEngine,
    ffmpeg_bin: str = "ffmpeg",
    duration_s: float,
    on_progress: ProgressFn | None = None,
) -> Path:
    """Render what ``plan`` asks for into ``output``.

    ``on_progress`` gets the share of the work done, 0 to 1, by source frames.

    A pending range on an existing output is spliced in by
    :func:`apply_range_revise`. Anything else (the first render, a pending
    span over the whole file, a scale change) renders every shot again.
    """
    require_supported_timing(source, ffmpeg_bin=ffmpeg_bin)
    pending = plan.pending
    if (
        pending is not None
        and output.is_file()
        and not _covers_all(pending, duration_s)
    ):
        return apply_range_revise(
            source,
            output,
            pending,
            output,
            engine=engine,
            ffmpeg_bin=ffmpeg_bin,
            duration_s=duration_s,
            on_progress=on_progress,
        )
    return render_shots(
        source,
        output,
        plan.spans,
        engine=engine,
        ffmpeg_bin=ffmpeg_bin,
        duration_s=duration_s,
        on_progress=on_progress,
    )


def shot_timeline(
    spans: tuple[UpscaleSpan, ...], duration_s: float
) -> list[UpscaleSpan]:
    """``spans`` in time order, checked to cover the clip once at one scale."""
    ordered = sorted(spans, key=lambda span: span.start_s)
    if not ordered:
        raise UpscaleError("the plan has no shots")
    if len({span.scale for span in ordered}) > 1:
        raise UpscaleError("every shot in one output must use the same scale")
    cursor = 0.0
    for span in ordered:
        if abs(span.start_s - cursor) > 0.05:
            raise UpscaleError(
                f"shots must cover the clip without gaps or overlaps "
                f"(one starts at {span.start_s:.2f}s, expected {cursor:.2f}s)"
            )
        cursor = span.end_s
    if cursor < duration_s - 0.05:
        raise UpscaleError(f"the last shot ends at {cursor:.2f}s, before the end")
    return ordered


def _coalesce(ordered: list[UpscaleSpan]) -> list[UpscaleSpan]:
    """Join neighbouring spans whose scale, strength, and look match."""
    out: list[UpscaleSpan] = []
    for span in ordered:
        last = out[-1] if out else None
        if last is not None and (
            last.scale,
            last.strength,
            last.look,
            last.regions,
            last.use_upscale,
        ) == (span.scale, span.strength, span.look, span.regions, span.use_upscale):
            out[-1] = replace(last, end_s=span.end_s)
        else:
            out.append(span)
    return out


# Where a part is cut, in frames before the frame it starts on. FFmpeg rounds
# a seek time to the nearest frame, so a cut half a frame early sits exactly on
# the rounding edge: with the time written to six decimals, 141.5/24 became
# 141.49999 and the part started a frame early, shifting the whole segment.
# A quarter frame keeps clear of both that edge and the frame time itself.
_CUT_MARGIN_FRAMES = 0.25


def _cut_time(frame: int, fps: float) -> float:
    """The seek time that starts (or ends) a part exactly at ``frame``."""
    return max(0.0, (frame - _CUT_MARGIN_FRAMES) / fps)


def render_shots(
    source: Path,
    output: Path,
    spans: tuple[UpscaleSpan, ...],
    *,
    engine: UpscaleEngine,
    ffmpeg_bin: str = "ffmpeg",
    duration_s: float,
    on_progress: ProgressFn | None = None,
) -> Path:
    """Restore each shot with its own look and join them into ``output``.

    Shot edges snap to source frames, so the parts hold every source frame
    exactly once. Each part goes through the delivery encode, which starts it
    on a keyframe, and the parts are joined by stream copy. Neighbouring
    shots with the same settings render as one part, so FlashVSR is not
    loaded once per shot when most shots look alike.
    """
    require_supported_timing(source, ffmpeg_bin=ffmpeg_bin)
    ordered = _coalesce(shot_timeline(spans, duration_s))
    work = output.parent / "shot_work"
    work.mkdir(parents=True, exist_ok=True)
    if len(ordered) == 1:
        return _upscale_full(
            source,
            output,
            ordered[0],
            work,
            engine=engine,
            ffmpeg_bin=ffmpeg_bin,
            on_progress=on_progress,
        )
    ffprobe_bin = _ffprobe_for(ffmpeg_bin)
    want, rate = video_stream_facts(source, ffprobe_bin=ffprobe_bin)
    fps = float(Fraction(rate))
    edges = [0, *(round(span.start_s * fps) for span in ordered[1:]), want]
    parts: list[Path] = []
    for index, span in enumerate(ordered):
        first, stop = edges[index], edges[index + 1]
        if stop <= first:
            continue
        # Each part's share of the work is its share of the source frames.
        part = _part_of(on_progress, first / want, (stop - first) / want)
        clip = trim_source(
            source,
            work / f"shot_{index}.mp4",
            _cut_time(first, fps),
            _cut_time(stop, fps),
            ffmpeg_bin=ffmpeg_bin,
            video_filter="setpts=PTS-STARTPTS",
        )
        raw, strength = restore_clip(
            engine,
            UpscaleRequest(
                clip,
                work / f"shot_{index}_up.mp4",
                scale=span.scale,
                strength=span.strength,
                look=span.look,
                regions=span.regions,
                on_progress=part,
            ),
            use_upscale=span.use_upscale,
            ffmpeg_bin=ffmpeg_bin,
        )
        parts.append(
            encode_delivery(
                raw,
                work / f"shot_{index}_delivery.mp4",
                ffmpeg_bin=ffmpeg_bin,
                blend_with=clip,
                strength=strength,
                look=span.look,
                regions=span.regions,
            )
        )
        if part is not None:
            part(1.0)
    silent = work / "shots.mp4"
    _concat(parts, silent, ffmpeg_bin=ffmpeg_bin)
    have, _rate = video_stream_facts(silent, ffprobe_bin=ffprobe_bin)
    if have != want:
        raise UpscaleError(
            f"the joined shots have {have} frames instead of {want}; "
            "the previous output is left unchanged"
        )
    silent.replace(output)
    _write_marker(output)
    mux_source_audio(output, source, ffmpeg_bin=ffmpeg_bin)
    return output


def mux_source_audio(
    video: Path,
    source: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    """Copy the source audio onto an upscaled picture. No audio is fine.

    The output is as long as the picture: longer audio is cut there, and
    shorter audio just ends early. (``-shortest`` would instead drop the
    last frames of a picture whose audio is shorter, which is common.) If
    the frame count still changes, the picture is kept without audio.
    """
    mixed = video.with_suffix(".mux.mp4")
    ffprobe_bin = _ffprobe_for(ffmpeg_bin)
    frames, rate = video_stream_facts(video, ffprobe_bin=ffprobe_bin)
    length = frames / Fraction(rate)
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
            "-t",
            f"{float(length):.6f}",
            # moov at the front, so a browser can start playing and seeking
            # before the whole file has been read.
            "-movflags",
            "+faststart",
            str(mixed),
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0 or not mixed.is_file():
        return video
    if video_stream_facts(mixed, ffprobe_bin=ffprobe_bin)[0] != frames:
        mixed.unlink(missing_ok=True)
        return video
    mixed.replace(video)
    return video


def _write_listing(path: Path, parts: list[Path]) -> Path:
    from .render import write_concat_list

    return write_concat_list(path, parts)


def _concat(parts: list[Path], dest: Path, *, ffmpeg_bin: str) -> None:
    from .render import build_concat_argv

    listing = _write_listing(dest.parent / "splice.txt", parts)
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
        raise UpscaleError("scale must be 1, 2, or 4")
    if value not in _SCALES:
        raise UpscaleError("scale must be 1, 2, or 4")
    return value


def _require_strength(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UpscaleError("strength must be a number from 0.0 to 1.0")
    strength = float(value)
    if strength < 0.0 or strength > 1.0:
        raise UpscaleError("strength must be between 0.0 and 1.0")
    return strength
