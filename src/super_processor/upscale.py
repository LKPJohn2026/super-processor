"""Local SPAN upscale.

FFmpeg probes, trims, concatenates, and copies audio. Each span (a shot) also
carries a :class:`~super_processor.look.ShotLook`: bounded FFmpeg clean-up
before SPAN and finishing after it, always in that order. Gemini may only
return a time range plus ``scale`` and ``strength``.
Shapes, colour, and layout always come from a plain lanczos upscale of the
same (cleaned) frames; ``strength`` is how much of SPAN's fine detail is
added on top, so 0.0 adds nothing and 1.0 adds all of it. Inside protected
regions (faces, hands, text) the share is lower. SPAN itself has no settings
(see :mod:`super_processor.span_sr`).
The real engine refuses to run without CUDA; tests use :class:`FakeUpscaleEngine`.

Work clips handed to the engine are encoded losslessly. The engine output is
encoded once into the delivery settings, with a keyframe every second. A range
revise snaps outward to those keyframes and stream-copies the head and tail, so
the parts of the video a note did not name are never encoded again.

Delivery video is always H.264: NVENC where this machine has it (see
:func:`delivery_encoder`), libx264 elsewhere.
"""

from __future__ import annotations

import contextlib
import contextvars
import importlib
import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import IO, Any, Protocol

from .atomic_io import unlink_quietly
from .cancel import Cancelled, stop_process
from .dpir import DpirError, ffmpeg_look, gpu_look, level_of, note_level
from .look import LookError, ShotLook
from .regions import Region, regions_from_dicts, strength_expr
from .tools import FILE_INPUT, resolve_argv, tool_path

UPSCALE_PLAN_FILE = "upscale_plan.json"
CHUNK_SECONDS = 8.0
CHUNK_OVERLAP_S = 0.5
# 3: H.264 from NVENC or libx264, named in the marker. Outputs from an
# older version, or from the other encoder, are not spliced into: a revise
# encodes them again first (see apply_range_revise), so one file never mixes
# encoders.
DELIVERY_VERSION = 3
NVENC = "h264_nvenc"
X264 = "libx264"
# Against a lossless 2880x2160 upscale (200 frames, these keyframe flags):
# x264 "faster" crf 16 made 88.7 MB at 44.44 dB (SSIM 0.98655) for about
# 48 CPU-seconds; NVENC p5 cq 17 made 81.8 MB at 44.29 dB (SSIM 0.98588)
# for next to none (cq 16: 92.9 MB, 44.97 dB; cq 18: 71.2 MB, 43.62 dB).
# The x264 encode was half the CPU of a SPAN render, which ran a 16-core CPU
# at its 95 C limit. ("faster" was within 0.15 dB of "medium" for 57% less.)
NVENC_PRESET = "p5"
NVENC_CQ = 17
X264_PRESET = "faster"
X264_CRF = 16
# Frame sizes NVENC's H.264 encoder takes on an RTX 5090: 128x96 and 4098
# wide were refused. Outside them the delivery is libx264, which every part
# of that output shares (they are all one size).
_NVENC_MIN = (145, 49)
_NVENC_MAX = (4096, 4096)
# NVENC holds VBR to the bitrate of the level it picks, the lowest the size
# and rate allow: 320x180 at 10 fps got level 1.1 (192 kbit/s), and moving
# grain came out at 26.9 dB where x264 made 41.8 (2.3 dB short of NVENC's
# own constant QP 17). So the level is at least 4.1 (62.5 Mbit/s High), the
# next one up when the size or rate needs it, as (name, MaxMBPS, MaxFS).
# Every width NVENC takes (<= 256 macroblocks) fits these levels' width limit.
_H264_LEVELS = (
    ("4.1", 245760, 8192),
    ("4.2", 522240, 8704),
    ("5.0", 589824, 22080),
    ("5.1", 983040, 36864),
    ("5.2", 2073600, 36864),
    ("6.0", 4177920, 139264),
    ("6.1", 8355840, 139264),
    ("6.2", 16711680, 139264),
)
KEYFRAME_INTERVAL_S = 1.0
_KEY_EPSILON_S = 0.02
# An engine may return a frame or two more or fewer than it was given (a
# decoder that drops a damaged frame, a stand-in engine). Up to this many (or
# 2% of the clip) are conformed back to the source count (the last frame
# repeated, or the surplus dropped); more than that, either way, means the
# clip is misaligned and is refused.
FRAME_SLACK = 8
# Shapes, colour, and layout come from the plain upscale of the source; only
# detail finer than this (a Gaussian sigma, in source pixels) comes from
# SPAN. SPAN can sharpen what is there but cannot move, reshape, or
# recolour it.
STRUCTURE_SIGMA = 1.5
# 1 is "do not upscale": the clip keeps its size, goes through clean-up and
# finishing, and never reaches SPAN (see :func:`restore_to_delivery`). SPAN only
# does 2x.
_SCALES = frozenset({1, 2})
# Plans saved while 4x existed still name it; they render at 2x.
_RETIRED_SCALES = {4: 2}
# One model load or upscale at a time: two wizard threads (a preview and a
# render) would otherwise share the GPU's memory between two copies.
_GPU_LOCK = threading.Lock()
_GPU_ADVICE = (
    "Close other programs that use the GPU (games, other AI tools) and try again."
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
    """SPAN ran out of GPU memory, even one frame at a time."""


def _gpu_out_of_memory(exc: BaseException) -> bool:
    """True for PyTorch's CUDA out-of-memory error, by name or by text."""
    return (
        type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()
    )


def _forget_frames(exc: BaseException) -> None:
    """Drop the locals ``exc``'s traceback keeps alive (tensors, the model).

    Releasing the GPU while they exist frees nothing: a failed preview once
    held 13 GiB until the server restarted.
    """
    import traceback

    traceback.clear_frames(exc.__traceback__)
    exc.__traceback__ = None


def _upscale_failure(exc: BaseException) -> UpscaleError:
    """The error to raise for a failed upscale call, holding no GPU memory.

    The traceback's frames are cleared (see :func:`_forget_frames`), and the
    new error does not chain to the old one, which would keep them.
    """
    text = str(exc).splitlines()[0] if str(exc) else repr(exc)
    if isinstance(exc, UpscaleError):
        message = str(exc)
    elif isinstance(exc, MemoryError) or "can't allocate memory" in text.lower():
        # torch's CPU allocator says "can't allocate memory": the chunk's
        # frames did not fit in system RAM, which a smaller batch cannot fix.
        message = f"the upscale ran out of system memory: {text}"
    else:
        message = f"SPAN failed: {text}"
    gpu = _gpu_out_of_memory(exc)
    _forget_frames(exc)
    if gpu and not isinstance(exc, GpuOutOfMemory):
        return GpuOutOfMemory(f"{message}. {_GPU_ADVICE}")
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
    # False: skip SPAN and only resize this span (see :func:`restore_to_delivery`).
    use_upscale: bool = True
    # The editor's DPIR level for this span (0 off); see :mod:`.dpir`.
    dpir: int = 0

    def __post_init__(self) -> None:
        _require_dpir(self.dpir)
        scale = _require_scale(self.scale)
        strength = _require_strength(self.strength)
        start = float(self.start_s)
        end = float(self.end_s)
        if not (math.isfinite(start) and math.isfinite(end)):
            raise UpscaleError("span start and end must be finite numbers")
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
            "dpir": self.dpir,
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
            scale = int(data.get("scale", 2))
            return cls(
                start_s=float(data["start_s"]),
                end_s=float(data["end_s"]),
                scale=_RETIRED_SCALES.get(scale, scale),
                strength=float(data.get("strength", 0.5)),
                source=str(data.get("source", "original")),
                look=look,
                regions=regions_from_dicts(data.get("regions")),
                use_upscale=data.get("use_upscale") is not False,
                dpir=level_of(data.get("dpir")),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise UpscaleError(
                "upscale span needs numbers for start_s, end_s and strength"
            ) from exc


@dataclass(frozen=True, slots=True)
class UpscalePlan:
    """Ordered spans covering a job."""

    spans: tuple[UpscaleSpan, ...]
    pending: UpscaleSpan | None = None
    # Requested knobs/range, independent of split spans and render scheduling.
    latest_revision: UpscaleSpan | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "spans": [span.to_dict() for span in self.spans],
        }
        if self.pending is not None:
            payload["pending"] = self.pending.to_dict()
        if self.latest_revision is not None:
            payload["latest_revision"] = self.latest_revision.to_dict()
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
        revision_raw = data.get("latest_revision")
        latest_revision = (
            UpscaleSpan.from_dict(revision_raw) if revision_raw is not None else None
        )
        return cls(
            spans=tuple(UpscaleSpan.from_dict(item) for item in raw_spans),
            pending=pending,
            latest_revision=latest_revision,
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
    # through it; SPAN reports once per block of frames.
    on_progress: ProgressFn | None = field(default=None, compare=False, repr=False)
    # DPIR level (0 off): on the GPU it replaces the look's hqdn3d.
    dpir: int = 0

    def __post_init__(self) -> None:
        _require_dpir(self.dpir)
        object.__setattr__(self, "scale", _require_scale(self.scale))
        object.__setattr__(self, "strength", _require_strength(self.strength))

    @property
    def ffmpeg_look(self) -> ShotLook:
        """The look for FFmpeg-only work: with DPIR asked for, its stand-in."""
        return ffmpeg_look(self.look, self.dpir)


def _require_dpir(level: Any) -> None:
    if level_of(level) != level or isinstance(level, bool):
        raise UpscaleError("dpir must be a whole number from 0 to 3")


class UpscaleEngine(Protocol):
    """Restore and upscale a clip.

    Implementations apply ``request.ffmpeg_look.clean_filter()`` to the
    frames they read (or, running DPIR, the look without its hqdn3d) and
    nothing else; finishing happens in the delivery encode.
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


def models_dir() -> Path:
    """Gitignored weights root. Override with ``SUPER_PROCESSOR_MODELS``."""
    raw = os.environ.get("SUPER_PROCESSOR_MODELS", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.cwd() / ".dogfood" / "models"


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
    note: str = "",
) -> UpscalePlan:
    """Store a range revise. A scale change re-renders the whole timeline.

    The patch trims the spans it overlaps instead of dropping them, so every
    other shot keeps its strength and look. One output file has one
    resolution, so a new scale cannot be spliced into part of it: every span
    takes the new scale, and the pending span covers the whole file, which
    :func:`render_plan` reads as "render every shot again".

    Result notes set ``preserve_shot_settings`` to retune only scale and
    strength, and the DPIR level of each shot in the range by the words of
    their ``note`` (see :func:`.dpir.note_level`). Split those patches at
    shot boundaries; a multi-shot revision uses the full-render path because
    a range splice accepts only one recipe.
    """
    patch = clamp_span(patch, duration_s)
    if _covers_all(patch, duration_s) and not preserve_shot_settings:
        full = replace(patch, start_s=0.0, end_s=duration_s)
        return UpscalePlan(spans=(full,), pending=full, latest_revision=patch)
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
                    dpir=note_level(note, span.dpir),
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
    return UpscalePlan(spans=(*kept, *patches), pending=pending, latest_revision=patch)


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
                *FILE_INPUT,
                "-i",
                str(request.source),
                "-vf",
                _join_filters(
                    request.ffmpeg_look.clean_filter(),
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


@dataclass(frozen=True)
class _Span:
    """The weights to load, already checked: SPAN's, DPIR's, or both."""

    weights: Path | None
    dpir_weights: Path | None = None

    def load(self) -> _Loaded:
        from .dpir import load_dpir
        from .span_sr import load_span

        loaded = _Loaded(None)
        try:
            if self.weights is not None:
                loaded.model = load_span(self.weights)
            if self.dpir_weights is not None:
                loaded.dpir = load_dpir(self.dpir_weights)
        except BaseException:
            _unload(loaded)
            raise
        return loaded


class _Loaded:
    """The networks on the GPU (SPAN, DPIR, either may be ``None``), and the
    batch they run at.

    ``batch`` is 0 (by frame size, see
    :func:`super_processor.span_sr.batch_size`) until a batch runs out of
    memory; the smaller one that fits is then kept for the rest of the load.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.dpir: Any = None
        self.batch = 0


def _span_ready(scale: int) -> _Span:
    """SPAN, checked for this scale, the GPU, and its weights, or a
    wizard-facing error."""
    from ._dist import GPU_INSTALL_HINT
    from .span_sr import SPAN_SCALE, resolve_span_weights

    if scale != SPAN_SCALE:
        raise UpscaleError(f"SPAN only upscales {SPAN_SCALE}x, not {scale}x")
    if not cuda_available():
        raise UpscaleError(
            "SPAN needs an NVIDIA GPU with CUDA. Refusing to fall back to an "
            "FFmpeg grade. Install the gpu extra "
            f"({GPU_INSTALL_HINT}) and an NVIDIA driver."
        )
    return _Span(resolve_span_weights(models_dir()))


def _dpir_weights() -> Path:
    """The DPIR weights, fetched if needed, or a wizard-facing error. A
    Stop during the download ends it (see :func:`.dpir.resolve_weights`)."""
    from .dpir import resolve_weights

    try:
        return resolve_weights(models_dir(), should_stop=_check)
    except DpirError as exc:
        raise UpscaleError(str(exc)) from exc


def dpir_runs_here() -> bool:
    """True when DPIR can run on this machine: PyTorch, spandrel, and CUDA.

    A broken install (any error while importing) counts as "no", so the
    scene falls back to hqdn3d instead of failing the screen that asks.
    """
    try:
        if not cuda_available():
            return False
        importlib.import_module("spandrel")
    except Exception:  # noqa: BLE001 - any import failure means no DPIR
        return False
    return True


def plain_delivery(
    request: UpscaleRequest, dest: Path, *, ffmpeg_bin: str = "ffmpeg"
) -> Path:
    """Deliver ``request.source`` resized by its scale with lanczos, without
    SPAN, in one encode.

    This is the blend's base at strength 0 (see :func:`blend_filter`): the
    look's clean step, the lanczos resize, the finish step. It is encoded
    straight into the delivery settings, decoded on NVDEC where it can be.
    It went through a libx264 work file at 2x first, then a 4K blend whose
    two blurs and three lookup tables a strength of 0 threw away: on 200
    frames of 1440x1080 that took 0.29 CPU-seconds a frame against 0.10 now
    (with NVENC; the lanczos resize is most of what is left).

    A span with DPIR goes through the GPU instead where DPIR can run (see
    :func:`dpir_delivery`), and gets the strongest hqdn3d where it cannot.
    """
    require_supported_timing(request.source, ffmpeg_bin=ffmpeg_bin)
    if request.dpir and dpir_runs_here():
        return dpir_delivery(request, dest)
    size = _frame_size(request.source)
    if size is None:
        raise UpscaleError(f"could not read the frame size of {request.source}")
    out = (size[0] * request.scale, size[1] * request.scale)
    encoder = delivery_encoder(ffmpeg_bin, out)
    rate = _frame_rate(request.source) if encoder == NVENC else None
    dest.parent.mkdir(parents=True, exist_ok=True)
    delivery_marker_path(dest).unlink(missing_ok=True)
    picture = _join_filters(
        _base_chain(out[0], out[1], request.ffmpeg_look),
        request.ffmpeg_look.finish_filter(),
    )
    tmp = dest.with_name(dest.stem + ".encoding" + dest.suffix)
    try:
        _run_ffmpeg(
            [
                ffmpeg_bin,
                *_QUIET,
                "-y",
                *(["-hwaccel", "cuda"] if cuda_decode_works(ffmpeg_bin) else []),
                *FILE_INPUT,
                "-i",
                str(request.source),
                "-map",
                "0:v:0",
                "-vf",
                picture,
                *_delivery_codec(encoder, out, rate),
                str(tmp),
            ],
            on_progress=request.on_progress,
            total_frames=_count_frames(request.source, ffmpeg_bin)
            if request.on_progress is not None
            else 0,
        )
        tmp.replace(dest)
    except BaseException:
        # A failed or stopped encode leaves no half-written file behind.
        unlink_quietly(tmp)
        raise
    look = request.ffmpeg_look
    _write_marker(
        dest,
        encoder,
        blend=_blend_record(0.0, request.regions),
        look=None if look.is_neutral() else look,
    )
    return dest


def dpir_delivery(request: UpscaleRequest, dest: Path) -> Path:
    """A pass-through span (no SPAN) cleaned by DPIR on the GPU, resized by
    its scale with lanczos, and encoded into the delivery settings at
    ``dest`` in one stream (see :func:`_stream_span`)."""
    streamed = replace(request, output=dest, strength=0.0)
    return SpanEngine()._run(_Span(None, _dpir_weights()), [streamed])[0]


def restore_to_delivery(
    engine: UpscaleEngine,
    request: UpscaleRequest,
    dest: Path,
    *,
    use_upscale: bool,
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    """Restore ``request`` and encode it into the delivery settings at ``dest``.

    Without SPAN (``use_upscale`` off, or scale 1) the clip is only resized
    (at scale 1 only cleaned, finished, and re-encoded) by
    :func:`plain_delivery`, which is the blend's base at strength 0, so a
    pass-through span matches its neighbours' size and frame timing.
    ``request.on_progress`` covers the whole part either way: SPAN blends
    and encodes in its own pass, and the encode after it only renames.
    """
    require_supported_timing(request.source, ffmpeg_bin=ffmpeg_bin)
    if not (use_upscale and request.scale > 1):
        return plain_delivery(request, dest, ffmpeg_bin=ffmpeg_bin)
    raw = engine.upscale(request)
    return encode_delivery(
        raw,
        dest,
        ffmpeg_bin=ffmpeg_bin,
        blend_with=request.source,
        strength=request.strength,
        look=request.ffmpeg_look,
        regions=request.regions,
        dpir=request.dpir,
    )


def upscale_all(
    engine: UpscaleEngine,
    requests: list[UpscaleRequest],
    *,
    ffmpeg_bin: str | None = None,
) -> list[Path]:
    """Batch clips, preflighting with the override or engine's FFmpeg selection."""
    if ffmpeg_bin is None:
        ffmpeg_bin = getattr(engine, "ffmpeg_bin", "ffmpeg")
    for request in requests:
        require_supported_timing(request.source, ffmpeg_bin=ffmpeg_bin)
    many = getattr(engine, "upscale_many", None)
    if callable(many):
        return list(many(requests))
    return [engine.upscale(request) for request in requests]


class SpanEngine:
    """2x upscale with SPAN (see :mod:`super_processor.span_sr`), on the GPU.

    A clip streams through three processes at once: an FFmpeg decoder (NVDEC
    where it can) reads the source through the look's clean step as yuv420p,
    the GPU upscales and blends it a block of frames at a time (see
    :func:`~super_processor.span_sr.restore_block`), and the delivery encoder
    (NVENC where it can) takes the finished yuv420p frames as they arrive.
    Nothing is written in between, so a 20-minute 1080p film needs no scratch
    space beyond its output, and host memory holds a few blocks of frames,
    never the clip. On 1440x1080 -> 2880x2160 that is 26 frames a second at
    0.044 CPU-seconds a frame; with the blend and x264 on the CPU it was 22
    at 0.43, and a 16-core CPU reached 95 C.
    """

    def upscale_many(self, requests: list[UpscaleRequest]) -> list[Path]:
        """Restore several short clips (previews) with one model load."""
        if not requests:
            return []
        for request in requests:
            require_supported_timing(request.source)
        backend = _span_ready(requests[0].scale)
        for request in requests[1:]:
            _span_ready(request.scale)
        if any(request.dpir for request in requests):
            backend = replace(backend, dpir_weights=_dpir_weights())
        return self._run(backend, requests)

    def upscale(self, request: UpscaleRequest) -> Path:
        return self.upscale_many([request])[0]

    def _run(self, backend: _Span, requests: list[UpscaleRequest]) -> list[Path]:
        failed: UpscaleError | None = None
        stopped = False
        with _GPU_LOCK:
            loaded = _load(backend)
            try:
                for request in requests:
                    _check()
                    _stream_span(loaded, request)
            except Cancelled as exc:
                # Its traceback holds the frames' tensors: drop them before
                # the GPU is released, and stop outside the lock.
                _forget_frames(exc)
                del exc
                stopped = True
            except Exception as exc:  # noqa: BLE001 - becomes an UpscaleError
                failed = _upscale_failure(exc)
                del exc
            finally:
                _unload(loaded)
        if stopped:
            raise Cancelled
        if failed is not None:
            raise failed from None
        missing = [r.output.name for r in requests if not r.output.is_file()]
        if missing:
            raise UpscaleError(f"SPAN produced no output for {missing[0]}")
        return [request.output for request in requests]


def _load(backend: _Span) -> _Loaded:
    """Load SPAN, or raise a wizard-facing error that holds no GPU memory."""
    try:
        return backend.load()
    except Exception as exc:  # noqa: BLE001 - becomes an UpscaleError
        failure = _upscale_failure(exc)
        del exc
        _release_gpu()
        raise failure from None


def _unload(loaded: _Loaded) -> None:
    _release_gpu(loaded.model)
    loaded.model = None
    if loaded.dpir is not None:
        _release_gpu(loaded.dpir)
        loaded.dpir = None
    _release_gpu()


# Source bytes (yuv420p) decoded per hand-off to the GPU. Each block's 2x
# output is four times as large, and at most two input blocks (queued) and
# two output blocks (one upscaling, one being encoded) exist at once: about
# 1.3 GB of host memory at any source size (57 frames of 1440x1080, 291 of
# 640x480).
STREAM_BLOCK_BYTES = 128 * 1024 * 1024


def yuv420_frame_bytes(width: int, height: int) -> int:
    """Bytes in one packed yuv420p frame, as FFmpeg's rawvideo lays it out:
    chroma planes of ceil(w/2) x ceil(h/2)."""
    return width * height + 2 * (-(-width // 2)) * (-(-height // 2))


def block_frames(width: int, height: int) -> int:
    """Frames per GPU hand-off for a ``width`` x ``height`` source."""
    return max(1, STREAM_BLOCK_BYTES // max(1, yuv420_frame_bytes(width, height)))


# A probe that works is kept for the life of the process. One that fails is
# asked again after this long: a failure that passes (another program holding
# every NVENC session when the wizard starts) would otherwise keep a
# long-running wizard on libx264, and make its next revise of an NVENC film
# re-encode the whole film, until a restart.
_PROBE_RETRY_S = 60.0


class _Probe:
    """``probe(ffmpeg_bin)``, kept once it is True; False is kept for
    :data:`_PROBE_RETRY_S` and then asked again."""

    def __init__(self, probe: Callable[[str], bool]) -> None:
        self._probe = probe
        self._works: set[str] = set()
        self._failed_at: dict[str, float] = {}
        self._lock = threading.Lock()

    def __call__(self, ffmpeg_bin: str) -> bool:
        with self._lock:
            if ffmpeg_bin in self._works:
                return True
            failed = self._failed_at.get(ffmpeg_bin)
            if failed is not None and time.monotonic() - failed < _PROBE_RETRY_S:
                return False
            if self._probe(ffmpeg_bin):
                self._works.add(ffmpeg_bin)
                self._failed_at.pop(ffmpeg_bin, None)
                return True
            self._failed_at[ffmpeg_bin] = time.monotonic()
            return False

    def cache_clear(self) -> None:
        with self._lock:
            self._works.clear()
            self._failed_at.clear()


@_Probe
def _cuda_decode_probe(ffmpeg_bin: str) -> bool:
    """True when ``ffmpeg_bin`` can open a CUDA device for decoding."""
    command = [
        ffmpeg_bin,
        *_QUIET,
        "-init_hw_device",
        "cuda",
        *_PROBE_PICTURE,
        "-f",
        "null",
        "-",
    ]
    return _succeeds(command)


def cuda_decode_works(ffmpeg_bin: str = "ffmpeg") -> bool:
    """Whether SPAN's and pass-through decoders ask for ``-hwaccel cuda``
    (NVDEC).

    A codec NVDEC cannot decode still decodes: FFmpeg falls back to its own
    decoder for it.
    """
    return _cuda_decode_probe(ffmpeg_bin)


def _stream_span(loaded: _Loaded, request: UpscaleRequest) -> Path:
    """Decode, upscale, blend, and encode ``request`` in one pass.

    Writes ``request.output`` as a delivery file (with its marker), so the
    caller's :func:`encode_delivery` of it only renames it. The output holds
    exactly the source's frames: a decoder a few frames short repeats the
    last one, a few too many are dropped, and more than that either way is
    an error. A failed decode is an error too, raised before the output is
    published.
    """
    from . import span_sr
    from .span_sr import SPAN_SCALE

    if loaded.model is None and request.scale not in (1, SPAN_SCALE):
        raise UpscaleError(f"a DPIR pass-through is 1x or {SPAN_SCALE}x")
    if request.dpir and loaded.dpir is None:
        raise UpscaleError("DPIR was asked for but is not loaded")
    scale = SPAN_SCALE if loaded.model is not None else request.scale
    dpir = request.dpir if loaded.dpir is not None else 0
    source = request.source
    size = _frame_size(source)
    if size is None:
        raise UpscaleError(f"could not read the frame size of {source.name}")
    want, rate = video_stream_facts(source)
    width, height = size
    frame_in = yuv420_frame_bytes(width, height)
    per_block = block_frames(width, height)
    slack = max(FRAME_SLACK, int(want * 0.02))
    # With DPIR on the GPU the decoder only deblocks; DPIR replaces hqdn3d.
    clean = gpu_look(request.look, dpir).clean_filter()
    decode = [
        tool_path("ffmpeg"),
        *_QUIET,
        "-nostdin",
        *(["-hwaccel", "cuda"] if cuda_decode_works() else []),
        *FILE_INPUT,
        "-i",
        str(source),
        *(["-vf", clean] if clean else []),
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-",
    ]
    # The strength map is drawn once per clip, on the GPU.
    blend = span_sr.prepare_blend(
        width,
        height,
        strength=request.strength,
        regions=request.regions,
        sigma=STRUCTURE_SIGMA * SPAN_SCALE,
    )
    report = request.on_progress
    blocks: queue.Queue[bytearray | None] = queue.Queue(maxsize=2)
    upscaled: queue.Queue[_Restored | None] = queue.Queue(maxsize=1)
    failures: list[BaseException] = []
    stop = threading.Event()
    # Set when the decoder's output ended (not when reading was stopped).
    decoded = threading.Event()

    with tempfile.TemporaryFile() as decode_errors:
        try:
            decoder = subprocess.Popen(
                resolve_argv(decode), stdout=subprocess.PIPE, stderr=decode_errors
            )
        except OSError as exc:
            raise UpscaleError(f"failed to start ffmpeg: {exc}") from exc

        def read() -> None:
            assert decoder.stdout is not None
            try:
                while not stop.is_set():
                    block = bytearray(frame_in * per_block)
                    got = _read_into(decoder.stdout, block)
                    whole = got - got % frame_in
                    if whole:
                        del block[whole:]
                        blocks.put(block)
                    if got < len(block) or not whole:
                        decoded.set()
                        break
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                failures.append(exc)
            finally:
                blocks.put(None)

        def write(encoder: IO[bytes]) -> None:
            written = received = 0
            last: Any = None
            ended = False
            try:
                while (restored := upscaled.get()) is not None:
                    received += restored.count
                    if received > want + slack:
                        # Stop at once: the rest would only keep the GPU busy.
                        raise UpscaleError(
                            f"SPAN read over {want + slack} frames of a "
                            f"{want}-frame clip ({source.name}); refusing to "
                            "splice a misaligned picture"
                        )
                    for index in range(min(restored.count, want - written)):
                        last = restored.frame(index)
                        encoder.write(last)
                        written += 1
                    if report is not None and want:
                        report(min(1.0, written / want))
                    del restored
                ended = True
                if written < want - slack or written == 0:
                    raise UpscaleError(
                        f"SPAN read {written} frames of a {want}-frame clip "
                        f"({source.name}); refusing to splice a misaligned picture"
                    )
                while written < want:
                    encoder.write(last)
                    written += 1
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                failures.append(exc)
                stop.set()
                # Keep taking pictures so the GPU loop never blocks on us,
                # up to its end mark (unless the loop above already took it).
                while not ended and upscaled.get() is not None:
                    pass

        def feed(encoder: IO[bytes]) -> None:
            reader = threading.Thread(target=read, name="span-decode", daemon=True)
            writer = threading.Thread(
                target=write, args=(encoder,), name="span-encode", daemon=True
            )
            reader.start()
            writer.start()
            decoded_all = False
            try:
                while (block := blocks.get()) is not None:
                    # A stop is seen between blocks, not only at progress.
                    _check()
                    count = len(block) // frame_in
                    restored = _restore_span(
                        loaded,
                        _Prepared(
                            source,
                            size,
                            count,
                            rate,
                            blend,
                            payload=block,
                            scale=scale,
                            dpir=dpir,
                        ),
                    )
                    del block
                    if stop.is_set():
                        break
                    upscaled.put(restored)
                    del restored
                else:
                    decoded_all = True
            except BaseException:
                stop.set()
                raise
            finally:
                upscaled.put(None)
                writer.join()
                stop.set()
                # Stopped early, the reader may be blocked on a full queue:
                # take blocks until its end mark (the loop took it otherwise).
                while not decoded_all and blocks.get() is not None:
                    pass
                reader.join()
            if decoded.is_set() and decoder.wait() != 0:
                # Raised here, the encoder's half-written file is thrown
                # away; it also names the cause of any frame shortfall.
                decode_errors.seek(0)
                detail = decode_errors.read().decode("utf-8", errors="replace")
                reason = detail.strip() or f"ffmpeg exited with {decoder.returncode}"
                raise UpscaleError(f"decoding {source.name} for SPAN failed: {reason}")
            if failures:
                raise failures[0]

        try:
            encode_delivery(
                None,
                request.output,
                blend_with=source,
                strength=request.strength,
                look=request.ffmpeg_look,
                regions=request.regions,
                frames_in=(width * scale, height * scale, rate, feed),
                dpir=dpir,
            )
        finally:
            _stop_ffmpeg(decoder, decoder.stdout)
    return request.output


# How long a streamed FFmpeg gets to end on its own once its pipe is closed
# (an encoder still has a few frames to compress) before it is killed.
_STOP_TIMEOUT_S = 30.0


def _stop_ffmpeg(proc: subprocess.Popen[bytes], pipe: IO[bytes] | None) -> None:
    """End an FFmpeg that frames stream through, even behind a launcher.

    Closing its pipe ends it the way a finished stream would: an encoder
    reads the end of its input, and a decoder fails to write and stops.
    Killing is only the fallback. On Windows, Chocolatey's ffmpeg.exe is a
    shim that starts the real one: killing the shim left the real encoder
    running with its output open, so the failed output could not be deleted
    (WinError 32).
    """
    if pipe is not None:
        with contextlib.suppress(OSError):
            pipe.close()
    try:
        proc.wait(timeout=_STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _read_into(stream: IO[bytes], buffer: bytearray) -> int:
    """Fill ``buffer`` from ``stream``; the bytes read (fewer only at the end)."""
    view = memoryview(buffer)
    got = 0
    while got < len(buffer):
        n = stream.readinto(view[got:])  # type: ignore[attr-defined]
        if not n:
            break
        got += n
    return got


def _restore_span(loaded: _Loaded, prepared: _Prepared) -> _Restored:
    """Upscale and blend one block of decoded frames on the GPU.

    GPU memory follows the batch (frames per call), not the block. A batch
    that does not fit (another program holds GPU memory) is halved until one
    does, down to a single frame; only then is it an error.
    """
    from . import dpir
    from .span_sr import batch_size, restore_block

    width, height = prepared.size
    raw: bytearray = prepared.payload
    prepared.payload = None
    batch = loaded.batch or batch_size(width, height)
    if prepared.dpir:
        batch = min(batch, dpir.batch_size(width, height))
    while True:
        try:
            pictures = restore_block(
                loaded.model,
                prepared.blend,
                raw,
                batch=batch,
                dpir=loaded.dpir if prepared.dpir else None,
                dpir_level=prepared.dpir,
                scale=prepared.scale,
            )
            break
        except Exception as exc:
            if not _gpu_out_of_memory(exc):
                raise
            reason = str(exc).splitlines()[0]
            _forget_frames(exc)
        if batch <= 1:
            raise GpuOutOfMemory(
                f"{'SPAN' if loaded.model is not None else 'DPIR'} ran out of "
                "GPU memory even one frame at a time. "
                f"{_GPU_ADVICE} ({reason})"
            )
        batch = loaded.batch = batch // 2
        _empty_cuda_cache()
    del raw
    size = (width * prepared.scale, height * prepared.scale)
    frame_out = yuv420_frame_bytes(*size)
    view = memoryview(pictures)
    return _Restored(
        count=len(pictures) // frame_out,
        size=size,
        frame=lambda index: view[index * frame_out : (index + 1) * frame_out],
    )


def _empty_cuda_cache() -> None:
    """Return PyTorch's cached GPU blocks to the driver, if CUDA is loaded."""
    import gc

    gc.collect()
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


def _release_gpu(model: Any | None = None) -> None:
    """Move the model off the GPU and return its cache to the driver.

    Deleting the name inside this function does not drop the caller's
    reference, so the caller sets its own variable to None and calls this
    again.
    """
    mover = getattr(model, "to", None)
    if mover is not None:
        with contextlib.suppress(Exception):
            mover("cpu")
    del model
    _return_freed_heap()
    _empty_cuda_cache()


def _return_freed_heap() -> None:
    """Hand the heap an upscale call freed back to the system (glibc only).

    Decoding and upscaling a chunk go through several GiB of host memory.
    Freed, it stays in glibc's heap, split over one arena per worker thread,
    so the wizard kept about 10 GiB between previews (measured with the
    FlashVSR engine SPAN replaced). Asking glibc to trim took a process from
    5.1 to 2.5 GiB after each call. Elsewhere this does nothing.
    """
    import ctypes
    import ctypes.util

    name = ctypes.util.find_library("c")
    if not name or not sys.platform.startswith("linux"):
        return
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL(name).malloc_trim(0)


@dataclass
class _Prepared:
    """A block ready for SPAN: its geometry, blend, and decoded frames."""

    source: Path
    size: tuple[int, int]
    frames: int
    rate: str
    blend: Any = None
    payload: Any = None
    scale: int = 2
    dpir: int = 0


@dataclass(frozen=True)
class _Restored:
    """The GPU's finished frames, in memory, as packed yuv420p."""

    count: int
    size: tuple[int, int]
    frame: Callable[[int], bytes | memoryview]


def video_stream_facts(video: Path, *, ffprobe_bin: str = "ffprobe") -> tuple[int, str]:
    """Frame count (packets, no decode) and ``r_frame_rate`` of the video."""
    try:
        completed = subprocess.run(
            resolve_argv(
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
                    *FILE_INPUT,
                    str(video),
                ]
            ),
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
            *FILE_INPUT,
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


def _delivery_marker(encoder: str = X264) -> dict[str, Any]:
    """What a delivery made with ``encoder`` records about its encode."""
    quality: dict[str, Any] = (
        {"preset": NVENC_PRESET, "cq": NVENC_CQ}
        if encoder == NVENC
        else {"preset": X264_PRESET, "crf": X264_CRF}
    )
    return {
        "version": DELIVERY_VERSION,
        "bframes": 0,
        "codec": encoder,
        **quality,
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


def is_delivery(video: Path, *, ffmpeg_bin: str = "ffmpeg") -> bool:
    """True when ``video`` was written by :func:`encode_delivery` (this
    version) with the encoder this machine picks for its size, so a part
    encoded now splices into it by stream copy."""
    data = _read_marker(video)
    if data is None:
        return False
    encoder = delivery_encoder(ffmpeg_bin, _frame_size(video))
    expected = _delivery_marker(encoder)
    return all(data.get(key) == value for key, value in expected.items())


def _write_marker(
    video: Path,
    encoder: str,
    *,
    blend: dict[str, Any] | None = None,
    look: ShotLook | None = None,
) -> None:
    data = _delivery_marker(encoder)
    if blend is not None:
        data["blend"] = blend
    if look is not None:
        data["look"] = look.to_dict()
    delivery_marker_path(video).write_text(json.dumps(data) + "\n", encoding="utf-8")


def _blend_record(
    strength: float, regions: tuple[Region, ...], dpir: int = 0
) -> dict[str, Any]:
    """What a blended encode was made with, so an identical one can be skipped."""
    record: dict[str, Any] = {
        "strength": strength,
        "structure_sigma": STRUCTURE_SIGMA,
        "regions": [region.to_dict() for region in regions],
    }
    if dpir:
        record["dpir"] = dpir
    return record


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
    the base's shapes and colour with SPAN's fine detail. The strength
    ``s`` is per pixel: lower inside protected ``regions``.

    Every two-input step takes the restored picture's branch as its first
    (main) input with ``eof_action=pass``, so the frame count the keyframe
    splice relies on never changes. The steps are lookup tables and a mask
    drawn once, not an expression per pixel, which ran at about 5 frames a
    second at 1280x960. The base is padded by a second of its last frame so a
    few missing frames do not leave the tail unmixed. The base gets the look's
    clean step, as the engine input did, and the mix gets its finish step.
    """
    _require_strength(strength)
    look = look or ShotLook()
    base = _join_filters(
        _base_chain(width, height, look), "tpad=stop_mode=clone:stop_duration=1"
    )
    finish = look.finish_filter()
    mixed = "[mix];[mix]" + finish + "[v]" if finish else "[v]"
    if sigma <= 0:
        graph = (
            f"[0:v]setpts=PTS-STARTPTS,format=yuv420p[rebuilt];[1:v]{base}[basemix];"
        )
    else:
        blur = f"gblur=sigma={sigma:.3f}"
        graph = (
            "[0:v]setpts=PTS-STARTPTS,format=yuv420p,split[up][upc];"
            f"[upc]{blur}[uplow];"
            f"[1:v]{base},split[basec][basemix];"
            f"[basec]{blur}[baselow];"
            # Restored detail, offset to mid grey so negative values survive.
            f"[up][uplow]{_lut2('x-y+128')}[detail];"
            # The base's coarse layer with the restored detail on top.
            f"[detail][baselow]{_lut2('y+x-128')}[rebuilt];"
        )
    share = strength_expr(strength, regions)
    if share == f"{strength:.4f}":
        # One share for every pixel: a lookup table, the same sum the
        # expression gave, at the speed of a table read.
        return graph + f"[rebuilt][basemix]{_lut2(f'y+(x-y)*{share}')}{mixed}"
    # A share that changes across the frame cannot be a table. It is drawn
    # once into two masks, the share and what is left of it. Each picture
    # is weighted by its mask against black, with that picture as the first
    # input so its frames keep their own timestamps and length; the sum then
    # takes the restored picture's frame count. In geq, W and H are the luma
    # size on every plane, so box corners are scaled by its subsampling.
    plane = share.replace("X/W", "X/(W*SW)").replace("Y/H", "Y/(H*SH)")
    # One frame (r=1, d=1): framesync repeats it for the whole picture.
    still = f"color=c=black:s={width}x{height}:r=1:d=1,format=yuv420p"

    def mask(level: str) -> str:
        return f"{still},geq=lum={level}:cb={level}:cr={level}"

    restored_mask = mask(f"'255*(1-({plane}))'")
    base_mask = mask(f"'255*({plane})'")
    return graph + (
        # maskedmerge gives first + (second - first) * mask: against black,
        # a mask of (1 - weight) leaves first * weight.
        f"{restored_mask}[keepr];"
        f"{base_mask}[keepb];"
        f"{mask('0')},split[zeror][zerob];"
        "[rebuilt][zeror][keepr]maskedmerge[rpart];"
        "[basemix][zerob][keepb]maskedmerge[bpart];"
        f"[rpart][bpart]{_lut2('x+y')}{mixed}"
    )


def _base_chain(width: int, height: int, look: ShotLook) -> str:
    """The plain upscale everything is mixed onto: the look's clean step,
    then lanczos to ``width`` x ``height``."""
    return _join_filters(
        "setpts=PTS-STARTPTS",
        look.clean_filter(),
        f"scale={width}:{height}:flags=lanczos,format=yuv420p",
    )


def _lut2(expr: str) -> str:
    """A two-input ``lut2`` on every plane; the first input sets the length."""
    planes = ":".join(f"c{index}='{expr}'" for index in range(3))
    return f"lut2={planes}:eof_action=pass"


def _join_filters(*parts: str) -> str:
    return ",".join(part for part in parts if part)


# Frames fed to the delivery encoder through a pipe: their width, height,
# frame rate, and a function that writes them (packed yuv420p) to the pipe.
FramesIn = tuple[int, int, str, Callable[[IO[bytes]], None]]


@contextlib.contextmanager
def one_encoder() -> Iterator[None]:
    """Decide each FFmpeg's delivery encoder once for everything inside.

    A failed NVENC probe is asked again after a while (see :class:`_Probe`),
    so its answer can change. Inside this, on this thread, it does not: the
    parts of one output, and the check that a part needs no second encode,
    all see the same encoder, so one file never mixes encoders.
    """
    if getattr(_PINNED, "choices", None) is not None:
        yield
        return
    _PINNED.choices = {}
    try:
        yield
    finally:
        _PINNED.choices = None


_PINNED = threading.local()


def _nvenc_here(ffmpeg_bin: str) -> bool:
    choices: dict[str, bool] | None = getattr(_PINNED, "choices", None)
    if choices is None:
        return nvenc_works(ffmpeg_bin)
    if ffmpeg_bin not in choices:
        choices[ffmpeg_bin] = nvenc_works(ffmpeg_bin)
    return choices[ffmpeg_bin]


@one_encoder()
def encode_delivery(
    source: Path | None,
    dest: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    concat: bool = False,
    blend_with: Path | None = None,
    strength: float = 1.0,
    look: ShotLook | None = None,
    regions: tuple[Region, ...] = (),
    on_progress: ProgressFn | None = None,
    frames_in: FramesIn | None = None,
    dpir: int = 0,
) -> Path:
    """Encode a picture once into the settings every splice part shares.

    A forced IDR frame every second, and no B-frames, give a range revise
    clean points to cut the current output by stream copy. ``concat`` reads
    ``source`` as a concat-demuxer listing. With ``blend_with``, the picture
    is mixed with a lanczos upscale of ``blend_with`` in the same encode,
    taking its coarse layer from that base (see :func:`blend_filter`). The
    look's finish step runs in this encode too. ``on_progress`` gets the
    share of frames encoded, 0 to 1, when the frame count can be read.
    With ``frames_in`` the picture comes from a pipe instead of ``source``
    (see :func:`_stream_span`), already blended on the GPU: ``blend_with``
    and the strength are only recorded. Its feeder reports its own progress.
    ``dpir`` is recorded with the blend: the level DPIR cleaned the picture
    at, so a file made without it is never reused for a span that asks.
    """
    look = look or ShotLook()
    record = _blend_record(strength, regions, dpir) if blend_with is not None else None
    finish = look.finish_filter()
    if frames_in is not None:
        if record is None:
            raise UpscaleError("a piped picture is always blended")
        return _encode_piped(frames_in, dest, look, ffmpeg_bin, record)
    if source is None:
        raise UpscaleError("encode_delivery needs a source or frames_in")
    if not concat and is_delivery(source, ffmpeg_bin=ffmpeg_bin):
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
        ["-f", "concat", "-safe", "0", *FILE_INPUT, "-i", str(source)]
        if concat
        else [
            *FILE_INPUT,
            "-i",
            str(source),
        ]
    )
    picture = ["-map", "0:v:0"]
    picture_file = _first_part(source) if concat else source
    size = _frame_size(picture_file)
    encoder = delivery_encoder(ffmpeg_bin, size)
    rate = _frame_rate(picture_file) if encoder == NVENC else None
    if blend_with is not None:
        if size is None:
            raise UpscaleError(f"could not read the frame size of {source}")
        base_size = _frame_size(blend_with)
        if base_size is None:
            raise UpscaleError(f"could not read the frame size of {blend_with}")
        sigma = STRUCTURE_SIGMA * size[0] / base_size[0]
        inputs.extend([*FILE_INPUT, "-i", str(blend_with)])
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
    try:
        _run_ffmpeg(
            [
                ffmpeg_bin,
                *_QUIET,
                "-y",
                *inputs,
                *picture,
                *_delivery_codec(encoder, size, rate),
                str(tmp),
            ],
            on_progress=on_progress,
            total_frames=_count_frames(
                blend_with if blend_with is not None else None if concat else source,
                ffmpeg_bin,
            )
            if on_progress is not None
            else 0,
        )
        tmp.replace(dest)
    except BaseException:
        # A failed or stopped encode leaves no half-written file behind.
        unlink_quietly(tmp)
        raise
    _write_marker(dest, encoder, blend=record, look=None if look.is_neutral() else look)
    return dest


_QUIET = ("-hide_banner", "-loglevel", "error")
# One black frame, for the probes below.
_PROBE_PICTURE = (
    "-f",
    "lavfi",
    "-i",
    "color=c=black:s=256x144:d=0.04",
    "-frames:v",
    "1",
)


def _succeeds(command: list[str]) -> bool:
    """True when ``command`` runs and exits 0; any failure is False.

    Only the exit code is read, so nothing is piped. On Windows a timeout
    kills Chocolatey's ffmpeg shim and then reads its pipes to the end; the
    real ffmpeg it started still held them, so a hung probe never returned.
    """
    try:
        completed = subprocess.run(
            resolve_argv(command),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


@_Probe
def _nvenc_probe(ffmpeg_bin: str) -> bool:
    """True when ``ffmpeg_bin`` can encode a frame with NVENC's H.264.

    FFmpeg lists h264_nvenc whenever it was built with it, GPU or not, so
    only an encode tells.
    """
    command = [
        ffmpeg_bin,
        *_QUIET,
        *_PROBE_PICTURE,
        "-c:v",
        NVENC,
        "-f",
        "null",
        "-",
    ]
    return _succeeds(command)


def nvenc_works(ffmpeg_bin: str = "ffmpeg") -> bool:
    """Whether NVENC's H.264 encoder works here (see :class:`_Probe`)."""
    return _nvenc_probe(ffmpeg_bin)


def delivery_encoder(
    ffmpeg_bin: str = "ffmpeg", size: tuple[int, int] | None = None
) -> str:
    """The H.264 encoder for a delivery of ``size`` (``None``: any size).

    NVENC where it works and takes the size, so the CPU does not encode;
    libx264 elsewhere (CI, Macs, AMD), in the same container and keyframe
    settings. Inside :func:`one_encoder` the answer does not change.
    """
    if not _nvenc_here(ffmpeg_bin):
        return X264
    if size is not None and not (
        _NVENC_MIN[0] <= size[0] <= _NVENC_MAX[0]
        and _NVENC_MIN[1] <= size[1] <= _NVENC_MAX[1]
    ):
        return X264
    return NVENC


def h264_level(size: tuple[int, int], rate: str) -> str | None:
    """The H.264 level NVENC is given for ``size`` at ``rate`` frames/s.

    ``None`` when the rate cannot be read or no level is big enough: NVENC
    then picks its own.
    """
    try:
        fps = Fraction(rate)
    except (ValueError, ZeroDivisionError):
        return None
    macroblocks = -(-size[0] // 16) * -(-size[1] // 16)
    for name, per_second, per_frame in _H264_LEVELS:
        if macroblocks <= per_frame and macroblocks * fps <= per_second:
            return name if fps > 0 else None
    return None


def _delivery_codec(
    encoder: str, size: tuple[int, int] | None = None, rate: str | None = None
) -> list[str]:
    """The encode every delivery part shares, so parts splice by stream copy.

    Parts of one output share its size and rate, so they share the level.
    """
    if encoder == NVENC:
        level = None if size is None or rate is None else h264_level(size, rate)
        quality = [
            "-c:v",
            NVENC,
            "-preset",
            NVENC_PRESET,
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            str(NVENC_CQ),
            "-b:v",
            "0",
            "-profile:v",
            "high",
            *(["-level", level] if level else []),
        ]
    else:
        quality = ["-c:v", X264, "-preset", X264_PRESET, "-crf", str(X264_CRF)]
    return [
        *quality,
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
    ]


def _encode_piped(
    frames_in: FramesIn,
    dest: Path,
    look: ShotLook,
    ffmpeg_bin: str,
    record: dict[str, Any],
) -> Path:
    """:func:`encode_delivery` of yuv420p frames written to the encoder's
    stdin. Only a look's finish step (eq, grain) still runs on the CPU."""
    width, height, rate, feed = frames_in
    dest.parent.mkdir(parents=True, exist_ok=True)
    delivery_marker_path(dest).unlink(missing_ok=True)
    tmp = dest.with_name(dest.stem + ".encoding" + dest.suffix)
    encoder_name = delivery_encoder(ffmpeg_bin, (width, height))
    finish = look.finish_filter()
    command = [
        ffmpeg_bin,
        *_QUIET,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        f"{width}x{height}",
        "-framerate",
        rate,
        "-i",
        "pipe:0",
        *(["-vf", finish] if finish else []),
        *_delivery_codec(encoder_name, (width, height), rate),
        str(tmp),
    ]
    # Errors go to a file: nothing reads the encoder's stderr while it runs.
    with tempfile.TemporaryFile() as errors:
        try:
            encoder = subprocess.Popen(
                resolve_argv(command), stdin=subprocess.PIPE, stderr=errors
            )
        except OSError as exc:
            raise UpscaleError(f"failed to start ffmpeg: {exc}") from exc
        assert encoder.stdin is not None
        try:
            feed(encoder.stdin)
            encoder.stdin.close()
        except BaseException as exc:
            # A feeder that fails because the encoder died reports the
            # encoder's own error instead (a broken pipe says nothing).
            # A stop stays a stop, whatever the encoder printed as it ended.
            _stop_ffmpeg(encoder, encoder.stdin)
            unlink_quietly(tmp)
            if isinstance(exc, Cancelled):
                raise
            errors.seek(0)
            detail = errors.read().decode("utf-8", errors="replace").strip()
            if detail:
                raise UpscaleError(f"encoding SPAN frames failed: {detail}") from None
            raise
        encoder.wait()
        errors.seek(0)
        detail = errors.read().decode("utf-8", errors="replace").strip()
    if encoder.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise UpscaleError(f"encoding SPAN frames failed: {detail or 'ffmpeg failed'}")
    tmp.replace(dest)
    _write_marker(
        dest, encoder_name, blend=record, look=None if look.is_neutral() else look
    )
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
            resolve_argv(
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
                    *FILE_INPUT,
                    str(video),
                ]
            ),
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
    return tool_path("ffprobe")


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
    command.extend([*FILE_INPUT, "-i", str(video)])
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


def _frame_rate(video: Path) -> str | None:
    """The video's ``r_frame_rate`` (``None`` when it cannot be read)."""
    from .probe import ProbeError, probe_file

    try:
        stream = probe_file(video).primary_video()
    except ProbeError:
        return None
    return None if stream is None else stream.r_frame_rate


def _frame_size(video: Path) -> tuple[int, int] | None:
    from .probe import ProbeError, probe_file

    try:
        stream = probe_file(video).primary_video()
    except ProbeError:
        return None
    if stream is None or not stream.width or not stream.height:
        return None
    return stream.width, stream.height


class RenderControl(Protocol):
    """What a caller can install around a render to stop it (see
    :func:`render_control`)."""

    def check(self) -> None:
        """Called often while FFmpeg or SPAN works; raises
        :class:`~super_processor.cancel.Cancelled` once the user has asked to
        stop. The running FFmpeg is then ended and the stop passes through."""

    def commit(self) -> None:
        """Called once, just before ``output`` is replaced. Raising here
        (the user stopped the render) leaves the old output in place;
        returning means the render can no longer be stopped."""


_CONTROL: contextvars.ContextVar[RenderControl | None] = contextvars.ContextVar(
    "super_processor_render_control", default=None
)


@contextlib.contextmanager
def render_control(control: RenderControl) -> Iterator[None]:
    """Install ``control`` for renders run on this thread inside the block."""
    token = _CONTROL.set(control)
    try:
        yield
    finally:
        _CONTROL.reset(token)


def _check() -> None:
    """Raise :class:`Cancelled` when the installed control says to stop."""
    control = _CONTROL.get()
    if control is not None:
        control.check()


def _commit() -> None:
    """The point of no return before a render replaces its output."""
    control = _CONTROL.get()
    if control is not None:
        control.commit()


def _commit_output(
    work_file: Path, output: Path, source: Path, *, ffmpeg_bin: str
) -> Path:
    """Finish ``work_file`` (its marker already beside it) and put it in place.

    The audio is muxed into the work file first, so ``output`` is replaced
    once, by a complete file: it is always either the old output or the new
    one, never a new picture still waiting for its audio. Anything that fails
    before the replace leaves the old output and its marker as they were.
    """
    mux_source_audio(work_file, source, ffmpeg_bin=ffmpeg_bin)
    _commit()
    work_file.replace(output)
    marker = delivery_marker_path(work_file)
    if marker.is_file():
        marker.replace(delivery_marker_path(output))
    else:
        delivery_marker_path(output).unlink(missing_ok=True)
    return output


@one_encoder()
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
    delivery = restore_to_delivery(
        engine,
        UpscaleRequest(
            source,
            work / "full_up.mp4",
            scale=span.scale,
            strength=span.strength,
            look=span.look,
            regions=span.regions,
            on_progress=on_progress,
            dpir=span.dpir,
        ),
        work / "full_delivery.mp4",
        use_upscale=span.use_upscale,
        ffmpeg_bin=ffmpeg_bin,
    )
    return _commit_output(delivery, output, source, ffmpeg_bin=ffmpeg_bin)


@one_encoder()
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
    on_revise_range: Callable[[float, float], None] | None = None,
) -> Path:
    """Re-upscale ``span`` from the original source and splice it into ``current``.

    The range widens out to the keyframes of ``current``. The head and tail are
    stream-copied, so repeated revises never re-encode the untouched parts.
    A ``current`` from before the delivery settings existed is encoded into
    them once first. ``on_revise_range`` reports the actual expanded interval
    after probing that delivery's keyframes, for seam checks.
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
    if not is_delivery(current, ffmpeg_bin=ffmpeg_bin):
        current = encode_delivery(current, work / "current.mp4", ffmpeg_bin=ffmpeg_bin)
    keys = keyframe_times(current, ffprobe_bin=_ffprobe_for(ffmpeg_bin))
    start, end = snap_to_keyframes(keys, span.start_s, span.end_s, duration_s)
    if on_revise_range is not None:
        on_revise_range(start, duration_s if end is None else end)
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
    middle = restore_to_delivery(
        engine,
        UpscaleRequest(
            clip,
            work / "range_up.mp4",
            scale=span.scale,
            strength=span.strength,
            look=span.look,
            regions=span.regions,
            on_progress=on_progress,
            dpir=span.dpir,
        ),
        work / "range_delivery.mp4",
        use_upscale=span.use_upscale,
        ffmpeg_bin=ffmpeg_bin,
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
    _write_marker(silent, delivery_encoder(ffmpeg_bin, _frame_size(silent)))
    return _commit_output(silent, output, source, ffmpeg_bin=ffmpeg_bin)


def render_plan(
    source: Path,
    output: Path,
    plan: UpscalePlan,
    *,
    engine: UpscaleEngine,
    ffmpeg_bin: str = "ffmpeg",
    duration_s: float,
    on_progress: ProgressFn | None = None,
    on_revise_range: Callable[[float, float], None] | None = None,
) -> Path:
    """Render what ``plan`` asks for into ``output``.

    ``on_progress`` gets the share of the work done, 0 to 1, by source frames.
    ``on_revise_range`` receives the actual keyframe-expanded interval when
    revising an existing output; full shot renders do not call it.

    A pending range on an existing output is spliced in by
    :func:`apply_range_revise`. Anything else (the first render, a pending
    span over the whole file, a scale change, an output at another scale
    than the pending range) renders every shot again.
    """
    require_supported_timing(source, ffmpeg_bin=ffmpeg_bin)
    pending = plan.pending
    if (
        pending is not None
        and output.is_file()
        and not _covers_all(pending, duration_s)
        # An output rendered at another scale (4x, before it was retired,
        # with the plan now loaded at 2x) cannot take a 2x range.
        and rendered_scale(source, output) in (None, pending.scale)
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
            on_revise_range=on_revise_range,
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


def rendered_scale(source: Path, output: Path) -> int | None:
    """The whole-number scale ``output`` has over ``source``; None when either
    size cannot be read or they are not a whole multiple of each other."""
    source_size = _frame_size(source)
    output_size = _frame_size(output)
    if source_size is None or output_size is None:
        return None
    (width, height), (out_width, out_height) = source_size, output_size
    if out_width % width or out_width // width != out_height / height:
        return None
    return out_width // width


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
    """Join neighbouring spans whose complete restoration settings match.

    Every field but the times is compared (the DPIR level too), so a field
    added to :class:`UpscaleSpan` later cannot be merged away unnoticed.
    """
    out: list[UpscaleSpan] = []
    for span in ordered:
        last = out[-1] if out else None
        if last is not None and _settings(last) == _settings(span):
            out[-1] = replace(last, end_s=span.end_s)
        else:
            out.append(span)
    return out


def _settings(span: UpscaleSpan) -> UpscaleSpan:
    """``span`` without its times, to compare restoration settings."""
    return replace(span, start_s=0.0, end_s=1.0)


# Where a part is cut, in frames before the frame it starts on. FFmpeg rounds
# a seek time to the nearest frame, so a cut half a frame early sits exactly on
# the rounding edge: with the time written to six decimals, 141.5/24 became
# 141.49999 and the part started a frame early, shifting the whole segment.
# A quarter frame keeps clear of both that edge and the frame time itself.
_CUT_MARGIN_FRAMES = 0.25


# What a source frame costs to render without SPAN (one lanczos resize and
# delivery encode, see :func:`plain_delivery`), against 1 for a frame SPAN
# upscales and blends. On 200 frames of 1440x1080 on an RTX 5090: 2.6 s
# against 7.6 s for SPAN at 26 fps. (With a 2x work file and an FFmpeg
# blend it was 5.0 s; with FlashVSR at 9.5 fps it was 0.05.)
PLAIN_FRAME_COST = 0.35


def _upscaled(span: UpscaleSpan) -> bool:
    return span.use_upscale and span.scale > 1


def _frame_cost(span: UpscaleSpan) -> float:
    return 1.0 if _upscaled(span) else PLAIN_FRAME_COST


def render_shot_parts(
    spans: tuple[UpscaleSpan, ...], duration_s: float, total: int, fps: float
) -> list[tuple[UpscaleSpan, int, int]]:
    """Coalesced render parts and their exact source-frame edges."""
    ordered = _coalesce(shot_timeline(spans, duration_s))
    edges = [0, *(round(span.start_s * fps) for span in ordered[1:]), total]
    return [
        (span, first, stop)
        for span, first, stop in zip(ordered, edges, edges[1:], strict=False)
        if stop > first
    ]


def _cut_time(frame: int, fps: float) -> float:
    """The seek time that starts (or ends) a part exactly at ``frame``."""
    return max(0.0, (frame - _CUT_MARGIN_FRAMES) / fps)


@one_encoder()
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
    shots with the same settings render as one part, so SPAN is not
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
    timeline = render_shot_parts(spans, duration_s, want, fps)
    # Each part's share of the work is its frames times what a frame costs,
    # so a part SPAN skips does not make the time left jump.
    costs = [(stop - first) * _frame_cost(span) for span, first, stop in timeline]
    total_cost = sum(costs) or 1.0
    parts: list[Path] = []
    done_cost = 0.0
    for index, (span, first, stop) in enumerate(timeline):
        whole = _part_of(on_progress, done_cost / total_cost, costs[index] / total_cost)
        done_cost += costs[index]
        clip = trim_source(
            source,
            work / f"shot_{index}.mp4",
            _cut_time(first, fps),
            _cut_time(stop, fps),
            ffmpeg_bin=ffmpeg_bin,
            video_filter="setpts=PTS-STARTPTS",
        )
        parts.append(
            restore_to_delivery(
                engine,
                UpscaleRequest(
                    clip,
                    work / f"shot_{index}_up.mp4",
                    scale=span.scale,
                    strength=span.strength,
                    look=span.look,
                    regions=span.regions,
                    on_progress=whole,
                    dpir=span.dpir,
                ),
                work / f"shot_{index}_delivery.mp4",
                use_upscale=span.use_upscale,
                ffmpeg_bin=ffmpeg_bin,
            )
        )
        if whole is not None:
            whole(1.0)
    silent = work / "shots.mp4"
    _concat(parts, silent, ffmpeg_bin=ffmpeg_bin)
    have, _rate = video_stream_facts(silent, ffprobe_bin=ffprobe_bin)
    if have != want:
        raise UpscaleError(
            f"the joined shots have {have} frames instead of {want}; "
            "the previous output is left unchanged"
        )
    _write_marker(silent, delivery_encoder(ffmpeg_bin, _frame_size(silent)))
    return _commit_output(silent, output, source, ffmpeg_bin=ffmpeg_bin)


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

    Audio MP4 can carry as it is (see ``_COPY_AUDIO``) is copied, not
    encoded again: re-encoding a 20-minute film's AAC track was 15 of the
    18 CPU-seconds this step took, a single-core burst that took a
    heat-soaked CPU from 86 C to 94.6 C at the end of a render. Other
    audio, or a copy FFmpeg refuses, is encoded to AAC.
    """
    ffprobe_bin = _ffprobe_for(ffmpeg_bin)
    frames, rate = video_stream_facts(video, ffprobe_bin=ffprobe_bin)
    length = frames / Fraction(rate)
    codecs = ["aac"]
    if _audio_codec(source, ffprobe_bin=ffprobe_bin) in _COPY_AUDIO:
        codecs.insert(0, "copy")
    for codec in codecs:
        mixed = _mux_audio(video, source, codec, length, ffmpeg_bin=ffmpeg_bin)
        if mixed is None:
            continue
        # A copy keeps the first packet's AAC config for the whole track, so
        # a recording whose audio changes format partway (stereo to 5.1, 48
        # to 44.1 kHz) muxes without complaint and then decodes garbled. An
        # encode reconfigures at the change; decoding costs far less.
        if codec == "copy" and not _audio_decodes(mixed, ffmpeg_bin=ffmpeg_bin):
            mixed.unlink(missing_ok=True)
            continue
        if video_stream_facts(mixed, ffprobe_bin=ffprobe_bin)[0] != frames:
            mixed.unlink(missing_ok=True)
            return video
        mixed.replace(video)
        return video
    return video


# Audio codecs every browser plays from MP4, so they are copied as they are.
_COPY_AUDIO = frozenset({"aac", "mp3"})


def _audio_codec(source: Path, *, ffprobe_bin: str) -> str | None:
    """The first audio stream's codec name, or None with no audio."""
    try:
        completed = subprocess.run(
            resolve_argv(
                [
                    ffprobe_bin,
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=codec_name",
                    "-of",
                    "csv=p=0",
                    *FILE_INPUT,
                    str(source),
                ]
            ),
            check=False,
            capture_output=True,
            # ffprobe echoes paths in its own bytes; a strict locale decode
            # of an odd one would crash the end of a finished render.
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return None
    name = (completed.stdout or "").strip().splitlines()
    return name[0].strip() if completed.returncode == 0 and name else None


def _audio_decodes(video: Path, *, ffmpeg_bin: str) -> bool:
    """True when ``video``'s audio decodes from end to end without an error."""
    try:
        completed = subprocess.run(
            resolve_argv(
                [
                    ffmpeg_bin,
                    "-hide_banner",
                    "-v",
                    "error",
                    "-xerror",
                    *FILE_INPUT,
                    "-i",
                    str(video),
                    "-map",
                    "0:a:0?",
                    "-f",
                    "null",
                    "-",
                ]
            ),
            check=False,
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and not completed.stderr.strip()


def _mux_audio(
    video: Path, source: Path, codec: str, length: Fraction, *, ffmpeg_bin: str
) -> Path | None:
    """``video``'s picture with ``source``'s audio as ``codec``, or None."""
    mixed = video.with_suffix(".mux.mp4")
    completed = subprocess.run(
        resolve_argv(
            [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                *FILE_INPUT,
                "-i",
                str(video),
                *FILE_INPUT,
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-c:v",
                "copy",
                "-c:a",
                codec,
                "-t",
                f"{float(length):.6f}",
                # moov at the front, so a browser can start playing and seeking
                # before the whole file has been read.
                "-movflags",
                "+faststart",
                str(mixed),
            ]
        ),
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0 or not mixed.is_file():
        mixed.unlink(missing_ok=True)
        return None
    return mixed


def _write_listing(path: Path, parts: list[Path]) -> Path:
    from .render import write_concat_list

    return write_concat_list(path, parts)


def _concat(parts: list[Path], dest: Path, *, ffmpeg_bin: str) -> None:
    from .render import build_concat_argv

    listing = _write_listing(dest.parent / "splice.txt", parts)
    _run_ffmpeg(build_concat_argv(listing, dest, ffmpeg_bin=ffmpeg_bin))


# How often a running FFmpeg is checked for a stop.
_POLL_S = 0.1
# How long the progress reader gets to finish once FFmpeg has been ended.
_READER_JOIN_S = 5.0


def _run_ffmpeg(
    command: list[str],
    *,
    on_progress: ProgressFn | None = None,
    total_frames: int = 0,
) -> None:
    """Run ``command``; with ``on_progress``, report frames out of ``total_frames``.

    The calling thread waits in short steps and asks the installed
    :class:`RenderControl` (see :func:`render_control`) whether to stop each
    time. A stop, or anything ``on_progress`` raises (the wizard raises
    :class:`Cancelled` from it), ends FFmpeg and is raised here.
    """
    watched = on_progress is not None and total_frames > 0
    if watched:
        command = [command[0], "-progress", "pipe:1", "-nostats", *command[1:]]
    # Errors go to a file: read from a pipe only after the progress, a long
    # error output would fill it and stall ffmpeg.
    with tempfile.TemporaryFile() as errors:
        try:
            proc = subprocess.Popen(
                resolve_argv(command),
                stdout=subprocess.PIPE if watched else subprocess.DEVNULL,
                stderr=errors,
                text=watched,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise UpscaleError(f"failed to start ffmpeg: {exc}") from exc
        failures: list[BaseException] = []
        reader: threading.Thread | None = None
        if watched:
            assert on_progress is not None
            reader = threading.Thread(
                target=_read_progress,
                args=(proc, on_progress, total_frames, failures),
                name="ffmpeg-progress",
                daemon=True,
            )
            reader.start()
        try:
            while True:
                try:
                    proc.wait(timeout=_POLL_S)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if failures:
                    raise failures[0]
                _check()
            if reader is not None:
                # The last progress lines are read before the result is.
                reader.join()
            if failures:
                raise failures[0]
        except BaseException:
            stop_process(proc)
            if reader is not None:
                reader.join(timeout=_READER_JOIN_S)
            raise
        finally:
            if proc.stdout is not None:
                with contextlib.suppress(OSError):
                    proc.stdout.close()
        errors.seek(0)
        detail = errors.read().decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise UpscaleError(detail.strip() or "ffmpeg failed")


def _read_progress(
    proc: subprocess.Popen[str],
    on_progress: ProgressFn,
    total_frames: int,
    failures: list[BaseException],
) -> None:
    """Report FFmpeg's ``-progress`` frame lines until its output ends.

    Runs on its own thread. What ``on_progress`` raises is kept in
    ``failures`` for :func:`_run_ffmpeg` to raise on the calling thread.
    """
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            if key == "frame" and value.isdigit():
                on_progress(min(1.0, int(value) / total_frames))
    except BaseException as exc:  # noqa: BLE001 - re-raised by _run_ffmpeg
        failures.append(exc)


def _count_frames(video: Path | None, ffmpeg_bin: str) -> int:
    """Frames in ``video`` for a progress report; 0 when unknown."""
    if video is None:
        return 0
    try:
        frames, _rate = video_stream_facts(video, ffprobe_bin=_ffprobe_for(ffmpeg_bin))
    except UpscaleError:
        return 0
    return frames


def _require_scale(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise UpscaleError("scale must be 1 or 2")
    if value not in _SCALES:
        raise UpscaleError("scale must be 1 or 2")
    return value


def _require_strength(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UpscaleError("strength must be a number from 0.0 to 1.0")
    strength = float(value)
    # NaN fails every comparison, so it is refused by name.
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise UpscaleError("strength must be between 0.0 and 1.0")
    return strength
