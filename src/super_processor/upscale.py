"""Local FlashVSR restoration upscale.

FFmpeg probes, trims, concatenates, and copies audio. Each span (a shot) also
carries a :class:`~super_processor.look.ShotLook`: bounded FFmpeg clean-up
before FlashVSR and finishing after it, always in that order. Gemini may only
return a time range plus ``scale`` and ``strength``.
``strength`` is the share of the FlashVSR picture in the output; the rest is a
plain lanczos upscale of the same (cleaned) frames, so 0.0 invents nothing and
1.0 is FlashVSR alone. FlashVSR itself always runs at its upstream settings.
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
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from .look import LookError, ShotLook

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
_SCALES = frozenset({2, 4})
# The upstream script is imported with a chdir into its folder, which is
# process-wide. Only one FlashVSR load or inference may run at a time.
_FLASHVSR_LOCK = threading.Lock()
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
    look: ShotLook = field(default_factory=ShotLook)

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


def look_at(plan: UpscalePlan, time_s: float) -> ShotLook:
    """The look of the latest span covering ``time_s``, or a neutral one."""
    for span in reversed(plan.spans):
        if span.start_s - 0.05 <= time_s < span.end_s:
            return span.look
    return ShotLook()


def _covers_all(span: UpscaleSpan, duration_s: float) -> bool:
    return span.start_s <= 0.05 and span.end_s >= duration_s - 0.05


def replace_overlapping(
    plan: UpscalePlan, patch: UpscaleSpan, duration_s: float
) -> UpscalePlan:
    """Store a range revise. A scale change re-renders the whole timeline.

    The patch trims the spans it overlaps instead of dropping them, so every
    other shot keeps its strength and look. One output file has one
    resolution, so a new scale cannot be spliced into part of it: every span
    takes the new scale, and the pending span covers the whole file, which
    :func:`render_plan` reads as "render every shot again".
    """
    patch = clamp_span(patch, duration_s)
    if _covers_all(patch, duration_s):
        full = replace(patch, start_s=0.0, end_s=duration_s)
        return UpscalePlan(spans=(full,), pending=full)
    spans = plan.spans
    rescale = patch.scale not in {span.scale for span in spans}
    if rescale:
        spans = tuple(replace(span, scale=patch.scale) for span in spans)
    kept: list[UpscaleSpan] = []
    for span in spans:
        if span.end_s <= patch.start_s + 0.05 or span.start_s >= patch.end_s - 0.05:
            kept.append(span)
            continue
        if patch.start_s - span.start_s > 0.05:
            kept.append(replace(span, end_s=patch.start_s))
        if span.end_s - patch.end_s > 0.05:
            kept.append(replace(span, start_s=patch.end_s))
    pending = patch
    if rescale:
        pending = replace(patch, start_s=0.0, end_s=duration_s)
    # The latest patch stays last; the wizard reads it as the prior knobs.
    return UpscalePlan(spans=(*kept, patch), pending=pending)


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
        with _FLASHVSR_LOCK:
            if duration > CHUNK_SECONDS + 0.05:
                return _run_flashvsr_chunked(
                    request, home=home, infer=infer, duration_s=duration
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


def _infer_loaded(
    module: Any,
    pipe: Any,
    source: Path,
    output: Path,
    *,
    scale: int,
) -> None:
    """Run one already-loaded pipeline on a short clip."""
    low, height, width, frames, fps = module.prepare_input_tensor(
        str(source),
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
    output.parent.mkdir(parents=True, exist_ok=True)
    module.save_video(pictures, str(output), fps=fps, quality=6)
    empty = getattr(getattr(module.torch, "cuda", None), "empty_cache", None)
    if empty is not None:
        empty()
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
    except UpscaleError:
        raise
    except Exception as exc:
        raise UpscaleError(f"FlashVSR failed: {exc}") from exc
    try:
        _infer_loaded(
            module,
            pipe,
            request.source,
            request.output,
            scale=request.scale,
        )
    except UpscaleError:
        raise
    except Exception as exc:
        raise UpscaleError(f"FlashVSR failed: {exc}") from exc
    finally:
        _unload_flashvsr(previous, path_added)
    if not request.output.is_file():
        raise UpscaleError("FlashVSR produced no output")
    return request.output


def _run_flashvsr_chunked(
    request: UpscaleRequest,
    *,
    home: Path,
    infer: Path,
    duration_s: float,
) -> Path:
    """Upscale overlapping ~8s pieces so the whole clip is not resident on GPU."""
    from .probe import probe_file

    chunks = planned_chunks(duration_s, scale=request.scale, strength=request.strength)
    work = request.output.parent / "flash_chunks"
    work.mkdir(parents=True, exist_ok=True)
    try:
        module, pipe, previous, path_added = _load_flashvsr(home, infer)
    except UpscaleError:
        raise
    except Exception as exc:
        raise UpscaleError(f"FlashVSR failed: {exc}") from exc
    pieces: list[Path] = []
    try:
        for index, span in enumerate(chunks):
            clip = trim_source(
                request.source,
                work / f"in_{index}.mp4",
                span.start_s,
                span.end_s,
                video_filter=request.look.clean_filter(),
            )
            raw = work / f"up_{index}.mp4"
            _infer_loaded(
                module,
                pipe,
                clip,
                raw,
                scale=span.scale,
            )
            # Every piece goes through the same lossless encode so the join
            # reads one set of codec parameters.
            kept = work / f"keep_{index}.mp4"
            raw_duration = float(probe_file(raw).duration_s or 0.0)
            skip = 0.0 if index == 0 else CHUNK_OVERLAP_S
            trim_source(raw, kept, skip, raw_duration)
            pieces.append(kept)
    except UpscaleError:
        raise
    except Exception as exc:
        raise UpscaleError(f"FlashVSR failed: {exc}") from exc
    finally:
        _unload_flashvsr(previous, path_added)
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
    video: Path, *, strength: float | None, look: ShotLook | None = None
) -> None:
    data = _delivery_marker()
    if strength is not None:
        data["strength"] = strength
    if look is not None:
        data["look"] = look.to_dict()
    delivery_marker_path(video).write_text(json.dumps(data) + "\n", encoding="utf-8")


def blend_filter(
    width: int, height: int, strength: float, look: ShotLook | None = None
) -> str:
    """Mix the restored picture (input 0) with a lanczos upscale of input 1.

    ``eof_action=pass`` ends on the restored picture, so the frame count the
    keyframe splice relies on never changes. The base is padded by a second
    of its last frame so a few missing frames do not leave the tail unmixed.
    The base gets the look's clean step, as the engine input did, and the mix
    gets its finish step.
    """
    weight = _require_strength(strength)
    look = look or ShotLook()
    base = _join_filters(
        "setpts=PTS-STARTPTS",
        look.clean_filter(),
        f"scale={width}:{height}:flags=lanczos,format=yuv420p",
        "tpad=stop_mode=clone:stop_duration=1",
    )
    finish = look.finish_filter()
    mixed = "[mix];[mix]" + finish + "[v]" if finish else "[v]"
    return (
        "[0:v]setpts=PTS-STARTPTS,format=yuv420p[up];"
        f"[1:v]{base}[base];"
        f"[up][base]blend=all_expr='A*{weight:.4f}+B*{1 - weight:.4f}'"
        f":eof_action=pass{mixed}"
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
) -> Path:
    """Encode a picture once into the settings every splice part shares.

    A forced IDR frame every second, and no B-frames, give a range revise
    clean points to cut the current output by stream copy. ``concat`` reads
    ``source`` as a concat-demuxer listing. With ``blend_with`` and a strength
    below 1, the picture is mixed with a lanczos upscale of ``blend_with`` in
    the same encode (see :func:`blend_filter`). The look's finish step runs in
    this encode too.
    """
    look = look or ShotLook()
    mixing = blend_with is not None and strength < 1.0
    finish = look.finish_filter()
    if not concat and is_delivery(source):
        done = _read_marker(source) or {}
        same_look = done.get("look", ShotLook().to_dict()) == look.to_dict()
        if (not mixing or done.get("strength") == strength) and same_look:
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
    if mixing:
        assert blend_with is not None
        size = _frame_size(_first_part(source) if concat else source)
        if size is None:
            raise UpscaleError(f"could not read the frame size of {source}")
        inputs.extend(["-i", str(blend_with)])
        picture = [
            "-filter_complex",
            blend_filter(size[0], size[1], strength, look),
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
    _write_marker(
        dest,
        strength=strength if blend_with is not None else None,
        look=None if look.is_neutral() else look,
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
) -> Path:
    raw = engine.upscale(
        UpscaleRequest(
            source,
            work / "full_up.mp4",
            scale=span.scale,
            strength=span.strength,
            look=span.look,
        )
    )
    encode_delivery(
        raw,
        output,
        ffmpeg_bin=ffmpeg_bin,
        blend_with=source,
        strength=span.strength,
        look=span.look,
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
) -> Path:
    """Re-upscale ``span`` from the original source and splice it into ``current``.

    The range widens out to the keyframes of ``current``. The head and tail are
    stream-copied, so repeated revises never re-encode the untouched parts.
    A ``current`` from before the delivery settings existed is encoded into
    them once first.
    """
    work = output.parent / "range_work"
    work.mkdir(parents=True, exist_ok=True)
    if _covers_all(span, duration_s) or not current.is_file():
        return _upscale_full(
            source, output, span, work, engine=engine, ffmpeg_bin=ffmpeg_bin
        )
    if not is_delivery(current):
        current = encode_delivery(current, work / "current.mp4", ffmpeg_bin=ffmpeg_bin)
    keys = keyframe_times(current, ffprobe_bin=_ffprobe_for(ffmpeg_bin))
    start, end = snap_to_keyframes(keys, span.start_s, span.end_s, duration_s)
    if start == 0.0 and end is None:
        return _upscale_full(
            source, output, span, work, engine=engine, ffmpeg_bin=ffmpeg_bin
        )
    clip = trim_source(
        source,
        work / "range.mp4",
        start,
        duration_s if end is None else end,
        ffmpeg_bin=ffmpeg_bin,
    )
    raw = engine.upscale(
        UpscaleRequest(
            clip,
            work / "range_up.mp4",
            scale=span.scale,
            strength=span.strength,
            look=span.look,
        )
    )
    middle = encode_delivery(
        raw,
        work / "range_delivery.mp4",
        ffmpeg_bin=ffmpeg_bin,
        blend_with=clip,
        strength=span.strength,
        look=span.look,
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
    _write_marker(output, strength=None)
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
) -> Path:
    """Render what ``plan`` asks for into ``output``.

    A pending range on an existing output is spliced in by
    :func:`apply_range_revise`. Anything else (the first render, a pending
    span over the whole file, a scale change) renders every shot again.
    """
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
        )
    return render_shots(
        source,
        output,
        plan.spans,
        engine=engine,
        ffmpeg_bin=ffmpeg_bin,
        duration_s=duration_s,
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


def render_shots(
    source: Path,
    output: Path,
    spans: tuple[UpscaleSpan, ...],
    *,
    engine: UpscaleEngine,
    ffmpeg_bin: str = "ffmpeg",
    duration_s: float,
) -> Path:
    """Restore each shot with its own look and join them into ``output``.

    Shot edges snap to source frames, so the parts hold every source frame
    exactly once. Each part goes through the delivery encode, which starts it
    on a keyframe, and the parts are joined by stream copy.
    """
    ordered = shot_timeline(spans, duration_s)
    work = output.parent / "shot_work"
    work.mkdir(parents=True, exist_ok=True)
    if len(ordered) == 1:
        return _upscale_full(
            source, output, ordered[0], work, engine=engine, ffmpeg_bin=ffmpeg_bin
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
        # Half a frame early on both ends, so float rounding of the frame
        # times cannot move a frame into the neighbouring shot. The clip then
        # restarts at zero, or the encoder pads the half-frame gap with a copy.
        clip = trim_source(
            source,
            work / f"shot_{index}.mp4",
            max(0.0, (first - 0.5) / fps),
            (stop - 0.5) / fps,
            ffmpeg_bin=ffmpeg_bin,
            video_filter="setpts=PTS-STARTPTS",
        )
        raw = engine.upscale(
            UpscaleRequest(
                clip,
                work / f"shot_{index}_up.mp4",
                scale=span.scale,
                strength=span.strength,
                look=span.look,
            )
        )
        parts.append(
            encode_delivery(
                raw,
                work / f"shot_{index}_delivery.mp4",
                ffmpeg_bin=ffmpeg_bin,
                blend_with=clip,
                strength=span.strength,
                look=span.look,
            )
        )
    silent = work / "shots.mp4"
    _concat(parts, silent, ffmpeg_bin=ffmpeg_bin)
    have, _rate = video_stream_facts(silent, ffprobe_bin=ffprobe_bin)
    if have != want:
        raise UpscaleError(
            f"the joined shots have {have} frames instead of {want}; "
            "the previous output is left unchanged"
        )
    silent.replace(output)
    _write_marker(output, strength=None)
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
