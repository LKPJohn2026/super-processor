"""Capped SeedVR2 restore, then RTX Video Super Resolution.

FFmpeg probes, decodes, and encodes. It does not decide the look. Gemini may
only retune :class:`UpscaleParams`. A note cannot raise the strength cap, and
a note about artificial detail cannot raise strength or VSR quality.
"""

from __future__ import annotations

import importlib
import os
import struct
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

_ALLOWED_KEYS = frozenset({"restore_strength", "scale", "vsr_quality"})
_FORBIDDEN_KEYS = frozenset(
    {
        "prompt",
        "prompts",
        "model",
        "model_name",
        "ffmpeg",
        "argv",
        "args",
        "cmd",
        "command",
        "filter",
        "filter_complex",
    }
)
_QUALITIES = ("LOW", "MEDIUM", "HIGH")
_QUALITY_RANK = {name: index for index, name in enumerate(_QUALITIES)}
_SCALES = frozenset({2, 3, 4})
_MAX_RESTORE_STRENGTH = 0.35
_ARTIFICIAL_PHRASES = (
    "artificial",
    "plastic",
    "over-sharpened",
    "over sharpened",
    "oversharpened",
    "not natural",
)
_SHARPNESS_PHRASES = (
    "sharp",
    "too soft",
    "softer",
    "crisp",
)
DEFAULT_CHUNK_FRAMES = 8
_WEIGHTS_ENV = "SEEDVR2_3B_WEIGHTS"


class UpscaleError(RuntimeError):
    """Raised when restore or VSR cannot run, or params are rejected."""


@dataclass(frozen=True, slots=True)
class UpscaleParams:
    """Allowlisted knobs for the wizard quality path."""

    restore_strength: float = 0.15
    scale: int = 2
    vsr_quality: str = "MEDIUM"

    def __post_init__(self) -> None:
        strength = _require_strength(self.restore_strength)
        scale = _require_scale(self.scale)
        quality = _require_quality(self.vsr_quality)
        object.__setattr__(self, "restore_strength", strength)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "vsr_quality", quality)

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "restore_strength": self.restore_strength,
            "scale": self.scale,
            "vsr_quality": self.vsr_quality,
        }

    @classmethod
    def from_dict(cls, data: Any) -> UpscaleParams:
        """Parse Gemini JSON. Extra keys, prompts, and argv are rejected."""
        if isinstance(data, UpscaleParams):
            return data
        if not isinstance(data, Mapping):
            raise UpscaleError("upscale params must be a JSON object")
        keys = set(data)
        forbidden = keys & _FORBIDDEN_KEYS
        if forbidden:
            listed = ", ".join(sorted(forbidden))
            raise UpscaleError(
                f"Gemini cannot emit a prompt, a model name, or FFmpeg argv ({listed})"
            )
        extra = keys - _ALLOWED_KEYS
        if extra:
            listed = ", ".join(sorted(str(item) for item in extra))
            raise UpscaleError(f"upscale params have unknown fields: {listed}")
        missing = _ALLOWED_KEYS - keys
        if missing:
            listed = ", ".join(sorted(missing))
            raise UpscaleError(f"upscale params missing {listed}")
        return cls(
            restore_strength=data["restore_strength"],
            scale=data["scale"],
            vsr_quality=data["vsr_quality"],
        )


def note_blocks_detail_increase(note: str) -> bool:
    """True when the note says the picture looks artificial or unnatural."""
    text = note.lower()
    return any(phrase in text for phrase in _ARTIFICIAL_PHRASES)


def note_requests_sharpness(note: str) -> bool:
    """True when the note asks for a sharper, less soft picture."""
    text = note.lower()
    return any(phrase in text for phrase in _SHARPNESS_PHRASES)


def enforce_note_direction(
    prior: UpscaleParams,
    proposed: UpscaleParams,
    note: str,
) -> UpscaleParams:
    """Hold or lower detail knobs when the note says the image looks unnatural.

    A sharpness request may raise ``restore_strength`` and ``vsr_quality``
    only inside the caps already enforced by :class:`UpscaleParams`.
    """
    if note_blocks_detail_increase(note):
        if proposed.restore_strength > prior.restore_strength:
            raise UpscaleError(
                "this note says the image looks unnatural, so restore "
                "strength cannot increase"
            )
        if _QUALITY_RANK[proposed.vsr_quality] > _QUALITY_RANK[prior.vsr_quality]:
            raise UpscaleError(
                "this note says the image looks unnatural, so VSR quality "
                "cannot increase"
            )
    return proposed


def align8(value: int) -> int:
    """Round a pixel size down to a multiple of 8 (minimum 8)."""
    if value < 8:
        raise UpscaleError("frame dimension must be at least 8 pixels")
    return value - (value % 8)


def output_size(width: int, height: int, scale: int) -> tuple[int, int]:
    """Scaled size used by RTX VSR, forced to a multiple of 8."""
    return align8(width * scale), align8(height * scale)


@dataclass(frozen=True, slots=True)
class HostChwFrame:
    """CHW float32 frame in ``[0, 1]`` before it is moved to CUDA."""

    data: bytes
    channels: int
    height: int
    width: int

    def __post_init__(self) -> None:
        expected = self.channels * self.height * self.width * 4
        if len(self.data) != expected:
            raise UpscaleError("CHW float32 buffer has the wrong size")


def rgb24_to_chw(frame: bytes, width: int, height: int) -> HostChwFrame:
    """Pack interleaved RGB bytes into planar float32 ``[0, 1]``."""
    expected = width * height * 3
    if len(frame) != expected:
        raise UpscaleError("decoded frame size does not match the source")
    pixels = width * height
    out = bytearray(pixels * 3 * 4)
    for index in range(pixels):
        base = index * 3
        red = frame[base] / 255.0
        green = frame[base + 1] / 255.0
        blue = frame[base + 2] / 255.0
        struct.pack_into("<f", out, index * 4, red)
        struct.pack_into("<f", out, (pixels + index) * 4, green)
        struct.pack_into("<f", out, (2 * pixels + index) * 4, blue)
    return HostChwFrame(bytes(out), 3, height, width)


def _cuda_chw_to_rgb24(image: Any) -> bytes:
    """Clone an ``nvvfx`` CHW float frame into RGB24 bytes."""
    torch = importlib.import_module("torch")
    tensor = torch.from_dlpack(image).detach().clone()
    array = tensor.float().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu()
    return bytes(array.numpy().tobytes())


def clone_dlpack_output(value: Any) -> bytes:
    """Copy a VSR DLPack buffer immediately so the producer can reuse it."""
    exporter = getattr(value, "__dlpack__", None)
    if exporter is not None:
        exported = exporter()
        if isinstance(exported, (bytes, bytearray, memoryview)):
            return bytes(exported)
    if hasattr(value, "clone"):
        cloned = value.clone()
        if hasattr(cloned, "cpu"):
            cloned = cloned.cpu()
        to_numpy = getattr(cloned, "numpy", None)
        if to_numpy is not None:
            array = to_numpy()
            return bytes(array.tobytes())
    raise UpscaleError("RTX VSR output has no DLPack buffer to clone")


class RestoreBackend(Protocol):
    """SeedVR2-3B at source size. Tests supply a fake."""

    def restore(
        self,
        frames: list[bytes],
        *,
        width: int,
        height: int,
        strength: float,
    ) -> list[bytes]:
        """Return restored RGB24 frames at the source size."""


class VsrBackend(Protocol):
    """``nvvfx.VideoSuperRes`` adapter. Tests supply a fake."""

    def run(
        self,
        frame: HostChwFrame,
        *,
        scale: int,
        quality: str,
        out_width: int,
        out_height: int,
    ) -> Any:
        """Return a DLPack frame at ``out_width`` × ``out_height`` RGB24."""


class FrameEncoder(Protocol):
    """Sink for aligned RGB24 frames. The default is ``hevc_nvenc``."""

    def write_frame(self, frame: bytes) -> None:
        """Write one output frame."""

    def close(self) -> None:
        """Finish the encode."""


def bundled_seedvr2_weights() -> Path:
    """FP8 3B file the standalone CLI downloads into its model directory."""
    return (
        seedvr_home()
        / "models"
        / "SEEDVR2"
        / "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
    )


def resolve_seedvr2_weights(path: Path | None = None) -> Path:
    """Locate SeedVR2-3B weights or raise a wizard-facing error."""
    if path is None:
        if _WEIGHTS_ENV in os.environ and not os.environ[_WEIGHTS_ENV].strip():
            path = None
        else:
            raw = os.environ.get(_WEIGHTS_ENV, "").strip()
            path = Path(raw).expanduser() if raw else None
            if path is None and bundled_seedvr2_weights().is_file():
                return bundled_seedvr2_weights()
    if path is None or not path.is_file():
        where = str(path) if path is not None else f"${_WEIGHTS_ENV}"
        raise UpscaleError(
            "SeedVR2-3B weights were not found "
            f"({where}). Download the SeedVR2-3B weights and set "
            f"{_WEIGHTS_ENV} to that file. ComfyUI is not required. "
            "An NVIDIA driver and an RTX GPU are required for the upscale pass."
        )
    return path


def seedvr_home() -> Path:
    """Checkout that contains ``inference_cli.py``. Override with ``SEEDVR2_HOME``."""
    raw = os.environ.get("SEEDVR2_HOME", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parents[2] / ".seedvr" / "seedvr2_videoupscaler"


def seedvr_python() -> Path | None:
    """Python for the SeedVR2 standalone runtime, when it is installed."""
    raw = os.environ.get("SEEDVR2_PYTHON", "").strip()
    if raw:
        path = Path(raw).expanduser()
        return path if path.is_file() else None
    candidate = seedvr_home() / ".venv" / "bin" / "python"
    script = seedvr_home() / "inference_cli.py"
    if candidate.is_file() and script.is_file():
        return candidate
    return None


def _runtime_missing() -> UpscaleError:
    return UpscaleError(
        "SeedVR2-3B weights are present, but the local restore runtime "
        "is not loaded. Install the standalone runtime at "
        f"{seedvr_home()}. ComfyUI is not required."
    )


class SeedVR2Restorer:
    """Default pass-1 backend. Missing weights raise :class:`UpscaleError`."""

    def __init__(self, weights: Path | None = None) -> None:
        self.weights = resolve_seedvr2_weights(weights)

    def restore(
        self,
        frames: list[bytes],
        *,
        width: int,
        height: int,
        strength: float,
    ) -> list[bytes]:
        del frames, width, height, strength
        raise _runtime_missing()

    def restore_video(
        self,
        source: Path,
        output: Path,
        *,
        width: int,
        height: int,
        strength: float,
        fps: float,
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        """Run SeedVR2 at source size, then mix it back by ``strength``."""
        del fps
        python = seedvr_python()
        if python is None:
            raise _runtime_missing()
        home = seedvr_home()
        raw = output.with_suffix(".seedvr.mp4")
        short_side = min(width, height)
        command = [
            str(python),
            str(home / "inference_cli.py"),
            str(source),
            "--output",
            str(raw),
            "--resolution",
            str(short_side),
            "--max_resolution",
            str(max(width, height)),
            "--dit_model",
            "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
            "--color_correction",
            "lab",
            "--cuda_device",
            "0",
            "--batch_size",
            "5",
            "--video_backend",
            "ffmpeg",
        ]
        completed = subprocess.run(
            command,
            cwd=home,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not raw.is_file():
            detail = (completed.stderr or completed.stdout).strip()
            tail = detail[-2000:] or str(completed.returncode)
            raise UpscaleError(f"SeedVR2 restore failed: {tail}")
        blend = subprocess.run(
            [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-i",
                str(raw),
                "-filter_complex",
                (
                    f"[1:v]scale={width}:{height}:flags=lanczos[restored];"
                    f"[0:v][restored]blend=all_opacity={strength}"
                ),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        raw.unlink(missing_ok=True)
        if blend.returncode != 0 or not output.is_file():
            detail = (blend.stderr or blend.stdout).strip()
            raise UpscaleError(
                "SeedVR2 blend failed: " + (detail[-2000:] or str(blend.returncode))
            )


class _NvvfxEffect:
    """Adapter for the installed ``nvvfx.VideoSuperRes`` constructor."""

    def __init__(self, factory: Any, *, quality: str) -> None:
        levels = getattr(factory, "QualityLevel", None)
        level = getattr(levels, quality, None) if levels is not None else None
        if level is None:
            raise UpscaleError(
                f"nvvfx.VideoSuperRes has no quality level {quality}."
            )
        self._effect = factory(quality=level)
        enter = getattr(self._effect, "__enter__", None)
        if enter is not None:
            enter()
        self._loaded_size: tuple[int, int] | None = None

    def prepare(self, out_width: int, out_height: int) -> None:
        if self._loaded_size == (out_width, out_height):
            return
        self._effect.output_width = out_width
        self._effect.output_height = out_height
        self._effect.load()
        self._loaded_size = (out_width, out_height)

    def run(self, frame: Any) -> Any:
        return self._effect.run(frame)


def open_video_super_res(*, scale: int, quality: str) -> Any:
    """Construct ``nvvfx.VideoSuperRes`` or raise a clear wizard error."""
    try:
        nvvfx = importlib.import_module("nvvfx")
    except ImportError as exc:
        raise UpscaleError(
            "RTX Video Super Resolution needs the nvvfx package, an NVIDIA "
            "driver, and an RTX GPU."
        ) from exc
    factory = getattr(nvvfx, "VideoSuperRes", None)
    if factory is None:
        raise UpscaleError(
            "nvvfx is installed but VideoSuperRes is missing. Install an RTX "
            "Video Effects build that provides nvvfx.VideoSuperRes."
        )
    try:
        return factory(scale=scale, quality=quality)
    except TypeError:
        try:
            return _NvvfxEffect(factory, quality=quality)
        except Exception as exc:
            raise UpscaleError(
                f"Could not start RTX Video Super Resolution: {exc}"
            ) from exc
    except Exception as exc:
        raise UpscaleError(
            f"Could not start RTX Video Super Resolution: {exc}"
        ) from exc


def host_chw_to_cuda(frame: HostChwFrame, *, torch_module: Any | None = None) -> Any:
    """Move a CHW float32 frame to CUDA for ``VideoSuperRes.run``."""
    torch = torch_module
    if torch is None:
        try:
            imported_torch = importlib.import_module("torch")
        except ImportError as exc:
            raise UpscaleError(
                "RTX Video Super Resolution needs PyTorch with CUDA so frames "
                "can be passed as CHW float32."
            ) from exc
        torch = imported_torch
    cuda = getattr(torch, "cuda", None)
    available = getattr(cuda, "is_available", None)
    if cuda is None or available is None or not available():
        raise UpscaleError(
            "RTX Video Super Resolution needs an NVIDIA driver and an RTX GPU."
        )
    tensor = torch.frombuffer(bytearray(frame.data), dtype=torch.float32)
    reshaped = tensor.reshape(frame.channels, frame.height, frame.width)
    return reshaped.contiguous().cuda()


class NvvfxVsr:
    """Pass 2. Clones the DLPack result before returning it to the pipeline."""

    def __init__(self, *, scale: int, quality: str) -> None:
        self._effect = open_video_super_res(scale=scale, quality=quality)
        self.scale = scale
        self.quality = quality

    def run(
        self,
        frame: HostChwFrame,
        *,
        scale: int,
        quality: str,
        out_width: int,
        out_height: int,
    ) -> Any:
        del scale, quality
        prepare = getattr(self._effect, "prepare", None)
        if prepare is not None:
            prepare(out_width, out_height)
        cuda_frame = host_chw_to_cuda(frame)
        produced = self._effect.run(cuda_frame)
        image = getattr(produced, "image", None)
        if image is not None:
            return _ClonedDlpack(_cuda_chw_to_rgb24(image))
        return _ClonedDlpack(clone_dlpack_output(produced))


class _ClonedDlpack:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __dlpack__(self) -> memoryview:
        return memoryview(self._payload)


def decode_rgb_chunks(
    source: Path,
    *,
    ffmpeg_bin: str,
    width: int,
    height: int,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
) -> Iterator[list[bytes]]:
    """Decode RGB24 frames from ``source`` in chunks. FFmpeg does not filter."""
    if chunk_frames < 1:
        raise UpscaleError("chunk_frames must be at least 1")
    frame_bytes = width * height * 3
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None
    chunk: list[bytes] = []
    try:
        while True:
            frame = _read_exact(proc.stdout, frame_bytes)
            if frame is None:
                break
            chunk.append(frame)
            if len(chunk) >= chunk_frames:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
    finally:
        proc.stdout.close()
        err = proc.stderr.read()
        proc.stderr.close()
        code = proc.wait()
    if code != 0:
        detail = err.decode("utf-8", errors="replace").strip()
        raise UpscaleError(f"FFmpeg decode failed: {detail or code}")


def build_nvenc_command(
    *,
    ffmpeg_bin: str,
    width: int,
    height: int,
    fps: float,
    output: Path,
) -> list[str]:
    """HEVC NVENC encode of raw RGB24. FFmpeg is the encoder only."""
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "hevc_nvenc",
        "-rc",
        "constqp",
        "-qp",
        "24",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]


class NvencEncoder:
    """Stream RGB frames into ``hevc_nvenc``."""

    def __init__(
        self,
        output: Path,
        *,
        ffmpeg_bin: str,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        self.output = output
        self._frame_bytes = width * height * 3
        command = build_nvenc_command(
            ffmpeg_bin=ffmpeg_bin,
            width=width,
            height=height,
            fps=fps,
            output=output,
        )
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write_frame(self, frame: bytes) -> None:
        if len(frame) != self._frame_bytes:
            raise UpscaleError("encode frame size does not match the VSR output")
        stdin = self._proc.stdin
        if stdin is None:
            raise UpscaleError("hevc_nvenc encoder is not accepting frames")
        try:
            stdin.write(frame)
        except BrokenPipeError as exc:
            raise UpscaleError(self._failure_text()) from exc

    def close(self) -> None:
        stdin = self._proc.stdin
        if stdin is not None:
            stdin.close()
        err = b""
        if self._proc.stderr is not None:
            err = self._proc.stderr.read()
            self._proc.stderr.close()
        code = self._proc.wait()
        if code != 0:
            detail = err.decode("utf-8", errors="replace").strip()
            raise UpscaleError(f"hevc_nvenc encode failed: {detail or code}")

    def _failure_text(self) -> str:
        err = b""
        if self._proc.stderr is not None:
            err = self._proc.stderr.read()
        return "hevc_nvenc encode failed: " + err.decode("utf-8", errors="replace")


def run_two_pass(
    source: Path,
    output: Path,
    params: UpscaleParams,
    *,
    ffmpeg_bin: str = "ffmpeg",
    width: int,
    height: int,
    fps: float,
    restorer: RestoreBackend | None = None,
    vsr: VsrBackend | None = None,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    encoder: FrameEncoder | None = None,
) -> Path:
    """Restore at source size when strength > 0, then scale with RTX VSR.

    ``restorer`` and ``vsr`` default to the real backends. Tests pass fakes so
    CI never loads SeedVR2 weights or ``nvvfx``.
    """
    if fps <= 0:
        raise UpscaleError("frame rate must be positive")
    out_w, out_h = output_size(width, height, params.scale)
    if params.restore_strength > 0 and restorer is None:
        restorer = SeedVR2Restorer()
    if vsr is None:
        vsr = NvvfxVsr(scale=params.scale, quality=params.vsr_quality)
    owned: NvencEncoder | None
    sink: FrameEncoder
    if encoder is None:
        owned = NvencEncoder(
            output,
            ffmpeg_bin=ffmpeg_bin,
            width=out_w,
            height=out_h,
            fps=fps,
        )
        sink = owned
    else:
        owned = None
        sink = encoder
    wrote = 0
    decode_from = source
    blended: Path | None = None
    file_restore = getattr(restorer, "restore_video", None)
    try:
        if params.restore_strength > 0 and file_restore is not None:
            blended = output.with_suffix(".restore.mp4")
            file_restore(
                source,
                blended,
                width=width,
                height=height,
                strength=params.restore_strength,
                fps=fps,
                ffmpeg_bin=ffmpeg_bin,
            )
            decode_from = blended
        for chunk in decode_rgb_chunks(
            decode_from,
            ffmpeg_bin=ffmpeg_bin,
            width=width,
            height=height,
            chunk_frames=chunk_frames,
        ):
            frames = chunk
            if params.restore_strength > 0 and file_restore is None:
                assert restorer is not None
                frames = restorer.restore(
                    frames,
                    width=width,
                    height=height,
                    strength=params.restore_strength,
                )
                if len(frames) != len(chunk):
                    raise UpscaleError(
                        "SeedVR2 restore must keep the source frame count"
                    )
            for frame in frames:
                if len(frame) != width * height * 3:
                    raise UpscaleError(
                        "SeedVR2 restore must keep the source frame size"
                    )
                chw = rgb24_to_chw(frame, width, height)
                produced = vsr.run(
                    chw,
                    scale=params.scale,
                    quality=params.vsr_quality,
                    out_width=out_w,
                    out_height=out_h,
                )
                cloned = clone_dlpack_output(produced)
                if len(cloned) != out_w * out_h * 3:
                    raise UpscaleError("RTX VSR frame is not the aligned output size")
                sink.write_frame(cloned)
                wrote += 1
        if wrote == 0:
            raise UpscaleError("FFmpeg decoded no video frames")
    except Exception:
        if owned is not None:
            _abandon_encoder(owned)
        raise
    finally:
        if blended is not None:
            blended.unlink(missing_ok=True)
    if owned is not None:
        owned.close()
    elif encoder is not None:
        encoder.close()
    return output


def _abandon_encoder(encoder: NvencEncoder) -> None:
    proc = encoder._proc
    if proc.stdin is not None:
        with suppress(BrokenPipeError):
            proc.stdin.close()
    if proc.stderr is not None:
        proc.stderr.read()
        proc.stderr.close()
    proc.wait()


def _read_exact(stream: Any, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        piece = stream.read(remaining)
        if not piece:
            if chunks:
                raise UpscaleError("FFmpeg decode ended mid-frame")
            return None
        chunks.append(piece)
        remaining -= len(piece)
    return b"".join(chunks)


def _require_strength(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UpscaleError("restore_strength must be a number from 0.0 to 0.35")
    strength = float(value)
    if strength < 0.0 or strength > _MAX_RESTORE_STRENGTH:
        raise UpscaleError(
            "restore_strength must be between 0.0 and 0.35; a note cannot "
            "raise that cap"
        )
    return strength


def _require_scale(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise UpscaleError("scale must be 2, 3, or 4")
    if value not in _SCALES:
        raise UpscaleError("scale must be 2, 3, or 4 and is applied only by RTX VSR")
    return value


def _require_quality(value: Any) -> str:
    if not isinstance(value, str):
        raise UpscaleError("vsr_quality must be LOW, MEDIUM, or HIGH")
    quality = value.strip().upper()
    if quality == "ULTRA":
        raise UpscaleError("vsr_quality ULTRA is not allowed")
    if quality not in _QUALITY_RANK:
        raise UpscaleError("vsr_quality must be LOW, MEDIUM, or HIGH")
    return quality
