"""SPAN: the single-frame 2x network the upscale runs.

SPAN is a small convolutional network (2xLiveActionV1_SPAN, about 9 MB) that
sees one frame at a time. It replaced FlashVSR v1.1 after a bake-off on an
RTX 5090: on SD, FlashVSR blended at strength 0.5 and plain SPAN came out
almost the same (consistency 41.9 vs 41.5 dB, detail 32.1 vs 32.6, added
shimmer +0.04 vs +0.07), at 97 against 3 minutes per 20-minute film and 26
against 2 GB of GPU memory. 1440x1080 at 2x did not fit FlashVSR in 32 GB at
all; SPAN takes it to 2880x2160 at about 30-40 frames a second in 4-9 GB.

The weights are Apache-2.0 since jcj83429 relicensed them in commit
``5d8cdd2`` of jcj83429/upscaling (2026-07-31); OpenModelDB still lists the
older CC-BY-NC-SA terms. They are loaded through spandrel (MIT; its SPAN
architecture is Apache-2.0). Both torch and spandrel are imported only when a
SPAN clip runs, so the package imports without the gpu extra.

Everything between the decoder and the encoder runs on the GPU too
(:func:`restore_block`): the colour conversion, the lanczos base, and the
detail blend the FFmpeg :func:`~super_processor.upscale.blend_filter` did on
the CPU. With only SPAN on the GPU, a 1440x1080 -> 2880x2160 render took 0.43
CPU-seconds a frame (the x264 encode about half, the 4K blend filter a
quarter, copies and pipes most of the rest) and ran a 16-core CPU at 95 C.
The GPU block, with NVDEC and NVENC around it, took 0.054 (1500 frames,
preflight included) at 26.8 frames a second instead of 22; on 200 frames
its output was within 0.3 levels on average of the old one (44.96 dB PSNR).
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .regions import FEATHER, KINDS, REGION_STRENGTH, Region
from .upscale import UpscaleError, yuv420_frame_bytes

SPAN_SCALE = 2
SPAN_DIR = "SPAN"
SPAN_FILE = "2xLiveActionV1_SPAN_490000.pth"
SPAN_SHA256 = "8b166c75831ea7f694d9058ee9c8df8148af8cc1d2b57e69e6581b15cab572f7"
# The first commit that carries the Apache-2.0 LICENSE and NOTICE for these
# (unchanged) bytes. Earlier commits say CC-BY-NC-SA 4.0.
SPAN_URL = (
    "https://github.com/jcj83429/upscaling/raw/"
    "5d8cdd2e17750b64be39ccab3a8763d91128fe15/2xLiveActionV1_SPAN/" + SPAN_FILE
)
SPAN_WEIGHTS_ENV = "SUPER_PROCESSOR_SPAN_WEIGHTS"
# Frames per GPU call: about four megapixels of input, which kept a 5090 busy
# in the bake-off without raising peak memory much (4.1 GB at 1440x1080).
_BATCH_PIXELS = 4_000_000
_MAX_BATCH = 8


class SpanError(UpscaleError):
    """The SPAN weights or runtime are missing or wrong."""


def span_weights_path(models_dir: Path) -> Path:
    """Where the weights are read from: ``SUPER_PROCESSOR_SPAN_WEIGHTS``, or
    ``<models>/SPAN/2xLiveActionV1_SPAN_490000.pth``."""
    raw = os.environ.get(SPAN_WEIGHTS_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    return models_dir / SPAN_DIR / SPAN_FILE


def resolve_span_weights(models_dir: Path) -> Path:
    """The weights file, checked by size and hash, or a wizard-facing error."""
    path = span_weights_path(models_dir)
    if not path.is_file():
        raise SpanError(
            f"SPAN weights were not found at {path}. Download {SPAN_URL} "
            f"(Apache-2.0) there, or set {SPAN_WEIGHTS_ENV} to the file."
        )
    digest = _sha256(path)
    if digest != SPAN_SHA256:
        raise SpanError(
            f"{path} is not the expected SPAN weights (sha256 {digest[:12]}..., "
            f"expected {SPAN_SHA256[:12]}...). Download it again from {SPAN_URL}."
        )
    return path


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def batch_size(width: int, height: int) -> int:
    """Frames per GPU call for a ``width`` x ``height`` source."""
    pixels = max(1, width * height)
    return max(1, min(_MAX_BATCH, round(_BATCH_PIXELS / pixels)))


def load_span(weights: Path) -> Any:  # pragma: no cover - needs CUDA and torch
    """The SPAN network on the GPU in half precision, ready to run.

    SPAN's Conv3XC blocks rebuild one fused 3x3 convolution from three
    branches on every forward call in eval mode; fusing them once here makes
    the same picture about twice as fast.
    """
    try:
        torch = importlib.import_module("torch")
        spandrel = importlib.import_module("spandrel")
    except ImportError as exc:
        raise SpanError(
            "SPAN needs PyTorch and spandrel. Install the gpu extra "
            "(pip install 'super-processor[gpu]')."
        ) from exc
    state = torch.load(weights, map_location="cpu", weights_only=True)
    for key in ("params_ema", "params", "state_dict"):
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            state = state[key]
            break
    described = spandrel.ModelLoader().load_from_state_dict(state)
    if described.scale != SPAN_SCALE:
        raise SpanError(f"{weights.name} is a {described.scale}x model, not 2x")
    model = described.model.eval()
    with torch.no_grad():
        _fuse_conv3xc(model)
    return model.cuda().half().to(memory_format=torch.channels_last)


def _fuse_conv3xc(module: Any) -> None:  # pragma: no cover - needs torch
    """Fuse every Conv3XC block once and make it run only the fused conv."""
    leaky_relu = importlib.import_module("torch.nn.functional").leaky_relu
    for child in module.modules():
        if type(child).__name__ != "Conv3XC":
            continue
        child.update_params()
        conv, relu = child.eval_conv, bool(child.has_relu)

        def forward(x: Any, conv: Any = conv, relu: bool = relu) -> Any:
            y = conv(x)
            return leaky_relu(y, 0.05) if relu else y

        child.forward = forward


@dataclass(frozen=True)
class Blend:
    """One clip's settings for :func:`restore_block`, made once per clip.

    ``share`` is the SPAN share per output pixel (1 x 1 x 2H x 2W, or 1 x 1 x
    1 x 1 when it is the same everywhere), ``sigma`` the detail blur in output
    pixels, and ``taps`` the two lanczos phase kernels.
    """

    width: int
    height: int
    share: Any
    sigma: float
    taps: Any
    device: str
    dtype: Any


def prepare_blend(
    width: int,
    height: int,
    *,
    strength: float,
    regions: tuple[Region, ...] = (),
    sigma: float,
    device: str = "cuda",
) -> Blend:  # pragma: no cover - needs torch
    """The :class:`Blend` for a ``width`` x ``height`` source."""
    torch = importlib.import_module("torch")
    out_w, out_h = width * SPAN_SCALE, height * SPAN_SCALE
    return Blend(
        width=width,
        height=height,
        share=strength_map(out_w, out_h, strength, regions, device=device),
        sigma=sigma,
        taps=lanczos_taps(device=device),
        device=device,
        # SPAN runs in half precision on the GPU; the CPU (tests) has no
        # fast half convolutions.
        dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    )


def restore_block(
    model: Any, blend: Blend, frames: bytearray, *, batch: int
) -> bytearray:  # pragma: no cover - needs torch
    """Upscale packed yuv420p source frames by 2, blended, as packed yuv420p.

    All of it runs on ``blend.device``, ``batch`` frames at a time (GPU
    memory follows the batch): yuv420p to RGB, SPAN, a lanczos 2x base, the
    detail swap and strength mix (see :func:`blend_detail`), and back to
    yuv420p. Only the 8-bit planes cross to the host, straight into the
    returned buffer, so nothing is copied again before the encoder's pipe.
    Bytes past the last whole frame are ignored.
    """
    torch = importlib.import_module("torch")
    width, height = blend.width, blend.height
    frame_in = yuv420_frame_bytes(width, height)
    out_w, out_h = width * SPAN_SCALE, height * SPAN_SCALE
    frame_out = yuv420_frame_bytes(out_w, out_h)
    count = len(frames) // frame_in
    out = bytearray(count * frame_out)
    if count == 0:
        return out
    source = torch.frombuffer(frames, dtype=torch.uint8)[: count * frame_in]
    source = source.view(count, frame_in)
    target = torch.frombuffer(out, dtype=torch.uint8).view(count, frame_out)
    with torch.inference_mode():
        for first in range(0, count, max(1, batch)):
            part = source[first : first + max(1, batch)].to(blend.device)
            rgb = yuv420_to_rgb(*split_yuv420(part, width, height))
            x = rgb.to(blend.dtype).contiguous(memory_format=torch.channels_last)
            up = model(x).float().clamp_(0.0, 1.0)
            del x
            base = lanczos2x(rgb, blend.taps)
            del rgb
            mixed = blend_detail(up, base, blend.share, blend.sigma)
            del up, base
            target[first : first + len(part)].copy_(pack_yuv420(mixed.clamp_(0, 1)))
            del mixed
    return out


# BT.601, limited range: what swscale uses for untagged yuv <-> rgb, so the
# GPU path keeps the colours the FFmpeg one had.
_KR, _KB = 0.299, 0.114


def split_yuv420(
    frames: Any, width: int, height: int
) -> tuple[Any, Any, Any]:  # pragma: no cover - needs torch
    """N packed yuv420p frames (uint8, N x bytes) as Y, U, V (N x 1 x h x w).

    The chroma planes are ceil(w/2) x ceil(h/2), as FFmpeg lays out an odd
    size.
    """
    count = frames.shape[0]
    cw, ch = -(-width // 2), -(-height // 2)
    luma, chroma = width * height, cw * ch
    y = frames[:, :luma].reshape(count, 1, height, width)
    u = frames[:, luma : luma + chroma].reshape(count, 1, ch, cw)
    v = frames[:, luma + chroma : luma + 2 * chroma].reshape(count, 1, ch, cw)
    return y, u, v


def yuv420_to_rgb(y: Any, u: Any, v: Any) -> Any:  # pragma: no cover - needs torch
    """uint8 Y, U, V planes to float RGB (N x 3 x h x w, 0-1).

    Chroma is upsampled bilinearly, then cropped to the luma size.
    """
    f = importlib.import_module("torch.nn.functional")
    torch = importlib.import_module("torch")
    height, width = y.shape[-2:]
    luma = (y.float() - 16.0) / 219.0

    def up(plane: Any) -> Any:
        full = f.interpolate(
            (plane.float() - 128.0) / 224.0,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        return full[..., :height, :width]

    cb, cr = up(u), up(v)
    kg = 1.0 - _KR - _KB
    r = luma + 2 * (1 - _KR) * cr
    b = luma + 2 * (1 - _KB) * cb
    g = luma - (2 * _KB * (1 - _KB) * cb + 2 * _KR * (1 - _KR) * cr) / kg
    return torch.cat([r, g, b], 1).clamp_(0.0, 1.0)


def rgb_to_yuv420(rgb: Any) -> tuple[Any, Any, Any]:  # pragma: no cover - torch
    """Float RGB (0-1) to uint8 Y, U, V planes; chroma is a 2x2 average.

    An odd size averages the last row or column with itself, so the chroma
    planes are ceil(w/2) x ceil(h/2) as in FFmpeg's layout.
    """
    f = importlib.import_module("torch.nn.functional")
    torch = importlib.import_module("torch")
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    luma = _KR * r + (1 - _KR - _KB) * g + _KB * b
    cb = (b - luma) / (2 * (1 - _KB))
    cr = (r - luma) / (2 * (1 - _KR))
    height, width = rgb.shape[-2:]
    pad = (0, width % 2, 0, height % 2)

    def level(plane: Any, scale: float, offset: float) -> Any:
        return (plane * scale + offset).round_().clamp_(0, 255).to(torch.uint8)

    def half(plane: Any) -> Any:
        if any(pad):
            plane = f.pad(plane, pad, mode="replicate")
        return f.avg_pool2d(plane, 2)

    return (
        level(luma, 219.0, 16.0),
        level(half(cb), 224.0, 128.0),
        level(half(cr), 224.0, 128.0),
    )


def pack_yuv420(rgb: Any) -> Any:  # pragma: no cover - needs torch
    """Float RGB frames as packed yuv420p rows (uint8, N x bytes)."""
    torch = importlib.import_module("torch")
    count = rgb.shape[0]
    planes = rgb_to_yuv420(rgb)
    return torch.cat([plane.reshape(count, -1) for plane in planes], 1)


# Lanczos with a = 3, as FFmpeg's ``flags=lanczos`` scale.
_LANCZOS_A = 3


def lanczos_taps(device: str = "cpu") -> Any:  # pragma: no cover - needs torch
    """Taps for the two output phases of a 2x lanczos upscale (2 x 7).

    Output pixel 2i sits a quarter of a source pixel before source pixel i,
    and 2i + 1 a quarter after it, as FFmpeg's scale places them.
    """
    torch = importlib.import_module("torch")

    def lanczos(x: float) -> float:
        if x == 0:
            return 1.0
        if abs(x) >= _LANCZOS_A:
            return 0.0
        px = math.pi * x
        return _LANCZOS_A * math.sin(px) * math.sin(px / _LANCZOS_A) / (px * px)

    rows = []
    for phase in (-0.25, 0.25):
        taps = [lanczos(phase - k) for k in range(-_LANCZOS_A, _LANCZOS_A + 1)]
        total = sum(taps)
        rows.append([tap / total for tap in taps])
    return torch.tensor(rows, dtype=torch.float32, device=device)


def lanczos2x(x: Any, taps: Any) -> Any:  # pragma: no cover - needs torch
    """A 2x lanczos upscale of float frames (N x C x h x w), separable,
    with the edge pixels repeated past the border (as FFmpeg's scale)."""
    f = importlib.import_module("torch.nn.functional")
    count, channels, height, width = x.shape
    k = taps.to(x.dtype)
    pad = k.shape[1] // 2
    flat = x.reshape(count * channels, 1, height, width)
    rows = f.conv2d(
        f.pad(flat, (pad, pad, 0, 0), mode="replicate"), k.view(2, 1, 1, -1)
    )
    # (NC, 2, h, w): the two phases, interleaved along the width.
    rows = rows.permute(0, 2, 3, 1).reshape(count * channels, 1, height, 2 * width)
    cols = f.conv2d(
        f.pad(rows, (0, 0, pad, pad), mode="replicate"), k.view(2, 1, -1, 1)
    )
    cols = cols.permute(0, 2, 1, 3).reshape(count, channels, 2 * height, 2 * width)
    return cols


def gaussian_blur(x: Any, sigma: float) -> Any:  # pragma: no cover - needs torch
    """A separable Gaussian blur (radius 3 sigma), edges repeated."""
    torch = importlib.import_module("torch")
    f = importlib.import_module("torch.nn.functional")
    radius = max(1, math.ceil(3 * sigma))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-(t * t) / (2 * sigma * sigma))
    k = (k / k.sum()).to(x.dtype)
    count, channels, height, width = x.shape
    y = f.pad(
        x.reshape(count * channels, 1, height, width), (radius,) * 4, mode="replicate"
    )
    y = f.conv2d(y, k.view(1, 1, 1, -1))
    y = f.conv2d(y, k.view(1, 1, -1, 1))
    return y.reshape(count, channels, height, width)


def blend_detail(
    up: Any, base: Any, share: Any, sigma: float
) -> Any:  # pragma: no cover - needs torch
    """``base + share * (detail(up) - detail(base))``, detail = x - blur(x).

    The base's shapes and colour with SPAN's fine detail, by ``share``: the
    FFmpeg :func:`~super_processor.upscale.blend_filter` sum. The blur is
    linear, so one blur of the difference does the work of two.
    """
    diff = up - base
    diff = diff - gaussian_blur(diff, sigma)
    return base + share * diff


def strength_map(
    width: int,
    height: int,
    strength: float,
    regions: tuple[Region, ...],
    *,
    device: str = "cpu",
) -> Any:  # pragma: no cover - needs torch
    """The SPAN share per output pixel, by the rule of
    :func:`~super_processor.regions.strength_expr`, at pixel centres.

    Without regions that lower it, one value (1 x 1 x 1 x 1) for the frame.
    """
    torch = importlib.import_module("torch")
    share = torch.full((1, 1, 1, 1), float(strength), device=device)
    xs = ((torch.arange(width, device=device) + 0.5) / width).view(1, -1)
    ys = ((torch.arange(height, device=device) + 0.5) / height).view(-1, 1)
    for kind in KINDS:
        boxes = [r for r in regions if r.kind == kind]
        ceiling = min(strength, REGION_STRENGTH[kind])
        if not boxes or ceiling >= strength:
            continue
        weight = torch.zeros((height, width), device=device)
        for box in boxes:
            inside = torch.minimum(
                torch.minimum(xs - box.x0, box.x1 - xs),
                torch.minimum(ys - box.y0, box.y1 - ys),
            )
            weight = torch.maximum(
                weight, ((inside + FEATHER) / FEATHER).clamp(0.0, 1.0)
            )
        term = (strength - (strength - ceiling) * weight).view(1, 1, height, width)
        share = torch.minimum(share, term)
    return share
