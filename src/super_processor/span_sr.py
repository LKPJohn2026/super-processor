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
"""

from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
from typing import Any

from .upscale import UpscaleError

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


def upscale_frames(
    model: Any, frames: Any, *, batch: int = 0
) -> Any:  # pragma: no cover - needs CUDA
    """Upscale ``frames`` (uint8, N x H x W x 3, on the CPU) by 2.

    Returns uint8 frames, N x 2H x 2W x 3, on the CPU. The frames go to the
    GPU ``batch`` at a time (0: :func:`batch_size`), so GPU memory follows
    the batch, not the clip.
    """
    torch = importlib.import_module("torch")
    count, height, width, _ = frames.shape
    out = torch.empty(
        (count, height * SPAN_SCALE, width * SPAN_SCALE, 3), dtype=torch.uint8
    )
    step = batch or batch_size(width, height)
    with torch.inference_mode():
        for first in range(0, count, step):
            part = frames[first : first + step].cuda(non_blocking=True)
            x = part.permute(0, 3, 1, 2).half().div_(255.0)
            x = x.contiguous(memory_format=torch.channels_last)
            y = model(x).float().clamp_(0.0, 1.0).mul_(255.0).round_()
            out[first : first + step] = y.to(torch.uint8).permute(0, 2, 3, 1).cpu()
    return out
