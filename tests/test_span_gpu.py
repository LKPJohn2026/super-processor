"""The GPU block on a real GPU: batches, precision, the weights, and the
blend against the FFmpeg path it replaced.

Every streaming test elsewhere swaps in a stand-in ``restore_block``, and
``test_gpu_blend.py`` checks the maths on the CPU; these are the only ones
that run it on CUDA. They need torch with CUDA (and, for the last two, the
weights under ``models_dir()``), so CI, which has no torch, skips them;
``-m "not gpu"`` skips them anywhere.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from super_processor import span_sr
from super_processor.regions import Region
from super_processor.span_sr import SpanError, load_span, resolve_span_weights
from super_processor.upscale import (
    STRUCTURE_SIGMA,
    blend_filter,
    models_dir,
    yuv420_frame_bytes,
)

torch: Any = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required"),
]

SIGMA = STRUCTURE_SIGMA * 2
_W, _H, _COUNT = 64, 48, 24


def _clip(path: Path, width: int = _W, height: int = _H, frames: int = _COUNT) -> Path:
    # Softened: testsrc2's one-pixel colour edges are where RGB and YUV
    # resampling differ most, and no film has them.
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"testsrc2=s={width}x{height}:r=24,gblur=sigma=1",
         "-frames:v", str(frames),
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )  # fmt: skip
    return path


def _decode(path: Path, pix_fmt: str) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", pix_fmt, "-"],
        check=True,
        capture_output=True,
    ).stdout  # fmt: skip


def _planes(raw: bytes | bytearray, width: int, height: int) -> list[Any]:
    """Y, U, V of every frame, as float tensors (frames x pixels)."""
    frames = torch.tensor(list(raw), dtype=torch.float32)
    frames = frames.view(-1, yuv420_frame_bytes(width, height))
    luma = width * height
    chroma = (width // 2) * (height // 2)
    return [
        frames[:, :luma],
        frames[:, luma : luma + chroma],
        frames[:, luma + chroma :],
    ]


def _nearest(x: Any) -> Any:
    return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")


def _weights_model() -> Any:
    try:
        weights = resolve_span_weights(models_dir())
    except SpanError as exc:
        pytest.skip(f"no SPAN weights: {exc}")
    return load_span(weights)


def test_cuda_matches_the_cpu_and_ignores_the_batch(tmp_path: Path) -> None:
    raw = bytearray(_decode(_clip(tmp_path / "in.mp4", frames=7), "yuv420p"))
    regions = (Region("faces", 0.2, 0.2, 0.6, 0.7),)

    def run(device: str, batch: int) -> bytearray:
        blend = span_sr.prepare_blend(
            _W, _H, strength=0.8, regions=regions, sigma=SIGMA, device=device
        )
        return span_sr.restore_block(_nearest, blend, raw, batch=batch)

    # 7 frames at a batch of 3: the last batch is short.
    gpu = run("cuda", 3)
    assert len(gpu) == 7 * yuv420_frame_bytes(2 * _W, 2 * _H)
    assert run("cuda", 7) == gpu
    cpu = run("cpu", 3)
    # The model runs in half precision on the GPU, in float on the CPU.
    planes = zip(
        _planes(gpu, 2 * _W, 2 * _H), _planes(cpu, 2 * _W, 2 * _H), strict=True
    )
    for a, b in planes:
        assert (a - b).abs().max().item() <= 2


def test_real_weights_keep_a_flat_colour() -> None:
    model = _weights_model()
    frame = bytes([150] * (_W * _H) + [100] * (_W * _H // 4) + [170] * (_W * _H // 4))
    blend = span_sr.prepare_blend(_W, _H, strength=1.0, sigma=SIGMA)
    out = span_sr.restore_block(model, blend, bytearray(frame * 5), batch=2)
    y, u, v = _planes(out, 2 * _W, 2 * _H)
    assert y.shape[0] == 5
    for plane, level, (w, h) in (
        (y, 150, (2 * _W, 2 * _H)),
        (u, 100, (_W, _H)),
        (v, 170, (_W, _H)),
    ):
        # Away from the border, SPAN leaves a flat colour as it is.
        middle = plane.view(5, h, w)[:, 8:-8, 8:-8]
        assert (middle - level).abs().max().item() <= 2


def _psnr(a: Any, b: Any) -> float:
    mse = ((a - b) ** 2).mean().item()
    return math.inf if mse == 0 else 10 * math.log10(255**2 / mse)


def test_real_weights_match_the_ffmpeg_blend(tmp_path: Path) -> None:
    """The GPU block against the path it replaced: SPAN on decoded RGB, then
    FFmpeg's lanczos base, gblur detail swap, and region mix."""
    model = _weights_model()
    width, height = 320, 240
    source = _clip(tmp_path / "in.mp4", width, height)
    strength = 0.6
    regions = (Region("text", 0.1, 0.1, 0.5, 0.4), Region("hands", 0.4, 0.5, 0.9, 0.9))
    out_w, out_h = 2 * width, 2 * height
    rgb = torch.tensor(list(_decode(source, "rgb24")), dtype=torch.uint8)
    rgb = rgb.view(-1, height, width, 3)
    with torch.inference_mode():
        x = rgb.cuda().permute(0, 3, 1, 2).half().div_(255.0)
        up = model(x.contiguous(memory_format=torch.channels_last))
        up = up.float().clamp_(0, 1).mul_(255).round_().to(torch.uint8)
        restored = up.permute(0, 2, 3, 1).cpu().contiguous()
    old = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{out_w}x{out_h}", "-framerate", "24", "-i", "-",
         "-i", str(source), "-filter_complex",
         blend_filter(out_w, out_h, strength, regions=regions, sigma=SIGMA),
         "-map", "[v]", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        input=bytes(restored.flatten().tolist()),
        check=True,
        capture_output=True,
    ).stdout  # fmt: skip
    blend = span_sr.prepare_blend(
        width, height, strength=strength, regions=regions, sigma=SIGMA
    )
    raw = bytearray(_decode(source, "yuv420p"))
    new = span_sr.restore_block(model, blend, raw, batch=4)
    assert len(new) == len(old)
    # Measured 46.2, 41.9, 40.9 dB on Y, U, V (chroma differs most: it is
    # resampled in RGB now, in YUV before), means within 0.13.
    pairs = zip(_planes(new, out_w, out_h), _planes(old, out_w, out_h), strict=True)
    for got, want in pairs:
        assert abs(got.mean().item() - want.mean().item()) < 1.0
        assert _psnr(got, want) > 38.0
