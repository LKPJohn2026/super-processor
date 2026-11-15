"""The GPU block's maths, run with torch on the CPU.

:func:`super_processor.span_sr.restore_block` does on the GPU what the
FFmpeg blend filter did on the CPU: yuv420p to RGB, SPAN, a lanczos base,
the detail swap and strength mix, and back to yuv420p. These check each
step against FFmpeg or a plain evaluation of the same rule. They need torch
(CI has none) but no GPU: everything here runs on the CPU device.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

import pytest

from super_processor import span_sr
from super_processor.regions import FEATHER, Region, strength_at
from super_processor.upscale import STRUCTURE_SIGMA, yuv420_frame_bytes

torch: Any = pytest.importorskip("torch")
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg required"
)
SIGMA = STRUCTURE_SIGMA * 2


def _smooth(height: int, width: int, low: int, high: int, seed: int) -> Any:
    """A uint8 plane of soft texture between ``low`` and ``high``."""
    gen = torch.Generator().manual_seed(seed)
    coarse = torch.rand((1, 1, max(2, height // 2), max(2, width // 2)), generator=gen)
    soft = torch.nn.functional.interpolate(
        coarse, size=(height, width), mode="bilinear", align_corners=False
    )
    return (low + (high - low) * soft).round().to(torch.uint8)


def _ramp(height: int, width: int, low: int, high: int) -> Any:
    """A uint8 plane rising evenly from left to right."""
    row = torch.linspace(low, high, width).round().to(torch.uint8)
    return row.view(1, 1, 1, width).expand(1, 1, height, width).clone()


@pytest.mark.parametrize(("width", "height"), [(32, 24), (33, 25), (31, 18)])
def test_yuv_rgb_round_trip_keeps_every_plane(width: int, height: int) -> None:
    cw, ch = -(-width // 2), -(-height // 2)
    # Inside the RGB gamut (no clamping), with chroma smooth enough that a
    # bilinear upsample and a 2x2 average give it back.
    y = _smooth(height, width, 40, 215, seed=1)
    u = _ramp(ch, cw, 120, 136)
    v = _ramp(ch, cw, 136, 122)
    rgb = span_sr.yuv420_to_rgb(y, u, v)
    assert tuple(rgb.shape) == (1, 3, height, width)
    back = span_sr.rgb_to_yuv420(rgb)
    for got, want in zip(back, (y, u, v), strict=True):
        assert got.shape == want.shape
        assert (got.int() - want.int()).abs().max().item() <= 1


@pytest.mark.parametrize(("width", "height"), [(32, 24), (33, 25)])
def test_packed_frames_split_and_pack_in_ffmpeg_layout(width: int, height: int) -> None:
    size = yuv420_frame_bytes(width, height)
    cw, ch = -(-width // 2), -(-height // 2)
    assert size == width * height + 2 * cw * ch
    frames = torch.arange(2 * size, dtype=torch.int64).remainder(251).to(torch.uint8)
    y, u, v = span_sr.split_yuv420(frames.view(2, size), width, height)
    assert tuple(y.shape) == (2, 1, height, width)
    assert tuple(u.shape) == tuple(v.shape) == (2, 1, ch, cw)
    assert torch.equal(y[1].flatten(), frames[size : size + width * height])
    assert torch.equal(v[0].flatten(), frames[width * height + cw * ch : size])
    # A 2x frame is always even, and packs to the 2x layout.
    packed = span_sr.pack_yuv420(torch.full((2, 3, 2 * height, 2 * width), 0.5))
    assert tuple(packed.shape) == (2, yuv420_frame_bytes(2 * width, 2 * height))


def test_lanczos_keeps_a_flat_picture_flat() -> None:
    flat = torch.full((2, 3, 9, 13), 0.4)
    out = span_sr.lanczos2x(flat, span_sr.lanczos_taps())
    assert tuple(out.shape) == (2, 3, 18, 26)
    assert torch.allclose(out, flat.new_full(out.shape, 0.4), atol=1e-6)


@needs_ffmpeg
def test_lanczos_matches_ffmpeg_scale() -> None:
    picture = _smooth(24, 32, 0, 255, seed=3)
    scaled = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", "32x24", "-i", "-", "-vf", "scale=2*iw:2*ih:flags=lanczos",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        input=bytes(picture.flatten().tolist()),
        check=True,
        capture_output=True,
    ).stdout  # fmt: skip
    ffmpeg = torch.tensor(list(scaled), dtype=torch.float32).view(1, 1, 48, 64)
    mine = span_sr.lanczos2x(picture.float(), span_sr.lanczos_taps())
    diff = (mine.round().clamp(0, 255) - ffmpeg).abs()
    # FFmpeg's taps are 14-bit fixed point: within a level, mostly exact.
    assert diff.max().item() <= 1
    assert diff.mean().item() < 0.15


def test_strength_zero_is_the_lanczos_base() -> None:
    gen = torch.Generator().manual_seed(5)
    base = torch.rand((1, 3, 20, 24), generator=gen)
    up = torch.rand((1, 3, 20, 24), generator=gen)
    share = span_sr.strength_map(24, 20, 0.0, ())
    assert torch.equal(span_sr.blend_detail(up, base, share, SIGMA), base)


def test_full_strength_takes_spans_detail_on_the_base_colour() -> None:
    gen = torch.Generator().manual_seed(6)
    source = torch.rand((1, 3, 16, 20), generator=gen)
    base = span_sr.lanczos2x(source, span_sr.lanczos_taps())
    rows = torch.arange(32).view(-1, 1)
    cols = torch.arange(40).view(1, -1)
    checker = ((rows + cols) % 2 * 2 - 1).float().expand(1, 3, 32, 40)
    # SPAN shifted the colour (coarse) and added a one-pixel texture (fine).
    up = base + 0.2 + 0.05 * checker
    share = span_sr.strength_map(40, 32, 1.0, ())
    mixed = span_sr.blend_detail(up, base, share, SIGMA)
    # The shift is gone, the texture is kept (away from the border, where
    # the repeated edge pixels make the texture blur to more than 0).
    inner = (..., slice(10, -10), slice(10, -10))
    assert (mixed - (base + 0.05 * checker))[inner].abs().max().item() < 1e-3
    share = span_sr.strength_map(40, 32, 0.5, ())
    half = span_sr.blend_detail(up, base, share, SIGMA)
    assert (half - (base + 0.025 * checker))[inner].abs().max().item() < 1e-3


def test_gaussian_blur_keeps_a_flat_picture_and_sums_to_one() -> None:
    flat = torch.full((1, 3, 10, 12), 0.7)
    assert torch.allclose(span_sr.gaussian_blur(flat, SIGMA), flat, atol=1e-6)
    spike = torch.zeros((1, 1, 41, 41))
    spike[..., 20, 20] = 1.0
    blurred = span_sr.gaussian_blur(spike, SIGMA)
    assert blurred.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert blurred[..., 20, 20].item() == blurred.max().item()


_REGIONS = (
    Region("faces", 0.10, 0.10, 0.40, 0.50),
    Region("faces", 0.30, 0.20, 0.60, 0.45),
    Region("hands", 0.50, 0.40, 0.80, 0.90),
    Region("text", 0.05, 0.70, 0.35, 0.95),
)


@pytest.mark.parametrize("strength", [0.9, 0.5, 0.2, 0.1])
def test_strength_map_follows_the_region_rule(strength: float) -> None:
    width, height = 160, 120
    share = span_sr.strength_map(width, height, strength, _REGIONS)
    assert tuple(share.shape) in {(1, 1, height, width), (1, 1, 1, 1)}
    full = share.expand(1, 1, height, width)[0, 0]
    # Every 7th pixel, plus pixels just inside and outside the feather.
    columns = sorted({*range(0, width, 7), 15, 16, 63, 64, 112, 113, width - 1})
    rows = sorted({*range(0, height, 7), 8, 9, 53, 54, 110, height - 1})
    for row in rows:
        for column in columns:
            want = strength_at(
                strength, _REGIONS, (column + 0.5) / width, (row + 0.5) / height
            )
            assert full[row, column].item() == pytest.approx(want, abs=1e-5)


def test_strength_map_shapes() -> None:
    # No regions, or none below the strength: one value for the frame.
    for regions in ((), (Region("faces", 0.1, 0.1, 0.5, 0.5),)):
        share = span_sr.strength_map(64, 48, 0.25, regions)
        assert tuple(share.shape) == (1, 1, 1, 1)
        assert share.item() == pytest.approx(0.25)
    one = (Region("text", 0.25, 0.25, 0.75, 0.75),)
    share = span_sr.strength_map(64, 48, 0.8, one)[0, 0]
    assert share[24, 32].item() == pytest.approx(0.15)
    assert share[0, 0].item() == pytest.approx(0.8)
    # Half a feather outside the box: half way back up.
    x = 0.75 + FEATHER / 2
    assert strength_at(0.8, one, x, 0.5) == pytest.approx(0.475)


def _nearest_model(x: Any) -> Any:
    return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")


@pytest.mark.parametrize(("width", "height"), [(24, 16), (25, 17)])
def test_restore_block_on_the_cpu(width: int, height: int) -> None:
    frame_in = yuv420_frame_bytes(width, height)
    cw, ch = -(-width // 2), -(-height // 2)
    planes = []
    for index in range(5):
        y = _smooth(height, width, 40, 215, seed=10 + index).flatten()
        u = _ramp(ch, cw, 120, 136).flatten()
        v = _ramp(ch, cw, 136, 122).flatten()
        planes.append(torch.cat([y, u, v]))
    raw = bytearray(bytes(torch.cat(planes).tolist()) + b"\x07" * 3)
    assert len(raw) == 5 * frame_in + 3
    blend = span_sr.prepare_blend(
        width, height, strength=0.0, sigma=SIGMA, device="cpu"
    )
    out = span_sr.restore_block(_nearest_model, blend, raw, batch=2)
    frame_out = yuv420_frame_bytes(2 * width, 2 * height)
    assert len(out) == 5 * frame_out
    # Strength 0 is the lanczos base, frame for frame, whatever the batch.
    assert span_sr.restore_block(_nearest_model, blend, raw, batch=5) == out
    assert span_sr.restore_block(_nearest_model, blend, raw, batch=0) == out
    stack = torch.stack(planes).view(5, frame_in)
    rgb = span_sr.yuv420_to_rgb(*span_sr.split_yuv420(stack, width, height))
    base = span_sr.lanczos2x(rgb, blend.taps).clamp(0, 1)
    want = span_sr.pack_yuv420(base)
    got = torch.tensor(list(out), dtype=torch.uint8).view(5, frame_out)
    assert torch.equal(got, want)
    assert span_sr.restore_block(_nearest_model, blend, bytearray(3), batch=2) == b""
