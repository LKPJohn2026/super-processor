"""SPAN on the real GPU: the batch loop, the colour conversion, the weights.

Every other SPAN test swaps in a stand-in ``upscale_frames``, so these are
the only ones that run the network path. They need torch with CUDA (and,
for the last one, the weights under ``models_dir()``), so CI, which has no
torch, skips them; ``-m "not gpu"`` skips them anywhere.
"""

from __future__ import annotations

from typing import Any

import pytest

from super_processor.span_sr import SpanError, load_span, resolve_span_weights
from super_processor.upscale import models_dir

torch: Any = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

# 7 frames at a batch of 3: the last batch is short.
_COUNT, _BATCH = 7, 3


def _frames() -> Any:
    """``_COUNT`` frames of a gradient, so every value survives the trip."""
    gen = torch.Generator().manual_seed(7)
    return torch.randint(0, 256, (_COUNT, 24, 32, 3), dtype=torch.uint8, generator=gen)


def test_batches_and_colour_conversion_round_trip_exactly() -> None:
    from super_processor.span_sr import upscale_frames

    # Nearest-neighbour 2x stands in for the network: whatever goes wrong in
    # the slicing, NHWC/NCHW permutes, or the 0-1 half conversion shows.
    model = torch.nn.Upsample(scale_factor=2, mode="nearest").cuda().half()
    frames = _frames()
    out = upscale_frames(model, frames, batch=_BATCH)
    assert out.dtype == torch.uint8 and out.device.type == "cpu"
    assert tuple(out.shape) == (_COUNT, 48, 64, 3)
    expected = frames.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    assert torch.equal(out, expected)


def test_real_weights_keep_a_flat_colour_and_ignore_the_batch() -> None:
    from super_processor.span_sr import upscale_frames

    try:
        weights = resolve_span_weights(models_dir())
    except SpanError as exc:
        pytest.skip(f"no SPAN weights: {exc}")
    model = load_span(weights)
    flat = torch.empty((_COUNT, 24, 32, 3), dtype=torch.uint8)
    flat[...] = torch.tensor([120, 64, 200], dtype=torch.uint8)
    out = upscale_frames(model, flat, batch=_BATCH)
    assert out.dtype == torch.uint8 and tuple(out.shape) == (_COUNT, 48, 64, 3)
    # Away from the border, SPAN leaves a flat colour as it is.
    middle = out[:, 8:-8, 8:-8].int()
    assert (middle - flat[:1, :1, :1].int()).abs().max().item() <= 2
    # The same frames one batch at a time come out the same.
    one = upscale_frames(model, _frames(), batch=1)
    whole = upscale_frames(model, _frames(), batch=_COUNT)
    assert (one.int() - whole.int()).abs().max().item() <= 1
