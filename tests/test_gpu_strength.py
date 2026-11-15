"""The strength rule the GPU blend draws its map from, in plain Python.

:func:`~super_processor.regions.strength_at` is what
``test_gpu_blend.py`` checks the GPU's strength map against, so here it is
checked against the FFmpeg expression it mirrors, without torch.
"""

from __future__ import annotations

import pytest

from super_processor.regions import FEATHER, Region, strength_at, strength_expr

_REGIONS = (
    Region("faces", 0.10, 0.10, 0.40, 0.50),
    Region("faces", 0.30, 0.20, 0.60, 0.45),
    Region("hands", 0.50, 0.40, 0.80, 0.90),
    Region("text", 0.05, 0.70, 0.35, 0.95),
)


def _ffmpeg_value(expr: str, x: float, y: float) -> float:
    """``expr`` (our own text, in FFmpeg's syntax) at normalised (x, y)."""
    names = {
        "min": min,
        "max": max,
        "clip": lambda v, low, high: min(max(v, low), high),
        "X": x,
        "Y": y,
        "W": 1.0,
        "H": 1.0,
    }
    return float(eval(expr, {"__builtins__": {}}, names))  # noqa: S307


@pytest.mark.parametrize("strength", [1.0, 0.5, 0.25, 0.15, 0.0])
def test_strength_at_is_the_ffmpeg_expression(strength: float) -> None:
    expr = strength_expr(strength, _REGIONS)
    points = [i / 40 + 0.004 for i in range(40)]
    for x in points:
        for y in points:
            assert strength_at(strength, _REGIONS, x, y) == pytest.approx(
                _ffmpeg_value(expr, x, y), abs=1e-4
            )


def test_strength_at_inside_outside_and_on_the_feather() -> None:
    face = (Region("faces", 0.2, 0.2, 0.6, 0.6),)
    assert strength_at(0.8, face, 0.4, 0.4) == pytest.approx(0.3)
    assert strength_at(0.8, face, 0.6 + FEATHER / 2, 0.4) == pytest.approx(0.55)
    assert strength_at(0.8, face, 0.6 + FEATHER, 0.4) == pytest.approx(0.8)
    assert strength_at(0.8, face, 0.95, 0.95) == pytest.approx(0.8)
    # Below the kind's ceiling a box changes nothing.
    assert strength_at(0.2, face, 0.4, 0.4) == pytest.approx(0.2)
    # Where kinds overlap, the lowest share wins.
    both = (*face, Region("text", 0.3, 0.3, 0.5, 0.5))
    assert strength_at(0.8, both, 0.4, 0.4) == pytest.approx(0.15)
    assert strength_at(0.8, (), 0.4, 0.4) == 0.8
