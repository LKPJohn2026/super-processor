"""Per-shot FFmpeg settings around the FlashVSR restore.

Every shot runs the same fixed pipeline. Only the numbers change:

1. clean: ``deblock`` then ``denoise``, on the source frames. FlashVSR reads
   compression blocks and noise as real detail and sharpens them into texture,
   so removing them first leaves it less to invent from. The plain-upscale base
   of the strength blend is cleaned the same way.
2. restore: FlashVSR at the shot's ``scale``, mixed with the cleaned base by
   ``strength`` (see :mod:`super_processor.upscale`).
3. finish: ``contrast``, ``brightness``, ``saturation``, ``gamma``, then
   ``grain``, on the restored picture.

Gemini and the editor only set values inside :data:`LOOK_BOUNDS`. Filter text
is built here from those numbers; no model output is ever passed to FFmpeg.
None of these filters add or drop frames.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


class LookError(ValueError):
    """Raised when a shot look is missing a field or a value is out of bounds."""


# name: (low, high, neutral). The neutral value leaves the picture unchanged.
LOOK_BOUNDS: dict[str, tuple[float, float, float]] = {
    "deblock": (0.0, 1.0, 0.0),
    "denoise": (0.0, 1.0, 0.0),
    "contrast": (0.8, 1.3, 1.0),
    "brightness": (-0.1, 0.1, 0.0),
    "saturation": (0.7, 1.3, 1.0),
    "gamma": (0.8, 1.25, 1.0),
    "grain": (0.0, 1.0, 0.0),
}

# hqdn3d at denoise=1.0: luma/chroma spatial, luma/chroma temporal. Temporal
# stays near the filter default so moving edges do not ghost.
_DENOISE_MAX = (6.0, 4.5, 6.0, 4.5)
# noise strength at grain=1.0, on a 0-100 scale. Seeded so a re-render of the
# same shot is identical.
_GRAIN_MAX = 12
_GRAIN_SEED = 1


@dataclass(frozen=True, slots=True)
class ShotLook:
    """The clean and finish values for one shot. Defaults change nothing."""

    deblock: float = 0.0
    denoise: float = 0.0
    contrast: float = 1.0
    brightness: float = 0.0
    saturation: float = 1.0
    gamma: float = 1.0
    grain: float = 0.0

    def __post_init__(self) -> None:
        for field in fields(self):
            low, high, _neutral = LOOK_BOUNDS[field.name]
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise LookError(f"{field.name} must be a number")
            value = float(value)
            if not low <= value <= high:
                raise LookError(f"{field.name} must be between {low:g} and {high:g}")
            object.__setattr__(self, field.name, value)

    def is_neutral(self) -> bool:
        return self == ShotLook()

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> ShotLook:
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise LookError("look must be an object")
        unknown = set(data) - set(LOOK_BOUNDS)
        if unknown:
            raise LookError(f"unknown look settings: {', '.join(sorted(unknown))}")
        return cls(**data)

    def clean_filter(self) -> str:
        """FFmpeg filters for step 1, or ``""`` when there is nothing to clean."""
        parts: list[str] = []
        if self.deblock > 0:
            d = self.deblock
            parts.append(
                "deblock=filter=strong:block=8"
                f":alpha={0.05 + 0.15 * d:.4f}:beta={0.03 + 0.10 * d:.4f}"
                f":gamma={0.03 + 0.10 * d:.4f}:delta={0.03 + 0.10 * d:.4f}"
            )
        if self.denoise > 0:
            ls, cs, lt, ct = (limit * self.denoise for limit in _DENOISE_MAX)
            parts.append(f"hqdn3d={ls:.3f}:{cs:.3f}:{lt:.3f}:{ct:.3f}")
        return ",".join(parts)

    def finish_filter(self) -> str:
        """FFmpeg filters for step 3, or ``""`` when nothing changes."""
        parts: list[str] = []
        if (self.contrast, self.brightness, self.saturation, self.gamma) != (
            1.0,
            0.0,
            1.0,
            1.0,
        ):
            parts.append(
                f"eq=contrast={self.contrast:.4f}:brightness={self.brightness:.4f}"
                f":saturation={self.saturation:.4f}:gamma={self.gamma:.4f}"
            )
        strength = round(_GRAIN_MAX * self.grain)
        if strength > 0:
            parts.append(f"noise=alls={strength}:allf=t:all_seed={_GRAIN_SEED}")
        return ",".join(parts)
