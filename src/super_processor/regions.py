"""Protected regions: where FlashVSR is held down inside a shot.

Faces, hands, and text are where invented detail shows first: a sixth finger,
a different iris, a sign that no longer reads. Gemini marks boxes around them
on three stills of the shot (start, middle, end), so one box covers where the
thing moves during the shot. Boxes are normalised to the frame (0-1), padded,
and feathered, so the same expression works on every plane and at any scale.

Inside a box, the FlashVSR share drops to the kind's ceiling in
:data:`REGION_STRENGTH`; it rises back to the shot's strength across
:data:`FEATHER` of the frame outside the box, so there is no seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Highest FlashVSR share allowed inside a box of each kind.
REGION_STRENGTH: dict[str, float] = {"faces": 0.3, "hands": 0.3, "text": 0.15}
KINDS = tuple(REGION_STRENGTH)
# Padding added on every side of a box, and the width of the soft edge, as a
# share of the frame.
PAD = 0.03
FEATHER = 0.03
MAX_REGIONS = 6


class RegionError(ValueError):
    """Raised when a stored region is malformed."""


@dataclass(frozen=True, slots=True)
class Region:
    """A box in normalised frame coordinates (0-1, origin top left)."""

    kind: str
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.kind not in REGION_STRENGTH:
            raise RegionError(f"unknown region kind {self.kind!r}")
        values = (self.x0, self.y0, self.x1, self.y1)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
            raise RegionError("region corners must be numbers")
        if not (0 <= self.x0 < self.x1 <= 1 and 0 <= self.y0 < self.y1 <= 1):
            raise RegionError("region corners must be inside the frame")
        for name in ("x0", "y0", "x1", "y1"):
            object.__setattr__(self, name, round(float(getattr(self, name)), 4))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Region:
        if not isinstance(data, dict):
            raise RegionError("a region must be an object")
        try:
            return cls(
                kind=str(data["kind"]),
                x0=data["x0"],
                y0=data["y0"],
                x1=data["x1"],
                y1=data["y1"],
            )
        except KeyError as exc:
            raise RegionError(f"region is missing {exc}") from exc


def regions_from_dicts(raw: Any) -> tuple[Region, ...]:
    """Stored regions; anything malformed is dropped."""
    if not isinstance(raw, list):
        return ()
    out: list[Region] = []
    for item in raw:
        try:
            out.append(Region.from_dict(item))
        except RegionError:
            continue
    return tuple(out[:MAX_REGIONS])


def region_from_model(kind: Any, box: Any) -> Region | None:
    """A padded region from Gemini's ``box_2d``: [ymin, xmin, ymax, xmax], 0-1000.

    Returns ``None`` for an unknown kind or a box that is not four numbers.
    """
    name = str(kind).strip().lower()
    if name not in REGION_STRENGTH or not isinstance(box, list) or len(box) != 4:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in box):
        return None
    ymin, xmin, ymax, xmax = (min(1000.0, max(0.0, float(v))) / 1000 for v in box)
    x0, x1 = sorted((xmin, xmax))
    y0, y1 = sorted((ymin, ymax))
    if x1 - x0 < 0.005 or y1 - y0 < 0.005:
        return None
    return Region(
        kind=name,
        x0=max(0.0, x0 - PAD),
        y0=max(0.0, y0 - PAD),
        x1=min(1.0, x1 + PAD),
        y1=min(1.0, y1 + PAD),
    )


def _box_weight(region: Region) -> str:
    """1 inside the box, falling to 0 over :data:`FEATHER` outside it."""
    inside = (
        f"min(min(X/W-{region.x0:.4f},{region.x1:.4f}-X/W),"
        f"min(Y/H-{region.y0:.4f},{region.y1:.4f}-Y/H))"
    )
    return f"clip(({inside}+{FEATHER:.4f})/{FEATHER:.4f},0,1)"


def strength_expr(strength: float, regions: tuple[Region, ...]) -> str:
    """Per-pixel FlashVSR share for an FFmpeg ``blend`` expression.

    Without regions this is just ``strength``. With them, each kind lowers the
    share towards its ceiling where its boxes are, and the lowest share wins
    where kinds overlap.
    """
    terms: list[str] = []
    for kind in KINDS:
        boxes = [r for r in regions if r.kind == kind]
        ceiling = min(strength, REGION_STRENGTH[kind])
        if not boxes or ceiling >= strength:
            continue
        weight = _box_weight(boxes[0])
        for box in boxes[1:]:
            weight = f"max({weight},{_box_weight(box)})"
        terms.append(f"({strength:.4f}-{strength - ceiling:.4f}*{weight})")
    if not terms:
        return f"{strength:.4f}"
    expr = terms[0]
    for term in terms[1:]:
        expr = f"min({expr},{term})"
    return expr
