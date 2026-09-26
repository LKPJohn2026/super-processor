"""Allowlisted local treatments for each segment problem."""

from __future__ import annotations

from dataclasses import dataclass

from .recipe import DEFAULT_OP_ORDER, OpName
from .segments import PROBLEM_NAMES

_ORDER = {name: index for index, name in enumerate(DEFAULT_OP_ORDER)}


class TreatmentError(ValueError):
    """Raised when a problem has no treatment list."""


@dataclass(frozen=True, slots=True)
class TreatmentStep:
    """One allowlisted operation and its parameters."""

    op: OpName
    params: tuple[tuple[str, float], ...]

    def as_dict(self) -> dict[str, float]:
        return dict(self.params)


@dataclass(frozen=True, slots=True)
class Treatment:
    """An ordered set of steps for one problem."""

    treatment_id: str
    problem: str
    steps: tuple[TreatmentStep, ...]


def _step(op: OpName, **params: float) -> TreatmentStep:
    return TreatmentStep(op, tuple(params.items()))


def _treatment(problem: str, name: str, steps: tuple[TreatmentStep, ...]) -> Treatment:
    indexes = [_ORDER[step.op] for step in steps]
    if indexes != sorted(indexes):
        raise TreatmentError(f"{problem}.{name} steps are out of allowlist order")
    return Treatment(f"{problem}.{name}", problem, steps)


# Shadow lift is a brightness raise on the contrast op. Contrast itself stays
# at 1.0 until a later step in the same family touches it. Silhouette uses a
# stronger lift than low light. Denoise never shares a treatment with sharpen.
_TABLE: dict[str, tuple[Treatment, ...]] = {
    "low_light": (
        _treatment(
            "low_light",
            "lift",
            (_step(OpName.CONTRAST, contrast=1.0, brightness=0.12, gamma=1.1),),
        ),
        _treatment(
            "low_light",
            "lift_denoise",
            (
                _step(OpName.CONTRAST, contrast=1.0, brightness=0.12, gamma=1.1),
                _step(OpName.DENOISE, strength=0.35),
            ),
        ),
        _treatment(
            "low_light",
            "lift_contrast",
            (_step(OpName.CONTRAST, contrast=1.2, brightness=0.12, gamma=1.05),),
        ),
    ),
    "low_contrast": (
        _treatment(
            "low_contrast",
            "mild",
            (_step(OpName.CONTRAST, contrast=1.25, brightness=0.0, gamma=1.0),),
        ),
        _treatment(
            "low_contrast",
            "strong",
            (_step(OpName.CONTRAST, contrast=1.6, brightness=0.0, gamma=1.0),),
        ),
        _treatment(
            "low_contrast",
            "contrast_sharpen",
            (
                _step(OpName.CONTRAST, contrast=1.25, brightness=0.0, gamma=1.0),
                _step(OpName.SHARPEN, luma_amount=0.5, luma_size=5.0),
            ),
        ),
    ),
    "silhouette": (
        _treatment(
            "silhouette",
            "lift",
            (_step(OpName.CONTRAST, contrast=1.0, brightness=0.28, gamma=1.25),),
        ),
        _treatment(
            "silhouette",
            "lift_denoise",
            (
                _step(OpName.CONTRAST, contrast=1.0, brightness=0.28, gamma=1.25),
                _step(OpName.DENOISE, strength=0.25),
            ),
        ),
        _treatment(
            "silhouette",
            "lift_contrast",
            (_step(OpName.CONTRAST, contrast=1.15, brightness=0.28, gamma=1.15),),
        ),
    ),
    "too_warm": (
        _treatment(
            "too_warm",
            "cool",
            (_step(OpName.WHITE_BALANCE, temperature=8000.0, tint=0.0),),
        ),
        _treatment(
            "too_warm",
            "cool_mild",
            (_step(OpName.WHITE_BALANCE, temperature=7200.0, tint=0.0),),
        ),
        _treatment(
            "too_warm",
            "cool_contrast",
            (
                _step(OpName.WHITE_BALANCE, temperature=8000.0, tint=0.0),
                _step(OpName.CONTRAST, contrast=1.15, brightness=0.0, gamma=1.0),
            ),
        ),
    ),
    "noisy": (
        _treatment("noisy", "light", (_step(OpName.DENOISE, strength=0.35),)),
        _treatment("noisy", "medium", (_step(OpName.DENOISE, strength=0.6),)),
        _treatment(
            "noisy",
            "denoise_contrast",
            (
                _step(OpName.CONTRAST, contrast=1.1, brightness=0.0, gamma=1.0),
                _step(OpName.DENOISE, strength=0.35),
            ),
        ),
    ),
    "shaky": (
        _treatment(
            "shaky",
            "mild",
            (_step(OpName.STABILIZE, shakiness=5.0, smoothing=10.0, max_crop_pct=5.0),),
        ),
        _treatment(
            "shaky",
            "strong",
            (
                _step(
                    OpName.STABILIZE, shakiness=8.0, smoothing=20.0, max_crop_pct=10.0
                ),
            ),
        ),
    ),
    "off_center": (
        _treatment(
            "off_center",
            "recenter",
            (_step(OpName.REFRAME_VERTICAL, padding=0.0, subject_cx=0.5),),
        ),
        _treatment(
            "off_center",
            "recenter_pad",
            (_step(OpName.REFRAME_VERTICAL, padding=0.04, subject_cx=0.5),),
        ),
    ),
}


def treatments_for(problem: str) -> tuple[Treatment, ...]:
    """Return the ordered local treatments for one primary problem."""
    try:
        return _TABLE[problem]
    except KeyError as exc:
        raise TreatmentError(f"no treatments for {problem}") from exc


def iter_treatments() -> tuple[Treatment, ...]:
    """Return every treatment, in problem order."""
    listed: list[Treatment] = []
    for problem in PROBLEM_NAMES:
        listed.extend(treatments_for(problem))
    return tuple(listed)
