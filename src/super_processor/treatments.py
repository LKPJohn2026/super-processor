"""Allowlisted local treatments for each segment problem."""

from __future__ import annotations

from dataclasses import dataclass

from .recipe import DEFAULT_OP_ORDER, OpName
from .segments import PROBLEM_NAMES

_ORDER = {name: index for index, name in enumerate(DEFAULT_OP_ORDER)}


class TreatmentError(ValueError):
    """Raised when a treatment list is missing or a treatment is illegal."""


# Denoise at or above this strength cannot share a treatment with sharpen.
STRONG_DENOISE = 0.5


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


FAULT_PROBLEMS: dict[str, frozenset[str]] = {
    "too-warm": frozenset({"too_warm"}),
    "too-cool": frozenset(),
    "too-dark": frozenset({"low_light", "silhouette"}),
    "too-bright": frozenset(),
    "not-sharp": frozenset({"low_contrast"}),
    "too-much-denoise": frozenset({"noisy"}),
    "too-much-contrast": frozenset({"low_contrast"}),
    "bad-trim": frozenset(),
}

FAULT_NAMES: tuple[str, ...] = (
    "too-warm",
    "too-cool",
    "too-dark",
    "too-bright",
    "not-sharp",
    "too-much-denoise",
    "too-much-contrast",
    "bad-trim",
)


def _step_param(treatment: Treatment, op: OpName, key: str) -> float | None:
    for step in treatment.steps:
        if step.op is op and key in step.as_dict():
            return float(step.as_dict()[key])
    return None


_SENTENCE_FAULTS: tuple[tuple[str, str], ...] = (
    ("too much denoise", "too-much-denoise"),
    ("too much contrast", "too-much-contrast"),
    ("not sharp", "not-sharp"),
    ("too dark", "too-dark"),
    ("too bright", "too-bright"),
    ("too warm", "too-warm"),
    ("too cool", "too-cool"),
    ("bad trim", "bad-trim"),
)


def faults_from_sentence(text: str) -> list[str]:
    """Map a sentence onto the fault list. Unmatched wording returns nothing."""
    cleaned = " ".join(text.strip().lower().split())
    found: list[str] = []
    for phrase, fault in _SENTENCE_FAULTS:
        if phrase in cleaned and fault not in found:
            found.append(fault)
    return found


def filter_treatments(problem: str, fault: str) -> tuple[Treatment, ...]:
    """Shrink one problem's treatment list according to a fixed fault."""
    options = treatments_for(problem)
    if fault not in FAULT_NAMES:
        raise TreatmentError(f"unknown fault {fault}")
    if fault == "too-warm":
        chosen = [
            treatment
            for treatment in options
            if (_step_param(treatment, OpName.WHITE_BALANCE, "temperature") or 0.0)
            > 6500.0
        ]
    elif fault == "too-cool":
        chosen = [
            treatment
            for treatment in options
            if (_step_param(treatment, OpName.WHITE_BALANCE, "temperature") or 10_000.0)
            < 6500.0
        ]
    elif fault == "too-dark":
        chosen = [
            treatment
            for treatment in options
            if (_step_param(treatment, OpName.CONTRAST, "brightness") or 0.0) > 0.0
        ]
    elif fault == "too-bright":
        chosen = [
            treatment
            for treatment in options
            if (_step_param(treatment, OpName.CONTRAST, "brightness") or 0.0) < 0.0
        ]
    elif fault == "not-sharp":
        chosen = [
            treatment
            for treatment in options
            if any(step.op is OpName.SHARPEN for step in treatment.steps)
        ]
    elif fault == "too-much-denoise":
        chosen = [
            treatment
            for treatment in options
            if (_step_param(treatment, OpName.DENOISE, "strength") or 0.0)
            < STRONG_DENOISE
        ]
    elif fault == "too-much-contrast":
        contrasts = [
            (_step_param(treatment, OpName.CONTRAST, "contrast") or 1.0, treatment)
            for treatment in options
        ]
        mildest = min(value for value, _treatment in contrasts)
        chosen = [treatment for value, treatment in contrasts if value == mildest]
    else:
        chosen = [
            treatment
            for treatment in options
            if any(step.op is OpName.TRIM for step in treatment.steps)
        ]
    if not chosen:
        return (options[0],)
    return tuple(chosen)


def treatments_for(problem: str) -> tuple[Treatment, ...]:
    """Return the ordered local treatments for one primary problem."""
    try:
        return _TABLE[problem]
    except KeyError as exc:
        raise TreatmentError(f"no treatments for {problem}") from exc


def treatment_by_id(treatment_id: str) -> Treatment:
    """Return the catalog treatment with this id."""
    for treatment in iter_treatments():
        if treatment.treatment_id == treatment_id:
            return treatment
    raise TreatmentError(f"unknown treatment {treatment_id}")


def iter_treatments() -> tuple[Treatment, ...]:
    """Return every treatment, in problem order."""
    listed: list[Treatment] = []
    for problem in PROBLEM_NAMES:
        listed.extend(treatments_for(problem))
    return tuple(listed)


def treatment_violation(treatment: Treatment) -> str | None:
    """Return why a treatment is illegal, or ``None`` when it is allowed.

    Steps must follow the recipe allowlist, each operation once. Strong denoise
    and sharpen cannot appear together.
    """
    indexes = [_ORDER[step.op] for step in treatment.steps]
    if indexes != sorted(indexes) or len(indexes) != len(set(indexes)):
        return "operation order does not match the allowlist"
    denoise = 0.0
    sharpen = False
    for step in treatment.steps:
        if step.op is OpName.DENOISE:
            denoise = float(step.as_dict().get("strength", 0.0))
        if step.op is OpName.SHARPEN:
            sharpen = True
    if sharpen and denoise >= STRONG_DENOISE:
        return "strong denoise and sharpen cannot share a treatment"
    return None


def require_legal_treatment(treatment: Treatment) -> None:
    """Raise when a treatment breaks order or pairs strong denoise with sharpen."""
    reason = treatment_violation(treatment)
    if reason is not None:
        raise TreatmentError(reason)
