"""Tests for the per-problem treatment table."""

from __future__ import annotations

import pytest

from super_processor.recipe import DEFAULT_OP_ORDER, OpName
from super_processor.segments import PROBLEM_NAMES
from super_processor.treatments import (
    Treatment,
    TreatmentError,
    TreatmentStep,
    iter_treatments,
    require_legal_treatment,
    treatments_for,
)
from super_processor.validator import PARAM_BOUNDS


def _brightness(problem: str, name: str) -> float:
    treatment = next(
        item for item in treatments_for(problem) if item.treatment_id.endswith(name)
    )
    contrast = next(step for step in treatment.steps if step.op is OpName.CONTRAST)
    return contrast.as_dict()["brightness"]


def test_every_problem_has_several_ordered_treatments() -> None:
    seen: set[str] = set()
    order = {name: index for index, name in enumerate(DEFAULT_OP_ORDER)}
    for problem in PROBLEM_NAMES:
        options = treatments_for(problem)
        assert len(options) >= 2
        for treatment in options:
            assert treatment.problem == problem
            assert treatment.treatment_id not in seen
            seen.add(treatment.treatment_id)
            indexes = [order[step.op] for step in treatment.steps]
            assert indexes == sorted(indexes)
            assert len(indexes) == len(set(indexes))
            for step in treatment.steps:
                for key, value in step.as_dict().items():
                    low, high = PARAM_BOUNDS[step.op][key]
                    assert low <= value <= high


def test_silhouette_lifts_shadows_harder_than_low_light() -> None:
    assert _brightness("silhouette", "lift") > _brightness("low_light", "lift")
    lift = next(
        item
        for item in treatments_for("silhouette")
        if item.treatment_id.endswith("lift")
    )
    assert lift.steps[0].as_dict()["contrast"] == 1.0
    touched = next(
        item
        for item in treatments_for("silhouette")
        if item.treatment_id.endswith("lift_contrast")
    )
    assert touched.steps[0].as_dict()["contrast"] > 1.0


def test_too_warm_cools_the_white_balance() -> None:
    for treatment in treatments_for("too_warm"):
        temperature = treatment.steps[0].as_dict()["temperature"]
        assert temperature > 6500.0


def test_denoise_and_sharpen_stay_apart() -> None:
    for treatment in iter_treatments():
        ops = {step.op for step in treatment.steps}
        assert not (OpName.DENOISE in ops and OpName.SHARPEN in ops)


def test_catalog_treatments_are_legal() -> None:
    for treatment in iter_treatments():
        require_legal_treatment(treatment)


def test_strong_denoise_with_sharpen_is_rejected() -> None:
    illegal = Treatment(
        treatment_id="noisy.harsh",
        problem="noisy",
        steps=(
            TreatmentStep(OpName.DENOISE, (("strength", 0.8),)),
            TreatmentStep(OpName.SHARPEN, (("luma_amount", 0.5), ("luma_size", 5.0))),
        ),
    )
    with pytest.raises(TreatmentError, match="sharpen"):
        require_legal_treatment(illegal)
    mild = Treatment(
        treatment_id="noisy.gentle",
        problem="noisy",
        steps=(
            TreatmentStep(OpName.DENOISE, (("strength", 0.2),)),
            TreatmentStep(OpName.SHARPEN, (("luma_amount", 0.4), ("luma_size", 5.0))),
        ),
    )
    require_legal_treatment(mild)


def test_out_of_order_steps_are_rejected() -> None:
    illegal = Treatment(
        treatment_id="noisy.reversed",
        problem="noisy",
        steps=(
            TreatmentStep(OpName.DENOISE, (("strength", 0.2),)),
            TreatmentStep(OpName.CONTRAST, (("contrast", 1.1),)),
        ),
    )
    with pytest.raises(TreatmentError, match="order"):
        require_legal_treatment(illegal)


def test_unknown_problem_is_refused() -> None:
    with pytest.raises(TreatmentError, match="no treatments"):
        treatments_for("face_swap")
