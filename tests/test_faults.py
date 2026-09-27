"""Tests for shrinking a treatment list by fault."""

from __future__ import annotations

import pytest

from super_processor.treatments import TreatmentError, filter_treatments


def test_not_sharp_keeps_only_the_sharpen_treatment() -> None:
    chosen = filter_treatments("low_contrast", "not-sharp")
    assert [item.treatment_id for item in chosen] == ["low_contrast.contrast_sharpen"]


def test_too_much_denoise_drops_the_strong_setting() -> None:
    chosen = filter_treatments("noisy", "too-much-denoise")
    assert "noisy.medium" not in [item.treatment_id for item in chosen]
    assert chosen
    assert all(
        step.as_dict().get("strength", 0.0) < 0.5
        for item in chosen
        for step in item.steps
        if step.op.value == "denoise"
    )


def test_a_fault_with_no_match_keeps_the_first_treatment() -> None:
    chosen = filter_treatments("too_warm", "too-cool")
    assert [item.treatment_id for item in chosen] == ["too_warm.cool"]


def test_unknown_fault_is_refused() -> None:
    with pytest.raises(TreatmentError, match="unknown fault"):
        filter_treatments("noisy", "make-it-cinematic")
