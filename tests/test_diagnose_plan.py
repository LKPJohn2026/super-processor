"""Tests for diagnosis, planning, QA, and model helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_processor.diagnose import build_diagnosis, write_diagnosis
from super_processor.estimators import LookEstimates, OpSuggestion, SampleWindow
from super_processor.models import apply_llm_plan_patch, list_models
from super_processor.plan import apply_instruction_patches, recipe_from_diagnosis
from super_processor.qa import QAReport, analyze_preview
from super_processor.recipe import OpName, empty_recipe
from super_processor.reframe import (
    ReframePath,
    SizeCapAssessment,
    SocialExportPlan,
)


def _estimates(source: str = "/tmp/clip.mp4") -> LookEstimates:
    return LookEstimates(
        schema_version=1,
        source_path=source,
        windows=[SampleWindow(label="head", start_s=0.0)],
        contrast=OpSuggestion(
            enabled=True,
            confidence=0.7,
            params={"contrast": 1.2, "brightness": 0.0, "gamma": 1.0},
            reason="test",
        ),
        white_balance=OpSuggestion(
            enabled=False,
            confidence=0.1,
            params={"temperature": 6500.0, "tint": 0.0},
        ),
        denoise=OpSuggestion(
            enabled=True,
            confidence=0.6,
            params={"strength": 0.4},
        ),
        stabilize=OpSuggestion(
            enabled=False,
            confidence=0.0,
            params={"shakiness": 5.0, "smoothing": 10.0, "max_crop_pct": 10.0},
        ),
        metrics={"luma_mean": 100.0},
    )


def test_build_diagnosis_and_plan(tmp_path: Path) -> None:
    estimates = _estimates(str(tmp_path / "clip.mp4"))
    social = SocialExportPlan(
        schema_version=1,
        source_path=estimates.source_path,
        reframe=ReframePath(
            mode="vertical_9_16",
            input_width=1920,
            input_height=1080,
            output_width=1080,
            output_height=1920,
            crop_x=420,
            crop_y=0,
            crop_w=1080,
            crop_h=1080,
            padding=0.0,
            subject_strategy="center",
        ),
        size_cap=SizeCapAssessment(
            max_size_mb=50.0,
            duration_s=60.0,
            avg_kbps=6800.0,
            floor_kbps=300.0,
            status="ok",
            message="ok",
        ),
    )
    diagnosis = build_diagnosis(estimates, social)
    assert "contrast" in diagnosis.summary
    write_diagnosis(tmp_path, diagnosis)
    assert (tmp_path / "diagnosis.json").is_file()

    recipe = recipe_from_diagnosis(
        diagnosis,
        job_id="abcd1234abcd1234",
        source_path=estimates.source_path,
        instruction="less denoise warmer",
    )
    denoise = next(op for op in recipe.ops if op.op is OpName.DENOISE)
    assert denoise.enabled is True
    assert denoise.params["strength"] == 0.15
    wb = next(op for op in recipe.ops if op.op is OpName.WHITE_BALANCE)
    assert wb.enabled is True
    assert wb.params["temperature"] == 5800.0
    reframe = next(op for op in recipe.ops if op.op is OpName.REFRAME_VERTICAL)
    assert reframe.enabled is True


def test_instruction_patches_and_safe_llm() -> None:
    recipe = empty_recipe("abcd1234abcd1234", "/tmp/x.mp4")
    for op in recipe.ops:
        if op.op is OpName.DENOISE:
            op.enabled = True
            op.params = {"strength": 0.5}
    patched = apply_instruction_patches(recipe, "no denoise")
    denoise = next(op for op in patched.ops if op.op is OpName.DENOISE)
    assert denoise.enabled is False
    again = apply_llm_plan_patch(patched, "more denoise", safe_mode=True)
    denoise2 = next(op for op in again.ops if op.op is OpName.DENOISE)
    assert denoise2.enabled is True


def test_list_models_and_missing_preview_qa(tmp_path: Path) -> None:
    payload = list_models(remote_frames_consent=True)
    assert payload["remote_frames_consent"] is True
    assert payload["endpoints"]
    report = analyze_preview(tmp_path / "missing.mp4")
    assert isinstance(report, QAReport)
    assert report.ok is False
    assert report.issues[0].code == "missing_preview"


def test_planner_available_and_advisory_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from super_processor.models import ModelEndpoint, planner_available

    endpoint = ModelEndpoint(
        name="ollama",
        base_url="http://127.0.0.1:9",
        model="x",
    )
    assert planner_available(endpoint) is False

    recipe = empty_recipe("abcd1234abcd1234", "/tmp/x.mp4")
    # Non-safe mode still falls back to NL patches when the endpoint is down.
    patched = apply_llm_plan_patch(
        recipe,
        "more contrast",
        safe_mode=False,
        endpoint_name="ollama",
    )
    contrast = next(op for op in patched.ops if op.op is OpName.CONTRAST)
    assert contrast.enabled is True
    assert contrast.params["contrast"] == 1.35
