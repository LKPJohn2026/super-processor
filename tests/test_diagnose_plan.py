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


def test_structured_llm_patch_clamps_and_ignores_unknown() -> None:
    from super_processor.models import apply_structured_patches, parse_structured_patch

    recipe = empty_recipe("abcd1234abcd1234", "/tmp/x.mp4")
    payload = parse_structured_patch(
        'noise {"patches":[{"op":"denoise","enabled":true,'
        '"params":{"strength":9.0,"bogus":1}},'
        '{"op":"shell","enabled":true}]}'
    )
    assert payload is not None
    patched = apply_structured_patches(recipe, payload)
    denoise = next(op for op in patched.ops if op.op is OpName.DENOISE)
    assert denoise.enabled is True
    assert denoise.params["strength"] == 1.0
    assert "bogus" not in denoise.params


def test_format_progress_status() -> None:
    from super_processor.worker import format_progress_status

    assert "encoding" in format_progress_status({})
    line = format_progress_status(
        {"out_time_ms": "5000", "speed": "1.2x"},
        duration_s=10.0,
    )
    assert "50.0%" in line
    assert "1.2x" in line
    clock = format_progress_status({"out_time": "00:00:02.5", "speed": "1x"})
    assert "2.5s" in clock


def test_resolve_secret_and_diagnose_ack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from super_processor.diagnose import diagnose_job_dir, write_diagnosis
    from super_processor.estimators import write_estimates
    from super_processor.models import resolve_secret
    from super_processor.probe import MediaFacts, StreamFacts, write_media_facts
    from super_processor.reframe import write_reframe_plan

    monkey_env = "SUPER_PROCESSOR_TEST_SECRET"
    monkeypatch.setenv(monkey_env, "tok")
    assert resolve_secret(monkey_env) == "tok"
    monkeypatch.delenv(monkey_env, raising=False)
    assert resolve_secret(monkey_env) is None

    source = str(tmp_path / "clip.mp4")
    estimates = _estimates(source)
    social = SocialExportPlan(
        schema_version=1,
        source_path=source,
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
            subject_strategy="saliency",
            subject_cx=0.3,
        ),
        size_cap=SizeCapAssessment(
            max_size_mb=1.0,
            duration_s=7200.0,
            avg_kbps=1.0,
            floor_kbps=300.0,
            status="infeasible",
            message="too small",
        ),
    )
    diagnosis = build_diagnosis(estimates, social)
    diagnosis.size_cap = dict(diagnosis.size_cap or {})
    diagnosis.size_cap["acknowledge_size_risk"] = 1.0
    recipe = recipe_from_diagnosis(
        diagnosis,
        job_id="abcd1234abcd1234",
        source_path=source,
    )
    encode = next(op for op in recipe.ops if op.op is OpName.ENCODE_HEVC_SIZE_CAP)
    assert encode.params.get("acknowledge_size_risk") == 1.0
    reframe = next(op for op in recipe.ops if op.op is OpName.REFRAME_VERTICAL)
    assert reframe.params["crop_x"] == 420.0
    assert reframe.params["subject_cx"] == 0.3

    facts = MediaFacts(
        schema_version=1,
        source_path=source,
        format_name="mp4",
        format_long_name="MP4",
        duration_s=7200.0,
        size_bytes=1,
        bit_rate=1,
        streams=[
            StreamFacts(
                index=0,
                codec_type="video",
                codec_name="h264",
                width=640,
                height=360,
            )
        ],
        has_video=True,
        has_audio=False,
        is_vfr=False,
    )
    write_media_facts(tmp_path, facts)
    write_estimates(tmp_path, estimates)
    write_reframe_plan(tmp_path, social)
    ack = diagnose_job_dir(tmp_path, max_size_mb=1.0, acknowledge_size_risk=True)
    assert ack.size_cap is not None
    assert ack.size_cap.get("acknowledge_size_risk") == 1.0
    write_diagnosis(tmp_path, ack)


def test_apply_llm_uses_structured_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    from super_processor import models as models_mod

    def fake_chat(
        endpoint: object,
        instruction: str,
        recipe: object,
    ) -> str:
        return (
            '{"patches":[{"op":"denoise","enabled":true,"params":{"strength":0.22}}]}'
        )

    monkeypatch.setattr(models_mod, "_advisory_chat", fake_chat)
    recipe = empty_recipe("abcd1234abcd1234", "/tmp/x.mp4")
    patched = apply_llm_plan_patch(
        recipe,
        "please tweak denoise",
        safe_mode=False,
    )
    denoise = next(op for op in patched.ops if op.op is OpName.DENOISE)
    assert denoise.enabled is True
    assert denoise.params["strength"] == 0.22


def test_list_models_and_missing_preview_qa(tmp_path: Path) -> None:
    payload = list_models(remote_frames_consent=True)
    assert payload["remote_frames_consent"] is True
    assert payload["endpoints"]
    report = analyze_preview(tmp_path / "missing.mp4")
    assert isinstance(report, QAReport)
    assert report.ok is False
    assert report.issues[0].code == "missing_preview"
    from super_processor.qa import write_qa_report

    path = write_qa_report(tmp_path, report)
    assert path.is_file()
    assert "missing_preview" in path.read_text(encoding="utf-8")


def test_diagnosis_roundtrip_loader(tmp_path: Path) -> None:
    from super_processor.diagnose import load_diagnosis

    estimates = _estimates(str(tmp_path / "clip.mp4"))
    diagnosis = build_diagnosis(estimates, None)
    write_diagnosis(tmp_path, diagnosis)
    loaded = load_diagnosis(tmp_path)
    assert loaded.summary == diagnosis.summary
    assert "contrast" in loaded.suggestions


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


def test_resolve_secret_keyring_and_diagnose_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    from super_processor.diagnose import diagnose_source
    from super_processor.models import resolve_secret
    from super_processor.probe import MediaFacts, StreamFacts

    fake_keyring = types.SimpleNamespace(
        get_password=lambda service, name: (
            "from-keyring" if name == "OPENAI_API_KEY" else None
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "keyring", fake_keyring)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resolve_secret("OPENAI_API_KEY") == "from-keyring"

    estimates = _estimates(str(tmp_path / "clip.mp4"))
    social = SocialExportPlan(
        schema_version=1,
        source_path=str(tmp_path / "clip.mp4"),
        reframe=ReframePath(
            mode="vertical_9_16",
            input_width=640,
            input_height=360,
            output_width=360,
            output_height=640,
            crop_x=140,
            crop_y=0,
            crop_w=360,
            crop_h=360,
            padding=0.0,
            subject_strategy="center",
            subject_cx=0.5,
        ),
        size_cap=SizeCapAssessment(
            max_size_mb=1.0,
            duration_s=7200.0,
            avg_kbps=1.0,
            floor_kbps=300.0,
            status="infeasible",
            message="too small",
        ),
    )
    monkeypatch.setattr(
        "super_processor.diagnose.estimate_look",
        lambda *args, **kwargs: estimates,
    )
    monkeypatch.setattr(
        "super_processor.diagnose.plan_social_export",
        lambda *args, **kwargs: social,
    )
    facts = MediaFacts(
        schema_version=1,
        source_path=str(tmp_path / "clip.mp4"),
        format_name="mp4",
        format_long_name="MP4",
        duration_s=7200.0,
        size_bytes=1,
        bit_rate=1,
        streams=[
            StreamFacts(
                index=0,
                codec_type="video",
                codec_name="h264",
                width=640,
                height=360,
            )
        ],
        has_video=True,
        has_audio=False,
        is_vfr=False,
    )
    diagnosis = diagnose_source(
        tmp_path / "clip.mp4",
        facts,
        max_size_mb=1.0,
        acknowledge_size_risk=True,
    )
    assert diagnosis.size_cap is not None
    assert diagnosis.size_cap.get("acknowledge_size_risk") == 1.0
    assert "acknowledged" in diagnosis.summary


def test_empty_instruction_and_parse_none() -> None:
    from super_processor.models import parse_structured_patch

    recipe = empty_recipe("abcd1234abcd1234", "/tmp/x.mp4")
    same = apply_llm_plan_patch(recipe, "   ", safe_mode=False)
    assert same is recipe
    assert parse_structured_patch("") is None
    assert parse_structured_patch("not json") is None
