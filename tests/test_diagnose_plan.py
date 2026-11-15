"""Diagnosis documents, the CV-first recipe planner, and the model helpers
around it (advisory LLM patches, secrets, QA, progress lines)."""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest

from super_processor.diagnose import (
    Diagnosis,
    DiagnosisError,
    build_diagnosis,
    diagnose_job_dir,
    diagnose_source,
    load_diagnosis,
    write_diagnosis,
)
from super_processor.estimators import (
    LookEstimates,
    OpSuggestion,
    SampleWindow,
    write_estimates,
)
from super_processor.models import (
    ModelEndpoint,
    apply_llm_plan_patch,
    apply_structured_patches,
    list_models,
    parse_structured_patch,
    planner_available,
    resolve_secret,
)
from super_processor.plan import (
    PlanError,
    apply_instruction_patches,
    plan_job,
    recipe_from_diagnosis,
)
from super_processor.probe import MediaFacts, StreamFacts, write_media_facts
from super_processor.qa import QAReport, analyze_preview, write_qa_report
from super_processor.recipe import (
    DEFAULT_OP_ORDER,
    OpName,
    Recipe,
    empty_recipe,
    load_recipe,
)
from super_processor.reframe import (
    ReframePath,
    SizeCapAssessment,
    SocialExportPlan,
    write_reframe_plan,
)
from super_processor.worker import format_progress_status

JOB_ID = "abcd1234abcd1234"


def _estimates(source: str = "/tmp/clip.mp4", *, enabled: bool = True) -> LookEstimates:
    return LookEstimates(
        schema_version=1,
        source_path=source,
        windows=[SampleWindow(label="head", start_s=0.0)],
        contrast=OpSuggestion(
            enabled=enabled,
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
            enabled=enabled,
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


def _social(
    source: str,
    *,
    status: str = "ok",
    max_size_mb: float = 50.0,
    output_height: int = 1920,
    subject_cx: float = 0.5,
) -> SocialExportPlan:
    return SocialExportPlan(
        schema_version=1,
        source_path=source,
        reframe=ReframePath(
            mode="vertical_9_16",
            input_width=1920,
            input_height=1080,
            output_width=output_height * 9 // 16,
            output_height=output_height,
            crop_x=420,
            crop_y=0,
            crop_w=1080,
            crop_h=1080,
            padding=0.0,
            subject_strategy="center" if subject_cx == 0.5 else "saliency",
            subject_cx=subject_cx,
        ),
        size_cap=SizeCapAssessment(
            max_size_mb=max_size_mb,
            duration_s=60.0,
            avg_kbps=6800.0,
            floor_kbps=300.0,
            status=status,
            message=status,
        ),
    )


def _facts(source: str, duration_s: float = 60.0) -> MediaFacts:
    return MediaFacts(
        schema_version=1,
        source_path=source,
        format_name="mp4",
        format_long_name="MP4",
        duration_s=duration_s,
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


def _op(recipe: Recipe, name: OpName) -> Any:
    return next(op for op in recipe.ops if op.op is name)


# --- diagnosis --------------------------------------------------------------


def test_diagnosis_summary_names_the_enabled_ops_and_the_size_cap() -> None:
    diagnosis = build_diagnosis(_estimates(), _social("/tmp/clip.mp4", status="warn"))
    assert diagnosis.summary == "Suggested ops: contrast, denoise; size-cap warn"
    assert diagnosis.reframe is not None and diagnosis.reframe["crop_x"] == 420
    assert diagnosis.size_cap is not None and diagnosis.size_cap["status"] == "warn"
    quiet = build_diagnosis(_estimates(enabled=False), None)
    assert quiet.summary == "No look/motion corrections suggested"
    assert quiet.reframe is None and quiet.size_cap is None


def test_diagnosis_round_trips_through_its_file(tmp_path: Path) -> None:
    diagnosis = build_diagnosis(_estimates(), _social("/tmp/clip.mp4"))
    write_diagnosis(tmp_path, diagnosis)
    loaded = load_diagnosis(tmp_path)
    assert loaded.to_dict() == diagnosis.to_dict()
    assert loaded.suggestions["denoise"].params == {"strength": 0.4}
    assert not (tmp_path / "diagnosis.tmp").exists()


def test_a_missing_or_damaged_diagnosis_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DiagnosisError, match="not found"):
        load_diagnosis(tmp_path)
    (tmp_path / "diagnosis.json").write_text("[1]", encoding="utf-8")
    with pytest.raises(DiagnosisError, match="must be an object"):
        load_diagnosis(tmp_path)


def test_diagnose_job_dir_uses_saved_estimates_and_acknowledges_an_infeasible_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = str(tmp_path / "clip.mp4")

    def must_not_sample(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("saved estimates and reframe plan should be used")

    monkeypatch.setattr("super_processor.diagnose.estimate_look", must_not_sample)
    monkeypatch.setattr("super_processor.diagnose.plan_social_export", must_not_sample)
    write_media_facts(tmp_path, _facts(source, 7200.0))
    write_estimates(tmp_path, _estimates(source))
    write_reframe_plan(tmp_path, _social(source, status="infeasible", max_size_mb=1.0))
    ack = diagnose_job_dir(tmp_path, max_size_mb=1.0, acknowledge_size_risk=True)
    assert ack.size_cap is not None
    assert ack.size_cap["acknowledge_size_risk"] == 1.0
    assert ack.summary.endswith("; size risk acknowledged")
    plain = diagnose_job_dir(tmp_path, max_size_mb=1.0)
    assert plain.size_cap is not None and "acknowledge_size_risk" not in plain.size_cap


def test_diagnose_job_dir_computes_what_is_not_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = str(tmp_path / "clip.mp4")
    calls: list[str] = []

    def estimate(path: Path, facts: MediaFacts) -> LookEstimates:
        calls.append(f"estimate {path}")
        return _estimates(source)

    def social(facts: MediaFacts, **kwargs: Any) -> SocialExportPlan:
        calls.append(f"social {kwargs['max_size_mb']}")
        return _social(source, status="ok")

    monkeypatch.setattr("super_processor.diagnose.estimate_look", estimate)
    monkeypatch.setattr("super_processor.diagnose.plan_social_export", social)
    write_media_facts(tmp_path, _facts(source))
    diagnosis = diagnose_job_dir(tmp_path, max_size_mb=50.0, acknowledge_size_risk=True)
    assert calls == [f"estimate {source}", "social 50.0"]
    # Only an infeasible cap is acknowledged; an ok one is left alone.
    assert diagnosis.size_cap is not None
    assert "acknowledge_size_risk" not in diagnosis.size_cap
    assert "acknowledged" not in diagnosis.summary


def test_diagnose_source_acknowledges_only_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = str(tmp_path / "clip.mp4")
    monkeypatch.setattr(
        "super_processor.diagnose.estimate_look",
        lambda *args, **kwargs: _estimates(source),
    )
    monkeypatch.setattr(
        "super_processor.diagnose.plan_social_export",
        lambda *args, **kwargs: _social(source, status="infeasible", max_size_mb=1.0),
    )
    facts = _facts(source, 7200.0)
    acked = diagnose_source(
        Path(source), facts, max_size_mb=1.0, acknowledge_size_risk=True
    )
    assert acked.size_cap is not None
    assert acked.size_cap["acknowledge_size_risk"] == 1.0
    assert "acknowledged" in acked.summary
    unacked = diagnose_source(Path(source), facts, max_size_mb=1.0)
    assert unacked.size_cap is not None
    assert "acknowledge_size_risk" not in unacked.size_cap


# --- planning ---------------------------------------------------------------


def test_recipe_takes_cv_suggestions_and_instruction_patches() -> None:
    estimates = _estimates()
    diagnosis = build_diagnosis(estimates, _social(estimates.source_path))
    recipe = recipe_from_diagnosis(
        diagnosis,
        job_id=JOB_ID,
        source_path=estimates.source_path,
        instruction="less denoise warmer",
    )
    assert [op.op for op in recipe.ops] == list(DEFAULT_OP_ORDER)
    contrast = _op(recipe, OpName.CONTRAST)
    assert contrast.enabled is True and contrast.params["contrast"] == 1.2
    denoise = _op(recipe, OpName.DENOISE)
    assert denoise.enabled is True and denoise.params["strength"] == 0.15
    wb = _op(recipe, OpName.WHITE_BALANCE)
    # "warmer" switches on an op the CV diagnosis left off.
    assert wb.enabled is True and wb.params["temperature"] == 5800.0
    assert _op(recipe, OpName.STABILIZE).enabled is False
    assert _op(recipe, OpName.REFRAME_VERTICAL).enabled is True


def test_social_plan_sets_the_reframe_crop_and_the_size_cap() -> None:
    diagnosis = build_diagnosis(
        _estimates(),
        _social("/tmp/clip.mp4", max_size_mb=40.0, output_height=1280, subject_cx=0.3),
    )
    recipe = recipe_from_diagnosis(
        diagnosis, job_id=JOB_ID, source_path="/tmp/clip.mp4"
    )
    reframe = _op(recipe, OpName.REFRAME_VERTICAL)
    assert reframe.params == {
        "padding": 0.0,
        "crop_x": 420.0,
        "crop_y": 0.0,
        "crop_w": 1080.0,
        "crop_h": 1080.0,
        "subject_cx": 0.3,
    }
    encode = _op(recipe, OpName.ENCODE_SIZE_CAP)
    assert encode.enabled is True
    # The cap encodes at the reframe's output height, without a risk flag.
    assert encode.params == {"max_size_mb": 40.0, "max_height": 1280.0}
    assert recipe.target.export.max_size_mb == 40.0


def test_an_acknowledged_size_risk_reaches_the_encode_op() -> None:
    diagnosis = build_diagnosis(
        _estimates(), _social("/tmp/clip.mp4", status="infeasible", max_size_mb=1.0)
    )
    assert diagnosis.size_cap is not None
    diagnosis.size_cap["acknowledge_size_risk"] = 1.0
    recipe = recipe_from_diagnosis(
        diagnosis, job_id=JOB_ID, source_path="/tmp/clip.mp4"
    )
    assert _op(recipe, OpName.ENCODE_SIZE_CAP).params["acknowledge_size_risk"] == 1.0


def test_without_a_social_plan_reframe_and_size_cap_stay_off() -> None:
    recipe = recipe_from_diagnosis(
        build_diagnosis(_estimates(), None), job_id=JOB_ID, source_path="/tmp/x.mp4"
    )
    assert _op(recipe, OpName.REFRAME_VERTICAL).enabled is False
    assert _op(recipe, OpName.ENCODE_SIZE_CAP).enabled is False


def test_instruction_patches_disable_and_ignore_unknown_words() -> None:
    recipe = empty_recipe(JOB_ID, "/tmp/x.mp4")
    _op(recipe, OpName.DENOISE).enabled = True
    _op(recipe, OpName.DENOISE).params = {"strength": 0.5}
    _op(recipe, OpName.STABILIZE).enabled = True
    patched = apply_instruction_patches(recipe, "No Denoise, no stabilisation")
    assert _op(patched, OpName.DENOISE).enabled is False
    assert _op(patched, OpName.DENOISE).params == {}
    assert _op(patched, OpName.STABILIZE).enabled is False
    before = patched.to_dict()
    assert apply_instruction_patches(patched, "make it cinematic").to_dict() == before
    assert apply_instruction_patches(patched, "   ").to_dict() == before
    cooled = apply_instruction_patches(patched, "cooler, more contrast")
    assert _op(cooled, OpName.WHITE_BALANCE).params["temperature"] == 7200.0
    assert _op(cooled, OpName.CONTRAST).params["contrast"] == 1.35


def test_plan_job_writes_the_recipe_and_needs_a_diagnosis(tmp_path: Path) -> None:
    source = str(tmp_path / "clip.mp4")
    write_media_facts(tmp_path, _facts(source))
    with pytest.raises(PlanError, match="diagnosis required"):
        plan_job(tmp_path, JOB_ID)
    write_diagnosis(tmp_path, build_diagnosis(_estimates(source), None))
    recipe = plan_job(tmp_path, JOB_ID, instruction="more denoise")
    assert recipe.source_path == source
    saved = load_recipe(tmp_path)
    assert saved.to_dict() == recipe.to_dict()
    assert _op(saved, OpName.DENOISE).params["strength"] == 0.55


# --- advisory LLM patches ---------------------------------------------------


def _no_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object) -> str:
        raise AssertionError("safe mode must not contact a model")

    monkeypatch.setattr("super_processor.models._advisory_chat", fail)


def test_safe_mode_applies_only_the_word_patches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_chat(monkeypatch)
    recipe = empty_recipe(JOB_ID, "/tmp/x.mp4")
    patched = apply_llm_plan_patch(recipe, "more denoise", safe_mode=True)
    assert _op(patched, OpName.DENOISE).enabled is True
    assert _op(patched, OpName.DENOISE).params["strength"] == 0.55
    # An empty instruction changes nothing, in either mode.
    assert apply_llm_plan_patch(recipe, "   ", safe_mode=False) is recipe


def test_structured_llm_patch_clamps_and_ignores_unknown() -> None:
    recipe = empty_recipe(JOB_ID, "/tmp/x.mp4")
    payload = parse_structured_patch(
        'noise {"patches":[{"op":"denoise","enabled":true,'
        '"params":{"strength":9.0,"bogus":1}},'
        '{"op":"shell","enabled":true}]}'
    )
    assert payload is not None
    patched = apply_structured_patches(recipe, payload)
    denoise = _op(patched, OpName.DENOISE)
    assert denoise.enabled is True
    assert denoise.params["strength"] == 1.0
    assert "bogus" not in denoise.params
    assert [op.op for op in patched.ops] == list(DEFAULT_OP_ORDER)
    assert parse_structured_patch("") is None
    assert parse_structured_patch("not json") is None


def test_apply_llm_uses_structured_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_chat(endpoint: object, instruction: str, recipe: object) -> str:
        return json.dumps(
            {
                "patches": [
                    {"op": "denoise", "enabled": True, "params": {"strength": 0.22}}
                ]
            }
        )

    monkeypatch.setattr("super_processor.models._advisory_chat", fake_chat)
    recipe = empty_recipe(JOB_ID, "/tmp/x.mp4")
    patched = apply_llm_plan_patch(recipe, "please tweak denoise", safe_mode=False)
    denoise = _op(patched, OpName.DENOISE)
    assert denoise.enabled is True
    assert denoise.params["strength"] == 0.22


def test_a_down_model_falls_back_to_the_word_patches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stub the chat as unreachable, so a model running on this machine is
    # never contacted by the test.
    def down(*_args: object) -> str:
        raise OSError("connection refused")

    monkeypatch.setattr("super_processor.models._advisory_chat", down)
    recipe = empty_recipe(JOB_ID, "/tmp/x.mp4")
    patched = apply_llm_plan_patch(
        recipe, "more contrast", safe_mode=False, endpoint_name="ollama"
    )
    contrast = _op(patched, OpName.CONTRAST)
    assert contrast.enabled is True
    assert contrast.params["contrast"] == 1.35


def test_planner_is_unavailable_when_nothing_listens() -> None:
    endpoint = ModelEndpoint(name="ollama", base_url="http://127.0.0.1:9", model="x")
    assert planner_available(endpoint) is False


def test_list_models_reports_consent_and_endpoints() -> None:
    payload = list_models(remote_frames_consent=True)
    assert payload["remote_frames_consent"] is True
    assert payload["endpoints"]


# --- secrets, QA, progress --------------------------------------------------


def test_resolve_secret_reads_the_environment_then_the_keyring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "SUPER_PROCESSOR_TEST_SECRET"
    monkeypatch.setenv(name, "tok")
    assert resolve_secret(name) == "tok"
    monkeypatch.delenv(name, raising=False)
    assert resolve_secret(name) is None
    fake_keyring = types.SimpleNamespace(
        get_password=lambda service, key: (
            "from-keyring" if key == "OPENAI_API_KEY" else None
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "keyring", fake_keyring)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resolve_secret("OPENAI_API_KEY") == "from-keyring"


def test_qa_reports_a_missing_preview(tmp_path: Path) -> None:
    report = analyze_preview(tmp_path / "missing.mp4")
    assert isinstance(report, QAReport)
    assert report.ok is False
    assert report.issues[0].code == "missing_preview"
    path = write_qa_report(tmp_path, report)
    assert "missing_preview" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("progress", "duration_s", "expected"),
    [
        ({}, None, "encoding… speed=?"),
        (
            {"out_time_ms": "5000000", "speed": "1.2x"},
            10.0,
            "encoding  50.0% (5.0/10.0s) speed=1.2x",
        ),
        ({"out_time": "00:00:02.5", "speed": "1x"}, None, "encoding 2.5s speed=1x"),
        # FFmpeg's out_time_ms is in microseconds, despite its name.
        ({"out_time_ms": "5000"}, None, "encoding 0.0s speed=?"),
        ({"out_time_ms": "2500000"}, None, "encoding 2.5s speed=?"),
        (
            {"out_time_ms": "2500000", "out_time": "00:00:02.500000"},
            10.0,
            "encoding  25.0% (2.5/10.0s) speed=?",
        ),
        (
            {"out_time_ms": "N/A", "out_time": "00:00:02.500000"},
            10.0,
            "encoding  25.0% (2.5/10.0s) speed=?",
        ),
        ({"out_time_ms": "N/A"}, None, "encoding… speed=?"),
        (
            {"out_time_ms": "12000000"},
            10.0,
            "encoding 100.0% (12.0/10.0s) speed=?",
        ),
    ],
)
def test_format_progress_status(
    progress: dict[str, str],
    duration_s: float | None,
    expected: str,
) -> None:
    assert format_progress_status(progress, duration_s=duration_s) == expected


def test_diagnosis_from_dict_fills_defaults() -> None:
    loaded = Diagnosis.from_dict({"source_path": "/tmp/x.mp4"})
    assert loaded.summary == ""
    assert loaded.suggestions == {}
    assert loaded.reframe is None and loaded.size_cap is None
