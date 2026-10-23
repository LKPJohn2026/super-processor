"""Segment loop through the wizard controller, with fake Gemini and GPU."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from super_processor.shot_plan import load_plan
from super_processor.upscale import FakeUpscaleEngine, UpscaleRequest
from super_processor.wizard import (
    SEGMENT_LOOP_ENV,
    WizardController,
    WizardError,
    WizardState,
    WizardStep,
    save_job_wizard_state,
    save_session_state,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg required"
)


def _clip(path: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=24:duration=8",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


class _Gemini:
    def __init__(self) -> None:
        self.notes: list[str] = []
        self.looks_calls: list[list[int]] = []

    def propose_segments(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.notes.append(kwargs["note"])
        assert len(kwargs["stills"]) == len(kwargs["times"]) > 2
        return [
            {"start_s": 0.0, "end_s": 4.0, "label": "first"},
            {"start_s": 3.5, "end_s": 8.0, "label": "second"},
        ]

    def propose_looks(self, **kwargs: Any) -> dict[str, Any]:
        self.looks_calls.append([shot["index"] for shot in kwargs["shots"]])
        return {
            "scale": 2,
            "shots": [
                {"index": shot["index"], "strength": 0.4, "look": {}, "reason": "ok"}
                for shot in kwargs["shots"]
            ],
        }

    def check_previews(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"index": shot["index"], "ok": True} for shot in kwargs["shots"]]


class _CountingEngine(FakeUpscaleEngine):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def upscale(self, request: UpscaleRequest) -> Path:
        self.calls += 1
        return super().upscale(request)


def _run(controller: WizardController, start: str) -> WizardState:
    assert getattr(controller, start)()
    assert controller.wait_idle(120)
    return controller.current_state()


@needs_ffmpeg
def test_segment_loop_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SEGMENT_LOOP_ENV, "1")
    clip = _clip(tmp_path / "in.mp4")
    gemini = _Gemini()
    engine = _CountingEngine()
    controller = WizardController(
        tmp_path / "jobs",
        gemini=gemini,  # type: ignore[arg-type]
        upscale_engine=engine,
    )
    save_session_state(controller.jobs_dir, WizardState(step=WizardStep.PICK_FILE))
    controller.handle_post("/pick", {"path": [str(clip)]})
    assert controller.current_state().step is WizardStep.SAMPLING

    state = _run(controller, "start_segmenting")
    assert state.step is WizardStep.SEGMENTS
    view = controller.api_view()["segments"]
    assert [(s["start_s"], s["label"]) for s in view["segments"]] == [
        (0.0, "first"),
        (3.5, "second"),
    ]

    controller.handle_post("/segments", {"action": ["note"], "note": [" "]})
    assert controller.current_state().error is not None
    controller.handle_post(
        "/segments", {"action": ["note"], "note": ["cut  at the zoom"]}
    )
    assert controller.current_state().step is WizardStep.PROPOSING
    assert _run(controller, "start_segmenting").step is WizardStep.SEGMENTS
    assert gemini.notes == ["", "cut at the zoom"]

    controller.handle_post("/segments", {"action": ["approve"]})
    state = _run(controller, "start_analyze")
    assert state.step is WizardStep.PREVIEW_SEGMENT
    assert gemini.looks_calls == [[0]]
    preview = controller.api_view()["segment_preview"]
    assert (preview["index"], preview["count"]) == (0, 2)
    assert preview["shot"]["after_url"].endswith("_r1_after.mp4")
    assert state.job_id
    job_dir = controller.store.job_dir(state.job_id)
    plan = load_plan(job_dir)
    assert plan is not None
    window = plan.recipes[0].preview_start
    assert window is not None and 0.0 <= window <= 1.0 + 1e-6

    controller.handle_post("/preview", {"action": ["redo"], "note": ["softer"]})
    assert _run(controller, "start_analyze").step is WizardStep.PREVIEW_SEGMENT
    plan = load_plan(job_dir)
    assert plan is not None
    assert plan.recipes[0].rev == 2
    assert plan.recipes[0].preview_start == window
    assert gemini.looks_calls == [[0], [0]]

    controller.handle_post("/preview", {"action": ["approve"]})
    assert controller.current_state().segment_index == 1
    assert _run(controller, "start_analyze").step is WizardStep.PREVIEW_SEGMENT
    assert gemini.looks_calls == [[0], [0], [1]]
    controller.handle_post("/preview", {"action": ["approve"]})
    assert controller.current_state().step is WizardStep.READY_TO_RENDER
    assert len(controller.api_view()["looks"]["shots"]) == 2

    controller.handle_post("/ready", {"action": ["wait"]})
    assert controller.current_state().step is WizardStep.READY_TO_RENDER
    controller.handle_post("/ready", {"action": ["render"]})
    previews = engine.calls
    assert _run(controller, "start_render").step is WizardStep.RESULT
    assert engine.calls > previews
    assert (job_dir / "output.mp4").is_file()


def test_segment_loop_posts_refuse_other_steps(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    save_session_state(controller.jobs_dir, WizardState(step=WizardStep.PICK_FILE))
    for path in ("/segments", "/preview", "/ready"):
        with pytest.raises(WizardError):
            controller.handle_post(path, {"action": ["approve"]})


def test_analyze_without_approved_segments_goes_back_to_pick(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    controller = WizardController(tmp_path / "jobs")
    job_id = controller.store.create(source).job_id
    state = WizardState(step=WizardStep.ANALYZING, job_id=job_id)
    save_job_wizard_state(controller.store.job_dir(job_id), state)
    save_session_state(controller.jobs_dir, state)
    controller.run_analyze()
    state = controller.current_state()
    assert state.step is WizardStep.PICK_FILE
    assert state.error is not None and "approved segments" in state.error


def test_pick_uses_shot_flow_without_the_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(SEGMENT_LOOP_ENV, raising=False)
    controller = WizardController(tmp_path / "jobs")
    assert controller.start_segmenting() is False
    assert controller.start_analyze() is False
