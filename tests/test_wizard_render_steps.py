"""Drive WizardController.render for each persisted step."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from super_processor.gemini import GeminiError
from super_processor.jobs import JobState
from super_processor.review import WizardServer
from super_processor.shot_plan import ShotPlan, ShotRecipe, save_plan
from super_processor.shots import Shot, ShotList, save_shots
from super_processor.upscale import FakeUpscaleEngine, UpscaleError
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    job_state_path,
    load_job_wizard_state,
    save_job_wizard_state,
    save_session_state,
    session_state_path,
)

TOKEN = "test-token"


def _seed_job(tmp_path: Path) -> tuple[WizardController, str, Path]:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not-a-real-video")
    controller = WizardController(tmp_path)
    manifest = controller.store.create(source)
    return controller, manifest.job_id, controller.store.job_dir(manifest.job_id)


def test_render_each_wizard_step(tmp_path: Path) -> None:
    controller, job_id, job_dir = _seed_job(tmp_path)
    # Each step's own heading, so a step that rendered another's page fails.
    steps = [
        (WizardStep.INTRO, '<p class="brand-hero">Super Processor</p>'),
        (WizardStep.LLM_CHOICE, "<h1>What type of LLM do you have now?</h1>"),
        (WizardStep.LOCAL_LLM_STUB, "<h1>Local LLM setup</h1>"),
        (WizardStep.SETUP, "<h1>Gemini API setup</h1>"),
        (WizardStep.PICK_FILE, "<h1>Pick a video file</h1>"),
        (WizardStep.SAMPLING, "<h1>Finding the segments</h1>"),
        (WizardStep.PROPOSING, "<h1>Finding the segments</h1>"),
        (WizardStep.SEGMENTS, "<h1>Segments</h1>"),
        (WizardStep.ANALYZING, "<title>Segment 1 of 0 · Super Processor</title>"),
        (WizardStep.PREVIEW_SEGMENT, "<title>Segment preview · Super Processor"),
        (WizardStep.READY_TO_RENDER, "<h1>Ready to render</h1>"),
        (WizardStep.RENDERING, "<h1>Upscaling</h1>"),
        (WizardStep.RESULT, "<h1>Result</h1>"),
        (WizardStep.DONE, "<h1>Done</h1>"),
    ]
    assert {step for step, _needle in steps} == set(WizardStep)
    for step, needle in steps:
        state = WizardState(step=step, job_id=job_id)
        save_job_wizard_state(job_dir, state)
        save_session_state(tmp_path, state)
        if step in RUN_STEPS:
            # Reopened there, as by an earlier server: interrupted.
            controller.open_job(job_id)
            assert "<h1>Carry on where you left off?</h1>" in controller.render()
            controller.resume()
        assert needle in controller.render()


RUN_STEPS = {
    WizardStep.SAMPLING,
    WizardStep.PROPOSING,
    WizardStep.ANALYZING,
    WizardStep.RENDERING,
}


def test_retired_steps_go_back_to_pick(tmp_path: Path) -> None:
    controller, job_id, job_dir = _seed_job(tmp_path)
    for retired in ("analyzing", "overview", "choose_split", "enhance"):
        old: dict[str, object] = {
            "schema_version": 1,
            "step": retired,
            "job_id": job_id,
            "layout_index": 0,
            "segment_index": 1,
            "highlights": ["h"],
            "enhance_cache": {"0": {}},
            "error": None,
        }
        session_state_path(tmp_path).write_text(json.dumps(old), encoding="utf-8")
        job_state_path(job_dir).write_text(json.dumps(old), encoding="utf-8")
        state = controller.current_state()
        assert state.step is WizardStep.PICK_FILE
        assert state.job_id is None
        assert state.error is not None and "pick the file again" in state.error
        assert "Pick" in controller.render()
    loaded = load_job_wizard_state(job_dir)
    assert loaded is not None
    assert loaded.to_dict() == {
        "schema_version": 3,
        "step": "pick_file",
        "job_id": None,
        "error": loaded.error,
        "segment_index": 0,
        "progress": None,
        "notice": None,
        "approved_count": 0,
        "stopped": None,
        "render_note": None,
    }


def test_segment_loop_state_round_trips(tmp_path: Path) -> None:
    controller, job_id, job_dir = _seed_job(tmp_path)
    state = WizardState(
        step=WizardStep.ANALYZING,
        job_id=job_id,
        segment_index=2,
        progress={"phase": "analyzing", "pct": 40, "message": "proposing"},
        approved_count=3,
        stopped="Stopped before the render finished",
        render_note="brighter",
    )
    save_job_wizard_state(job_dir, state)
    save_session_state(tmp_path, state)
    assert controller.current_state() == state
    raw = {
        "schema_version": 3,
        "step": "segments",
        "segment_index": -1,
        "progress": "junk",
    }
    loaded = WizardState.from_dict(raw)
    assert loaded.step is WizardStep.SEGMENTS
    assert (loaded.segment_index, loaded.progress) == (0, None)
    assert (loaded.approved_count, loaded.stopped, loaded.render_note) == (
        0,
        None,
        None,
    )


def test_a_state_file_from_before_approved_count_still_loads() -> None:
    """Older files have no approved_count: every earlier segment was approved."""
    old = {
        "schema_version": 3,
        "step": "preview_segment",
        "job_id": "job00001",
        "segment_index": 2,
        "progress": None,
        "notice": None,
    }
    loaded = WizardState.from_dict(old)
    assert loaded.step is WizardStep.PREVIEW_SEGMENT
    assert (loaded.segment_index, loaded.approved_count) == (2, 2)
    assert loaded.stopped is None and loaded.render_note is None
    # A saved count never trails the segment on screen, and junk is ignored.
    behind = WizardState.from_dict({**old, "approved_count": 1})
    assert behind.approved_count == 2
    junk = WizardState.from_dict(
        {**old, "approved_count": True, "stopped": 3, "render_note": ["x"]}
    )
    assert (junk.approved_count, junk.stopped, junk.render_note) == (2, None, None)
    ahead = WizardState.from_dict({**old, "approved_count": 4})
    assert ahead.approved_count == 4
    assert WizardState.from_dict(ahead.to_dict()) == ahead


class _BrokenEngine:
    def upscale(self, request: object) -> Path:
        raise UpscaleError("SPAN needs an NVIDIA GPU with CUDA")


class _NoGemini:
    """Every Gemini call fails, so each loop falls back to the measurements."""

    def propose_segments(self, **_kwargs: object) -> list[dict[str, object]]:
        raise GeminiError("no key in tests")

    def propose_looks(self, **_kwargs: object) -> dict[str, object]:
        raise GeminiError("no key in tests")

    def check_previews(self, **_kwargs: object) -> list[dict[str, object]]:
        raise GeminiError("no key in tests")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_failed_first_upscale_goes_back_to_ready_to_render(tmp_path: Path) -> None:
    clip = tmp_path / "real.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:size=160x90:rate=10:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
    )
    controller = WizardController(
        tmp_path / "jobs",
        gemini=_NoGemini(),  # type: ignore[arg-type]
        upscale_engine=_BrokenEngine(),
    )
    save_session_state(controller.jobs_dir, WizardState(step=WizardStep.PICK_FILE))
    controller.handle_post("/pick", {"path": [str(clip)]})
    job_id = controller.current_state().job_id
    assert job_id is not None
    assert controller.current_state().step is WizardStep.SAMPLING
    controller.run_segmenting()
    assert controller.current_state().step is WizardStep.SEGMENTS
    # Gemini is down, so the whole clip is one segment.
    assert "could not propose segments" in controller.render()
    controller.handle_post("/segments", {"action": ["approve"]})
    assert controller.current_state().step is WizardStep.ANALYZING
    # Without a GPU the preview fails, and the editor is back on the segments.
    controller.run_analyze()
    state = controller.current_state()
    assert state.step is WizardStep.SEGMENTS
    assert state.error is not None and "preview failed" in state.error
    assert "CUDA" in controller.render()
    # With previews working, approve the segment; then the full render fails.
    controller._upscale = FakeUpscaleEngine()
    controller.handle_post("/segments", {"action": ["approve"]})
    controller.run_analyze()
    assert controller.current_state().step is WizardStep.PREVIEW_SEGMENT
    assert "could not propose" in controller.render()
    controller.handle_post("/preview", {"action": ["approve"]})
    controller.handle_post("/ready", {"action": ["render"]})
    assert controller.current_state().step is WizardStep.RENDERING
    controller._upscale = _BrokenEngine()
    controller.run_render()
    # The approved scenes are kept: back to "Ready to render" on the same
    # job, which is not marked failed, so the editor can simply try again.
    state = controller.current_state()
    assert state.step is WizardStep.READY_TO_RENDER
    assert state.job_id == job_id
    assert (state.segment_index, state.approved_count) == (1, 1)
    assert state.error is not None
    assert "no result yet" in state.error and "CUDA" in state.error
    assert controller.store.load(job_id).state is not JobState.FAILED
    assert controller.api_view()["run"]["outcome"] == "failed"
    page = controller.render()
    assert "Ready to render" in page and "CUDA" in page
    # With the GPU back, the same render goes through to the result.
    controller._upscale = FakeUpscaleEngine()
    controller.handle_post("/ready", {"action": ["render"]})
    assert controller.current_state().step is WizardStep.RENDERING
    controller.run_render()
    state = controller.current_state()
    assert (state.step, state.job_id, state.error) == (
        WizardStep.RESULT,
        job_id,
        None,
    )


def _seed_scenes(job_dir: Path) -> None:
    """Two approved scenes, as the segment loop leaves them."""
    save_shots(
        job_dir,
        ShotList(shots=[Shot(0.0, 4.0, label="first"), Shot(4.0, 8.0, label="second")]),
    )
    save_plan(
        job_dir,
        ShotPlan(
            scale=2,
            recipes=[
                ShotRecipe(strength=0.4, rev=1, reason="planned"),
                ShotRecipe(strength=0.3, rev=1, reason="planned"),
            ],
        ),
    )


def _fail_render(controller: WizardController) -> WizardState:
    def _boom() -> None:
        raise UpscaleError("SPAN needs an NVIDIA GPU with CUDA")

    controller._run_render = _boom  # type: ignore[method-assign]
    controller.resume()
    assert controller.start_render() is True
    assert controller.wait_idle(timeout=5)
    return controller.current_state()


def test_a_failed_first_render_keeps_the_approved_scenes(tmp_path: Path) -> None:
    controller, job_id, job_dir = _seed_job(tmp_path)
    _seed_scenes(job_dir)
    controller._save(WizardState(step=WizardStep.RENDERING, job_id=job_id))
    state = _fail_render(controller)
    assert (state.step, state.job_id) == (WizardStep.READY_TO_RENDER, job_id)
    assert (state.segment_index, state.approved_count) == (2, 2)
    assert state.error is not None and "CUDA" in state.error
    assert controller.store.load(job_id).state is not JobState.FAILED
    assert (job_dir / "shots.json").is_file()
    assert (job_dir / "shot_plan.json").is_file()


@pytest.mark.parametrize("damage", ["truncated_shots", "plan_without_recipes"])
def test_a_failed_first_render_with_unreadable_scenes_goes_back_to_pick(
    tmp_path: Path, damage: str
) -> None:
    controller, job_id, job_dir = _seed_job(tmp_path)
    _seed_scenes(job_dir)
    if damage == "truncated_shots":
        text = (job_dir / "shots.json").read_text(encoding="utf-8")
        (job_dir / "shots.json").write_text(text[: len(text) // 2], encoding="utf-8")
    else:
        raw = json.loads((job_dir / "shot_plan.json").read_text(encoding="utf-8"))
        del raw["recipes"]
        (job_dir / "shot_plan.json").write_text(json.dumps(raw), encoding="utf-8")
    controller._save(WizardState(step=WizardStep.RENDERING, job_id=job_id))
    state = _fail_render(controller)
    assert (state.step, state.job_id) == (WizardStep.PICK_FILE, None)
    assert state.error is not None
    assert "no result yet" in state.error and "CUDA" in state.error
    assert controller.store.load(job_id).state is JobState.FAILED
    assert controller.start_render() is False


def test_start_new_job_from_done_and_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, job_id, job_dir = _seed_job(tmp_path)
    (job_dir / "output.mp4").write_bytes(b"result")
    for step in (WizardStep.DONE, WizardStep.RESULT):
        state = WizardState(step=step, job_id=job_id)
        save_job_wizard_state(job_dir, state)
        save_session_state(tmp_path, state)
        assert 'action="/new"' in controller.render()
        monkeypatch.setattr(
            "super_processor.wizard.resolve_gemini_api_key", lambda: "key"
        )
        controller.handle_post("/new", {})
        fresh = controller.current_state()
        assert fresh.step is WizardStep.PICK_FILE
        assert fresh.job_id is None
        # The finished job is left exactly as it was, to reopen later.
        assert (job_dir / "output.mp4").read_bytes() == b"result"
        kept = load_job_wizard_state(job_dir)
        assert kept is not None and (kept.step, kept.job_id) == (step, job_id)
    monkeypatch.setattr("super_processor.wizard.resolve_gemini_api_key", lambda: None)
    controller.handle_post("/new", {})
    assert controller.current_state().step is WizardStep.SETUP


def test_start_new_job_over_the_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("super_processor.wizard.resolve_gemini_api_key", lambda: "k")
    controller, job_id, _job_dir = _seed_job(tmp_path)
    save_session_state(tmp_path, WizardState(step=WizardStep.DONE, job_id=job_id))
    server = WizardServer(tmp_path, token=TOKEN, controller=controller)
    base = server.start()
    try:
        request = Request(
            base + "/api/new",
            data=b"{}",
            headers={"Content-Type": "application/json", "X-Wizard-Token": TOKEN},
            method="POST",
        )
        with urlopen(request) as response:
            view = json.loads(response.read().decode())
        assert (view["step"], view["job_id"]) == ("pick_file", None)
        assert controller.current_state().step is WizardStep.PICK_FILE
    finally:
        server.stop()
