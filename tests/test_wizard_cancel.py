"""Stopping a wizard run: segmenting and the render, with fake Gemini and FFmpeg.

No test here needs a GPU, FFmpeg, the network or a Gemini key: probing is
faked, and a render's FFmpeg is a Python child that sleeps.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from super_processor.gemini import GeminiVideo
from super_processor.jobs import JobState
from super_processor.segment_loop import (
    LoopSegment,
    SegmentList,
    SegmentStill,
    load_segment_list,
    save_segment_list,
)
from super_processor.upscale import (
    UPSCALE_PLAN_FILE,
    FakeUpscaleEngine,
    UpscalePlan,
    _commit,
    _run_ffmpeg,
    default_span,
    save_upscale_plan,
)
from super_processor.wizard import (
    CANCEL_MARKER,
    GEMINI_VIDEO_FILE,
    WORKER_PID_FILE,
    WizardConflict,
    WizardController,
    WizardError,
    WizardState,
    WizardStep,
    load_job_wizard_state,
    save_job_wizard_state,
    save_session_state,
)

_PLAN_BEFORE = "upscale_plan_before_note.json"


class _Facts:
    duration_s = 8.0

    def primary_video(self) -> None:
        return None


class _Gemini:
    """Proposes two segments; ``block`` holds the answer until released."""

    def __init__(self, *, block: bool = False) -> None:
        self.block = block
        self.entered = threading.Event()
        self.release = threading.Event()
        self.answered = threading.Event()

    def propose_segments(self, **kwargs: Any) -> dict[str, Any]:
        self.entered.set()
        if self.block:
            self.release.wait(10)
        self.answered.set()
        return {
            "segments": [
                {"start_s": 0.0, "end_s": 4.0, "label": "new first"},
                {"start_s": 4.0, "end_s": 8.0, "label": "new second"},
            ]
        }


class _UploadingGemini(_Gemini):
    """Its upload hangs until released, then reports progress late."""

    def __init__(self) -> None:
        super().__init__()
        self.uploading = threading.Event()
        self.late: list[BaseException] = []
        self.finished = threading.Event()

    def upload_video(self, path: Path, *, on_progress: Any = None) -> GeminiVideo:
        self.uploading.set()
        self.release.wait(10)
        try:
            on_progress(1.0)
        except BaseException as exc:
            self.late.append(exc)
            raise
        finally:
            self.finished.set()
        raise AssertionError("a stopped upload must not get this far")


@pytest.fixture(autouse=True)
def _no_ffprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("super_processor.wizard.probe_file", lambda _path: _Facts())
    monkeypatch.setattr(
        "super_processor.wizard.load_segment_stills",
        lambda _job_dir: [
            SegmentStill(t_s=t, path=f"segment_stills/t_{int(t * 1000):08d}.jpg")
            for t in (1.0, 3.0, 5.0)
        ],
    )


def _job(
    tmp_path: Path, step: WizardStep, gemini: Any = None
) -> tuple[WizardController, str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not really a video")
    controller = WizardController(
        tmp_path / "jobs", gemini=gemini, upscale_engine=FakeUpscaleEngine()
    )
    job_id = controller.store.create(source).job_id
    controller.store.transition(job_id, JobState.PROBED)
    job_dir = controller.store.job_dir(job_id)
    state = WizardState(step=step, job_id=job_id)
    save_job_wizard_state(job_dir, state)
    save_session_state(controller.jobs_dir, state)
    return controller, job_id, job_dir


def _wait(condition: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def _slow_render(started: threading.Event) -> Any:
    """A render whose FFmpeg runs for half a minute unless it is stopped."""

    def render(*_args: Any, **_kwargs: Any) -> None:
        started.set()
        _run_ffmpeg([sys.executable, "-c", "import time; time.sleep(30)"])

    return render


# Segmenting -------------------------------------------------------------


def test_a_stop_while_gemini_proposes_goes_back_to_pick(tmp_path: Path) -> None:
    gemini = _Gemini(block=True)
    controller, job_id, job_dir = _job(tmp_path, WizardStep.PROPOSING, gemini)
    controller.resume()
    assert controller.start_segmenting()
    assert gemini.entered.wait(5)
    view = controller.api_view()
    assert view["cancellable"] is True and view["cancel_requested"] is False

    began = time.monotonic()
    controller.handle_post("/cancel", {"job_id": [job_id]})
    # Gemini is still thinking; the run does not wait for it.
    assert controller.wait_idle(5)
    assert time.monotonic() - began < 2
    assert not gemini.answered.is_set()

    state = controller.current_state()
    assert (state.step, state.job_id, state.error) == (WizardStep.PICK_FILE, None, None)
    assert state.stopped is not None and "stopped" in state.stopped
    kept = load_job_wizard_state(job_dir)
    assert kept is not None
    assert (kept.step, kept.job_id) == (WizardStep.PICK_FILE, job_id)
    manifest = controller.store.load(job_id)
    assert manifest.state is JobState.PROBED
    assert manifest.notes.get("stopped")
    assert not (job_dir / CANCEL_MARKER).exists()
    view = controller.api_view()
    assert view["run"]["outcome"] == "stopped"
    assert view["stopped"] == state.stopped
    assert view["cancellable"] is False
    assert "stopped finding the segments" in view["stopped"]

    # The answer that comes in after the stop is dropped.
    gemini.release.set()
    assert gemini.answered.wait(5)
    time.sleep(0.3)
    assert load_segment_list(job_dir) is None
    assert controller.current_state().step is WizardStep.PICK_FILE


def test_a_stopped_upload_never_saves_after_the_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gemini = _UploadingGemini()
    controller, job_id, job_dir = _job(tmp_path, WizardStep.PROPOSING, gemini)
    copy = job_dir / "gemini_upload.mp4"
    part = job_dir / "gemini_upload.part.mp4"

    def fake_copy(*_args: Any, **_kwargs: Any) -> Path:
        copy.write_bytes(b"copy")
        part.write_bytes(b"half")
        return copy

    monkeypatch.setattr("super_processor.wizard.make_gemini_copy", fake_copy)
    controller.resume()
    assert controller.start_segmenting()
    assert gemini.uploading.wait(5)
    controller.cancel()
    assert controller.wait_idle(5)
    landed = controller.current_state()
    assert landed.step is WizardStep.PICK_FILE

    # The upload left behind reports progress late: it must not save.
    gemini.release.set()
    assert gemini.finished.wait(5)
    assert gemini.late and type(gemini.late[0]).__name__ == "Cancelled"
    assert controller.current_state() == landed
    kept = load_job_wizard_state(job_dir)
    assert kept is not None and kept.step is WizardStep.PICK_FILE
    assert kept.progress is None
    # The half-made copy goes; the finished one, maybe still open, stays.
    assert not part.exists() and copy.exists()
    # A stop is not an upload failure: the next try uploads again.
    assert not (job_dir / GEMINI_VIDEO_FILE).exists()
    assert not gemini.entered.is_set()


def test_a_stopped_re_propose_keeps_the_segments(tmp_path: Path) -> None:
    gemini = _Gemini(block=True)
    controller, job_id, job_dir = _job(tmp_path, WizardStep.SEGMENTS, gemini)
    earlier = [
        LoopSegment(0.0, 5.0, label="old first"),
        LoopSegment(5.0, 8.0, label="old second"),
    ]
    save_segment_list(job_dir, SegmentList(segments=earlier))
    (job_dir / CANCEL_MARKER).touch()  # left from an earlier stop
    controller.handle_post(
        "/segments", {"action": ["note"], "note": ["split at the door"]}
    )
    assert controller.current_state().step is WizardStep.PROPOSING
    # Sending the note dropped the old stop, so the new run is not stopped.
    assert not (job_dir / CANCEL_MARKER).exists()
    controller.resume()
    assert controller.start_segmenting()
    assert gemini.entered.wait(5)
    controller.cancel(job_id=job_id)
    assert controller.wait_idle(5)
    state = controller.current_state()
    assert (state.step, state.job_id) == (WizardStep.SEGMENTS, job_id)
    assert state.stopped is not None and "segments from before" in state.stopped
    found = load_segment_list(job_dir)
    assert found is not None
    assert [s.label for s in found.segments] == ["old first", "old second"]
    assert found.note == "split at the door"
    assert "segments from before" in controller.api_view()["stopped"]
    # The next action clears the message.
    controller.handle_post("/segments", {"action": ["bogus"]})
    assert controller.current_state().stopped is None


def test_a_stop_racing_the_segment_save_is_refused_and_leaves_no_marker(
    tmp_path: Path,
) -> None:
    gemini = _Gemini()
    controller, job_id, job_dir = _job(tmp_path, WizardStep.PROPOSING, gemini)
    refused: list[str] = []
    real_save = controller._save

    def hooked(state: WizardState, *, arm: bool = True) -> None:
        if state.step is WizardStep.SEGMENTS and not refused:
            # The segments are committed; a stop now must not undo them.
            with pytest.raises(WizardError, match="can no longer be stopped") as info:
                controller.cancel()
            refused.append(str(info.value))
        real_save(state, arm=arm)

    controller._save = hooked  # type: ignore[method-assign]
    controller.resume()
    assert controller.start_segmenting()
    assert controller.wait_idle(5)
    assert refused
    state = controller.current_state()
    assert state.step is WizardStep.SEGMENTS and state.stopped is None
    assert not (job_dir / CANCEL_MARKER).exists()
    assert controller.api_view()["run"]["outcome"] == "done"


def test_stopping_an_interrupted_analysis_keeps_the_scenes_approved(
    tmp_path: Path,
) -> None:
    controller, job_id, job_dir = _job(tmp_path, WizardStep.ANALYZING)
    segments = [LoopSegment(0.0, 4.0, label="a"), LoopSegment(4.0, 8.0, label="b")]
    save_segment_list(job_dir, SegmentList(segments=segments, approved=True))
    state = controller.current_state()
    state.segment_index = state.approved_count = 1
    save_job_wizard_state(job_dir, state)
    save_session_state(controller.jobs_dir, state)
    controller = WizardController(tmp_path / "jobs")
    controller.handle_post("/cancel", {"job_id": [job_id]})
    state = controller.current_state()
    assert state.step is WizardStep.SEGMENTS
    assert state.stopped is not None and "approve them" in state.stopped
    assert controller.api_view()["run"]["outcome"] == "stopped"
    # Approving the same segments again carries on from the scene it was on.
    controller.handle_post("/segments", {"action": ["approve"]})
    state = controller.current_state()
    assert (state.step, state.segment_index) == (WizardStep.ANALYZING, 1)
    assert state.approved_count == 1 and state.stopped is None


# The render -------------------------------------------------------------


def test_a_stop_during_the_first_render_goes_back_to_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    monkeypatch.setattr("super_processor.wizard.render_plan", _slow_render(started))
    controller, job_id, job_dir = _job(tmp_path, WizardStep.RENDERING)
    (job_dir / "shot_work").mkdir()
    (job_dir / "shot_work" / "half.mp4").write_bytes(b"x")
    monkeypatch.setattr(controller, "_approved_scene_count", lambda _job_dir: 3)
    controller.resume()
    assert controller.start_render()
    assert started.wait(5)
    began = time.monotonic()
    controller.handle_post("/cancel", {})
    assert controller.wait_idle(5)
    assert time.monotonic() - began < 2

    state = controller.current_state()
    assert (state.step, state.job_id) == (WizardStep.READY_TO_RENDER, job_id)
    assert state.error is None and state.progress is None
    assert state.stopped is not None and "Press Render" in state.stopped
    assert state.segment_index == state.approved_count == 3
    assert controller.store.load(job_id).state is JobState.PROBED
    assert not (job_dir / "shot_work").exists()
    assert not (job_dir / CANCEL_MARKER).exists()
    run = controller.api_view()["run"]
    assert (run["kind"], run["outcome"]) == ("render", "stopped")


def test_a_stopped_revise_keeps_the_old_output_and_its_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    monkeypatch.setattr("super_processor.wizard.render_plan", _slow_render(started))
    controller, _job_id, job_dir = _job(tmp_path, WizardStep.RENDERING)
    old = UpscalePlan(spans=(default_span(8.0),))
    (job_dir / "output.mp4").write_bytes(b"the old result")
    (job_dir / _PLAN_BEFORE).write_text(json.dumps(old.to_dict()), encoding="utf-8")
    revised = replace(old, pending=replace(old.spans[0], strength=0.9))
    save_upscale_plan(job_dir, revised)
    controller.resume()
    assert controller.start_render()
    assert started.wait(5)
    controller.cancel()
    assert controller.wait_idle(5)

    state = controller.current_state()
    assert state.step is WizardStep.RESULT
    assert state.stopped is not None and "the one you had before" in state.stopped
    assert (job_dir / "output.mp4").read_bytes() == b"the old result"
    assert json.loads((job_dir / UPSCALE_PLAN_FILE).read_text()) == old.to_dict()
    assert not (job_dir / _PLAN_BEFORE).exists()


def test_a_stop_after_the_render_commits_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    committed = threading.Event()
    release = threading.Event()

    def render(_source: Path, output: Path, *_args: Any, **_kwargs: Any) -> None:
        _commit()
        committed.set()
        release.wait(5)
        output.write_bytes(b"new")

    monkeypatch.setattr("super_processor.wizard.render_plan", render)
    monkeypatch.setattr(
        "super_processor.wizard.video_stream_facts", lambda *a, **k: (8, "24")
    )
    monkeypatch.setattr("super_processor.wizard.seam_candidates", lambda *a, **k: [])
    controller, _job_id, job_dir = _job(tmp_path, WizardStep.RENDERING)
    monkeypatch.setattr(controller, "_check_seams", lambda *_args: None)
    controller.resume()
    assert controller.start_render()
    assert committed.wait(5)
    assert controller.api_view()["cancellable"] is False
    with pytest.raises(WizardError, match="render is finishing"):
        controller.cancel()
    assert not (job_dir / CANCEL_MARKER).exists()
    release.set()
    assert controller.wait_idle(5)
    state = controller.current_state()
    assert state.step is WizardStep.RESULT and state.stopped is None
    assert controller.api_view()["run"]["outcome"] == "done"


def test_a_stop_as_the_worker_finishes_is_refused_and_leaves_no_marker(
    tmp_path: Path,
) -> None:
    controller, job_id, job_dir = _job(tmp_path, WizardStep.RENDERING)

    def finish() -> None:
        controller._save(WizardState(step=WizardStep.RESULT, job_id=job_id))

    controller._run_render = finish  # type: ignore[method-assign]
    outcome: list[str] = []
    real_clear = controller._clear_progress
    real_state = controller._state

    def racing_clear() -> None:
        # The run has ended but its thread still tidies up. The stop read
        # the step just before the result was saved.
        stale = [WizardState(step=WizardStep.RENDERING, job_id=job_id)]
        controller._state = lambda: stale.pop() if stale else real_state()  # type: ignore[method-assign]
        try:
            controller.cancel()
        except WizardError as exc:
            outcome.append(str(exc))
        finally:
            controller._state = real_state  # type: ignore[method-assign]
        real_clear()

    controller._clear_progress = racing_clear  # type: ignore[method-assign]
    controller.resume()
    assert controller.start_render()
    assert controller.wait_idle(5)
    assert outcome == ["it has already finished"]
    assert not (job_dir / CANCEL_MARKER).exists()
    assert controller.current_state().step is WizardStep.RESULT


def test_a_stop_asked_before_a_restart_waits_for_the_editor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("an interrupted render must not start by itself")

    monkeypatch.setattr("super_processor.wizard.render_plan", never)
    controller, job_id, job_dir = _job(tmp_path, WizardStep.RENDERING)
    (job_dir / CANCEL_MARKER).touch()
    # A new server on the same folder does not start the interrupted render.
    assert controller.api_view()["interrupted"] is True
    assert controller.start_render() is False
    assert controller.current_state().step is WizardStep.RENDERING
    # Stopping it lands the stop that was asked for.
    controller.handle_post("/cancel", {"job_id": [job_id]})
    state = controller.current_state()
    assert (state.step, state.job_id) == (WizardStep.READY_TO_RENDER, job_id)
    assert state.stopped is not None
    assert not (job_dir / CANCEL_MARKER).exists()
    assert controller.api_view()["run"]["outcome"] == "stopped"


def test_resuming_drops_a_stop_asked_before_the_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rendered = threading.Event()

    def render(*_args: Any, **_kwargs: Any) -> None:
        rendered.set()
        raise RuntimeError("no GPU here")

    monkeypatch.setattr("super_processor.wizard.render_plan", render)
    controller, job_id, job_dir = _job(tmp_path, WizardStep.RENDERING)
    (job_dir / CANCEL_MARKER).touch()
    controller.handle_post("/resume", {"job_id": [job_id]})
    assert not (job_dir / CANCEL_MARKER).exists()
    assert controller.api_view()["interrupted"] is False
    assert controller.start_render() is True
    assert controller.wait_idle(5)
    # The editor chose to carry on, so the render ran instead of stopping.
    assert rendered.is_set()
    assert controller.api_view()["run"]["outcome"] == "failed"


def test_an_interrupted_run_can_be_stopped_without_a_worker(tmp_path: Path) -> None:
    controller, job_id, job_dir = _job(tmp_path, WizardStep.RENDERING)
    assert controller.api_view()["cancellable"] is True
    controller.handle_post("/cancel", {"job_id": [job_id]})
    state = controller.current_state()
    assert state.step is WizardStep.READY_TO_RENDER and state.stopped is not None
    run = controller.api_view()["run"]
    assert (run["kind"], run["job_id"], run["outcome"]) == ("render", job_id, "stopped")
    assert not (job_dir / CANCEL_MARKER).exists()
    with pytest.raises(WizardError, match="nothing running to stop"):
        controller.cancel()


def test_a_double_stop_is_one_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    monkeypatch.setattr("super_processor.wizard.render_plan", _slow_render(started))
    controller, _job_id, job_dir = _job(tmp_path, WizardStep.RENDERING)
    controller.resume()
    assert controller.start_render()
    assert started.wait(5)
    # Holding the state lock keeps the stop from landing between the clicks.
    with controller._state_lock:
        controller.cancel()
        controller.cancel()
        view = controller.api_view()
        assert view["cancel_requested"] is True and view["cancellable"] is False
        assert (job_dir / CANCEL_MARKER).exists()
    assert controller.wait_idle(5)
    assert controller.current_state().step is WizardStep.READY_TO_RENDER
    assert controller.api_view()["cancel_requested"] is False
    with pytest.raises(WizardError, match="nothing running to stop"):
        controller.cancel()


def test_a_stop_names_the_run_and_job_it_is_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    monkeypatch.setattr("super_processor.wizard.render_plan", _slow_render(started))
    controller, job_id, _job_dir = _job(tmp_path, WizardStep.RENDERING)
    controller.resume()
    assert controller.start_render()
    assert started.wait(5)
    run_id = controller.api_view()["run"]["id"]
    with pytest.raises(WizardConflict, match="already ended"):
        controller.handle_post("/cancel", {"run_id": ["an-older-run"]})
    with pytest.raises(WizardConflict, match="no longer open"):
        controller.handle_post("/cancel", {"job_id": ["job-elsewhere"]})
    assert controller.api_view()["cancel_requested"] is False
    controller.handle_post("/cancel", {"job_id": [job_id], "run_id": [run_id]})
    assert controller.wait_idle(5)
    assert controller.current_state().step is WizardStep.READY_TO_RENDER


def test_only_running_steps_can_be_stopped(tmp_path: Path) -> None:
    for step in (WizardStep.SEGMENTS, WizardStep.RESULT):
        controller, _job_id, job_dir = _job(tmp_path / step.value, step)
        assert controller.api_view()["cancellable"] is False
        with pytest.raises(WizardError, match="nothing running to stop"):
            controller.handle_post("/cancel", {})
        assert not (job_dir / CANCEL_MARKER).exists()
        assert controller.current_state().step is step


# Server shutdown --------------------------------------------------------


def test_a_server_shutdown_mid_render_ends_its_ffmpeg_and_keeps_the_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    monkeypatch.setattr("super_processor.wizard.render_plan", _slow_render(started))
    controller, job_id, job_dir = _job(tmp_path, WizardStep.RENDERING)
    controller.resume()
    assert controller.start_render()
    assert started.wait(5)
    began = time.monotonic()
    controller.shutdown(timeout=5)
    # The render's FFmpeg was stopped, not left running for half a minute.
    assert not controller.busy()
    assert time.monotonic() - began < 3

    # No stop was landed: the job keeps its step and asks to resume (Q1).
    state = controller.current_state()
    assert (state.step, state.job_id) == (WizardStep.RENDERING, job_id)
    assert state.stopped is None
    assert not (job_dir / CANCEL_MARKER).exists()
    assert not (job_dir / WORKER_PID_FILE).exists()
    again = WizardController(controller.jobs_dir, upscale_engine=FakeUpscaleEngine())
    assert again.api_view()["interrupted"] is True
    # And nothing starts behind the stopping server.
    controller.prefetch = True
    assert controller.start_prefetch() is False


def test_a_shutdown_leaves_a_stop_the_editor_asked_for_to_land(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    monkeypatch.setattr("super_processor.wizard.render_plan", _slow_render(started))
    controller, _job_id, _job_dir = _job(tmp_path, WizardStep.RENDERING)
    monkeypatch.setattr(controller, "_approved_scene_count", lambda _job_dir: 3)
    controller.resume()
    assert controller.start_render()
    assert started.wait(5)
    controller.handle_post("/cancel", {})
    controller.shutdown(timeout=5)
    assert not controller.busy()
    assert controller.current_state().step is WizardStep.READY_TO_RENDER


def test_an_editor_stop_after_a_shutdown_began_is_still_landed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    monkeypatch.setattr("super_processor.wizard.render_plan", _slow_render(started))
    controller, _job_id, _job_dir = _job(tmp_path, WizardStep.RENDERING)
    monkeypatch.setattr(controller, "_approved_scene_count", lambda _job_dir: 3)
    controller.resume()
    assert controller.start_render()
    assert started.wait(5)
    run = controller._run
    assert run is not None
    # A shutdown marked the run abandoned, but its stop has not landed yet
    # when the editor's own stop arrives.
    run.abandoned = True
    controller.handle_post("/cancel", {})
    assert controller.wait_idle(5)
    assert controller.current_state().step is WizardStep.READY_TO_RENDER
