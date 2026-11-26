"""Edge paths of the wizard's steps: refusals, bad input, and failed workers.

None of these need FFmpeg: the jobs are seeded on disk with the files a step
reads, so each test reaches its branch directly.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from super_processor.jobs import JobState
from super_processor.segment_loop import (
    LoopSegment,
    SegmentList,
    load_segment_list,
    save_segment_list,
)
from super_processor.shot_plan import (
    finish,
    load_plan,
    new_plan,
    plan_path,
    save_plan,
)
from super_processor.shots import Shot, ShotList, save_shots, shots_path
from super_processor.upscale import load_upscale_plan
from super_processor.wizard import (
    WizardController,
    WizardError,
    WizardState,
    WizardStep,
    job_state_path,
    load_job_wizard_state,
    save_job_wizard_state,
    save_session_state,
    session_state_path,
)


def _job(tmp_path: Path, step: WizardStep, **state: object) -> WizardController:
    """A controller whose session and job both sit on ``step``."""
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not-a-real-video")
    controller = WizardController(tmp_path / "jobs")
    job_id = controller.store.create(source).job_id
    saved = WizardState(step=step, job_id=job_id, **state)  # type: ignore[arg-type]
    save_job_wizard_state(controller.store.job_dir(job_id), saved)
    save_session_state(controller.jobs_dir, saved)
    return controller


def _job_dir(controller: WizardController) -> Path:
    job_id = controller.current_state().job_id
    assert job_id
    return controller.store.job_dir(job_id)


def _two_segments(job_dir: Path, *, approved: bool) -> None:
    save_segment_list(
        job_dir,
        SegmentList(
            segments=[LoopSegment(0.0, 3.0, "a"), LoopSegment(3.0, 6.0, "b")],
            approved=approved,
        ),
    )


def _planned(job_dir: Path, scale: int = 2, *, previewed: bool = False) -> None:
    shots = ShotList([Shot(0.0, 3.0), Shot(3.0, 6.0)])
    save_shots(job_dir, shots)
    plan = new_plan(shots, scale)
    if previewed:
        # Every scene previewed as it is now, as on "Ready to render".
        plan = finish(plan, list(range(len(plan.recipes))))
    save_plan(job_dir, plan)


# The pick ------------------------------------------------------------------


def test_picking_a_missing_file_stays_on_the_pick(tmp_path: Path) -> None:
    controller = WizardController(tmp_path)
    save_session_state(tmp_path, WizardState(step=WizardStep.PICK_FILE))
    controller.handle_post("/pick", {"path": [str(tmp_path / "missing.mp4")]})
    state = controller.current_state()
    assert state.step is WizardStep.PICK_FILE and state.job_id is None
    assert state.error == f"file not found: {tmp_path / 'missing.mp4'}"
    assert controller.store.list_jobs() == []


@pytest.mark.parametrize(
    ("name", "kept"),
    [
        ("My Trailer (final).MP4", "My_Trailer__final_.MP4"),
        # A name is only a name: no folder in it may move the file.
        ("../../outside/evil.mp4", "evil.mp4"),
        ("C:\\Users\\me\\clip.mov", "clip.mov"),
    ],
)
def test_an_upload_is_kept_under_uploads_with_a_safe_name(
    tmp_path: Path, name: str, kept: str
) -> None:
    controller = WizardController(tmp_path)
    target = controller.upload_target(name)
    assert target.parent == controller.jobs_dir / "uploads"
    assert target.name.endswith("-" + kept)


@pytest.mark.parametrize("name", ["notes.txt", "", "clip.mp4.exe", "mp4"])
def test_an_upload_that_is_not_a_video_is_refused(tmp_path: Path, name: str) -> None:
    controller = WizardController(tmp_path)
    with pytest.raises(WizardError, match="only video files"):
        controller.upload_target(name)


# State files ---------------------------------------------------------------


def test_a_state_file_that_is_not_an_object_is_reported(tmp_path: Path) -> None:
    session_state_path(tmp_path).write_text("[1]\n", encoding="utf-8")
    with pytest.raises(WizardError, match="corrupt wizard session"):
        WizardController(tmp_path).current_state()


def test_a_state_file_that_is_not_json_is_reported(tmp_path: Path) -> None:
    session_state_path(tmp_path).write_text('{"step": "res', encoding="utf-8")
    with pytest.raises(WizardError, match="corrupt"):
        WizardController(tmp_path).current_state()


def test_a_state_file_that_is_not_utf8_is_reported(tmp_path: Path) -> None:
    session_state_path(tmp_path).write_bytes(b'{"step": "\xff\xfe"}\n')
    with pytest.raises(WizardError, match="corrupt wizard session"):
        WizardController(tmp_path).current_state()


@pytest.mark.parametrize("step", ['"from-a-newer-build"', "5"])
def test_a_state_file_with_an_unknown_step_is_reported(
    tmp_path: Path, step: str
) -> None:
    session_state_path(tmp_path).write_text(f'{{"step": {step}}}', encoding="utf-8")
    with pytest.raises(WizardError, match="corrupt wizard session"):
        WizardController(tmp_path).current_state()


def test_a_state_file_nested_too_deep_is_reported(tmp_path: Path) -> None:
    session_state_path(tmp_path).write_text("[" * 100_000, encoding="utf-8")
    with pytest.raises(WizardError, match="corrupt wizard session"):
        WizardController(tmp_path).current_state()


def test_a_job_state_file_that_is_not_json_is_reported(tmp_path: Path) -> None:
    job_state_path(tmp_path).write_text("not json\n", encoding="utf-8")
    with pytest.raises(WizardError, match="corrupt wizard_state.json"):
        load_job_wizard_state(tmp_path)


def test_the_job_state_wins_over_the_session(tmp_path: Path) -> None:
    # Workers save the job's own file; the session can lag behind it.
    controller = _job(tmp_path, WizardStep.RENDERING)
    job_dir = _job_dir(controller)
    job_id = controller.current_state().job_id
    save_job_wizard_state(job_dir, WizardState(step=WizardStep.RESULT, job_id=job_id))
    assert controller.current_state().step is WizardStep.RESULT


def test_the_file_step_needs_a_gemini_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Picking a file starts Gemini work, so it is only offered with a key.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    controller = _job(tmp_path, WizardStep.PREVIEW_SEGMENT)
    _two_segments(_job_dir(controller), approved=True)
    assert controller.navigation() == {"setup": True, "file": False, "segments": True}
    controller.handle_post("/goto", {"to": ["file"]})
    state = controller.current_state()
    assert state.step is WizardStep.PREVIEW_SEGMENT
    assert state.error == "you cannot go to file from here"
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert controller.navigation()["file"] is True
    controller.handle_post("/goto", {"to": ["file"]})
    assert controller.current_state().step is WizardStep.PICK_FILE


# Segments ------------------------------------------------------------------


def test_approving_new_segments_plans_them_from_the_start(tmp_path: Path) -> None:
    controller = _job(tmp_path, WizardStep.SEGMENTS, segment_index=1)
    job_dir = _job_dir(controller)
    _two_segments(job_dir, approved=False)
    _planned(job_dir)  # left over from segments that were sent back
    controller.handle_post("/segments", {"action": ["approve"]})
    state = controller.current_state()
    assert state.step is WizardStep.ANALYZING and state.segment_index == 0
    found = load_segment_list(job_dir)
    assert found is not None and found.approved
    assert not shots_path(job_dir).exists() and not plan_path(job_dir).exists()


def test_approving_the_same_segments_again_keeps_the_work(tmp_path: Path) -> None:
    controller = _job(tmp_path, WizardStep.SEGMENTS, segment_index=1)
    job_dir = _job_dir(controller)
    _two_segments(job_dir, approved=True)
    _planned(job_dir)
    controller.handle_post("/segments", {"action": ["approve"]})
    state = controller.current_state()
    assert state.step is WizardStep.ANALYZING and state.segment_index == 1
    assert shots_path(job_dir).is_file() and plan_path(job_dir).is_file()


def test_a_segments_note_sends_them_back_to_gemini(tmp_path: Path) -> None:
    controller = _job(tmp_path, WizardStep.SEGMENTS)
    job_dir = _job_dir(controller)
    _two_segments(job_dir, approved=True)
    note = "split   at the\ncar  " + "x" * 600
    controller.handle_post("/segments", {"action": ["note"], "note": [note]})
    state = controller.current_state()
    assert state.step is WizardStep.PROPOSING and state.error is None
    found = load_segment_list(job_dir)
    assert found is not None and not found.approved
    # Whitespace is collapsed and the note capped, as Gemini is sent it.
    assert found.note.startswith("split at the car x")
    assert len(found.note) == 500


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({"action": ["note"], "note": ["   "]}, "say what to change"),
        ({"action": ["shuffle"]}, "unknown action 'shuffle'"),
        ({}, "unknown action ''"),
    ],
)
def test_a_bad_segments_post_keeps_the_segments(
    tmp_path: Path, fields: dict[str, list[str]], error: str
) -> None:
    controller = _job(tmp_path, WizardStep.SEGMENTS)
    job_dir = _job_dir(controller)
    _two_segments(job_dir, approved=True)
    before = (job_dir / "segments.json").read_text(encoding="utf-8")
    controller.handle_post("/segments", fields)
    state = controller.current_state()
    assert state.step is WizardStep.SEGMENTS
    assert state.error is not None and error in state.error
    assert (job_dir / "segments.json").read_text(encoding="utf-8") == before


def test_segments_post_without_a_segment_list_is_refused(tmp_path: Path) -> None:
    controller = _job(tmp_path, WizardStep.SEGMENTS)
    with pytest.raises(WizardError, match="segments are missing"):
        controller.handle_post("/segments", {"action": ["approve"]})


# Segment previews ----------------------------------------------------------


def _back_on_preview(controller: WizardController) -> None:
    """Skip the analyze worker: the next segment's preview is ready."""
    state = controller.current_state()
    state.step = WizardStep.PREVIEW_SEGMENT
    save_session_state(controller.jobs_dir, state)
    save_job_wizard_state(_job_dir(controller), state)


def test_approving_a_preview_moves_to_the_next_segment_then_ready(
    tmp_path: Path,
) -> None:
    controller = _job(tmp_path, WizardStep.PREVIEW_SEGMENT)
    _planned(_job_dir(controller))
    controller.handle_post("/preview", {"action": ["approve"]})
    state = controller.current_state()
    assert (state.step, state.segment_index) == (WizardStep.ANALYZING, 1)
    _back_on_preview(controller)
    controller.handle_post("/preview", {"action": ["approve"]})
    state = controller.current_state()
    assert (state.step, state.segment_index) == (WizardStep.READY_TO_RENDER, 2)


def test_a_redo_note_goes_to_that_segment_only(tmp_path: Path) -> None:
    controller = _job(tmp_path, WizardStep.PREVIEW_SEGMENT, segment_index=1)
    job_dir = _job_dir(controller)
    _planned(job_dir)
    controller.handle_post("/preview", {"action": ["redo"], "note": ["less grain"]})
    assert controller.current_state().step is WizardStep.ANALYZING
    plan = load_plan(job_dir)
    assert plan is not None
    # New recipes start stale; the note is what marks the segment redone.
    assert [r.note for r in plan.recipes] == ["", "less grain"]
    assert plan.recipes[1].stale


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({"action": ["redo"], "note": [" "]}, "say what to change"),
        ({"action": ["zoom"]}, "unknown action 'zoom'"),
    ],
)
def test_a_bad_preview_post_stays_on_the_preview(
    tmp_path: Path, fields: dict[str, list[str]], error: str
) -> None:
    controller = _job(tmp_path, WizardStep.PREVIEW_SEGMENT)
    job_dir = _job_dir(controller)
    _planned(job_dir)
    before = plan_path(job_dir).read_text(encoding="utf-8")
    controller.handle_post("/preview", fields)
    state = controller.current_state()
    assert state.step is WizardStep.PREVIEW_SEGMENT
    assert state.error is not None and error in state.error
    assert plan_path(job_dir).read_text(encoding="utf-8") == before


# Ready to render -----------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({"action": ["preview"]}, "send action=render"),
        ({"action": ["render"], "scale": ["4"]}, "pick one of the upscale options"),
        ({"action": ["render"], "scale": ["2x"]}, "pick one of the upscale options"),
    ],
)
def test_a_bad_ready_post_does_not_start_the_render(
    tmp_path: Path, fields: dict[str, list[str]], error: str
) -> None:
    controller = _job(tmp_path, WizardStep.READY_TO_RENDER)
    job_dir = _job_dir(controller)
    _planned(job_dir, previewed=True)
    controller.handle_post("/ready", fields)
    state = controller.current_state()
    assert state.step is WizardStep.READY_TO_RENDER
    assert state.error is not None and error in state.error
    assert load_upscale_plan(job_dir) is None
    plan = load_plan(job_dir)
    assert plan is not None and plan.scale == 2


def test_ready_post_writes_the_render_plan_at_the_chosen_scale(
    tmp_path: Path,
) -> None:
    controller = _job(tmp_path, WizardStep.READY_TO_RENDER)
    job_dir = _job_dir(controller)
    _planned(job_dir, previewed=True)
    controller.handle_post("/ready", {"action": ["render"], "scale": ["1"]})
    assert controller.current_state().step is WizardStep.RENDERING
    upscale = load_upscale_plan(job_dir)
    assert upscale is not None
    assert [(s.start_s, s.end_s, s.scale) for s in upscale.spans] == [
        (0.0, 3.0, 1),
        (3.0, 6.0, 1),
    ]


def test_ready_post_without_a_plan_is_refused(tmp_path: Path) -> None:
    controller = _job(tmp_path, WizardStep.READY_TO_RENDER)
    with pytest.raises(WizardError, match="segment settings are missing"):
        controller.handle_post("/ready", {"action": ["render"]})


# Result --------------------------------------------------------------------


def test_an_empty_result_note_asks_for_one(tmp_path: Path) -> None:
    controller = _job(tmp_path, WizardStep.RESULT)
    controller.handle_post("/result", {"note": ["   "]})
    state = controller.current_state()
    assert state.step is WizardStep.RESULT
    assert state.error is not None and "describe what to change" in state.error
    controller.handle_post("/result", {"mood": ["happy"]})
    assert controller.current_state().step is WizardStep.DONE


def test_a_result_note_with_no_job_is_refused(tmp_path: Path) -> None:
    controller = WizardController(tmp_path)
    save_session_state(tmp_path, WizardState(step=WizardStep.RESULT))
    controller.handle_post("/result", {"note": ["brighter"]})
    state = controller.current_state()
    assert state.step is WizardStep.RESULT and state.error == "no job to revise"


# Failed and running workers ------------------------------------------------


def _boom() -> None:
    raise RuntimeError("stills vanished")


def test_a_segmenting_worker_crash_fails_the_job_and_goes_back_to_pick(
    tmp_path: Path,
) -> None:
    controller = _job(tmp_path, WizardStep.SAMPLING)
    job_id = controller.current_state().job_id
    assert job_id
    controller.run_segmenting = _boom  # type: ignore[method-assign]
    controller.resume()
    assert controller.start_segmenting() is True
    assert controller.wait_idle(timeout=5)
    state = controller.current_state()
    assert state.step is WizardStep.PICK_FILE and state.job_id is None
    assert state.error == "Finding the segments failed: stills vanished"
    assert controller.store.load(job_id).state is JobState.FAILED
    # The job keeps its own record, so reopening it shows why it stopped.
    kept = load_job_wizard_state(controller.store.job_dir(job_id))
    assert kept is not None and kept.job_id == job_id
    assert kept.step is WizardStep.PICK_FILE and kept.error == state.error


def test_an_analyze_worker_crash_goes_back_to_the_segments(tmp_path: Path) -> None:
    controller = _job(tmp_path, WizardStep.ANALYZING, segment_index=1)
    controller.run_analyze = _boom  # type: ignore[method-assign]
    controller.resume()
    assert controller.start_analyze() is True
    assert controller.wait_idle(timeout=5)
    state = controller.current_state()
    assert state.step is WizardStep.SEGMENTS and state.job_id is not None
    assert state.error == "The segment preview failed: stills vanished"


def test_a_sampling_crash_inside_the_run_is_landed_as_failed(
    tmp_path: Path,
) -> None:
    controller = _job(tmp_path, WizardStep.SAMPLING)
    job_id = controller.current_state().job_id
    assert job_id

    def _crash(state: WizardState) -> None:
        raise RuntimeError("stills vanished")

    controller._sample = _crash  # type: ignore[method-assign]
    controller.resume()
    assert controller.start_segmenting() is True
    assert controller.wait_idle(timeout=5)
    state = controller.current_state()
    assert state.step is WizardStep.PICK_FILE
    assert state.error == "Finding the segments failed: stills vanished"
    assert controller.store.load(job_id).state is JobState.FAILED
    run = controller.api_view()["run"]
    assert run is not None
    assert (run["kind"], run["job_id"], run["outcome"]) == (
        "segmenting",
        job_id,
        "failed",
    )


def test_an_analyze_crash_inside_the_run_is_landed_as_failed(tmp_path: Path) -> None:
    controller = _job(tmp_path, WizardStep.ANALYZING, segment_index=1)

    def _crash(state: WizardState) -> None:
        raise RuntimeError("stills vanished")

    controller._analyze = _crash  # type: ignore[method-assign]
    controller.resume()
    assert controller.start_analyze() is True
    assert controller.wait_idle(timeout=5)
    state = controller.current_state()
    assert state.step is WizardStep.SEGMENTS
    assert state.error == "The segment preview failed: stills vanished"
    run = controller._public_run(state)
    assert run is not None
    assert (run["kind"], run["outcome"]) == ("analyze", "failed")


def test_while_a_worker_runs_navigation_is_off_and_progress_is_live(
    tmp_path: Path,
) -> None:
    controller = _job(tmp_path, WizardStep.RENDERING)
    progress = {"phase": "rendering", "pct": 40, "message": "Upscaling"}
    state = controller.current_state()
    state.progress = progress
    save_session_state(controller.jobs_dir, state)
    save_job_wizard_state(_job_dir(controller), state)
    entered, release = threading.Event(), threading.Event()

    def _slow() -> None:
        entered.set()
        release.wait(5)

    controller._run_render = _slow  # type: ignore[method-assign]
    controller.resume()
    assert controller.start_render() is True
    try:
        assert entered.wait(5)
        assert controller.navigation() == {
            "setup": False,
            "file": False,
            "segments": False,
        }
        assert controller.api_view()["progress"] == progress
    finally:
        release.set()
        assert controller.wait_idle(timeout=5)
    # The worker's clear removed it; even a left-over record is hidden once
    # nothing runs, so a dead worker never shows a frozen bar.
    assert controller.current_state().progress is None
    state = controller.current_state()
    state.progress = progress
    save_session_state(controller.jobs_dir, state)
    save_job_wizard_state(_job_dir(controller), state)
    assert controller.api_view()["progress"] is None


# Scenes approved so far ----------------------------------------------------


def test_a_state_file_without_approved_count_counts_the_earlier_scenes(
    tmp_path: Path,
) -> None:
    controller = _job(tmp_path, WizardStep.PREVIEW_SEGMENT, segment_index=1)
    job_dir = _job_dir(controller)
    _two_segments(job_dir, approved=True)
    _planned(job_dir)
    for path in (job_state_path(job_dir), session_state_path(controller.jobs_dir)):
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["approved_count"]
        path.write_text(json.dumps(data), encoding="utf-8")
    assert controller.current_state().approved_count == 1
    job = controller.api_view()["job"]
    assert [scene["status"] for scene in job["scenes"]] == ["approved", "current"]
    assert job["approved_count"] == 1 and job["current_index"] == 1


def test_an_approved_count_past_the_scenes_is_clamped(tmp_path: Path) -> None:
    controller = _job(
        tmp_path, WizardStep.READY_TO_RENDER, segment_index=2, approved_count=9
    )
    job_dir = _job_dir(controller)
    _two_segments(job_dir, approved=True)
    _planned(job_dir)
    job = controller.api_view()["job"]
    assert (job["scene_count"], job["approved_count"]) == (2, 2)
    assert [scene["status"] for scene in job["scenes"]] == ["approved", "approved"]
    with pytest.raises(WizardError, match="scene 3 cannot be opened yet"):
        controller.handle_post("/scene", {"index": ["2"]})


def test_a_rebuilt_plan_starts_again_from_the_first_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _job(tmp_path, WizardStep.ANALYZING, segment_index=1, approved_count=2)
    job_dir = _job_dir(controller)
    _two_segments(job_dir, approved=True)
    # shots.json and shot_plan.json are gone: the plan is measured again.
    monkeypatch.setattr(
        "super_processor.wizard.shots_from_segments",
        lambda *args, **kwargs: ShotList([Shot(0.0, 3.0), Shot(3.0, 6.0)]),
    )
    worked: list[tuple[list[int], int, int]] = []
    monkeypatch.setattr(
        controller,
        "_work_on",
        lambda state, *args, **kwargs: worked.append(
            (args[4], state.segment_index, state.approved_count)
        ),
    )
    controller.run_analyze()
    assert worked == [([0], 0, 0)]


def test_a_failed_pick_shows_no_scene_strip(tmp_path: Path) -> None:
    controller = _job(
        tmp_path, WizardStep.READY_TO_RENDER, segment_index=2, approved_count=2
    )
    job_dir = _job_dir(controller)
    _two_segments(job_dir, approved=True)
    _planned(job_dir)
    assert len(controller.api_view()["job"]["scenes"]) == 2
    state = controller.current_state()
    state.step = WizardStep.PICK_FILE
    save_session_state(controller.jobs_dir, state)
    save_job_wizard_state(job_dir, state)
    controller.handle_post("/pick", {"path": [str(tmp_path / "missing.mp4")]})
    view = controller.api_view()
    assert view["step"] == "pick_file" and view["error"]
    job = view["job"]
    assert job["stage"] == "file" and job["scenes"] == []
    assert job["current_index"] is None


def test_a_more_denoise_note_turns_dpir_on_for_that_scene_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from super_processor import wizard

    wizard._dpir_on_gpu.cache_clear()
    monkeypatch.setattr(wizard, "dpir_runs_here", lambda: True)
    controller = _job(tmp_path, WizardStep.PREVIEW_SEGMENT, segment_index=1)
    job_dir = _job_dir(controller)
    _planned(job_dir)
    controller.handle_post(
        "/preview", {"action": ["redo"], "note": ["still noisy, more denoise"]}
    )
    plan = load_plan(job_dir)
    assert plan is not None
    assert [r.dpir for r in plan.recipes] == [0, 2]
    view = controller._segment_preview_view(job_dir, 1)
    assert (view["shot"]["dpir"], view["shot"]["dpir_on_gpu"]) == (2, True)
    other = controller._segment_preview_view(job_dir, 0)
    assert (other["shot"]["dpir"], other["shot"]["dpir_on_gpu"]) == (0, None)
    _back_on_preview(controller)
    controller.handle_post("/preview", {"action": ["redo"], "note": ["no dpir"]})
    plan = load_plan(job_dir)
    assert plan is not None
    assert [r.dpir for r in plan.recipes] == [0, 0]
    wizard._dpir_on_gpu.cache_clear()
