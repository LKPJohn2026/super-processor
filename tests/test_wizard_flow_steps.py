"""The wizard's segment-loop posts and workers, driven from files on disk.

These build the job folder (segments, shots, plan) directly, so each check
runs without FFmpeg, Gemini, or a GPU: the controller's own decisions are
what is under test.
"""

from __future__ import annotations

import json
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
    ShotPlan,
    ShotRecipe,
    load_plan,
    plan_path,
    save_plan,
)
from super_processor.shots import Shot, ShotList, save_shots, shots_path
from super_processor.upscale import UpscaleError, UpscaleRequest, load_upscale_plan
from super_processor.wizard import (
    WizardController,
    WizardError,
    WizardState,
    WizardStep,
    load_job_wizard_state,
)


class _NoGpu:
    """Any GPU pass is a failure: the step under test must not need one."""

    def __init__(self) -> None:
        self.calls = 0

    def upscale(self, request: UpscaleRequest) -> Path:
        self.calls += 1
        raise UpscaleError("no GPU pass expected here")


class _NoGemini:
    """Any Gemini call is a failure: the step under test must not need one."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"Gemini was asked to {name}")


def _job(
    tmp_path: Path, step: WizardStep, *, segment_index: int = 0
) -> tuple[WizardController, Path]:
    engine = _NoGpu()
    controller = WizardController(
        tmp_path / "jobs",
        gemini=_NoGemini(),  # type: ignore[arg-type]
        upscale_engine=engine,
    )
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not a real video")
    job_id = controller.store.create(source).job_id
    controller._save(WizardState(step=step, job_id=job_id, segment_index=segment_index))
    return controller, controller.store.job_dir(job_id)


def _segments(job_dir: Path, *, approved: bool) -> SegmentList:
    found = SegmentList(
        segments=[
            LoopSegment(0.0, 4.0, label="first"),
            LoopSegment(4.0, 8.0, label="second"),
        ],
        approved=approved,
    )
    save_segment_list(job_dir, found)
    return found


def _planned(job_dir: Path, *, scale: int = 2, rev: int = 1) -> None:
    """Shots and a plan for both segments, previewed ``rev`` times."""
    save_shots(
        job_dir,
        ShotList(shots=[Shot(0.0, 4.0, label="first"), Shot(4.0, 8.0, label="second")]),
    )
    save_plan(
        job_dir,
        ShotPlan(
            scale=scale,
            recipes=[
                ShotRecipe(strength=0.4, rev=rev, stale=rev == 0, reason="planned"),
                ShotRecipe(strength=0.3, rev=rev, stale=rev == 0, reason="planned"),
            ],
        ),
    )


# Segments ------------------------------------------------------------------


def test_approving_new_segments_drops_the_plan_of_the_old_ones(
    tmp_path: Path,
) -> None:
    # Segments Gemini proposed again after a note are new: the shots and
    # settings of the earlier cut no longer line up with them.
    controller, job_dir = _job(tmp_path, WizardStep.SEGMENTS, segment_index=1)
    _segments(job_dir, approved=False)
    _planned(job_dir)
    controller.handle_post("/segments", {"action": ["approve"]})
    state = controller.current_state()
    assert (state.step, state.segment_index) == (WizardStep.ANALYZING, 0)
    assert not shots_path(job_dir).exists() and not plan_path(job_dir).exists()
    found = load_segment_list(job_dir)
    assert found is not None and found.approved


def test_approving_the_same_segments_again_keeps_the_approved_work(
    tmp_path: Path,
) -> None:
    # Back to the segments and approve again without a change: the previews
    # already made are kept, and the loop carries on where it was.
    controller, job_dir = _job(tmp_path, WizardStep.SEGMENTS, segment_index=1)
    _segments(job_dir, approved=True)
    _planned(job_dir)
    controller.handle_post("/segments", {"action": ["approve"]})
    state = controller.current_state()
    assert (state.step, state.segment_index) == (WizardStep.ANALYZING, 1)
    plan = load_plan(job_dir)
    assert plan is not None and [r.rev for r in plan.recipes] == [1, 1]


def test_a_segment_note_is_tidied_and_sends_the_cut_back(tmp_path: Path) -> None:
    controller, job_dir = _job(tmp_path, WizardStep.SEGMENTS)
    _segments(job_dir, approved=True)
    controller.handle_post(
        "/segments",
        {"action": ["note"], "note": ["  cut \n at   the door " + "x" * 600]},
    )
    assert controller.current_state().step is WizardStep.PROPOSING
    found = load_segment_list(job_dir)
    assert found is not None and not found.approved
    assert found.note.startswith("cut at the door x") and len(found.note) == 500


def test_segment_posts_refuse_unknown_actions_and_missing_files(
    tmp_path: Path,
) -> None:
    controller, job_dir = _job(tmp_path, WizardStep.SEGMENTS)
    with pytest.raises(WizardError, match="segments are missing"):
        controller.handle_post("/segments", {"action": ["approve"]})
    (job_dir / "segments.json").write_text("{", encoding="utf-8")
    with pytest.raises(WizardError, match="cannot read segments.json"):
        controller.handle_post("/segments", {"action": ["approve"]})
    _segments(job_dir, approved=False)
    controller.handle_post("/segments", {"action": ["shuffle"]})
    state = controller.current_state()
    assert state.step is WizardStep.SEGMENTS
    assert state.error == "unknown action 'shuffle'"


# Analyze -------------------------------------------------------------------


def test_analyze_shows_a_finished_preview_again_without_redoing_it(
    tmp_path: Path,
) -> None:
    # Approving the same segments again lands on a segment whose preview is
    # already made: it is shown, not sent to Gemini and the GPU again.
    controller, job_dir = _job(tmp_path, WizardStep.ANALYZING, segment_index=1)
    _segments(job_dir, approved=True)
    _planned(job_dir)
    before = plan_path(job_dir).read_text(encoding="utf-8")
    controller.run_analyze()
    state = controller.current_state()
    assert state.step is WizardStep.PREVIEW_SEGMENT, state.error
    assert state.segment_index == 1
    engine = controller.upscale_engine()
    assert isinstance(engine, _NoGpu) and engine.calls == 0
    assert plan_path(job_dir).read_text(encoding="utf-8") == before


def test_analyze_past_the_last_segment_is_ready_to_render(tmp_path: Path) -> None:
    controller, job_dir = _job(tmp_path, WizardStep.ANALYZING, segment_index=2)
    _segments(job_dir, approved=True)
    _planned(job_dir)
    controller.run_analyze()
    assert controller.current_state().step is WizardStep.READY_TO_RENDER


def test_a_plan_saved_at_4x_loads_and_previews_at_2x(tmp_path: Path) -> None:
    # ShotPlan.from_dict does the clamping as the plan is read, so analyze
    # never sees a 4x plan; this proves such a job still reaches its preview.
    # The clamp itself is pinned in test_shot_plan.py.
    controller, job_dir = _job(tmp_path, WizardStep.ANALYZING)
    _segments(job_dir, approved=True)
    _planned(job_dir)
    raw = json.loads(plan_path(job_dir).read_text(encoding="utf-8"))
    raw["scale"] = 4
    plan_path(job_dir).write_text(json.dumps(raw), encoding="utf-8")
    controller.run_analyze()
    assert controller.current_state().step is WizardStep.PREVIEW_SEGMENT
    plan = load_plan(job_dir)
    assert plan is not None and plan.scale == 2


def test_a_crash_while_analyzing_goes_back_to_the_segments(tmp_path: Path) -> None:
    # A damaged shots.json raises out of the analyze step. The worker must
    # still land the editor on the segments with the reason, not leave the
    # page spinning on "analyzing".
    controller, job_dir = _job(tmp_path, WizardStep.ANALYZING)
    _segments(job_dir, approved=True)
    shots_path(job_dir).write_text("{", encoding="utf-8")
    controller.resume()
    assert controller.start_analyze()
    assert controller.wait_idle(30)
    state = controller.current_state()
    assert state.step is WizardStep.SEGMENTS
    assert state.error is not None
    assert state.error.startswith("The segment preview failed: shots.json")
    assert state.progress is None


def test_finding_segments_on_an_unreadable_file_fails_the_job(tmp_path: Path) -> None:
    controller, job_dir = _job(tmp_path, WizardStep.PROPOSING)
    job_id = controller.current_state().job_id
    assert job_id
    controller.resume()
    assert controller.start_segmenting()
    assert controller.wait_idle(30)
    state = controller.current_state()
    assert state.step is WizardStep.PICK_FILE and state.job_id is None
    assert state.error is not None
    assert state.error.startswith("Finding the segments failed:")
    assert controller.store.load(job_id).state is JobState.FAILED
    # The job keeps its own record of why it stopped.
    kept = load_job_wizard_state(job_dir)
    assert kept is not None and kept.error == state.error


# Preview -------------------------------------------------------------------


def test_approving_the_last_preview_is_ready_to_render(tmp_path: Path) -> None:
    controller, job_dir = _job(tmp_path, WizardStep.PREVIEW_SEGMENT)
    _segments(job_dir, approved=True)
    _planned(job_dir)
    controller.handle_post("/preview", {"action": ["approve"]})
    state = controller.current_state()
    assert (state.step, state.segment_index) == (WizardStep.ANALYZING, 1)
    controller._save(
        WizardState(
            step=WizardStep.PREVIEW_SEGMENT, job_id=state.job_id, segment_index=1
        )
    )
    controller.handle_post("/preview", {"action": ["approve"]})
    state = controller.current_state()
    assert (state.step, state.segment_index) == (WizardStep.READY_TO_RENDER, 2)


def test_preview_posts_refuse_unknown_actions_and_a_missing_plan(
    tmp_path: Path,
) -> None:
    controller, job_dir = _job(tmp_path, WizardStep.PREVIEW_SEGMENT)
    with pytest.raises(WizardError, match="settings are missing"):
        controller.handle_post("/preview", {"action": ["approve"]})
    _planned(job_dir)
    controller.handle_post("/preview", {"action": ["skip"]})
    state = controller.current_state()
    assert (state.step, state.segment_index) == (WizardStep.PREVIEW_SEGMENT, 0)
    assert state.error == "unknown action 'skip'"


# Ready ---------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["4", "3", "0", "two", "-1", "2.0"])
def test_ready_refuses_a_scale_it_did_not_offer(tmp_path: Path, raw: str) -> None:
    controller, job_dir = _job(tmp_path, WizardStep.READY_TO_RENDER)
    _planned(job_dir)
    controller.handle_post("/ready", {"action": ["render"], "scale": [raw]})
    state = controller.current_state()
    assert state.step is WizardStep.READY_TO_RENDER
    assert state.error == "pick one of the upscale options shown"
    assert load_upscale_plan(job_dir) is None
    plan = load_plan(job_dir)
    assert plan is not None and plan.scale == 2


@pytest.mark.parametrize(("raw", "expected"), [("1", 1), ("2", 2), ("", 2)])
def test_ready_renders_every_segment_at_the_chosen_scale(
    tmp_path: Path, raw: str, expected: int
) -> None:
    controller, job_dir = _job(tmp_path, WizardStep.READY_TO_RENDER)
    _planned(job_dir)
    controller.handle_post("/ready", {"action": ["render"], "scale": [raw]})
    state = controller.current_state()
    assert state.step is WizardStep.RENDERING and state.error is None
    upscale = load_upscale_plan(job_dir)
    assert upscale is not None
    assert [
        (span.start_s, span.end_s, span.scale, span.strength) for span in upscale.spans
    ] == [(0.0, 4.0, expected, 0.4), (4.0, 8.0, expected, 0.3)]
    plan = load_plan(job_dir)
    assert plan is not None and plan.scale == expected


def test_ready_without_the_render_action_waits(tmp_path: Path) -> None:
    controller, job_dir = _job(tmp_path, WizardStep.READY_TO_RENDER)
    _planned(job_dir)
    controller.handle_post("/ready", {"action": ["wait"]})
    state = controller.current_state()
    assert state.step is WizardStep.READY_TO_RENDER
    assert state.error == "send action=render to start the full render"
    assert load_upscale_plan(job_dir) is None


def test_ready_refuses_settings_that_no_longer_match_the_segments(
    tmp_path: Path,
) -> None:
    controller, job_dir = _job(tmp_path, WizardStep.READY_TO_RENDER)
    with pytest.raises(WizardError, match="settings are missing"):
        controller.handle_post("/ready", {"action": ["render"]})
    _planned(job_dir)
    save_shots(job_dir, ShotList(shots=[Shot(0.0, 8.0, label="one")]))
    controller.handle_post("/ready", {"action": ["render"]})
    state = controller.current_state()
    assert state.step is WizardStep.READY_TO_RENDER
    assert state.error is not None and "no longer match" in state.error
    assert load_upscale_plan(job_dir) is None


# Result --------------------------------------------------------------------


def test_result_needs_a_note_unless_the_editor_is_happy(tmp_path: Path) -> None:
    controller, job_dir = _job(tmp_path, WizardStep.RESULT)
    for fields in ({}, {"note": ["   "]}, {"mood": ["revise"]}):
        controller.handle_post("/result", fields)
        state = controller.current_state()
        assert state.step is WizardStep.RESULT
        assert state.error is not None and "describe what to change" in state.error
    assert load_upscale_plan(job_dir) is None
    controller.handle_post("/result", {"mood": ["happy"], "note": ["ignored"]})
    state = controller.current_state()
    assert (state.step, state.error) == (WizardStep.DONE, None)


def test_result_note_without_a_job_is_refused(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    controller._save(WizardState(step=WizardStep.RESULT))
    controller.handle_post("/result", {"note": ["less contrast"]})
    state = controller.current_state()
    assert (state.step, state.error) == (WizardStep.RESULT, "no job to revise")


def test_workers_start_only_on_their_own_step(tmp_path: Path) -> None:
    controller, _job_dir = _job(tmp_path, WizardStep.READY_TO_RENDER)
    assert controller.start_render() is False
    assert controller.start_analyze() is False
    assert controller.start_segmenting() is False
    assert controller.wait_idle(1)
