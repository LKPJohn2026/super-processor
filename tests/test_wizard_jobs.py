"""The job list, opening a job, and resuming a job left mid-step.

No test here needs a GPU, FFmpeg, the network or a Gemini key: runs are
replaced by small fakes, and stills are written by hand.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.client import HTTPConnection
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

from super_processor.jobs import JobState
from super_processor.review import WizardServer
from super_processor.segment_loop import (
    LoopSegment,
    SegmentList,
    save_segment_list,
)
from super_processor.wizard import (
    CANCEL_MARKER,
    WORKER_PID_FILE,
    WizardController,
    WizardError,
    WizardState,
    WizardStep,
    load_job_wizard_state,
    save_job_wizard_state,
    save_session_state,
)

TOKEN = "jobs-token"
RUN_STEPS = (
    WizardStep.SAMPLING,
    WizardStep.PROPOSING,
    WizardStep.ANALYZING,
    WizardStep.RENDERING,
)


def _job(
    controller: WizardController,
    tmp_path: Path,
    step: WizardStep,
    *,
    name: str = "clip.mp4",
    session: bool = True,
) -> tuple[str, Path]:
    source = tmp_path / name
    source.write_bytes(b"not really a video")
    job_id = controller.store.create(source).job_id
    controller.store.transition(job_id, JobState.PROBED)
    job_dir = controller.store.job_dir(job_id)
    state = WizardState(step=step, job_id=job_id)
    save_job_wizard_state(job_dir, state)
    if session:
        save_session_state(controller.jobs_dir, state)
    return job_id, job_dir


def _still(job_dir: Path, t_ms: int) -> None:
    path = job_dir / "segment_stills" / f"t_{t_ms:08d}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"still {t_ms}".encode())


def _request(
    base: str, method: str, path: str, payload: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    parsed = urlparse(base)
    connection = HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port)
    body = b"" if payload is None else json.dumps(payload).encode()
    headers = {"Accept": "application/json", "X-Wizard-Token": TOKEN}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read() or b"{}")
    finally:
        connection.close()


# Resuming a job left mid-step (decision Q1) --------------------------------


@pytest.mark.parametrize("step", RUN_STEPS)
def test_a_step_left_by_an_earlier_server_waits_for_the_editor(
    tmp_path: Path, step: WizardStep
) -> None:
    controller = WizardController(tmp_path / "jobs")
    job_id, _job_dir = _job(controller, tmp_path, step)
    calls: list[str] = []
    for name in ("_run_render", "run_segmenting", "run_analyze"):
        setattr(controller, name, lambda name=name: calls.append(name))
    view = controller.api_view()
    assert view["interrupted"] is True
    assert view["busy"] is False
    # Nothing starts GPU or Gemini work by itself.
    assert controller.start_render() is False
    assert controller.start_segmenting() is False
    assert controller.start_analyze() is False
    assert calls == []
    page = controller.render()
    assert 'action="/resume"' in page and "Carry on where you left off?" in page
    assert 'http-equiv="refresh"' not in page
    controller.handle_post("/resume", {"job_id": [job_id]})
    assert controller.api_view()["interrupted"] is False
    assert 'http-equiv="refresh"' in controller.render()
    starts = {
        WizardStep.SAMPLING: controller.start_segmenting,
        WizardStep.PROPOSING: controller.start_segmenting,
        WizardStep.ANALYZING: controller.start_analyze,
        WizardStep.RENDERING: controller.start_render,
    }
    assert starts[step]() is True
    assert controller.wait_idle(5)
    assert len(calls) == 1


def test_a_post_that_moves_to_a_run_step_is_not_interrupted(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    _job_id, _job_dir = _job(controller, tmp_path, WizardStep.READY_TO_RENDER)
    seen: list[bool] = []

    def dispatch(path: str, fields: dict[str, list[str]], state: WizardState) -> None:
        state.step = WizardStep.RENDERING
        controller._save(state)
        # Armed before the state reached the disk.
        seen.append(controller.interrupted(controller.current_state()))

    controller._dispatch_post = dispatch  # type: ignore[method-assign]
    controller._run_render = lambda: None  # type: ignore[method-assign]
    controller.handle_post("/ready", {})
    assert seen == [False]
    assert controller.api_view()["interrupted"] is False
    assert controller.start_render() is True
    assert controller.wait_idle(5)


def test_a_new_controller_on_the_same_folder_forgets_what_was_armed(
    tmp_path: Path,
) -> None:
    controller = WizardController(tmp_path / "jobs")
    _job(controller, tmp_path, WizardStep.RENDERING)
    controller.resume()
    assert controller.api_view()["interrupted"] is False
    # A restart: the arming lived only in the old server's memory.
    again = WizardController(tmp_path / "jobs")
    assert again.api_view()["interrupted"] is True
    assert again.start_render() is False


def test_resume_is_refused_off_a_run_step_and_for_another_job(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    job_id, _job_dir = _job(controller, tmp_path, WizardStep.SEGMENTS)
    with pytest.raises(WizardError, match="nothing to resume"):
        controller.handle_post("/resume", {})
    state = controller.current_state()
    state.step = WizardStep.RENDERING
    save_session_state(controller.jobs_dir, state)
    save_job_wizard_state(controller.store.job_dir(job_id), state)
    with pytest.raises(WizardError, match="no longer open"):
        controller.handle_post("/resume", {"job_id": ["ffffffffffffffff"]})
    assert controller.api_view()["interrupted"] is True


def test_resume_clears_the_old_error_and_drops_a_stale_stop(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    job_id, job_dir = _job(controller, tmp_path, WizardStep.SAMPLING)
    state = controller.current_state()
    state.error = "the server went away"
    save_session_state(controller.jobs_dir, state)
    save_job_wizard_state(job_dir, state)
    (job_dir / CANCEL_MARKER).touch()
    controller.resume(job_id=job_id)
    assert controller.current_state().error is None
    assert not (job_dir / CANCEL_MARKER).exists()


def test_an_interrupted_analysis_can_be_stopped_back_to_the_segments(
    tmp_path: Path,
) -> None:
    controller = WizardController(tmp_path / "jobs")
    job_id, job_dir = _job(controller, tmp_path, WizardStep.ANALYZING)
    save_segment_list(
        job_dir,
        SegmentList(
            segments=[LoopSegment(0.0, 4.0, "first"), LoopSegment(4.0, 8.0, "next")],
            approved=True,
        ),
    )
    view = controller.api_view()
    assert view["interrupted"] is True and view["cancellable"] is True
    assert 'action="/cancel"' in controller.render()
    controller.handle_post("/cancel", {"job_id": [job_id]})
    state = controller.current_state()
    assert state.step is WizardStep.SEGMENTS
    assert state.stopped is not None and "approved segments" in state.stopped
    run = controller.api_view()["run"]
    assert (run["kind"], run["outcome"]) == ("analyze", "stopped")


def test_a_resumed_analysis_is_not_stoppable(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    _job(controller, tmp_path, WizardStep.ANALYZING)
    controller.resume()
    assert controller.api_view()["cancellable"] is False
    with pytest.raises(WizardError, match="nothing running to stop"):
        controller.handle_post("/cancel", {})
    assert controller.current_state().step is WizardStep.ANALYZING


def test_run_1_on_an_interrupted_job_is_refused_over_http(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    job_id, _job_dir = _job(controller, tmp_path, WizardStep.RENDERING)
    entered = threading.Event()
    controller._run_render = entered.set  # type: ignore[method-assign]
    server = WizardServer(controller.jobs_dir, controller=controller, token=TOKEN)
    base = server.start()
    try:
        status, view = _request(base, "GET", "/api/render?run=1")
        assert status == 200
        assert view["interrupted"] is True and view["step"] == "rendering"
        assert not entered.wait(0.2)
        status, view = _request(base, "POST", "/api/resume", {"job_id": job_id})
        assert status == 200 and view["interrupted"] is False
        _request(base, "GET", "/api/render?run=1")
        assert entered.wait(5)
    finally:
        server.stop()


# Opening a job -------------------------------------------------------------


def test_opening_a_job_shows_it_where_it_was_left(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    first, first_dir = _job(controller, tmp_path, WizardStep.SEGMENTS)
    second, second_dir = _job(
        controller, tmp_path, WizardStep.RENDERING, name="b.mp4", session=False
    )
    state = load_job_wizard_state(second_dir)
    assert state is not None
    state.progress = {"phase": "rendering", "pct": 40, "message": "left over"}
    state.render_note = "less grain"
    save_job_wizard_state(second_dir, state)
    controller.handle_post("/jobs/open", {"job": [second], "job_id": [first]})
    opened = controller.current_state()
    assert (opened.job_id, opened.step) == (second, WizardStep.RENDERING)
    # The render it was left on may still be resumed, so it keeps its note.
    assert opened.progress is None and opened.render_note == "less grain"
    # Reopened on a run step, its work waits for the editor.
    assert controller.api_view()["interrupted"] is True
    assert controller.start_render() is False
    # Off a render, a left-over note belongs to no render and is dropped.
    first_state = load_job_wizard_state(first_dir)
    assert first_state is not None
    first_state.render_note = "a note for a render that never ran"
    save_job_wizard_state(first_dir, first_state)
    controller.handle_post("/jobs/open", {"job": [first]})
    assert controller.current_state().job_id == first
    assert controller.current_state().render_note is None


def test_opening_a_job_disarms_a_resumed_step(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    job_id, _job_dir = _job(controller, tmp_path, WizardStep.RENDERING)
    controller.resume()
    controller.open_job(job_id)
    assert controller.api_view()["interrupted"] is True


@pytest.mark.parametrize("job", ["", "../etc", "ffffffffffffffff"])
def test_opening_an_unknown_job_is_refused(tmp_path: Path, job: str) -> None:
    controller = WizardController(tmp_path / "jobs")
    job_id, _job_dir = _job(controller, tmp_path, WizardStep.SEGMENTS)
    with pytest.raises(WizardError):
        controller.handle_post("/jobs/open", {"job": [job]})
    assert controller.current_state().job_id == job_id


def test_opening_a_job_is_refused_while_a_run_works(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    _job(controller, tmp_path, WizardStep.RENDERING)
    other, _other_dir = _job(
        controller, tmp_path, WizardStep.RESULT, name="b.mp4", session=False
    )
    entered, release = threading.Event(), threading.Event()

    def slow() -> None:
        entered.set()
        release.wait(5)

    controller._run_render = slow  # type: ignore[method-assign]
    controller.resume()
    assert controller.start_render()
    try:
        assert entered.wait(5)
        with pytest.raises(WizardError, match="still running"):
            controller.handle_post("/jobs/open", {"job": [other]})
    finally:
        release.set()
        assert controller.wait_idle(5)


# The job list --------------------------------------------------------------


def test_the_job_list_is_read_only_and_empty_without_a_folder(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    assert controller.jobs_view() == []
    assert not (tmp_path / "jobs").exists()


def test_the_job_list_shows_each_wizard_job_newest_first(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    paused, paused_dir = _job(controller, tmp_path, WizardStep.SEGMENTS, name="a.mp4")
    done, done_dir = _job(controller, tmp_path, WizardStep.RESULT, name="b.mp4")
    (done_dir / "output.mp4").write_bytes(b"video")
    left, _left_dir = _job(
        controller, tmp_path, WizardStep.RENDERING, name="c.mp4", session=False
    )
    failed, failed_dir = _job(
        controller, tmp_path, WizardStep.PICK_FILE, name="d.mp4", session=False
    )
    controller.store.transition(failed, JobState.FAILED, error="boom")
    # A job the command line made, without the wizard, is not listed.
    (tmp_path / "e.mp4").write_bytes(b"x")
    controller.store.create(tmp_path / "e.mp4")
    _still(paused_dir, 1000)
    _still(paused_dir, 5000)
    _still(paused_dir, 9000)
    order = [failed, left, done, paused]
    for age, job_id in enumerate(order):
        stamp = time.time() - age * 100
        os.utime(controller.store.job_dir(job_id) / "wizard_state.json", (stamp, stamp))
    controller.store.update_notes(failed, {})  # manifest time is now: newest
    jobs = controller.jobs_view()
    assert [job["job_id"] for job in jobs] == order
    by_id = {job["job_id"]: job for job in jobs}
    assert by_id[done]["status"] == "result"
    assert by_id[done]["output_url"].startswith(f"/jobs/{done}/output.mp4?v=")
    assert by_id[done]["open"] is True
    assert by_id[paused]["status"] == "paused"
    assert by_id[paused]["thumb_url"] == f"/jobs/{paused}/thumb.jpg"
    assert by_id[paused]["file_name"] == "a.mp4"
    assert by_id[left]["resumes_work"] is True
    assert by_id[left]["interrupted"] is True
    assert by_id[left]["stage"] == "render"
    assert by_id[failed]["status"] == "failed"
    assert by_id[done]["resumes_work"] is False
    assert by_id[done]["thumb_url"] is None
    assert all(job["error"] is None for job in jobs)


def test_a_kept_version_counts_as_a_result(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    job_id, job_dir = _job(controller, tmp_path, WizardStep.READY_TO_RENDER)
    (job_dir / "outputs").mkdir()
    (job_dir / "outputs" / "v001.mp4").write_bytes(b"kept")
    (entry,) = controller.jobs_view()
    assert entry["job_id"] == job_id and entry["status"] == "result"


def test_a_damaged_job_is_listed_with_its_error(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    good, _good_dir = _job(controller, tmp_path, WizardStep.SEGMENTS)
    bad, bad_dir = _job(
        controller, tmp_path, WizardStep.SEGMENTS, name="b.mp4", session=False
    )
    (bad_dir / "wizard_state.json").write_text("{not json", encoding="utf-8")
    by_id = {job["job_id"]: job for job in controller.jobs_view()}
    assert by_id[good]["error"] is None
    assert by_id[bad]["error"] == "corrupt wizard_state.json"
    assert by_id[bad]["step"] is None


def test_the_title_is_the_picked_name_without_the_upload_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Video:
        width, height = 640, 480

    class _Facts:
        has_video = True
        duration_s = 8.0

        def primary_video(self) -> _Video:
            return _Video()

    monkeypatch.setattr("super_processor.wizard.probe_file", lambda _path: _Facts())
    monkeypatch.setattr(
        "super_processor.wizard.require_supported_timing", lambda *_a, **_k: None
    )
    controller = WizardController(tmp_path / "jobs")
    upload = controller.upload_target("Family VHS 1994.mp4")
    upload.write_bytes(b"x")
    controller.pick_upload(upload, "Family VHS 1994.mp4")
    state = controller.current_state()
    assert state.step is WizardStep.SAMPLING and state.job_id
    assert controller.store.load(state.job_id).notes["title"] == "Family VHS 1994.mp4"
    # The pick moved to a run step itself, so it is not interrupted.
    assert controller.api_view()["interrupted"] is False
    (entry,) = controller.jobs_view()
    assert entry["title"] == "Family VHS 1994.mp4"
    assert entry["file_name"] == "Family_VHS_1994.mp4"
    assert controller.api_view()["job"]["title"] == "Family VHS 1994.mp4"


def test_the_job_list_over_http(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    first, _first_dir = _job(controller, tmp_path, WizardStep.SEGMENTS)
    second, _second_dir = _job(
        controller, tmp_path, WizardStep.RESULT, name="b.mp4", session=False
    )
    server = WizardServer(controller.jobs_dir, controller=controller, token=TOKEN)
    base = server.start()
    try:
        status, listed = _request(base, "GET", "/api/jobs")
        assert status == 200
        assert {job["job_id"] for job in listed["jobs"]} == {first, second}
        status, view = _request(base, "POST", "/api/jobs/open", {"job": second})
        assert status == 200 and view["job_id"] == second
        status, refused = _request(base, "POST", "/api/jobs/open", {"job": "nope"})
        assert status == 400 and "invalid job id" in refused["error"]
    finally:
        server.stop()


# Another server working on the same job ------------------------------------


def test_a_run_leaves_its_process_id_in_the_job_while_it_works(
    tmp_path: Path,
) -> None:
    controller = WizardController(tmp_path / "jobs")
    job_id, job_dir = _job(controller, tmp_path, WizardStep.RENDERING)
    seen: list[str] = []

    def render() -> None:
        seen.append((job_dir / WORKER_PID_FILE).read_text(encoding="utf-8").split()[0])
        listed = {job["job_id"]: job for job in controller.jobs_view()}
        seen.append(listed[job_id]["status"])

    controller._run_render = render  # type: ignore[method-assign]
    controller.resume()
    assert controller.start_render()
    assert controller.wait_idle(5)
    assert seen == [str(os.getpid()), "running"]
    assert not (job_dir / WORKER_PID_FILE).exists()


def test_a_job_another_server_works_on_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = WizardController(tmp_path / "jobs")
    job_id, job_dir = _job(controller, tmp_path, WizardStep.RENDERING)
    other, _other_dir = _job(
        controller, tmp_path, WizardStep.RENDERING, name="b.mp4", session=False
    )
    elsewhere = os.getpid() + 1
    (job_dir / WORKER_PID_FILE).write_text(f"{elsewhere}\n", encoding="utf-8")
    alive = {elsewhere}
    monkeypatch.setattr("super_processor.wizard._pid_alive", lambda pid: pid in alive)
    by_id = {job["job_id"]: job for job in controller.jobs_view()}
    assert by_id[job_id]["status"] == "running_elsewhere"
    assert by_id[job_id]["interrupted"] is False
    assert by_id[other]["status"] == "paused"
    view = controller.api_view()
    assert view["running_elsewhere"] is True and view["cancellable"] is False
    assert "Working in another window" in controller.render()
    assert 'action="/resume"' not in controller.render()
    with pytest.raises(WizardError, match="another Super Processor window"):
        controller.resume()
    with pytest.raises(WizardError, match="another Super Processor window"):
        controller.cancel()
    controller._armed = (job_id, "render")
    assert controller.start_render() is False
    controller.open_job(other)
    with pytest.raises(WizardError, match="another Super Processor window"):
        controller.open_job(job_id)
    # Once that process has gone, the file it left behind means nothing.
    alive.clear()
    controller.open_job(job_id)
    assert controller.api_view()["running_elsewhere"] is False
    assert controller.api_view()["interrupted"] is True


def test_a_process_that_was_given_a_dead_runs_id_does_not_block_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A server that crashed left its file; the id now belongs to another
    # process, which started later than the run that wrote it.
    controller = WizardController(tmp_path / "jobs")
    job_id, job_dir = _job(controller, tmp_path, WizardStep.RENDERING)
    elsewhere = os.getpid() + 1
    (job_dir / WORKER_PID_FILE).write_text(f"{elsewhere} boot:100\n", encoding="utf-8")
    starts = {elsewhere: "boot:999"}
    monkeypatch.setattr("super_processor.wizard._pid_alive", lambda pid: True)
    monkeypatch.setattr("super_processor.wizard._process_start", starts.get)
    (entry,) = controller.jobs_view()
    assert entry["job_id"] == job_id and entry["status"] != "running_elsewhere"
    assert controller.api_view()["running_elsewhere"] is False
    controller.open_job(job_id)
    controller.resume()
    # The run that wrote it, still going, does block the job.
    starts[elsewhere] = "boot:100"
    assert controller.jobs_view()[0]["status"] == "running_elsewhere"
    # And so does a file whose start cannot be checked here.
    starts.clear()
    assert controller.jobs_view()[0]["status"] == "running_elsewhere"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_a_process_start_tells_two_live_processes_apart() -> None:
    from super_processor.wizard import _process_start

    mine = _process_start(os.getpid())
    assert mine is not None and mine == _process_start(os.getpid())
    # A child started a few clock ticks after us has a start of its own.
    # The parent is no use here: a shell can fork pytest on its own tick.
    time.sleep(0.05)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert _process_start(child.pid) not in (None, mine)
    finally:
        child.kill()
        child.wait()
    assert _process_start(0) is None
    assert _process_start(2**22 + 12345) is None


@pytest.mark.parametrize("text", ["", "not a pid", "-4", "0"])
def test_an_unreadable_process_id_file_is_ignored(tmp_path: Path, text: str) -> None:
    controller = WizardController(tmp_path / "jobs")
    job_id, job_dir = _job(controller, tmp_path, WizardStep.RESULT)
    (job_dir / WORKER_PID_FILE).write_text(text, encoding="utf-8")
    (entry,) = controller.jobs_view()
    assert entry["job_id"] == job_id and entry["status"] == "paused"
