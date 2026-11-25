"""``super-processor review JOB_ID``: the wizard opened on one job."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

import pytest
from wizard_client import get

from super_processor.jobs import JobStore
from super_processor.review import WizardServer
from super_processor.wizard import (
    WizardController,
    WizardError,
    WizardState,
    WizardStep,
    load_job_wizard_state,
    save_job_wizard_state,
    save_session_state,
)


@pytest.mark.parametrize("session", ["other", "empty", "missing"])
def test_reopened_wizard_uses_requested_job(tmp_path: Path, session: str) -> None:
    store = JobStore(tmp_path / "jobs")
    (tmp_path / "clip.mp4").write_bytes(b"source")
    requested_id = "aaaaaaaaaaaaaaaa"
    other_id = "bbbbbbbbbbbbbbbb"
    for job_id in (requested_id, other_id):
        store.create(tmp_path / "clip.mp4", job_id=job_id)
        job_dir = store.job_dir(job_id)
        save_job_wizard_state(
            job_dir, WizardState(step=WizardStep.RESULT, job_id=job_id)
        )
        (job_dir / "output.mp4").write_bytes(job_id.encode())
    other_state = load_job_wizard_state(store.job_dir(other_id))
    assert other_state is not None
    if session == "other":
        save_session_state(store.root, other_state)
    elif session == "empty":
        save_session_state(store.root, WizardState())

    server = WizardServer(store.root, job_id=requested_id)
    base = server.start()
    try:
        with get(base + "/api/state", server.token) as response:
            assert json.load(response)["job_id"] == requested_id
        with get(base + "/output.mp4", server.token) as response:
            assert response.read() == requested_id.encode()

        # Another window selecting a different job must not redirect this one.
        save_session_state(store.root, other_state)
        request = Request(
            base + "/api/result",
            data=b'{"mood": "happy"}',
            headers={
                "Content-Type": "application/json",
                "X-Wizard-Token": server.token,
            },
            method="POST",
        )
        with urlopen(request) as response:
            view = json.load(response)
        assert view["job_id"] == requested_id
        assert view["step"] == "done"
        requested = load_job_wizard_state(store.job_dir(requested_id))
        assert requested is not None and requested.step is WizardStep.DONE
        assert load_job_wizard_state(store.job_dir(other_id)) == other_state
    finally:
        server.stop()


def test_a_job_reopened_mid_render_waits_to_be_resumed(tmp_path: Path) -> None:
    (tmp_path / "clip.mp4").write_bytes(b"source")
    store = JobStore(tmp_path / "jobs")
    job_id = "aaaaaaaaaaaaaaaa"
    store.create(tmp_path / "clip.mp4", job_id=job_id)
    save_job_wizard_state(
        store.job_dir(job_id), WizardState(step=WizardStep.RENDERING, job_id=job_id)
    )
    server = WizardServer(store.root, job_id=job_id)
    base = server.start()
    try:
        started = Request(
            base + "/api/render?run=1", headers={"X-Wizard-Token": server.token}
        )
        with urlopen(started) as response:
            view = json.load(response)
        assert view["interrupted"] is True and view["busy"] is False
        assert view["step"] == "rendering"
        # The app asks "Carry on where you left off?" until a resume.
        with get(base + "/api/state", server.token) as response:
            assert json.load(response)["interrupted"] is True
    finally:
        server.stop()


def test_reopened_controller_can_start_a_new_session(tmp_path: Path) -> None:
    (tmp_path / "clip.mp4").write_bytes(b"source")
    store = JobStore(tmp_path / "jobs")
    job_id = "aaaaaaaaaaaaaaaa"
    store.create(tmp_path / "clip.mp4", job_id=job_id)
    saved = WizardState(step=WizardStep.DONE, job_id=job_id)
    save_job_wizard_state(store.job_dir(job_id), saved)
    controller = WizardController(store.root, job_id=job_id)

    controller.start_new_job()
    fresh = controller.current_state()
    assert fresh.job_id is None
    assert fresh.step in {WizardStep.PICK_FILE, WizardStep.SETUP}
    save_session_state(store.root, saved)
    assert controller.current_state() == fresh
    assert load_job_wizard_state(store.job_dir(job_id)) == saved


def test_reopened_controller_requires_job_wizard_state(tmp_path: Path) -> None:
    (tmp_path / "clip.mp4").write_bytes(b"source")
    store = JobStore(tmp_path / "jobs")
    job_id = "aaaaaaaaaaaaaaaa"
    store.create(tmp_path / "clip.mp4", job_id=job_id)
    with pytest.raises(WizardError, match="no wizard state"):
        WizardController(store.root, job_id=job_id)
