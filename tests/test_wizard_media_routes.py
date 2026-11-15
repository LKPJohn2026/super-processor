"""Which job files the wizard server hands out, and the setup key check."""

from __future__ import annotations

from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlparse

import pytest

from super_processor.gemini import credentials_path
from super_processor.review import WizardServer
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    save_session_state,
)


def _get(base: str, path: str) -> tuple[int, str | None, bytes]:
    parsed = urlparse(base)
    connection = HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type"), response.read()
    finally:
        connection.close()


def test_media_routes_serve_only_the_current_jobs_named_files(
    tmp_path: Path,
) -> None:
    controller = WizardController(tmp_path / "jobs")
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    job_id = controller.store.create(source).job_id
    job_dir = controller.store.job_dir(job_id)
    files = {
        "output.mp4": b"output",
        "previews/shot_000_r1_after.mp4": b"after",
        "previews/shot_000_r1_after.jpg": b"poster",
        "previews/notes.mp4": b"not a preview name",
        "segment_stills/t_00001000.jpg": b"grid still",
        "segment_stills/seg_00.ppm": b"P5",
        "shot_stills/000000001.jpg": b"shot still",
    }
    for name, payload in files.items():
        (job_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (job_dir / name).write_bytes(payload)
    # A grid still that points out of the job is never followed.
    outside = tmp_path / "secret.jpg"
    outside.write_bytes(b"secret")
    (job_dir / "segment_stills" / "t_00002000.jpg").symlink_to(outside)

    server = WizardServer(controller.jobs_dir, controller=controller)
    base = server.start()
    try:
        # Before a job is picked there is nothing to serve.
        assert _get(base, "/output.mp4")[0] == 404
        save_session_state(
            controller.jobs_dir, WizardState(step=WizardStep.RESULT, job_id=job_id)
        )
        served = {
            "/output.mp4": ("video/mp4", b"output"),
            "/previews/shot_000_r1_after.mp4": ("video/mp4", b"after"),
            "/previews/shot_000_r1_after.jpg": ("image/jpeg", b"poster"),
            "/segment_stills/t_00001000.jpg": ("image/jpeg", b"grid still"),
            "/shot_stills/000000001.jpg": ("image/jpeg", b"shot still"),
        }
        for path, (kind, payload) in served.items():
            assert _get(base, path) == (200, kind, payload), path
        for refused in (
            "/previews/notes.mp4",
            "/previews/shot_009_r1_after.mp4",
            "/previews/../output.mp4",
            "/segment_stills/t_00002000.jpg",
            # Still names of the removed split flow.
            "/segment_stills/seg_00.ppm",
            "/segment_stills/seg_00.bmp",
            "/clip.mp4",
            "/wizard_state.json",
        ):
            assert _get(base, refused)[0] == 404, refused
    finally:
        server.stop()


@pytest.mark.parametrize(
    "fields", [{"api_key": ["  "]}, {"api_key": [""]}, {"skip": ["1"]}]
)
def test_setup_refuses_an_empty_key_and_saves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fields: dict[str, list[str]]
) -> None:
    # "Use existing key" without a saved key falls through to the empty key.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    controller = WizardController(tmp_path)
    controller.handle_post("/intro", {})
    controller.handle_post("/llm", {"choice": ["gemini"]})
    controller.handle_post("/setup", fields)
    state = controller.current_state()
    assert state.step is WizardStep.SETUP
    assert state.error == "API key is empty"
    assert not credentials_path().exists()
    assert "API key is empty" in controller.render()
