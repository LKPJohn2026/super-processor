"""Which job files the wizard server hands out, and the setup key check."""

from __future__ import annotations

from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlparse

import pytest

from super_processor.gemini import credentials_path
from super_processor.review import WizardServer
from super_processor.segment_loop import (
    LoopSegment,
    SegmentList,
    SegmentStill,
    save_segment_list,
)
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    save_session_state,
)

TOKEN = "media-token"


def _get(base: str, path: str, method: str = "GET") -> tuple[int, str | None, bytes]:
    parsed = urlparse(base)
    connection = HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port)
    try:
        connection.request(method, path, headers={"X-Wizard-Token": TOKEN})
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

    server = WizardServer(controller.jobs_dir, controller=controller, token=TOKEN)
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


def test_job_scoped_routes_serve_any_jobs_files_by_its_id(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    shown = controller.store.create(source).job_id
    other = controller.store.create(source).job_id
    other_dir = controller.store.job_dir(other)
    files = {
        "output.mp4": b"other output",
        "outputs/v001.mp4": b"first version",
        "outputs/v002.mp4.part": b"half copied",
        "segment_stills/t_00001000.jpg": b"early",
        "segment_stills/t_00005000.jpg": b"middle",
        "segment_stills/t_00009000.jpg": b"late",
        "previews/shot_000_r1_after.mp4": b"after",
    }
    for name, payload in files.items():
        (other_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (other_dir / name).write_bytes(payload)
    outside = tmp_path / "secret.mp4"
    outside.write_bytes(b"secret")
    (other_dir / "outputs" / "v003.mp4").symlink_to(outside)
    save_session_state(
        controller.jobs_dir, WizardState(step=WizardStep.RESULT, job_id=shown)
    )
    server = WizardServer(controller.jobs_dir, controller=controller, token=TOKEN)
    base = server.start()
    try:
        served = {
            f"/jobs/{other}/output.mp4": ("video/mp4", b"other output"),
            f"/jobs/{other}/outputs/v001.mp4": ("video/mp4", b"first version"),
            # The still in the middle of the video stands for the job.
            f"/jobs/{other}/thumb.jpg": ("image/jpeg", b"middle"),
            f"/jobs/{other}/segment_stills/t_00001000.jpg": ("image/jpeg", b"early"),
            f"/jobs/{other}/previews/shot_000_r1_after.mp4": ("video/mp4", b"after"),
        }
        for path, (kind, payload) in served.items():
            assert _get(base, path) == (200, kind, payload), path
            assert _get(base, path, "HEAD")[:2] == (200, kind), path
        for refused in (
            # The shown job has no output, and the bare route is its own.
            f"/jobs/{shown}/output.mp4",
            f"/jobs/{shown}/thumb.jpg",
            "/output.mp4",
            "/thumb.jpg",
            f"/jobs/{other}/outputs/v002.mp4.part",
            f"/jobs/{other}/outputs/v003.mp4",
            f"/jobs/{other}/outputs/../output.mp4",
            f"/jobs/{other}/manifest.json",
            f"/jobs/{other}/wizard_state.json",
            "/jobs/ffffffffffffffff/output.mp4",
            "/jobs/NOT-AN-ID/output.mp4",
            "/jobs/../output.mp4",
        ):
            assert _get(base, refused)[0] == 404, refused
            assert _get(base, refused, "HEAD")[0] == 404, refused
        # Removed after the page got its URL: a plain 404, for HEAD too.
        (other_dir / "output.mp4").unlink()
        assert _get(base, f"/jobs/{other}/output.mp4", "HEAD")[0] == 404
        assert _get(base, f"/jobs/{other}/output.mp4")[0] == 404
    finally:
        server.stop()


@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError])
def test_a_file_that_cannot_be_opened_is_a_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[OSError]
) -> None:
    controller = WizardController(tmp_path / "jobs")
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    job_id = controller.store.create(source).job_id
    (controller.store.job_dir(job_id) / "output.mp4").write_bytes(b"video")
    server = WizardServer(controller.jobs_dir, controller=controller, token=TOKEN)
    base = server.start()
    real_open = Path.open

    def locked(self: Path, *args: object, **kwargs: object) -> object:
        # Removed (or locked by a Windows player) between the check and open.
        if self.name == "output.mp4":
            raise error("in use")
        return real_open(self, *args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(Path, "open", locked)
    try:
        for method in ("GET", "HEAD"):
            path = f"/jobs/{job_id}/output.mp4"
            assert _get(base, path, method)[0] == 404, method
    finally:
        server.stop()


def test_api_errors_are_json(tmp_path: Path) -> None:
    server = WizardServer(tmp_path / "jobs", token=TOKEN)
    base = server.start()
    try:
        status, kind, body = _get(base, "/api/nothing-here")
        assert (status, kind) == (404, "application/json")
        assert b"unknown endpoint" in body
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


def test_the_job_summary_stills_name_their_job_and_are_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = WizardController(tmp_path / "jobs")
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    job_id = controller.store.create(source).job_id
    job_dir = controller.store.job_dir(job_id)
    save_segment_list(
        job_dir,
        SegmentList(
            segments=[LoopSegment(0.0, 4.0, "a"), LoopSegment(4.0, 8.0, "b")],
            approved=True,
        ),
    )
    stills = [
        SegmentStill(t_s=t, path=f"segment_stills/t_{int(t * 1000):08d}.jpg")
        for t in (2.0, 6.0)
    ]
    for still in stills:
        (job_dir / still.path).parent.mkdir(parents=True, exist_ok=True)
        (job_dir / still.path).write_bytes(still.path.encode())
    monkeypatch.setattr(
        "super_processor.wizard.load_segment_stills", lambda _job_dir: stills
    )
    save_session_state(
        controller.jobs_dir,
        WizardState(step=WizardStep.PREVIEW_SEGMENT, job_id=job_id),
    )
    scenes = controller.api_view()["job"]["scenes"]
    urls = [scene["still_url"] for scene in scenes]
    assert urls == [f"/jobs/{job_id}/{still.path}" for still in stills]

    # Another window opens a different job: the URLs still show this one's.
    other = controller.store.create(source).job_id
    save_session_state(
        controller.jobs_dir, WizardState(step=WizardStep.RESULT, job_id=other)
    )
    server = WizardServer(controller.jobs_dir, controller=controller, token=TOKEN)
    base = server.start()
    try:
        for url, still in zip(urls, stills, strict=True):
            assert _get(base, url) == (200, "image/jpeg", still.path.encode())
    finally:
        server.stop()
