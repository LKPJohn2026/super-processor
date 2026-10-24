"""Going back between steps, choosing or dropping a file, and the upscale choice."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from http.client import HTTPConnection
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import pytest
from loop_driver import analyze, pick_and_segment, preview, to_ready

from super_processor.file_dialog import FileDialogError, choose_video
from super_processor.review import WizardServer
from super_processor.shot_plan import load_plan
from super_processor.upscale import (
    FakeUpscaleEngine,
    UpscaleRequest,
    load_upscale_plan,
    video_stream_facts,
)
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    save_session_state,
)
from super_processor.wizard_pages import with_navigation

TOKEN = "test-token"
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg required"
)


def _clip(path: Path, seconds: int = 6) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=320x180:rate=12:duration={seconds}",
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
        self.proposed_upscale = True

    def propose_segments(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {"start_s": 0.0, "end_s": 3.0, "label": "first", "contains": []},
            {"start_s": 3.0, "end_s": 6.0, "label": "second", "contains": []},
        ]

    def propose_looks(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "scale": 2,
            "shots": [
                {
                    "index": shot["index"],
                    "strength": 0.4,
                    "use_upscale": self.proposed_upscale,
                    "look": {},
                    "reason": "ok",
                }
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


def _controller(tmp_path: Path, engine: FakeUpscaleEngine | None = None) -> Any:
    return WizardController(
        tmp_path / "jobs",
        gemini=_Gemini(),  # type: ignore[arg-type]
        upscale_engine=engine or FakeUpscaleEngine(),
    )


# File dialog ---------------------------------------------------------------


def _done(code: int, out: str = "", err: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], code, stdout=out, stderr=err)


def test_linux_dialog_goes_through_the_portal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "super_processor.file_dialog.shutil.which", lambda name: f"/usr/bin/{name}"
    )
    seen: list[tuple[list[str], dict[str, str]]] = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append((argv, kwargs["env"]))
        return _done(0, "/videos/trailer.mp4\n")

    assert choose_video(Path("/videos"), run=run) == Path("/videos/trailer.mp4")
    argv, env = seen[0]
    assert argv[0] == "zenity" and "--file-selection" in argv
    assert env["GTK_USE_PORTAL"] == "1"
    assert any("*.mp4" in arg for arg in argv)


def test_dialog_cancel_falls_back_and_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "super_processor.file_dialog.shutil.which", lambda name: f"/usr/bin/{name}"
    )
    assert choose_video(run=lambda argv, **kw: _done(1)) is None

    calls: list[str] = []

    def broken(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv[0])
        if argv[0] == "zenity":
            raise OSError("no display")
        return _done(0, "/v/b.mkv")

    assert choose_video(run=broken) == Path("/v/b.mkv")
    assert calls == ["zenity", "kdialog"]
    with pytest.raises(FileDialogError, match="dialog failed"):
        choose_video(run=lambda argv, **kw: _done(2, err="cannot open display"))
    monkeypatch.setattr("super_processor.file_dialog.shutil.which", lambda name: None)
    with pytest.raises(FileDialogError, match="no system file dialog"):
        choose_video(run=lambda argv, **kw: _done(0, "x"))


@pytest.mark.parametrize(
    ("platform", "program"), [("darwin", "osascript"), ("win32", "powershell")]
)
def test_other_desktops_have_their_own_dialog(
    monkeypatch: pytest.MonkeyPatch, platform: str, program: str
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    seen: list[str] = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append(argv[0])
        return _done(0, "/Users/me/clip.mov")

    assert choose_video(Path("/Users/me"), run=run) == Path("/Users/me/clip.mov")
    assert seen == [program]


# Navigation ----------------------------------------------------------------


def test_navigation_turns_rail_steps_into_links() -> None:
    page = (
        '<span class="step done" data-phase="setup">Setup</span>'
        '<span class="step done" data-phase="file">File</span>'
        '<span class="step active" data-phase="segments">Segments</span>'
        "<!--back:file--><!--back:setup-->"
    )
    linked = with_navigation(page, {"setup": True, "file": False, "segments": False})
    assert 'action="/goto"' in linked and 'value="setup"' in linked
    assert '<span class="step done" data-phase="file">File</span>' in linked
    assert "← Back to Gemini setup" in linked
    assert "<!--back" not in linked and "Choose another file" not in linked


@needs_ffmpeg
def test_go_back_and_forward_between_setup_file_and_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    controller = _controller(tmp_path)
    clip = _clip(tmp_path / "in.mp4")
    pick_and_segment(controller, clip)
    assert controller.navigation() == {"setup": True, "file": True, "segments": False}
    page = controller.render()
    assert 'name="to" value="file"' in page and "Choose another file" in page

    controller.handle_post("/goto", {"to": ["file"]})
    assert controller.current_state().step is WizardStep.PICK_FILE
    assert controller.navigation()["segments"] is True
    controller.handle_post("/goto", {"to": ["setup"]})
    assert controller.current_state().step is WizardStep.SETUP
    controller.handle_post("/goto", {"to": ["segments"]})
    assert controller.current_state().step is WizardStep.SEGMENTS

    # Back from a preview keeps the approved work; approving resumes there.
    controller.handle_post("/segments", {"action": ["approve"]})
    analyze(controller)
    controller.handle_post("/preview", {"action": ["approve"]})
    assert controller.current_state().segment_index == 1
    controller.handle_post("/goto", {"to": ["segments"]})
    assert controller.current_state().step is WizardStep.SEGMENTS
    controller.handle_post("/segments", {"action": ["approve"]})
    assert controller.current_state().segment_index == 1

    controller.handle_post("/goto", {"to": ["nowhere"]})
    assert "cannot go to nowhere" in (controller.current_state().error or "")


def test_no_navigation_before_setup(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    assert controller.navigation() == {"setup": False, "file": False, "segments": False}
    controller.handle_post("/goto", {"to": ["setup"]})
    assert controller.current_state().step is WizardStep.INTRO


# Choose or drop a file -----------------------------------------------------


@needs_ffmpeg
def test_browse_picks_what_the_dialog_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _clip(tmp_path / "in.mp4")
    controller = _controller(tmp_path)
    save_session_state(controller.jobs_dir, WizardState(step=WizardStep.PICK_FILE))
    monkeypatch.setattr("super_processor.wizard.choose_video", lambda: None)
    controller.handle_post("/browse", {})
    assert controller.current_state().step is WizardStep.PICK_FILE

    def no_dialog() -> Path:
        raise FileDialogError("no system file dialog is available here")

    monkeypatch.setattr("super_processor.wizard.choose_video", no_dialog)
    controller.handle_post("/browse", {})
    assert "no system file dialog" in (controller.current_state().error or "")
    monkeypatch.setattr("super_processor.wizard.choose_video", lambda: clip)
    controller.handle_post("/browse", {})
    assert controller.current_state().step is WizardStep.SAMPLING


def _upload(
    base: str, body: bytes, name: str, *, token: str = TOKEN
) -> tuple[int, dict[str, Any]]:
    parsed = urlparse(base)
    connection = HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port)
    connection.request(
        "POST",
        "/api/upload",
        body=body,
        headers={
            "X-Wizard-Token": token,
            "X-File-Name": quote(name),
            "Content-Type": "application/octet-stream",
        },
    )
    response = connection.getresponse()
    data = json.loads(response.read().decode())
    return response.status, data


@needs_ffmpeg
def test_dropped_file_is_uploaded_and_picked(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "in.mp4")
    controller = _controller(tmp_path)
    save_session_state(controller.jobs_dir, WizardState(step=WizardStep.PICK_FILE))
    server = WizardServer(controller.jobs_dir, token=TOKEN, controller=controller)
    base = server.start()
    try:
        status, data = _upload(base, b"x", "a.mp4", token="wrong")
        assert status == 403
        status, data = _upload(base, b"x", "notes.txt")
        assert status == 415 and "only video files" in data["error"]
        status, data = _upload(base, b"not a video", "broken.mp4")
        assert status == 200 and data["step"] == "pick_file"
        assert data["error"].startswith("broken.mp4 could not be read as a video")
        uploads = controller.jobs_dir / "uploads"
        assert list(uploads.iterdir()) == []
        status, data = _upload(base, clip.read_bytes(), "My Trailer (final).mp4")
        assert status == 200 and data["step"] == "sampling", data
        (kept,) = uploads.iterdir()
        assert kept.name.endswith("-My_Trailer__final_.mp4")
        assert kept.read_bytes() == clip.read_bytes()
        page = controller.render()
        assert "Finding the segments" in page
    finally:
        server.stop()


def test_pick_page_offers_dialog_drop_and_path() -> None:
    from super_processor.wizard_pages import render_pick

    page = render_pick()
    assert 'action="/browse"' in page and 'id="dropzone"' in page
    assert 'xhr.open("POST", "/upload")' in page
    assert "<summary>Type a path instead</summary>" in page


# The upscale choice --------------------------------------------------------


@needs_ffmpeg
def test_editor_can_turn_flashvsr_off_for_a_segment(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    job_dir = to_ready(controller, _clip(tmp_path / "in.mp4"))
    controller.handle_post("/goto", {"to": ["segments"]})
    controller.handle_post("/segments", {"action": ["approve"]})
    state = analyze(controller)
    assert state.step is WizardStep.READY_TO_RENDER

    controller = _controller(tmp_path / "again")
    pick_and_segment(controller, _clip(tmp_path / "in2.mp4"))
    controller.handle_post("/segments", {"action": ["approve"]})
    analyze(controller)
    shot = preview(controller)["shot"]
    assert shot["use_upscale"] is True and shot["upscale_override"] is None
    assert "Gemini chose: this segment uses FlashVSR" in controller.render()
    controller.handle_post("/preview", {"action": ["upscale"], "on": ["0"]})
    assert controller.current_state().step is WizardStep.ANALYZING
    analyze(controller)
    shot = preview(controller)["shot"]
    # Gemini proposed FlashVSR again; the editor's choice holds.
    assert shot["use_upscale"] is False and shot["upscale_override"] is False
    page = controller.render()
    assert "You chose: this segment skips FlashVSR" in page
    assert "Use FlashVSR for this segment" in page
    assert job_dir.is_dir()


@needs_ffmpeg
def test_ready_screen_offers_the_upscale_choice(tmp_path: Path) -> None:
    engine = _CountingEngine()
    controller = _controller(tmp_path, engine)
    clip = _clip(tmp_path / "in.mp4")
    job_dir = to_ready(controller, clip)
    view = controller.api_view()["looks"]
    assert view["scales"] == [1, 2, 4] and view["source_size"] == [320, 180]
    page = controller.render()
    assert "Don't upscale" in page and "Upscale 4×" in page and "1280×720" in page
    assert 'name="scale" value="2" checked' in page

    controller.handle_post("/ready", {"action": ["render"], "scale": ["8"]})
    assert "pick one of the upscale options" in (controller.current_state().error or "")
    controller.handle_post("/ready", {"action": ["render"], "scale": ["1"]})
    assert controller.current_state().step is WizardStep.RENDERING
    plan = load_plan(job_dir)
    upscale = load_upscale_plan(job_dir)
    assert plan is not None and plan.scale == 1
    assert upscale is not None and {span.scale for span in upscale.spans} == {1}
    engine.calls = 0
    controller.run_render()
    assert controller.current_state().step is WizardStep.RESULT
    assert engine.calls == 0
    out = job_dir / "output.mp4"
    assert video_stream_facts(out)[0] == video_stream_facts(clip)[0]
    size = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert size == "320,180"
