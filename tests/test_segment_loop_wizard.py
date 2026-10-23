"""Segment loop through the wizard controller, with fake Gemini and GPU."""

from __future__ import annotations

import json
import shutil
import subprocess
from http.client import HTTPConnection
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
from loop_driver import to_ready

from super_processor.review import WizardServer
from super_processor.shot_plan import load_plan
from super_processor.upscale import FakeUpscaleEngine, UpscaleRequest
from super_processor.wizard import (
    WizardController,
    WizardError,
    WizardState,
    WizardStep,
    save_job_wizard_state,
    save_session_state,
)

TOKEN = "test-token"

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
    page = controller.render()
    assert "0:00–0:03.5 · first" in page and "Propose again" in page

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
    page = controller.render()
    assert "Segment 1 of 2" in page and "Preview a different part" in page
    assert preview["shot"]["after_url"] in page
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
    page = controller.render()
    assert "Render the whole video" in page and "· second:" in page

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


def test_workers_start_only_on_their_steps(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    assert controller.start_segmenting() is False
    assert controller.start_analyze() is False


def test_old_shot_flow_states_go_back_to_pick() -> None:
    for step in ("finding_shots", "shots", "planning", "looks"):
        state = WizardState.from_dict({"schema_version": 3, "step": step})
        assert state.step is WizardStep.PICK_FILE
        assert state.error is not None and "shot flow" in state.error


def _call(
    base: str, method: str, path: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    parsed = urlparse(base)
    connection = HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port)
    body = b"" if payload is None else json.dumps(payload).encode()
    headers = {"Accept": "application/json", "X-Wizard-Token": TOKEN}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    assert response.status == 200, raw
    loaded = json.loads(raw.decode())
    assert isinstance(loaded, dict)
    return loaded


@needs_ffmpeg
def test_segment_loop_over_the_json_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _clip(tmp_path / "in.mp4")
    controller = WizardController(
        tmp_path / "jobs",
        gemini=_Gemini(),  # type: ignore[arg-type]
        upscale_engine=FakeUpscaleEngine(),
    )
    save_session_state(controller.jobs_dir, WizardState(step=WizardStep.PICK_FILE))
    server = WizardServer(controller.jobs_dir, token=TOKEN, controller=controller)
    base = server.start()
    try:
        assert _call(base, "POST", "/api/pick", {"path": str(clip)})["step"] == (
            "sampling"
        )
        _call(base, "GET", "/api/segment?run=1")
        assert controller.wait_idle(120)
        view = _call(base, "GET", "/api/state")
        assert view["step"] == "segments" and len(view["segments"]["segments"]) == 2
        assert view["progress"] is None
        _call(base, "POST", "/api/segments", {"action": "approve"})
        for _ in range(2):
            _call(base, "GET", "/api/analyze?run=1")
            assert controller.wait_idle(120)
            view = _call(base, "GET", "/api/state")
            assert view["step"] == "preview_segment"
            assert view["segment_preview"]["shot"]["before_url"]
            _call(base, "POST", "/api/preview", {"action": "approve"})
        view = _call(base, "GET", "/api/state")
        assert view["step"] == "ready_to_render"
        assert _call(base, "POST", "/api/ready", {"action": "render"})["step"] == (
            "rendering"
        )
    finally:
        server.stop()


def test_busy_pages_show_progress() -> None:
    from super_processor.wizard_pages import render_analyzing, render_sampling

    page = render_analyzing(1, 3, {"pct": 140, "message": "Checking <it>"})
    assert "Segment 2 of 3" in page
    assert 'value="100"' in page and "Checking &lt;it&gt;" in page
    assert "/analyze?run=1" in page
    assert "<progress" not in render_sampling(None)
    assert "<progress" not in render_sampling({"pct": "x"})


class _Recording(_Gemini):
    """Records the chat each segment proposal sees, as the real client would."""

    def __init__(self) -> None:
        super().__init__()
        self.chats: list[list[str]] = []

    def propose_segments(self, **kwargs: Any) -> list[dict[str, Any]]:
        from super_processor.gemini import load_chat

        self.chats.append([turn.text for turn in load_chat(kwargs["job_dir"])])
        return super().propose_segments(**kwargs)


@needs_ffmpeg
def test_result_note_can_restart_from_the_segments(tmp_path: Path) -> None:
    from super_processor.gemini import ChatTurn, append_chat_turn

    clip = _clip(tmp_path / "in.mp4")
    gemini = _Recording()
    controller = WizardController(
        tmp_path / "jobs",
        gemini=gemini,  # type: ignore[arg-type]
        upscale_engine=FakeUpscaleEngine(),
    )
    job_dir = to_ready(controller, clip)
    append_chat_turn(job_dir, ChatTurn(role="model", text='{"segments": []}'))
    controller.handle_post("/ready", {"action": ["render"]})
    controller.run_render()
    assert controller.current_state().step is WizardStep.RESULT
    page = controller.render()
    assert 'name="restart" value="1"' in page
    for name in ("segments.json", "shot_plan.json", "upscale_plan.json", "output.mp4"):
        assert (job_dir / name).is_file()
    chat_before = (job_dir / "gemini_chat.json").read_text(encoding="utf-8")

    controller.handle_post("/result", {"note": [""], "restart": ["1"]})
    assert controller.current_state().step is WizardStep.RESULT
    controller.handle_post(
        "/result", {"note": ["too  sharp, faces look fake"], "restart": ["1"]}
    )
    state = controller.current_state()
    assert state.step is WizardStep.SAMPLING
    assert (state.segment_index, state.progress, state.error) == (0, None, None)
    for name in (
        "segments.json",
        "shots.json",
        "shot_plan.json",
        "upscale_plan.json",
        "output.mp4",
        "render_report.json",
        "previews",
        "shot_stills",
        "shot_work",
    ):
        assert not (job_dir / name).exists(), name
    chat = (job_dir / "gemini_chat.json").read_text(encoding="utf-8")
    assert chat.startswith(chat_before.rsplit("]", 1)[0].rstrip())

    assert _run(controller, "start_segmenting").step is WizardStep.SEGMENTS
    assert len(gemini.chats) == 2
    assert any("too sharp, faces look fake" in text for text in gemini.chats[-1])
    assert len(gemini.chats[-1]) == len(gemini.chats[0]) + 2


def test_restart_leaves_symlinked_folders_alone(tmp_path: Path) -> None:
    from super_processor.wizard import restart_from_segmentation

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.mp4").write_bytes(b"x")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "previews").symlink_to(outside, target_is_directory=True)
    (job_dir / "gemini_chat.json").write_text("{}", encoding="utf-8")
    restart_from_segmentation(job_dir)
    assert not (job_dir / "previews").exists()
    assert (outside / "keep.mp4").is_file()
    assert (job_dir / "gemini_chat.json").is_file()
