"""Segment loop through the wizard controller, with fake Gemini and GPU."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from http.client import HTTPConnection
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
from loop_driver import to_ready

from super_processor.review import WizardServer
from super_processor.segment_loop import (
    LoopSegment,
    SegmentList,
    SegmentLoopError,
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
from super_processor.shots import Shot, ShotList, load_shots, save_shots, shots_path
from super_processor.upscale import (
    FakeUpscaleEngine,
    UpscaleError,
    UpscaleRequest,
    UpscaleSpan,
    load_upscale_plan,
    save_upscale_plan,
)
from super_processor.wizard import (
    WizardController,
    WizardError,
    WizardState,
    WizardStep,
    save_job_wizard_state,
    save_session_state,
    strength_summary,
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
    assert view["segments"][0]["end_s"] == 3.5 and view["approved"] is False
    for item in view["segments"]:
        assert item["still_url"].startswith("/segment_stills/t_")

    controller.handle_post("/segments", {"action": ["note"], "note": [" "]})
    assert controller.current_state().error is not None
    controller.handle_post(
        "/segments", {"action": ["note"], "note": ["cut  at the zoom"]}
    )
    assert controller.current_state().step is WizardStep.PROPOSING
    assert _run(controller, "start_segmenting").step is WizardStep.SEGMENTS
    assert gemini.notes == ["", "cut at the zoom"]
    # The new proposal is a new list: a page showing the first one is stale.
    assert controller.api_view()["segments"]["revision"] == 1

    controller.handle_post("/segments", {"action": ["approve"]})
    state = _run(controller, "start_analyze")
    assert state.step is WizardStep.PREVIEW_SEGMENT
    assert gemini.looks_calls == [[0]]
    preview = controller.api_view()["segment_preview"]
    assert (preview["index"], preview["count"]) == (0, 2)
    assert preview["shot"]["after_url"].partition("?")[0].endswith("_r1_after.mp4")
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
    looks = controller.api_view()["looks"]
    assert [shot["label"] for shot in looks["shots"]] == ["first", "second"]
    # Both sizes are offered, with the plan's scale chosen.
    assert looks["scales"] == [1, 2] and looks["scale"] in looks["scales"]

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


class _Crashing(_Gemini):
    """A bug in the client: not a GeminiError, so nothing below catches it."""

    def propose_segments(self, **kwargs: Any) -> list[dict[str, Any]]:
        raise KeyError("segments")


@needs_ffmpeg
def test_a_crash_while_finding_segments_fails_the_job_and_goes_back_to_pick(
    tmp_path: Path,
) -> None:
    from super_processor.jobs import JobState

    controller = WizardController(
        tmp_path / "jobs",
        gemini=_Crashing(),  # type: ignore[arg-type]
        upscale_engine=FakeUpscaleEngine(),
    )
    save_session_state(controller.jobs_dir, WizardState(step=WizardStep.PICK_FILE))
    controller.handle_post("/pick", {"path": [str(_clip(tmp_path / "in.mp4"))]})
    job_id = controller.current_state().job_id
    assert job_id
    state = _run(controller, "start_segmenting")
    # Not left on "Finding the segments" with a page that refreshes forever.
    assert (state.step, state.job_id) == (WizardStep.PICK_FILE, None)
    assert state.error == "Finding the segments failed: 'segments'"
    assert state.progress is None
    assert controller.store.load(job_id).state is JobState.FAILED
    view = controller.api_view()
    assert view["step"] == "pick_file"
    assert view["error"] == "Finding the segments failed: 'segments'"


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
        still = view["segments"]["segments"][0]["still_url"]
        connection = HTTPConnection("127.0.0.1", urlparse(base).port)
        connection.request("GET", still, headers={"X-Wizard-Token": TOKEN})
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "image/jpeg"
        response.read()
        for bad in ("/segment_stills/../segments.json", "/segment_stills/t_1.jpg"):
            connection.request("GET", bad, headers={"X-Wizard-Token": TOKEN})
            refused = connection.getresponse()
            refused.read()
            assert refused.status == 404
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
    first_url = controller.api_view()["looks"]["shots"][0]["after_url"]
    append_chat_turn(job_dir, ChatTurn(role="model", text='{"segments": []}'))
    controller.handle_post("/ready", {"action": ["render"]})
    controller.run_render()
    assert controller.current_state().step is WizardStep.RESULT
    view = controller.api_view()
    assert view["step"] == "result" and view["output_ready"] is True
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

    # The new plan's first preview has the same file name as the old one's;
    # its URL is still new, so a browser cannot show the cached old one.
    controller.handle_post("/segments", {"action": ["approve"]})
    assert _run(controller, "start_analyze").step is WizardStep.PREVIEW_SEGMENT
    again = controller.api_view()["segment_preview"]["shot"]["after_url"]
    assert again.partition("?")[0] == first_url.partition("?")[0]
    assert again != first_url


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


@needs_ffmpeg
def test_render_worker_shows_frames_done(tmp_path: Path) -> None:
    seen: list[Any] = []

    class _Watching(FakeUpscaleEngine):
        def upscale(self, request: UpscaleRequest) -> Path:
            if request.on_progress is not None:
                request.on_progress(0.5)
                seen.append(controller.api_view()["progress"])
            return super().upscale(request)

    controller = WizardController(
        tmp_path / "jobs",
        gemini=_Gemini(),  # type: ignore[arg-type]
        upscale_engine=_Watching(),
    )
    to_ready(controller, _clip(tmp_path / "in.mp4"))
    controller.handle_post("/ready", {"action": ["render"]})
    assert _run(controller, "start_render").step is WizardStep.RESULT
    # Both segments share settings, so they render as one part.
    assert len(seen) == 1
    shown = seen[0]
    assert {k: shown[k] for k in ("phase", "pct", "message")} == {
        "phase": "rendering",
        "pct": 47,
        "message": "Upscaling on the GPU: 50% of the work",
    }
    # 47% of the render step is that far into the render's share of the job.
    assert shown["overall"] == round(38 + 62 * 0.475)
    # One report is not enough to know the speed yet.
    assert shown["eta_s"] is None
    assert shown["elapsed_s"] >= 0
    assert controller.current_state().progress is None


@needs_ffmpeg
def test_only_a_render_from_a_result_note_counts_on_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from super_processor import wizard as wizard_module

    seen: list[Any] = []

    class _Watching(FakeUpscaleEngine):
        def upscale(self, request: UpscaleRequest) -> Path:
            if request.on_progress is not None:
                request.on_progress(0.5)
                seen.append(controller.api_view()["progress"]["overall"])
            return super().upscale(request)

    controller = WizardController(
        tmp_path / "jobs",
        gemini=_Gemini(),  # type: ignore[arg-type]
        upscale_engine=_Watching(),
    )
    to_ready(controller, _clip(tmp_path / "in.mp4"))
    controller.handle_post("/ready", {"action": ["render"]})
    assert _run(controller, "start_render").step is WizardStep.RESULT
    state = controller.current_state()
    assert state.job_id
    job_dir = controller.store.job_dir(state.job_id)
    assert (job_dir / "output.mp4").is_file()

    # Rendering again with an output on disk, but not from a note (the job
    # went back through the segments): still the render band of the job,
    # even though /ready now keeps the plan the output holds.
    state.step = WizardStep.READY_TO_RENDER
    controller._save(state)
    controller.handle_post("/ready", {"action": ["render"]})
    assert (job_dir / wizard_module._PLAN_BEFORE_NOTE).is_file()
    assert _run(controller, "start_render").step is WizardStep.RESULT
    # A result note leaves a pending span on the plan: a revise.
    plan = load_upscale_plan(job_dir)
    assert plan is not None
    save_upscale_plan(job_dir, replace(plan, pending=plan.spans[0]))
    state = controller.current_state()
    state.step = WizardStep.RENDERING
    controller._save(state)
    assert _run(controller, "start_render").step is WizardStep.RESULT
    first, again, revise = seen
    assert first == again == round(38 + 62 * 0.475)
    assert revise == round(47.5)


def test_task_progress_saves_again_after_a_while_within_one_percent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from super_processor import wizard as wizard_module

    controller = WizardController(
        tmp_path / "jobs",
        gemini=_Gemini(),  # type: ignore[arg-type]
        upscale_engine=FakeUpscaleEngine(),
    )
    state = controller.current_state()
    saves: list[str] = []
    monkeypatch.setattr(
        controller,
        "_progress",
        lambda *args, **kwargs: saves.append(args[3]),
    )
    now = [1000.0]
    monkeypatch.setattr("super_processor.wizard.time.time", lambda: now[0])
    report = controller._task_progress(
        state, "proposing", 0, 10, "gemini-upload", lambda done: f"{done:.3f}"
    )
    report(0.50)
    report(0.51)  # same whole percent, no time passed: not saved
    now[0] += wizard_module._PROGRESS_SAVE_S + 0.1
    report(0.52)  # same whole percent, but the time left would go stale
    report(0.80)  # a new whole percent
    assert saves == ["0.500", "0.520", "0.800"]


class _Gemini4x(_Gemini):
    """Asks for 4x, as Gemini did for the trailer, and records what it was offered."""

    def __init__(self) -> None:
        super().__init__()
        self.offered: list[tuple[int, ...]] = []

    def propose_looks(self, **kwargs: Any) -> dict[str, Any]:
        self.offered.append(tuple(kwargs["scales"]))
        return {**super().propose_looks(**kwargs), "scale": 4}


class _ScaleEngine(FakeUpscaleEngine):
    def __init__(self) -> None:
        super().__init__()
        self.scales: list[int] = []

    def upscale(self, request: UpscaleRequest) -> Path:
        self.scales.append(request.scale)
        return super().upscale(request)


@needs_ffmpeg
def test_previews_are_at_the_plan_scale_and_4x_is_ignored(tmp_path: Path) -> None:
    # 4x was removed: Gemini still answering 4 leaves the plan and its
    # previews at 2x.
    clip = _clip(tmp_path / "in.mp4")
    gemini = _Gemini4x()
    engine = _ScaleEngine()
    controller = WizardController(
        tmp_path / "jobs",
        gemini=gemini,  # type: ignore[arg-type]
        upscale_engine=engine,
    )
    save_session_state(controller.jobs_dir, WizardState(step=WizardStep.PICK_FILE))
    controller.handle_post("/pick", {"path": [str(clip)]})
    assert _run(controller, "start_segmenting").step is WizardStep.SEGMENTS
    controller.handle_post("/segments", {"action": ["approve"]})
    assert _run(controller, "start_analyze").step is WizardStep.PREVIEW_SEGMENT

    assert gemini.offered == [(2,)]
    assert engine.scales == [2]
    state = controller.current_state()
    assert state.job_id
    plan = load_plan(controller.store.job_dir(state.job_id))
    assert plan is not None and plan.scale == 2


def test_result_strength_describes_every_restored_span() -> None:
    # Run 9 ended on credits that skip upscaling, and the Result line read
    # "strength 0.00" although the restored segments ran at 0.30-0.45.
    spans = (
        UpscaleSpan(0.0, 6.9, strength=0.45),
        UpscaleSpan(6.9, 12.8, strength=0.3),
        UpscaleSpan(12.8, 17.7, strength=0.4),
        UpscaleSpan(42.3, 52.2, strength=0.0, use_upscale=False),
    )
    summary = strength_summary(spans)
    assert summary == "strength 0.30–0.45 · upscaling off for 9.9 s"


def test_result_strength_for_one_recipe_or_none() -> None:
    # A revision split leaves equal strengths; they read as one value.
    same = (UpscaleSpan(0.0, 2.0, strength=0.4), UpscaleSpan(2.0, 5.0, strength=0.4))
    assert strength_summary(same) == "strength 0.40"
    off = (UpscaleSpan(0.0, 5.0, use_upscale=False),)
    assert strength_summary(off) == "upscaling off"


@needs_ffmpeg
def test_a_failed_render_after_going_back_to_the_segments_keeps_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A result, then back to the segments and a new render at another scale.
    # The new render fails before it replaces output.mp4, so the result
    # screen comes back with the old output and the plan that output holds.
    controller = WizardController(
        tmp_path / "jobs",
        gemini=_Gemini(),  # type: ignore[arg-type]
        upscale_engine=FakeUpscaleEngine(),
    )
    job_dir = to_ready(controller, _clip(tmp_path / "in.mp4"))
    controller.handle_post("/ready", {"action": ["render"]})
    assert _run(controller, "start_render").step is WizardStep.RESULT
    output = (job_dir / "output.mp4").read_bytes()
    plan_file = job_dir / "upscale_plan.json"
    held = plan_file.read_text(encoding="utf-8")
    assert json.loads(held)["spans"][0]["scale"] == 2

    controller.handle_post("/goto", {"to": ["segments"]})
    assert controller.current_state().step is WizardStep.SEGMENTS
    controller.handle_post("/segments", {"action": ["approve"]})
    state = controller.current_state()
    while state.step in (WizardStep.ANALYZING, WizardStep.PREVIEW_SEGMENT):
        if state.step is WizardStep.ANALYZING:
            controller.run_analyze()
        else:
            controller.handle_post("/preview", {"action": ["approve"]})
        state = controller.current_state()
    assert state.step is WizardStep.READY_TO_RENDER, state.error
    controller.handle_post("/ready", {"action": ["render"], "scale": ["1"]})
    assert json.loads(plan_file.read_text(encoding="utf-8"))["spans"][0]["scale"] == 1

    def _no_gpu(*_args: Any, **_kwargs: Any) -> Path:
        raise UpscaleError("SPAN needs an NVIDIA GPU with CUDA")

    monkeypatch.setattr("super_processor.wizard.render_plan", _no_gpu)
    state = _run(controller, "start_render")
    assert state.step is WizardStep.RESULT
    assert state.error is not None and "CUDA" in state.error
    assert (job_dir / "output.mp4").read_bytes() == output
    assert plan_file.read_text(encoding="utf-8") == held
    assert controller.api_view()["run"]["outcome"] == "failed"


@needs_ffmpeg
def test_a_failure_after_the_output_is_written_keeps_the_new_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The new output is in place when the seam report cannot be written:
    # the result screen shows it, and the plan stays the one it holds.
    from super_processor import wizard as wizard_module

    controller = WizardController(
        tmp_path / "jobs",
        gemini=_Gemini(),  # type: ignore[arg-type]
        upscale_engine=FakeUpscaleEngine(),
    )
    job_dir = to_ready(controller, _clip(tmp_path / "in.mp4"))
    controller.handle_post("/ready", {"action": ["render"]})
    assert _run(controller, "start_render").step is WizardStep.RESULT
    plan = load_upscale_plan(job_dir)
    assert plan is not None
    # A result note: the old plan is kept beside the revised one.
    revised = replace(plan, pending=replace(plan.spans[0], strength=0.9))
    (job_dir / wizard_module._PLAN_BEFORE_NOTE).write_text(
        json.dumps(plan.to_dict()), encoding="utf-8"
    )
    save_upscale_plan(job_dir, revised)
    state = controller.current_state()
    state.step = WizardStep.RENDERING
    controller._save(state)
    before = (job_dir / "output.mp4").stat().st_mtime_ns

    def _full_disk(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(controller, "_check_seams", _full_disk)
    state = _run(controller, "start_render")
    assert state.step is WizardStep.RESULT
    assert state.error == "no space left on device"
    assert (job_dir / "output.mp4").stat().st_mtime_ns != before
    assert not (job_dir / wizard_module._PLAN_BEFORE_NOTE).exists()
    kept = load_upscale_plan(job_dir)
    assert kept is not None and kept.pending is None
    assert kept.spans == revised.spans


# Scene edits by hand ---------------------------------------------------------


@needs_ffmpeg
def test_new_segments_from_a_note_get_new_preview_urls(tmp_path: Path) -> None:
    controller = WizardController(
        tmp_path / "jobs",
        gemini=_Gemini(),  # type: ignore[arg-type]
        upscale_engine=FakeUpscaleEngine(),
    )
    save_session_state(controller.jobs_dir, WizardState(step=WizardStep.PICK_FILE))
    controller.handle_post("/pick", {"path": [str(_clip(tmp_path / "in.mp4"))]})
    assert _run(controller, "start_segmenting").step is WizardStep.SEGMENTS
    controller.handle_post("/segments", {"action": ["approve"]})
    assert _run(controller, "start_analyze").step is WizardStep.PREVIEW_SEGMENT
    first = controller.api_view()["segment_preview"]["shot"]
    controller.handle_post("/goto", {"to": ["segments"]})
    controller.handle_post("/segments", {"action": ["note"], "note": ["cut later"]})
    assert _run(controller, "start_segmenting").step is WizardStep.SEGMENTS
    controller.handle_post("/segments", {"action": ["approve"]})
    assert _run(controller, "start_analyze").step is WizardStep.PREVIEW_SEGMENT
    again = controller.api_view()["segment_preview"]["shot"]
    for side in ("before_url", "after_url"):
        assert again[side].partition("?")[0] == first[side].partition("?")[0]
        assert again[side] != first[side]
    # The poster still of the same plan is there to serve.
    state = controller.current_state()
    assert state.job_id
    poster = again["after_url"].partition("?")[0].replace("_after.mp4", "_after.jpg")
    assert (controller.store.job_dir(state.job_id) / poster.lstrip("/")).is_file()


def _segments_job(
    tmp_path: Path, segments: list[LoopSegment], *, approved: bool = False
) -> tuple[WizardController, Path]:
    """A job on the Segments screen with ``segments`` saved; no FFmpeg."""
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not a real video")
    controller = WizardController(tmp_path / "jobs", gemini=_Gemini())  # type: ignore[arg-type]
    job_id = controller.store.create(source).job_id
    job_dir = controller.store.job_dir(job_id)
    save_segment_list(job_dir, SegmentList(segments=segments, approved=approved))
    controller._save(WizardState(step=WizardStep.SEGMENTS, job_id=job_id))
    return controller, job_dir


def _twelve_seconds() -> list[LoopSegment]:
    return [
        LoopSegment(0.0, 5.0, label="kitchen"),
        LoopSegment(5.0, 12.0, label="street"),
    ]


def _saved(job_dir: Path) -> SegmentList:
    found = load_segment_list(job_dir)
    assert found is not None
    return found


def test_scenes_can_be_moved_split_and_joined_by_hand(tmp_path: Path) -> None:
    controller, job_dir = _segments_job(tmp_path, _twelve_seconds())
    view = controller.api_view()["segments"]
    assert (view["revision"], view["approved"], view["duration_s"]) == (0, False, 12.0)
    assert (view["min_segment_s"], view["max_segments"]) == (3.0, 10)
    assert [item["index"] for item in view["segments"]] == [0, 1]

    post = controller.handle_post
    post(
        "/segments",
        {"action": ["move"], "index": ["0"], "at": ["6"], "revision": ["0"]},
    )
    post(
        "/segments",
        {"action": ["split"], "index": ["1"], "at": ["9"], "revision": ["1"]},
    )
    found = _saved(job_dir)
    assert [(s.start_s, s.end_s) for s in found.segments] == [
        (0.0, 6.0),
        (6.0, 9.0),
        (9.0, 12.0),
    ]
    assert found.revision == 2
    post("/segments", {"action": ["merge"], "index": ["0"], "revision": ["2"]})
    found = _saved(job_dir)
    assert [(s.start_s, s.end_s, s.label) for s in found.segments] == [
        (0.0, 9.0, "kitchen"),
        (9.0, 12.0, "street"),
    ]
    assert (found.revision, controller.current_state().error) == (3, None)
    assert controller.api_view()["segments"]["revision"] == 3


def test_an_edit_the_rules_refuse_shows_why_and_changes_nothing(
    tmp_path: Path,
) -> None:
    controller, job_dir = _segments_job(tmp_path, _twelve_seconds())
    before = (job_dir / "segments.json").read_text(encoding="utf-8")
    for fields in (
        {"action": ["split"], "index": ["0"], "at": ["2"]},
        {"action": ["move"], "index": ["0"], "at": ["10"]},
        {"action": ["move"], "index": ["0"], "at": ["soon"]},
        {"action": ["merge"], "index": ["1"]},
    ):
        controller.handle_post("/segments", {**fields, "revision": ["0"]})
        state = controller.current_state()
        assert state.step is WizardStep.SEGMENTS and state.error, fields
    assert (job_dir / "segments.json").read_text(encoding="utf-8") == before
    # An index the page could not have sent is refused outright.
    for raw in ("-1", "1e2", "٣", "1000", ""):
        with pytest.raises(WizardError, match="pick one of the scenes"):
            controller.handle_post(
                "/segments", {"action": ["merge"], "index": [raw], "revision": ["0"]}
            )


def test_an_edit_from_a_page_showing_older_scenes_is_refused(tmp_path: Path) -> None:
    controller, job_dir = _segments_job(tmp_path, _twelve_seconds())
    move = {"action": ["move"], "index": ["0"], "at": ["6"]}
    # A change by hand must say which list it changes.
    for revision in ([], [""], ["x"], ["1"]):
        with pytest.raises(WizardError, match="scenes changed"):
            controller.handle_post("/segments", {**move, "revision": revision})
    controller.handle_post("/segments", {**move, "revision": ["0"]})
    with pytest.raises(WizardError, match="scenes changed"):
        controller.handle_post("/segments", {**move, "revision": ["0"]})
    # Approve checks the revision too, when the page sends one, but does not
    # change it.
    with pytest.raises(WizardError, match="scenes changed"):
        controller.handle_post("/segments", {"action": ["approve"], "revision": ["0"]})
    assert not _saved(job_dir).approved
    controller.handle_post("/segments", {"action": ["approve"], "revision": ["1"]})
    found = _saved(job_dir)
    assert (found.approved, found.revision) == (True, 1)


def test_a_stale_revision_over_the_api_is_a_400(tmp_path: Path) -> None:
    controller, _job_dir = _segments_job(tmp_path, _twelve_seconds())
    server = WizardServer(controller.jobs_dir, token=TOKEN, controller=controller)
    base = server.start()
    try:
        parsed = urlparse(base)
        connection = HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port)
        for revision, status in ((0, 200), (0, 400)):
            body = json.dumps(
                {"action": "move", "index": 0, "at": 6.5, "revision": revision}
            ).encode()
            connection.request(
                "POST",
                "/api/segments",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Wizard-Token": TOKEN,
                },
            )
            response = connection.getresponse()
            answer = json.loads(response.read().decode())
            assert response.status == status, answer
        assert "scenes changed" in answer["error"]
    finally:
        server.stop()


def _planned_job(tmp_path: Path) -> tuple[WizardController, Path]:
    """Approved scenes with their shots, plan, and previews on disk."""
    controller, job_dir = _segments_job(tmp_path, _twelve_seconds(), approved=True)
    save_shots(
        job_dir, ShotList(shots=[Shot(0.0, 5.0, label="kitchen"), Shot(5.0, 12.0)])
    )
    save_plan(
        job_dir, ShotPlan(scale=2, recipes=[ShotRecipe(rev=1), ShotRecipe(rev=1)])
    )
    previews = job_dir / "previews"
    previews.mkdir()
    for name in ("shot_000_r1_after.mp4", "shot_000_r1_before.mp4"):
        (previews / name).write_bytes(b"x")
    for name in ("upscale_plan_before_note.json", "render_report.json"):
        (job_dir / name).write_text("{}", encoding="utf-8")
    return controller, job_dir


def test_an_edit_that_changes_nothing_keeps_the_approval(tmp_path: Path) -> None:
    controller, job_dir = _planned_job(tmp_path)
    state = controller.current_state()
    state.segment_index = state.approved_count = 1
    controller._save(state)
    # The cut is already at 5 s.
    controller.handle_post(
        "/segments",
        {"action": ["move"], "index": ["0"], "at": ["5.0001"], "revision": ["0"]},
    )
    found = _saved(job_dir)
    assert (found.approved, found.revision) == (True, 0)
    controller.handle_post("/segments", {"action": ["approve"], "revision": ["0"]})
    state = controller.current_state()
    assert (state.step, state.segment_index, state.approved_count) == (
        WizardStep.ANALYZING,
        1,
        1,
    )
    assert shots_path(job_dir).is_file() and plan_path(job_dir).is_file()


def test_approving_edited_scenes_clears_the_old_plan_even_with_a_preview_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, job_dir = _planned_job(tmp_path)
    controller.handle_post(
        "/segments",
        {"action": ["move"], "index": ["0"], "at": ["6"], "revision": ["0"]},
    )
    assert not _saved(job_dir).approved
    real_unlink = Path.unlink

    def held(self: Path, missing_ok: bool = False) -> None:
        # Windows: a browser still plays the old preview.
        if self.name == "shot_000_r1_after.mp4":
            raise PermissionError(13, "the file is open in another program")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", held)
    controller.handle_post("/segments", {"action": ["approve"], "revision": ["1"]})
    state = controller.current_state()
    assert (state.step, state.error) == (WizardStep.ANALYZING, None)
    assert _saved(job_dir).approved
    for name in (
        "shots.json",
        "shot_plan.json",
        "previews/shot_000_r1_before.mp4",
        "upscale_plan_before_note.json",
        "render_report.json",
    ):
        assert not (job_dir / name).exists(), name
    assert (job_dir / "previews" / "shot_000_r1_after.mp4").is_file()


def test_approving_is_refused_while_the_old_plan_cannot_be_cleared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("super_processor.atomic_io.RETRY_S", 0)
    controller, job_dir = _planned_job(tmp_path)
    controller.handle_post(
        "/segments",
        {"action": ["move"], "index": ["0"], "at": ["6"], "revision": ["0"]},
    )
    real_unlink = Path.unlink

    def held(self: Path, missing_ok: bool = False) -> None:
        if self.name == "shots.json":
            raise PermissionError(13, "the file is open in another program")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", held)
    with pytest.raises(WizardError, match="could not clear"):
        controller.handle_post("/segments", {"action": ["approve"]})
    # Never approved over shots measured from the old cut.
    assert not _saved(job_dir).approved
    assert controller.current_state().step is WizardStep.SEGMENTS


def _approved_whole_video(
    tmp_path: Path, segment_end: float, shot_end: float
) -> tuple[WizardController, Path]:
    controller, job_dir = _segments_job(
        tmp_path, [LoopSegment(0.0, segment_end, label="whole video")], approved=True
    )
    save_shots(job_dir, ShotList(shots=[Shot(0.0, shot_end, label="whole video")]))
    save_plan(job_dir, ShotPlan(scale=2, recipes=[ShotRecipe(rev=1, stale=False)]))
    state = controller.current_state()
    state.step = WizardStep.ANALYZING
    controller._save(state)
    return controller, job_dir


@pytest.mark.parametrize(
    ("segment_end", "shot_end"), [(12.3456789, 12.346), (12.346, 12.3456789)]
)
def test_a_whole_video_segment_is_not_measured_again_over_rounding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    segment_end: float,
    shot_end: float,
) -> None:
    # segments.json from before the fallback was rounded, or shots.json
    # measured from it: the same span to the millisecond.
    def measured(*args: Any, **kwargs: Any) -> ShotList:
        raise AssertionError("the segment was measured again")

    monkeypatch.setattr("super_processor.wizard.shots_from_segments", measured)
    controller, job_dir = _approved_whole_video(tmp_path, segment_end, shot_end)
    controller.run_analyze()
    state = controller.current_state()
    assert state.step is WizardStep.PREVIEW_SEGMENT, state.error
    shots = load_shots(job_dir)
    assert shots is not None
    assert shots.shots[0].end_s == pytest.approx(shot_end, abs=1e-6)


def test_shots_measured_from_other_scenes_are_measured_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measured: list[list[tuple[float, float]]] = []

    def measure(
        source: Path, job_dir: Path, segments: list[LoopSegment], **kwargs: Any
    ) -> ShotList:
        measured.append([(s.start_s, s.end_s) for s in segments])
        raise SegmentLoopError("measured again")

    monkeypatch.setattr("super_processor.wizard.shots_from_segments", measure)
    controller, _job_dir = _approved_whole_video(tmp_path, 12.0, 9.0)
    controller.run_analyze()
    assert measured == [[(0.0, 12.0)]]
    state = controller.current_state()
    assert state.error is not None and "measured again" in state.error
