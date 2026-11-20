"""Previewing the next scene in the background (``review --prefetch``)."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from loop_driver import approve_segments, pick_and_segment

from super_processor.gemini import (
    ChatTurn,
    GeminiVideo,
    append_chat_turns,
    load_chat,
)
from super_processor.shot_plan import PREVIEWS_DIR, load_plan
from super_processor.upscale import FakeUpscaleEngine, UpscaleError, UpscaleRequest
from super_processor.wizard import (
    GEMINI_VIDEO_FILE,
    WizardController,
    WizardState,
    WizardStep,
)

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
    """Two scenes; keeps its turns the way the real client does."""

    def propose_segments(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {"start_s": 0.0, "end_s": 3.0, "label": "first", "contains": []},
            {"start_s": 3.0, "end_s": 6.0, "label": "second", "contains": []},
        ]

    def _log(self, kwargs: dict[str, Any], what: str) -> None:
        indexes = [shot["index"] for shot in kwargs["shots"]]
        turns = [
            ChatTurn(role="user", text=f"{what} {indexes}"),
            ChatTurn(role="model", text="{}"),
        ]
        kept = kwargs.get("turns_out")
        if kept is not None:
            kept.extend(turns)
        else:
            append_chat_turns(kwargs["job_dir"], turns)

    def propose_looks(self, **kwargs: Any) -> dict[str, Any]:
        self._log(kwargs, "propose")
        return {
            "scale": 2,
            "shots": [
                {
                    "index": shot["index"],
                    "strength": 0.4,
                    "use_upscale": True,
                    "look": {},
                    "reason": "ok",
                }
                for shot in kwargs["shots"]
            ],
        }

    def check_previews(self, **kwargs: Any) -> list[dict[str, Any]]:
        self._log(kwargs, "check")
        return [{"index": shot["index"], "ok": True} for shot in kwargs["shots"]]


class _VideoGemini(_Gemini):
    """Can upload, and counts each upload."""

    def __init__(self) -> None:
        self.uploads = 0

    def upload_video(self, path: Path, **kwargs: Any) -> GeminiVideo:
        self.uploads += 1
        return GeminiVideo(name="files/new", uri="u", expires_at=time.time() + 1e5)

    def video_alive(self, video: GeminiVideo) -> bool:
        return True


class _HeldGemini(_VideoGemini):
    """Holds the call named by ``hold`` until ``gate`` opens (or 30 s)."""

    def __init__(self) -> None:
        super().__init__()
        self.hold = ""
        self.entered = threading.Event()
        self.gate = threading.Event()

    def _wait(self, name: str) -> None:
        if name == self.hold:
            self.entered.set()
            self.gate.wait(30)

    def video_alive(self, video: GeminiVideo) -> bool:
        self._wait("video_alive")
        return True

    def propose_looks(self, **kwargs: Any) -> dict[str, Any]:
        self._wait("propose_looks")
        return super().propose_looks(**kwargs)

    def check_previews(self, **kwargs: Any) -> list[dict[str, Any]]:
        self._wait("check_previews")
        return super().check_previews(**kwargs)


class _Engine(FakeUpscaleEngine):
    """Records which thread upscales; can hold or fail an upscale."""

    def __init__(self) -> None:
        super().__init__()
        self.threads: list[str] = []
        self.entered = threading.Event()
        self.gate: threading.Event | None = None
        self.fail = False

    def upscale(self, request: UpscaleRequest) -> Path:
        self.threads.append(threading.current_thread().name)
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(10)
        if self.fail:
            raise UpscaleError("the GPU fell over")
        return super().upscale(request)


@pytest.fixture
def made() -> Iterator[list[WizardController]]:
    """Controllers made by a test; each prefetch is dropped and waited for."""
    controllers: list[WizardController] = []
    yield controllers
    for controller in controllers:
        controller.shutdown()
        assert controller.wait_prefetch(5)


def _controller(
    tmp_path: Path,
    made: list[WizardController],
    *,
    gemini: _Gemini | None = None,
    engine: _Engine | None = None,
    prefetch: bool = True,
) -> WizardController:
    controller = WizardController(
        tmp_path / "jobs",
        gemini=gemini or _Gemini(),  # type: ignore[arg-type]
        upscale_engine=engine or _Engine(),
        prefetch=prefetch,
    )
    made.append(controller)
    return controller


def _on_first_scene(controller: WizardController, tmp_path: Path) -> Path:
    """Pick a two-scene clip and preview its first scene. The job dir."""
    job_id = pick_and_segment(controller, _clip(tmp_path / "clip.mp4")).job_id
    assert job_id
    state = approve_segments(controller)
    assert state.step is WizardStep.PREVIEW_SEGMENT, state.error
    assert state.segment_index == 0
    return controller.store.job_dir(job_id)


def _texts(job_dir: Path) -> list[str]:
    return [turn.text for turn in load_chat(job_dir) if turn.role == "user"]


def _scene_files(job_dir: Path, index: int) -> list[str]:
    folder = job_dir / PREVIEWS_DIR
    return sorted(
        path.relative_to(folder).as_posix()
        for path in folder.rglob(f"shot_{index:03d}_*")
    )


def test_prefetch_is_off_unless_asked_for(
    tmp_path: Path, made: list[WizardController]
) -> None:
    controller = _controller(tmp_path, made, prefetch=False)
    assert WizardController(tmp_path / "other").prefetch is False
    # Off, it refuses before it reads anything.
    assert controller.start_prefetch() is False
    assert controller.wait_prefetch(0)


@needs_ffmpeg
def test_the_next_scene_is_previewed_ahead_and_adopted(
    tmp_path: Path, made: list[WizardController], monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _Engine()
    controller = _controller(tmp_path, made, engine=engine)
    job_dir = _on_first_scene(controller, tmp_path)
    saves: list[str] = []
    real_save = controller._save

    def _save(state: WizardState, **kwargs: Any) -> None:
        saves.append(threading.current_thread().name)
        real_save(state, **kwargs)

    monkeypatch.setattr(controller, "_save", _save)
    before = load_plan(job_dir)
    assert before is not None and before.recipes[1].rev == 0

    assert controller.start_prefetch()
    # The same scene is not started twice.
    assert controller.start_prefetch() is False
    assert controller.wait_prefetch(30)
    view = controller.api_view()["segment_preview"]
    assert view["index"] == 0
    assert view["next"] == {"index": 1, "ready": True, "running": False}
    # Made in memory only: the plan, the chat, and the state are untouched.
    after = load_plan(job_dir)
    assert after is not None and after.recipes[1] == before.recipes[1]
    assert not any("[1]" in text for text in _texts(job_dir))
    assert "wizard-prefetch" in engine.threads
    assert "wizard-prefetch" not in saves
    # The work clips were named for that run and removed with it.
    assert not list((job_dir / PREVIEWS_DIR / "work").glob("shot_001_*"))

    controller.handle_post("/preview", {"action": ["approve"]})
    calls = len(engine.threads)
    controller.run_analyze()
    state = controller.current_state()
    assert state.step is WizardStep.PREVIEW_SEGMENT and state.segment_index == 1
    # Adopted: no second preview on the GPU.
    assert len(engine.threads) == calls
    plan = load_plan(job_dir)
    assert plan is not None
    assert plan.recipes[1].rev == 1 and not plan.recipes[1].stale
    assert plan.recipes[1].reason == "ok"
    files = _scene_files(job_dir, 1)
    assert {"shot_001_r1_after.mp4", "shot_001_r1_before.mp4"} <= set(files)
    assert all(name.startswith("shot_001_r1_") for name in files)
    # Gemini's turns for that scene land in the chat with the adoption.
    assert _texts(job_dir)[-2:] == ["propose [1]", "check [1]"]
    assert "wizard-prefetch" not in saves
    # Nothing is left to make ahead on the last scene.
    assert controller.start_prefetch() is False
    assert controller.api_view()["segment_preview"]["next"] is None


@needs_ffmpeg
def test_a_failed_prefetch_is_made_again_once_in_the_foreground(
    tmp_path: Path, made: list[WizardController]
) -> None:
    engine = _Engine()
    controller = _controller(tmp_path, made, engine=engine)
    job_dir = _on_first_scene(controller, tmp_path)
    engine.fail = True
    assert controller.start_prefetch()
    assert controller.wait_prefetch(30)
    assert controller.api_view()["segment_preview"]["next"] == {
        "index": 1,
        "ready": False,
        "running": False,
    }
    assert _scene_files(job_dir, 1) == []
    engine.fail = False
    controller.handle_post("/preview", {"action": ["approve"]})
    engine.threads.clear()
    controller.run_analyze()
    state = controller.current_state()
    assert state.step is WizardStep.PREVIEW_SEGMENT, state.error
    assert engine.threads == ["MainThread"]
    plan = load_plan(job_dir)
    assert plan is not None and plan.recipes[1].rev == 1
    assert not any(name.startswith("shot_001_r2") for name in _scene_files(job_dir, 1))


@needs_ffmpeg
def test_a_prefetch_still_running_is_waited_for_and_adopted(
    tmp_path: Path, made: list[WizardController]
) -> None:
    engine = _Engine()
    controller = _controller(tmp_path, made, engine=engine)
    job_dir = _on_first_scene(controller, tmp_path)
    engine.threads.clear()
    engine.gate = threading.Event()
    engine.entered.clear()
    assert controller.start_prefetch()
    assert engine.entered.wait(30)
    assert controller.api_view()["segment_preview"]["next"]["running"] is True
    controller.handle_post("/preview", {"action": ["approve"]})
    threading.Timer(0.5, engine.gate.set).start()
    controller.run_analyze()
    state = controller.current_state()
    assert state.step is WizardStep.PREVIEW_SEGMENT and state.segment_index == 1
    # Only the prefetch used the GPU for it.
    assert set(engine.threads) == {"wizard-prefetch"}
    plan = load_plan(job_dir)
    assert plan is not None and plan.recipes[1].rev == 1


@needs_ffmpeg
def test_a_redo_drops_the_prefetch_with_its_files_and_turns(
    tmp_path: Path, made: list[WizardController]
) -> None:
    engine = _Engine()
    controller = _controller(tmp_path, made, engine=engine)
    job_dir = _on_first_scene(controller, tmp_path)
    engine.gate = threading.Event()
    engine.entered.clear()
    assert controller.start_prefetch()
    assert engine.entered.wait(30)
    controller.handle_post("/preview", {"action": ["redo"], "note": ["warmer, please"]})
    # The redo's own analysis drops the prefetch before it uses the GPU.
    threading.Timer(0.5, engine.gate.set).start()
    controller.run_analyze()
    state = controller.current_state()
    assert state.step is WizardStep.PREVIEW_SEGMENT and state.segment_index == 0
    assert controller.api_view()["segment_preview"]["next"] is None
    assert _scene_files(job_dir, 1) == []
    plan = load_plan(job_dir)
    assert plan is not None and plan.recipes[1].rev == 0
    assert plan.recipes[0].rev == 2
    assert not any("[1]" in text for text in _texts(job_dir))


@needs_ffmpeg
def test_leaving_the_scenes_drops_a_finished_prefetch(
    tmp_path: Path, made: list[WizardController]
) -> None:
    controller = _controller(tmp_path, made)
    job_dir = _on_first_scene(controller, tmp_path)
    assert controller.start_prefetch()
    assert controller.wait_prefetch(30)
    assert _scene_files(job_dir, 1)
    controller.handle_post("/goto", {"to": ["segments"]})
    assert controller.current_state().step is WizardStep.SEGMENTS
    assert _scene_files(job_dir, 1) == []
    assert not any("[1]" in text for text in _texts(job_dir))


@needs_ffmpeg
@pytest.mark.parametrize("call", ["video_alive", "propose_looks", "check_previews"])
def test_leaving_does_not_wait_for_a_prefetch_gemini_call(
    tmp_path: Path, made: list[WizardController], call: str
) -> None:
    gemini = _HeldGemini()
    controller = _controller(tmp_path, made, gemini=gemini)
    job_dir = _on_first_scene(controller, tmp_path)
    assert (job_dir / GEMINI_VIDEO_FILE).is_file()
    gemini.hold = call
    try:
        assert controller.start_prefetch()
        assert gemini.entered.wait(30)
        began = time.monotonic()
        controller.handle_post("/goto", {"to": ["segments"]})
        # Gemini has not answered, and the POST did not wait for it.
        assert time.monotonic() - began < 5
        assert controller.current_state().step is WizardStep.SEGMENTS
        # The prefetch stops at once, leaving the call to finish on its own.
        assert controller.wait_prefetch(5)
        assert not gemini.gate.is_set()
        assert _scene_files(job_dir, 1) == []
        assert not list((job_dir / PREVIEWS_DIR / "work").glob("shot_001_*"))
    finally:
        gemini.gate.set()
    assert not any("[1]" in text for text in _texts(job_dir))


@needs_ffmpeg
def test_a_redo_does_not_wait_for_a_prefetch_gemini_call(
    tmp_path: Path, made: list[WizardController]
) -> None:
    gemini = _HeldGemini()
    controller = _controller(tmp_path, made, gemini=gemini)
    job_dir = _on_first_scene(controller, tmp_path)
    gemini.hold = "propose_looks"
    try:
        assert controller.start_prefetch()
        assert gemini.entered.wait(30)
        # The redo's own propose_looks is not held: only the first call is.
        gemini.hold = ""
        controller.handle_post("/preview", {"action": ["redo"], "note": ["softer"]})
        began = time.monotonic()
        controller.run_analyze()
        assert time.monotonic() - began < 20
        assert not gemini.gate.is_set()
    finally:
        gemini.gate.set()
    state = controller.current_state()
    assert state.step is WizardStep.PREVIEW_SEGMENT and state.segment_index == 0
    assert _scene_files(job_dir, 1) == []
    plan = load_plan(job_dir)
    assert plan is not None and plan.recipes[1].rev == 0


@needs_ffmpeg
def test_a_prefetch_is_not_merged_once_its_plan_is_gone(
    tmp_path: Path, made: list[WizardController], monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _Engine()
    controller = _controller(tmp_path, made, engine=engine)
    job_dir = _on_first_scene(controller, tmp_path)
    assert controller.start_prefetch()
    assert controller.wait_prefetch(30)
    controller.handle_post("/preview", {"action": ["approve"]})

    # Hold the post lock, as a POST would, until the merge asks for it.
    lock = controller._post_lock
    asked = threading.Event()

    class _Spy:
        def acquire(self, *args: Any, **kwargs: Any) -> bool:
            asked.set()
            return lock.acquire(*args, **kwargs)

        def release(self) -> None:
            lock.release()

        def __enter__(self) -> _Spy:
            lock.acquire()
            return self

        def __exit__(self, *exc: object) -> None:
            lock.release()

    # In use, POSTs are refused while the analysis runs, so the plan cannot
    # go away here; this forces it to show the merge reads the plan again
    # under the post lock rather than trusting what it read before.
    monkeypatch.setattr(controller, "_post_lock", _Spy())
    lock.acquire()
    engine.threads.clear()
    worker = threading.Thread(target=controller.run_analyze, name="foreground")
    try:
        worker.start()
        assert asked.wait(30)
        # What approving new segments does to the plan meanwhile.
        (job_dir / "shot_plan.json").unlink()
    finally:
        lock.release()
    worker.join(60)
    assert not worker.is_alive()
    # The prefetch's result was not merged: the foreground made the scene
    # itself (saving a plan of its own), and the prefetch's Gemini turns
    # were dropped.
    assert engine.threads == ["foreground"]
    assert _texts(job_dir).count("propose [1]") == 1
    plan = load_plan(job_dir)
    assert plan is not None and plan.recipes[1].rev == 1


@needs_ffmpeg
def test_an_expired_upload_is_not_replaced_ahead_of_time(
    tmp_path: Path, made: list[WizardController]
) -> None:
    gemini = _VideoGemini()
    controller = _controller(tmp_path, made, gemini=gemini)
    job_dir = _on_first_scene(controller, tmp_path)
    old = GeminiVideo(name="files/old", uri="u", expires_at=time.time() - 10)
    record = job_dir / GEMINI_VIDEO_FILE
    record.write_text(json.dumps({"video": old.to_dict(), "error": None}))
    kept = record.read_bytes()
    gemini.uploads = 0
    assert controller.start_prefetch()
    assert controller.wait_prefetch(30)
    assert controller.api_view()["segment_preview"]["next"]["ready"] is True
    assert gemini.uploads == 0
    assert record.read_bytes() == kept


@needs_ffmpeg
def test_prefetch_starts_after_the_worker_and_never_beside_one(
    tmp_path: Path, made: list[WizardController]
) -> None:
    controller = _controller(tmp_path, made)
    job_dir = _on_first_scene(controller, tmp_path)
    # While a run holds the work lock, nothing starts beside it.
    with controller._work_lock:
        assert controller.start_prefetch() is False
    # A redo through the worker starts the prefetch once the worker ends.
    controller.handle_post("/preview", {"action": ["redo"], "note": ["softer"]})
    assert controller.start_analyze()
    assert controller.wait_idle(60)
    deadline = time.time() + 30
    while controller.api_view()["segment_preview"]["next"] is None:
        assert time.time() < deadline
        time.sleep(0.05)
    assert controller.wait_prefetch(30)
    assert controller.api_view()["segment_preview"]["next"]["ready"] is True
    # A stopping server drops it, and its files with it.
    controller.shutdown()
    assert _scene_files(job_dir, 1) == []
