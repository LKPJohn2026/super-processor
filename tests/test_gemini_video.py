"""Gemini watches the uploaded video, and chooses the output scale."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from loop_driver import (
    approve_preview,
    approve_segments,
    pick_and_segment,
    to_ready,
)

from super_processor.gemini import (
    DEFAULT_GEMINI_MODEL,
    GeminiClient,
    GeminiError,
    GeminiVideo,
    _has_video,
    _parse_expiry,
    video_part,
)
from super_processor.segment_loop import (
    GEMINI_COPY_NAME,
    load_segment_list,
    make_gemini_copy,
)
from super_processor.shot_plan import load_plan
from super_processor.upscale import FakeUpscaleEngine, UpscaleRequest
from super_processor.wizard import GEMINI_VIDEO_FILE, WizardController, WizardStep

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg required"
)

VIDEO = GeminiVideo(
    name="files/abc", uri="https://example.test/files/abc", expires_at=2e9
)


class _Transport:
    """Answers generateContent with ``reply`` and the File API from a script."""

    def __init__(self, reply: dict[str, Any], states: list[str] | None = None):
        self.reply = reply
        self.states = states or ["ACTIVE"]
        self.bodies: list[dict[str, Any]] = []
        self.uploaded: list[Path] = []
        self.sent: list[float] = []

    def generate(self, *, model: str, api_key: str, body: dict[str, Any]) -> Any:
        self.bodies.append(body)
        text = json.dumps(self.reply)
        return {"candidates": [{"content": {"parts": [{"text": text}]}}]}

    def upload_file(self, *, path: Path, on_progress: Any = None, **_: Any) -> Any:
        self.uploaded.append(path)
        if on_progress is not None:
            for done in (0.5, 1.0):
                on_progress(done)
                self.sent.append(done)
        return self._file()

    def get_file(self, *, api_key: str, name: str) -> Any:
        return self._file()

    def _file(self) -> dict[str, Any]:
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return {
            "name": "files/abc",
            "uri": "https://example.test/files/abc",
            "mimeType": "video/mp4",
            "state": state,
            "expirationTime": "2026-11-14T10:00:00.123456789Z",
        }


def test_the_video_is_sent_with_1_fps_and_the_scale_is_asked() -> None:
    reply = {
        "segments": [{"start_s": 0, "end_s": 9, "label": "x", "contains": []}],
        "scale": 4,
        "scale_reason": "  A small, soft  480p film. ",
    }
    transport = _Transport(reply)
    client = GeminiClient(api_key="k", transport=transport)
    out = client.propose_segments(
        video=VIDEO,
        duration_s=9.0,
        scales=(1, 2, 4),
        source_size=(640, 480),
    )
    assert out["scale"] == 4
    assert out["scale_reason"] == "A small, soft 480p film."
    body = transport.bodies[-1]
    parts = body["contents"][-1]["parts"]
    # The video first, then the request about it.
    assert parts[0] == {
        "fileData": {"mimeType": "video/mp4", "fileUri": VIDEO.uri},
        "videoMetadata": {"fps": 1.0},
    }
    assert "sampled at 1 frame per second" in parts[1]["text"]
    assert "The source is 640x480" in parts[1]["text"]
    assert "Use 2 unless" in parts[1]["text"]
    schema = body["generationConfig"]["responseSchema"]
    assert set(schema["required"]) == {"segments", "scale", "scale_reason"}
    assert _has_video(body)


@pytest.mark.parametrize("answer", [8, "4", True, None, 2.5])
def test_a_scale_not_offered_is_dropped(answer: Any) -> None:
    reply = {
        "segments": [{"start_s": 0, "end_s": 9, "label": "x", "contains": []}],
        "scale": answer,
        "scale_reason": "why",
    }
    client = GeminiClient(api_key="k", transport=_Transport(reply))
    out = client.propose_segments(video=VIDEO, duration_s=9.0, scales=(1, 2))
    assert out["scale"] is None
    assert out["scale_reason"] == ""


def test_stills_alone_ask_no_scale() -> None:
    transport = _Transport(
        {"segments": [{"start_s": 0, "end_s": 9, "label": "x", "contains": []}]}
    )
    client = GeminiClient(api_key="k", transport=transport)
    with pytest.raises(GeminiError, match="one time per still"):
        client.propose_segments(duration_s=9.0)
    assert not transport.bodies


def test_each_shot_gets_its_stretch_of_the_video(tmp_path: Path) -> None:
    still = tmp_path / "s.jpg"
    still.write_bytes(b"\xff\xd8jpeg")
    transport = _Transport(
        {"scale": 2, "shots": [{"index": 0, "strength": 0.3, "look": {}}]}
    )
    client = GeminiClient(api_key="k", transport=transport)
    client.propose_looks(
        shots=[{"index": 0, "start_s": 12.5, "end_s": 30.0}],
        stills=[still],
        bounds={},
        scales=(2,),
        protected_strength=0.3,
        video=VIDEO,
    )
    parts = transport.bodies[-1]["contents"][-1]["parts"]
    clip = next(part for part in parts if "fileData" in part)
    assert clip["videoMetadata"] == {
        "fps": 1.0,
        "startOffset": "12.500s",
        "endOffset": "30.000s",
    }
    assert "stretch of the video" in parts[0]["text"]


def test_upload_waits_until_the_file_is_active(tmp_path: Path) -> None:
    copy = tmp_path / "copy.mp4"
    copy.write_bytes(b"video")
    transport = _Transport({}, states=["PROCESSING", "PROCESSING", "ACTIVE"])
    client = GeminiClient(api_key="k", transport=transport)
    naps: list[float] = []
    video = client.upload_video(copy, sleep=naps.append)
    assert video.uri == "https://example.test/files/abc"
    assert len(naps) == 2
    assert video.expires_at == pytest.approx(
        _parse_expiry("2026-11-14T10:00:00.123456Z")
    )
    assert client.video_alive(video)


def test_a_failed_or_slow_upload_is_an_error(tmp_path: Path) -> None:
    copy = tmp_path / "copy.mp4"
    copy.write_bytes(b"video")
    failed = GeminiClient(api_key="k", transport=_Transport({}, states=["FAILED"]))
    with pytest.raises(GeminiError, match="could not process"):
        failed.upload_video(copy, sleep=lambda _s: None)
    slow = GeminiClient(api_key="k", transport=_Transport({}, states=["PROCESSING"]))
    ticks = iter(range(0, 10_000, 100))
    with pytest.raises(GeminiError, match="too long"):
        slow.upload_video(copy, clock=lambda: float(next(ticks)), sleep=lambda _s: None)


def test_a_transport_without_files_cannot_upload(tmp_path: Path) -> None:
    class _TextOnly:
        def generate(self, **_: Any) -> Any:
            raise AssertionError("not called")

    client = GeminiClient(api_key="k", transport=_TextOnly())
    with pytest.raises(GeminiError, match="cannot upload"):
        client.upload_video(tmp_path / "x.mp4")
    assert not client.video_alive(VIDEO)


def test_video_parts_and_expiry_helpers() -> None:
    assert video_part(VIDEO, start_s=-1.0)["videoMetadata"]["startOffset"] == "0.000s"
    assert _parse_expiry("not a time") == 0.0
    assert _parse_expiry(None) == 0.0
    assert GeminiVideo.from_dict(VIDEO.to_dict()) == VIDEO
    assert GeminiVideo.from_dict({"name": 1}) is None
    assert VIDEO.usable(now=2e9 - 7200)
    assert not VIDEO.usable(now=2e9 - 60)


def _clip(path: Path, size: str = "320x180", audio: bool = True) -> Path:
    command = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i"]
    command += [f"testsrc2=size={size}:rate=24:duration=8"]
    if audio:
        command += ["-f", "lavfi", "-i", "sine=duration=8", "-c:a", "aac"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)]
    subprocess.run(command, check=True)
    return path


def _streams(video: Path) -> list[str]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height",
            "-of",
            "csv=p=0",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return out.split()


@needs_ffmpeg
def test_the_copy_for_gemini_is_silent_and_small(tmp_path: Path) -> None:
    source = _clip(tmp_path / "in.mp4", size="1920x1080")
    seen: list[float] = []
    copy = make_gemini_copy(source, tmp_path, 8.0, on_progress=seen.append)
    assert copy == tmp_path / GEMINI_COPY_NAME
    assert _streams(copy) == ["video,960,540"]
    assert seen and seen[-1] <= 1.0
    tall = _clip(tmp_path / "tall.mp4", size="480x854", audio=False)
    assert _streams(make_gemini_copy(tall, tmp_path, 8.0)) == ["video,480,854"]


class _Watcher:
    """A fake Gemini that can upload, and picks ``scale`` for the video."""

    def __init__(self, scale: int = 4, *, upload_fails: bool = False) -> None:
        self.scale = scale
        self.upload_fails = upload_fails
        self.uploads: list[Path] = []
        self.segment_calls: list[dict[str, Any]] = []
        self.looks_calls: list[dict[str, Any]] = []

    def upload_video(self, path: Path, *, on_progress: Any = None) -> GeminiVideo:
        if self.upload_fails:
            raise GeminiError("upload refused")
        self.uploads.append(path)
        if on_progress is not None:
            on_progress(1.0)
        return VIDEO

    def video_alive(self, video: GeminiVideo) -> bool:
        return True

    def propose_segments(self, **kwargs: Any) -> dict[str, Any]:
        self.segment_calls.append(kwargs)
        return {
            "segments": [
                {"start_s": 0.0, "end_s": 4.0, "label": "first", "contains": []},
                {"start_s": 4.0, "end_s": 8.0, "label": "second", "contains": []},
            ],
            "scale": self.scale if self.scale in kwargs["scales"] else None,
            "scale_reason": "small and soft",
        }

    def propose_looks(self, **kwargs: Any) -> dict[str, Any]:
        self.looks_calls.append(kwargs)
        return {
            "scale": 2,
            "shots": [
                {"index": shot["index"], "strength": 0.4, "look": {}, "reason": "ok"}
                for shot in kwargs["shots"]
            ],
        }

    def check_previews(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"index": shot["index"], "ok": True} for shot in kwargs["shots"]]


class _Scales(FakeUpscaleEngine):
    def __init__(self) -> None:
        super().__init__()
        self.scales: list[int] = []

    def upscale(self, request: UpscaleRequest) -> Path:
        self.scales.append(request.scale)
        return super().upscale(request)


@needs_ffmpeg
def test_gemini_watches_the_video_and_chooses_4x(tmp_path: Path) -> None:
    gemini = _Watcher(scale=4)
    engine = _Scales()
    controller = WizardController(
        tmp_path / "jobs",
        gemini=gemini,  # type: ignore[arg-type]
        upscale_engine=engine,
    )
    job_id = pick_and_segment(controller, _clip(tmp_path / "in.mp4")).job_id
    assert job_id
    job_dir = controller.store.job_dir(job_id)
    # One silent copy, uploaded once, watched instead of the stills.
    assert [path.name for path in gemini.uploads] == [GEMINI_COPY_NAME]
    asked = gemini.segment_calls[0]
    assert asked["video"] == VIDEO and "stills" not in asked
    assert asked["scales"] == (1, 2, 4)
    assert asked["source_size"] == (320, 180)
    found = load_segment_list(job_dir)
    assert found is not None
    assert (found.scale, found.scale_reason, found.error) == (4, "small and soft", None)
    view = controller.api_view()["segments"]
    assert view["scale"] == 4 and view["scale_reason"] == "small and soft"

    approve_segments(controller)
    plan = load_plan(job_dir)
    assert plan is not None and plan.scale == 4
    # Previews stay at 2x; the planner watches its segment of the upload.
    assert engine.scales and set(engine.scales) == {2}
    assert gemini.looks_calls[0]["video"] == VIDEO
    assert gemini.looks_calls[0]["scales"] == (4,)
    assert len(gemini.uploads) == 1
    progress = controller.current_state().progress
    assert progress is not None and progress["elapsed_s"] < 60
    while approve_preview(controller).step is WizardStep.PREVIEW_SEGMENT:
        pass
    assert controller.current_state().step is WizardStep.READY_TO_RENDER
    ready = controller.api_view()["looks"]
    assert ready["scale"] == 4 and ready["scale_reason"] == "small and soft"


@needs_ffmpeg
def test_a_small_gpu_keeps_gemini_from_4x(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUPER_PROCESSOR_GPU_VRAM_GB", "16")
    gemini = _Watcher(scale=4)
    controller = WizardController(
        tmp_path / "jobs",
        gemini=gemini,  # type: ignore[arg-type]
        upscale_engine=FakeUpscaleEngine(),
    )
    job_dir = to_ready(controller, _clip(tmp_path / "in.mp4"))
    assert gemini.segment_calls[0]["scales"] == (1, 2)
    plan = load_plan(job_dir)
    assert plan is not None and plan.scale == 2
    assert controller.api_view()["looks"]["scales"] == [1, 2]


@needs_ffmpeg
def test_gemini_may_keep_the_size(tmp_path: Path) -> None:
    gemini = _Watcher(scale=1)
    engine = _Scales()
    controller = WizardController(
        tmp_path / "jobs",
        gemini=gemini,  # type: ignore[arg-type]
        upscale_engine=engine,
    )
    job_dir = to_ready(controller, _clip(tmp_path / "in.mp4"))
    plan = load_plan(job_dir)
    assert plan is not None and plan.scale == 1
    # Without upscaling, FlashVSR never runs, not even for previews.
    assert engine.scales == []


@needs_ffmpeg
def test_a_failed_upload_falls_back_to_the_stills(tmp_path: Path) -> None:
    gemini = _Watcher(scale=2, upload_fails=True)
    controller = WizardController(
        tmp_path / "jobs",
        gemini=gemini,  # type: ignore[arg-type]
        upscale_engine=FakeUpscaleEngine(),
    )
    job_id = pick_and_segment(controller, _clip(tmp_path / "in.mp4")).job_id
    assert job_id
    asked = gemini.segment_calls[0]
    assert "video" not in asked and len(asked["stills"]) == len(asked["times"]) > 2
    found = load_segment_list(controller.store.job_dir(job_id))
    assert found is not None and found.scale == 2
    # The split worked; only how Gemini saw the video is worth a word.
    assert found.error is None
    view = controller.api_view()["segments"]
    assert view["video_problem"] == "upload refused"
    from super_processor.wizard_pages import render_segments

    page = render_segments(view)
    assert "Gemini saw stills instead of the video: upload refused" in page
    assert "could not split" not in page
    record = json.loads(
        (controller.store.job_dir(job_id) / GEMINI_VIDEO_FILE).read_text()
    )
    assert record == {"video": None, "error": "upload refused"}
    # A failed upload is not retried on every call.
    approve_segments(controller)
    assert "video" not in gemini.looks_calls[0]


class _TimesOut(_Transport):
    def generate(self, *, model: str, api_key: str, body: dict[str, Any]) -> Any:
        self.bodies.append(body)
        raise GeminiError("Gemini request failed: timed out")


def test_a_video_timeout_ends_the_failover(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "SUPER_PROCESSOR_GEMINI_CANDIDATES", "gemini-2.0-flash,gemini-2.0-pro"
    )
    transport = _TimesOut({})
    client = GeminiClient(api_key="k", transport=transport)
    with pytest.raises(GeminiError, match="timed out watching the video"):
        client.propose_segments(video=VIDEO, duration_s=9.0, scales=(1, 2))
    # Every model would read the same video: one try, not one per model.
    assert len(transport.bodies) == 1
    transport.bodies.clear()
    with pytest.raises(GeminiError, match="All candidate"):
        client._generate_with_failover({"contents": [{"parts": [{"text": "x"}]}]})
    assert len(transport.bodies) == len(client.candidate_models()) > 1


class _Busy(_Transport):
    """The first model is busy; any other answers."""

    models: list[str]

    def generate(self, *, model: str, api_key: str, body: dict[str, Any]) -> Any:
        self.models.append(model)
        if len(self.models) == 1:
            raise GeminiError("Gemini HTTP 503: high demand", status=503)
        return super().generate(model=model, api_key=api_key, body=body)


def test_a_busy_model_is_skipped_at_once_for_a_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SUPER_PROCESSOR_GEMINI_CANDIDATES", "gemini-2.0-flash,gemini-2.0-pro"
    )
    slept: list[float] = []
    monkeypatch.setattr("super_processor.gemini.time.sleep", slept.append)
    segment = {"start_s": 0, "end_s": 9, "label": "x", "contains": []}
    reply = {"segments": [segment], "scale": 2, "scale_reason": "ok"}
    transport = _Busy(reply)
    transport.models = []
    client = GeminiClient(api_key="k", transport=transport)
    client.propose_segments(video=VIDEO, duration_s=9.0, scales=(1, 2))
    # The busy first model is not retried: straight on to the next one.
    assert transport.models == [DEFAULT_GEMINI_MODEL, "gemini-2.0-flash"]
    assert slept == []


@needs_ffmpeg
def test_odd_sizes_give_an_even_copy(tmp_path: Path) -> None:
    def odd(name: str, size: str) -> Path:
        path = tmp_path / name
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=size={size}:rate=24:duration=2",
                "-c:v",
                "ffv1",
                str(path),
            ],
            check=True,
        )
        return path

    small = make_gemini_copy(odd("small.mkv", "853x479"), tmp_path, 2.0)
    assert _streams(small) == ["video,852,478"]
    tall = make_gemini_copy(odd("tall.mkv", "721x1281"), tmp_path, 2.0)
    assert _streams(tall) == ["video,540,960"]


def test_a_flood_of_ffmpeg_errors_does_not_stall_the_copy(tmp_path: Path) -> None:
    fake = tmp_path / "ffmpeg"
    fake.write_text(
        "#!/bin/sh\n"
        "for last; do :; done\n"
        # Far more than a pipe holds, before any progress.
        "yes e | head -c 300000 >&2\n"
        "echo out_time_us=1000000\n"
        ': > "$last"\n'
        "exit 1\n"
    )
    fake.chmod(0o755)
    seen: list[float] = []
    from super_processor.segment_loop import SegmentLoopError

    with pytest.raises(SegmentLoopError, match="copy for Gemini"):
        make_gemini_copy(
            tmp_path / "in.mp4",
            tmp_path,
            2.0,
            ffmpeg_bin=str(fake),
            on_progress=seen.append,
        )
    assert seen == [0.5]
    assert not (tmp_path / "gemini_upload.part.mp4").exists()


@needs_ffmpeg
def test_a_missing_key_falls_back_like_before(tmp_path: Path) -> None:
    controller = WizardController(
        tmp_path / "jobs",
        gemini=None,
        upscale_engine=FakeUpscaleEngine(),
    )

    def no_key() -> Any:
        raise GeminiError("no Gemini API key")

    controller.gemini = no_key  # type: ignore[method-assign]
    state = pick_and_segment(controller, _clip(tmp_path / "in.mp4"))
    assert state.step is WizardStep.SEGMENTS
    assert state.job_id
    found = load_segment_list(controller.store.job_dir(state.job_id))
    assert found is not None and found.error is not None
    assert "no Gemini API key" in found.error
    assert [s.label for s in found.segments] == ["whole video"]


class _WatchFailsOnce(_Watcher):
    def __init__(self) -> None:
        super().__init__(scale=2)
        self.failed = False

    def propose_segments(self, **kwargs: Any) -> dict[str, Any]:
        if "video" in kwargs and not self.failed:
            self.failed = True
            self.segment_calls.append(kwargs)
            raise GeminiError("Gemini HTTP 503: high demand")
        return super().propose_segments(**kwargs)


@needs_ffmpeg
def test_one_failed_watch_keeps_the_upload(tmp_path: Path) -> None:
    gemini = _WatchFailsOnce()
    controller = WizardController(
        tmp_path / "jobs",
        gemini=gemini,  # type: ignore[arg-type]
        upscale_engine=FakeUpscaleEngine(),
    )
    job_id = pick_and_segment(controller, _clip(tmp_path / "in.mp4")).job_id
    assert job_id
    job_dir = controller.store.job_dir(job_id)
    assert "stills" in gemini.segment_calls[-1]
    assert "high demand" in controller.api_view()["segments"]["video_problem"]
    record = json.loads((job_dir / GEMINI_VIDEO_FILE).read_text())
    assert record["video"] == VIDEO.to_dict()
    # The next request watches the same upload again; nothing re-uploads.
    approve_segments(controller)
    assert gemini.looks_calls[0]["video"] == VIDEO
    assert len(gemini.uploads) == 1
