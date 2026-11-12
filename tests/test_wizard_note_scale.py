"""A result note must not change the output size on its own."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pytest

from super_processor.gemini import GeminiClient
from super_processor.probe import probe_file
from super_processor.review import WizardServer
from super_processor.shot_plan import allowed_scales, note_scale, reply_scale
from super_processor.upscale import (
    FakeUpscaleEngine,
    UpscaleError,
    UpscaleRequest,
    UpscaleSpan,
    load_upscale_plan,
)
from super_processor.wizard import WizardController, WizardStep, strength_summary

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg required"
)


class _Transport:
    """Answer by schema; the note's reply is the only thing a test varies."""

    def __init__(self, revise_scale: Any) -> None:
        self.revise_scale = revise_scale
        self.revise_prompts: list[str] = []

    def generate(
        self, *, model: str, api_key: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        props = body["generationConfig"]["responseSchema"].get("properties", {})
        out: dict[str, Any]
        if "segments" in props:
            out = {"segments": [{"start_s": 0, "end_s": 6, "label": "card"}]}
        elif "start_s" in props:
            self.revise_prompts.append(json.dumps(body["contents"][-1]))
            out = {
                "start_s": 1,
                "end_s": 3,
                "scale": self.revise_scale,
                "strength": 0.2,
            }
        elif "ok" in json.dumps(body["generationConfig"]["responseSchema"]):
            out = {"shots": [{"index": 0, "ok": True, "problems": [], "note": "ok"}]}
        else:
            out = {"scale": 2, "shots": [{"index": 0, "strength": 0.4, "reason": "x"}]}
        return {"candidates": [{"content": {"parts": [{"text": json.dumps(out)}]}}]}


class _Engine(FakeUpscaleEngine):
    """Record each engine call, and fail at ``fail_scale`` when it is set."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_scale: int | None = None
        self.scales: list[int] = []

    def upscale(self, request: UpscaleRequest) -> Path:
        self.scales.append(request.scale)
        if request.scale == self.fail_scale:
            raise UpscaleError("simulated GPU failure")
        return super().upscale(request)


def _clip(path: Path, size: str) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=gray:duration=6:size={size}:rate=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )  # fmt: skip
    return path


def _render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    size: str,
    ready_scale: str,
    revise_scale: Any,
) -> tuple[WizardController, _Transport, Path]:
    """Drive a real wizard through a first render at ``ready_scale``."""
    monkeypatch.setenv("GEMINI_API_KEY", "note-scale-key")
    transport = _Transport(revise_scale)
    controller = WizardController(
        tmp_path / "jobs",
        gemini=GeminiClient(api_key="note-scale-key", transport=transport),
        upscale_engine=_Engine(),
    )
    clip = _clip(tmp_path / "clip.mp4", size)
    controller.handle_post("/intro", {})
    controller.handle_post("/llm", {"choice": ["gemini"]})
    controller.handle_post("/setup", {"skip": ["1"]})
    controller.handle_post("/pick", {"path": [str(clip)]})
    controller.run_segmenting()
    controller.handle_post("/segments", {"action": ["approve"]})
    controller.run_analyze()
    controller.handle_post("/preview", {"action": ["approve"]})
    controller.handle_post("/ready", {"action": ["render"], "scale": [ready_scale]})
    controller.run_render()
    state = controller.current_state()
    assert state.step is WizardStep.RESULT, state.error
    assert state.job_id
    return controller, transport, controller.store.job_dir(state.job_id)


def _output_url(controller: WizardController) -> str:
    url = controller.api_view()["output_url"]
    assert isinstance(url, str)
    return url


def _size(video: Path) -> tuple[int | None, int | None]:
    primary = probe_file(video).primary_video()
    assert primary is not None
    return primary.width, primary.height


@pytest.mark.parametrize(
    ("note", "current", "expected"),
    [
        # A named size wins, whatever the model replies.
        ("make it 4x", 2, 4),
        ("Make it 2× please", 1, 2),
        ("2x", 1, 2),
        ("2 x please", 1, 2),
        ("render at 2x from 2s to 5s", 1, 2),
        ("less detail, make it 4x", 2, 4),
        ("go back to 2x, less contrast", 4, 2),
        ("not 4x, 2x", 1, 2),
        # Everything else keeps the size.
        ("", 2, 2),
        ("don't make it 4x", 2, 2),
        ("don’t make it 4x", 2, 2),
        ("don't upscale", 2, 2),
        ("keep the same size, less contrast", 1, 1),
        ("make it bigger", 2, 2),
        ("double it", 1, 1),
        ("less artificial detail from 2s to 5s", 2, 2),
        ("reduce contrast", 1, 1),
        ("from 00:02 to 00:05", 2, 2),
        ("a 1x1 grid shows in the corner", 2, 2),
        ("1 x 1 blocks", 2, 2),
        ("16x9 looks off", 2, 2),
        ("make it 1.5x", 2, 2),
        ("use a 2x4 board", 2, 2),
        ("the bigger problem is contrast", 1, 1),
        ("the sign reads 4x and flickers", 1, 1),
        ("looks like 2x speed", 1, 1),
        ("a 4x truck drives past", 1, 1),
        ("4x truck drives past", 1, 1),
    ],
)
def test_note_scale(note: str, current: int, expected: int) -> None:
    assert note_scale(note, current, (1, 2, 4)) == expected


@pytest.mark.parametrize(
    ("note", "allowed"), [("make it 4x", (1, 2)), ("make it 3x", (1, 2, 4))]
)
def test_note_for_a_size_the_source_lacks_is_refused(
    note: str, allowed: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="kept 2x and rendered nothing"):
        note_scale(note, 2, allowed)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (2, 2),
        (2.0, 2),
        (3, None),
        (2.5, None),
        (True, None),
        ("4", None),
        (None, None),
        (float("inf"), None),
        (float("nan"), None),
        (10**400, None),
    ],
)
def test_reply_scale(raw: Any, expected: int | None) -> None:
    assert reply_scale(raw) == expected


def test_scale_one_spans_report_flashvsr_off() -> None:
    assert strength_summary((UpscaleSpan(0.0, 6.0, scale=1, strength=0.2),)) == (
        "FlashVSR off"
    )


def test_unknown_size_does_not_unlock_4x() -> None:
    assert allowed_scales(0, 0) == (2,)
    assert allowed_scales(854, 480) == (2, 4)
    assert allowed_scales(1920, 1080) == (2,)


@needs_ffmpeg
@pytest.mark.parametrize(
    "note",
    [
        "less contrast from 1s to 3s",
        "keep the same size, less contrast",
        "don't upscale",
    ],
)
def test_note_on_dont_upscale_render_cannot_change_the_picture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, note: str
) -> None:
    controller, transport, job_dir = _render(
        tmp_path, monkeypatch, size="160x90", ready_scale="1", revise_scale=2
    )
    before = (job_dir / "output.mp4").read_bytes()
    controller.handle_post("/result", {"note": [note]})
    state = controller.current_state()
    assert state.step is WizardStep.RESULT
    assert state.error is not None and "make it 2x" in state.error
    assert transport.revise_prompts == []
    assert (job_dir / "output.mp4").read_bytes() == before
    plan = load_upscale_plan(job_dir)
    assert plan is not None and plan.pending is None
    assert {span.scale for span in plan.spans} == {1}


@needs_ffmpeg
def test_note_asking_for_size_on_dont_upscale_render_is_honoured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, transport, job_dir = _render(
        tmp_path, monkeypatch, size="160x90", ready_scale="1", revise_scale=2
    )
    controller.handle_post("/result", {"note": ["make it 2x"]})
    assert "only one of: 1, 2, 4" in transport.revise_prompts[-1]
    assert "Keep scale=1" in transport.revise_prompts[-1]
    plan = load_upscale_plan(job_dir)
    assert plan is not None
    assert {span.scale for span in plan.spans} == {2}
    controller.run_render()
    state = controller.current_state()
    assert state.step is WizardStep.RESULT, state.error
    assert _size(job_dir / "output.mp4") == (320, 180)


@needs_ffmpeg
def test_named_size_wins_over_the_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, _transport, job_dir = _render(
        tmp_path, monkeypatch, size="160x90", ready_scale="1", revise_scale=4
    )
    controller.handle_post("/result", {"note": ["make it 2x"]})
    controller.run_render()
    state = controller.current_state()
    assert state.step is WizardStep.RESULT and state.error is None
    assert _size(job_dir / "output.mp4") == (320, 180)


@needs_ffmpeg
@pytest.mark.parametrize("reply", [float("inf"), float("nan")])
def test_non_finite_reply_keeps_scale_without_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reply: float
) -> None:
    controller, _transport, job_dir = _render(
        tmp_path, monkeypatch, size="160x90", ready_scale="2", revise_scale=reply
    )
    controller.handle_post("/result", {"note": ["less contrast"]})
    state = controller.current_state()
    assert state.step is WizardStep.RENDERING, state.error
    assert state.error is None and state.notice is None
    plan = load_upscale_plan(job_dir)
    assert plan is not None
    assert {span.scale for span in plan.spans} == {2}


@needs_ffmpeg
def test_size_the_source_lacks_renders_nothing_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, transport, job_dir = _render(
        tmp_path, monkeypatch, size="1000x90", ready_scale="2", revise_scale=4
    )
    before = load_upscale_plan(job_dir)
    controller.handle_post("/result", {"note": ["make it 4x"]})
    state = controller.current_state()
    assert state.step is WizardStep.RESULT
    assert state.error is not None and "1x, 2x" in state.error
    assert transport.revise_prompts == []
    assert load_upscale_plan(job_dir) == before


@needs_ffmpeg
def test_source_limit_reaches_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, transport, _job_dir = _render(
        tmp_path, monkeypatch, size="1000x90", ready_scale="2", revise_scale=2
    )
    controller.handle_post("/result", {"note": ["less contrast"]})
    assert "only one of: 1, 2;" in transport.revise_prompts[-1]


@needs_ffmpeg
def test_note_may_still_change_to_an_allowed_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, _transport, job_dir = _render(
        tmp_path, monkeypatch, size="160x90", ready_scale="2", revise_scale=4
    )
    controller.handle_post("/result", {"note": ["make it 4x"]})
    plan = load_upscale_plan(job_dir)
    assert plan is not None
    assert {span.scale for span in plan.spans} == {4}


@needs_ffmpeg
def test_reply_ignored_for_a_plain_note_is_reported_after_the_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, _transport, job_dir = _render(
        tmp_path, monkeypatch, size="160x90", ready_scale="2", revise_scale=4
    )
    controller.handle_post("/result", {"note": ["less contrast from 1s to 3s"]})
    state = controller.current_state()
    assert state.step is WizardStep.RENDERING and state.error is None
    controller.run_render()
    state = controller.current_state()
    assert state.step is WizardStep.RESULT
    assert state.error is not None and state.error.startswith("Kept 2x")
    assert 'say "make it 4x"' in state.error
    assert state.notice is None
    assert _size(job_dir / "output.mp4") == (320, 180)


@needs_ffmpeg
@pytest.mark.parametrize(
    ("ready", "note", "reply", "failed_scale", "size"),
    [
        ("2", "make it 4x", 4, 4, (320, 180)),
        ("4", "make it 2x", 2, 2, (640, 360)),
    ],
)
def test_plain_note_after_a_failed_rescale_uses_the_size_on_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ready: str,
    note: str,
    reply: int,
    failed_scale: int,
    size: tuple[int, int],
) -> None:
    controller, transport, job_dir = _render(
        tmp_path, monkeypatch, size="160x90", ready_scale=ready, revise_scale=reply
    )
    before = load_upscale_plan(job_dir)
    engine = controller.upscale_engine()
    assert isinstance(engine, _Engine)
    engine.fail_scale = failed_scale
    controller.handle_post("/result", {"note": [note]})
    controller.run_render()
    state = controller.current_state()
    assert state.step is WizardStep.RESULT and state.error == "simulated GPU failure"
    assert _size(job_dir / "output.mp4") == size
    assert load_upscale_plan(job_dir) == before

    engine.fail_scale = None
    engine.scales.clear()
    controller.handle_post("/result", {"note": ["less contrast from 1s to 3s"]})
    assert f"Keep scale={ready}" in transport.revise_prompts[-1]
    controller.run_render()
    state = controller.current_state()
    assert state.step is WizardStep.RESULT
    assert state.error is None or state.error.startswith("Kept")
    assert _size(job_dir / "output.mp4") == size
    assert set(engine.scales) == {int(ready)}
    plan = load_upscale_plan(job_dir)
    assert plan is not None and plan.pending is None
    assert {span.scale for span in plan.spans} == {int(ready)}


@needs_ffmpeg
def test_output_url_changes_only_when_a_revise_replaces_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A browser can replay what it buffered for an unchanged URL."""
    controller, _transport, job_dir = _render(
        tmp_path, monkeypatch, size="160x90", ready_scale="1", revise_scale=2
    )
    first = _output_url(controller)
    assert first.startswith("/output.mp4?v=")
    assert f'src="{first}"' in controller.render()
    assert _output_url(controller) == first

    # Refused: the file stays, and so does the URL.
    controller.handle_post("/result", {"note": ["less contrast"]})
    assert _output_url(controller) == first

    controller.handle_post("/result", {"note": ["make it 2x"]})
    controller.run_render()
    assert _size(job_dir / "output.mp4") == (320, 180)
    second = _output_url(controller)
    assert second != first
    assert f'src="{second}"' in controller.render()


@needs_ffmpeg
def test_output_url_with_a_version_serves_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, _transport, job_dir = _render(
        tmp_path, monkeypatch, size="160x90", ready_scale="1", revise_scale=1
    )
    server = WizardServer(tmp_path / "jobs", controller=controller)
    base = server.start()
    try:
        request = Request(
            base + _output_url(controller), headers={"Range": "bytes=0-3"}
        )
        with urlopen(request) as response:
            assert response.status == 206
            assert response.read() == (job_dir / "output.mp4").read_bytes()[:4]
    finally:
        server.stop()
