"""CI coverage: full localhost wizard website from intro through done."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest
from wizard_client import get

from super_processor.gemini import GeminiClient
from super_processor.jobs import JobState
from super_processor.output_versions import archive_output, list_versions
from super_processor.probe import probe_file
from super_processor.review import WizardServer
from super_processor.segment_loop import LoopSegment, SegmentList, save_segment_list
from super_processor.upscale import (
    FakeUpscaleEngine,
    UpscalePlan,
    _commit,
    default_span,
    load_upscale_plan,
    save_upscale_plan,
)
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    save_job_wizard_state,
    save_session_state,
)

TOKEN = "test-token"


class _FakeTransport:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = list(payloads)

    def generate(
        self,
        *,
        model: str,
        api_key: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        payload = self.payloads.pop(0)
        return {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}


def _tiny_clip(path: Path, seconds: int = 12) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=gray:duration={seconds}:size=160x90:rate=10",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            str(path),
        ],
        check=True,
    )
    return path


def _get(base: str, path: str = "/") -> str:
    with urlopen(Request(base + path, headers={"X-Wizard-Token": TOKEN})) as response:
        return str(response.read().decode())


def _post(base: str, path: str, fields: dict[str, str]) -> str:
    req = Request(
        base + path,
        data=urlencode(fields).encode(),
        headers={"X-Wizard-Token": TOKEN},
        method="POST",
    )
    with urlopen(req) as response:
        return str(response.read().decode())


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_wizard_website_full_flow_intro_to_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive pick → fake upscale → range revise → happy over HTTP."""
    monkeypatch.setenv("GEMINI_API_KEY", "ci-full-flow-key")
    clip = _tiny_clip(tmp_path / "clip.mp4")
    gemini = GeminiClient(
        api_key="ci-full-flow-key",
        transport=_FakeTransport(
            [
                {
                    "segments": [
                        {
                            "start_s": 0,
                            "end_s": 12,
                            "label": "flat gray card",
                            "contains": ["made_up"],
                        }
                    ]
                },
                {
                    "scale": 2,
                    "shots": [
                        {
                            "index": 0,
                            "strength": 0.4,
                            "look": {
                                "deblock": 0.2,
                                "denoise": 0,
                                "contrast": 1.05,
                                "brightness": 0,
                                "saturation": 1,
                                "gamma": 1,
                                "grain": 0,
                            },
                            "reason": "Flat card; light deblock, a touch of contrast.",
                        }
                    ],
                },
                {
                    "shots": [
                        {
                            "index": 0,
                            "ok": True,
                            "problems": [],
                            "note": "Cleaner, nothing invented.",
                            "strength": 0.4,
                            "look": {
                                "deblock": 0.2,
                                "denoise": 0,
                                "contrast": 1.05,
                                "brightness": 0,
                                "saturation": 1,
                                "gamma": 1,
                                "grain": 0,
                            },
                        }
                    ]
                },
                {
                    "start_s": 2,
                    "end_s": 5,
                    "scale": 2,
                    "strength": 0.2,
                },
            ]
        ),
    )
    controller = WizardController(
        tmp_path,
        gemini=gemini,
        upscale_engine=FakeUpscaleEngine(),
    )
    server = WizardServer(tmp_path, token=TOKEN, controller=controller)
    base = server.start()
    try:
        intro = _get(base)
        assert "brand-hero" in intro
        assert "Super Processor" in intro
        assert "Continue" in intro

        llm = _post(base, "/intro", {})
        assert "What type of LLM" in llm
        assert "I don’t have a local LLM" in llm

        setup = _post(base, "/llm", {"choice": "gemini"})
        assert "Gemini API setup" in setup

        pick = _post(base, "/setup", {"skip": "1"})
        assert "Pick a video file" in pick

        sampling = _post(base, "/pick", {"path": str(clip)})
        assert "Finding the segments" in sampling
        assert controller.current_state().step is WizardStep.SAMPLING
        _get(base, "/segment?run=1")
        assert controller.wait_idle(timeout=60)
        found = _get(base)
        assert "<h1>Segments</h1>" in found
        assert "1. flat gray card" in found and "0:00–0:12" in found
        assert "Approve 1 segment<" in found
        assert "made_up" not in found

        analyzing = _post(base, "/segments", {"action": "approve"})
        assert "<h1>Segment 1 of 1</h1>" in analyzing
        _get(base, "/analyze?run=1")
        assert controller.wait_idle(timeout=60)
        looks = _get(base)
        assert "Approve and finish" in looks
        assert "strength 0.40 · deblock 0.20 · contrast 1.05" in looks
        assert "Flat card; light deblock" in looks
        assert "Check passed. Cleaner, nothing invented." in looks
        # The plan's generation keeps a cached preview of an older plan away.
        assert 'src="/previews/shot_000_r1_after.mp4?g=' in looks
        assert 'poster="/previews/shot_000_r1_after.jpg?g=' in looks
        with get(base + "/previews/shot_000_r1_after.jpg", TOKEN) as response:
            assert response.headers.get_content_type() == "image/jpeg"

        ready = _post(base, "/preview", {"action": "approve"})
        assert "<h1>Ready to render</h1>" in ready
        waiting = _post(base, "/ready", {"action": "render"})
        assert "<h1>Upscaling</h1>" in waiting
        assert controller.current_state().step is WizardStep.RENDERING

        started = _get(base, "/render?run=1")
        assert "<h1>Upscaling</h1>" in started
        assert controller.wait_idle(timeout=60)
        result = _get(base)
        assert "<h1>Result</h1>" in result
        assert "A. I am happy" in result
        assert "Scale 2" in result
        assert "strength 0.40" in result
        assert controller.current_state().step is WizardStep.RESULT
        job_id = controller.current_state().job_id
        assert job_id
        output = controller.store.job_dir(job_id) / "output.mp4"
        primary = probe_file(output).primary_video()
        assert primary is not None and (primary.width, primary.height) == (320, 180)
        with get(base + "/output.mp4", TOKEN) as response:
            assert response.headers.get_content_type() == "video/mp4"
            assert len(response.read()) > 0

        revising = _post(
            base,
            "/result",
            {"note": "reduce artificial detail from 2s to 5s"},
        )
        assert "<h1>Upscaling</h1>" in revising
        assert controller.current_state().step is WizardStep.RENDERING

        _get(base, "/render?run=1")
        assert controller.wait_idle(timeout=60)
        revised = _get(base)
        assert "<h1>Result</h1>" in revised
        # Gemini's 0.20 covers 2-5 s only; the rest keeps the approved 0.40.
        assert "strength 0.20–0.40" in revised
        plan = load_upscale_plan(controller.store.job_dir(job_id))
        assert plan is not None
        assert sorted((s.start_s, s.end_s, s.strength) for s in plan.spans) == [
            (0.0, 2.0, 0.4),
            (2.0, 5.0, 0.2),
            (5.0, 12.0, 0.4),
        ]
        primary = probe_file(output).primary_video()
        assert primary is not None and (primary.width, primary.height) == (320, 180)
        # The first result and the revise are both kept, and served by job.
        versions = controller.api_view()["versions"]
        assert [v["name"] for v in versions] == ["v001.mp4", "v002.mp4"]
        assert versions[1]["note"] == "reduce artificial detail from 2s to 5s"
        assert versions[1]["current"] is True
        with get(base + versions[0]["url"], TOKEN) as response:
            assert response.headers.get_content_type() == "video/mp4"
            assert len(response.read()) > 0

        done = _post(base, "/result", {"mood": "happy"})
        assert "<h1>Done</h1>" in done
        assert controller.current_state().step is WizardStep.DONE
    finally:
        server.stop()


def test_wizard_website_local_llm_stub_then_gemini_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LLM B route shows deferred stub UI, then returns to Gemini setup."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    controller = WizardController(tmp_path)
    server = WizardServer(tmp_path, token=TOKEN, controller=controller)
    base = server.start()
    try:
        _post(base, "/intro", {})
        unsure = _post(base, "/llm", {"choice": "both"})
        assert "pick how you want to run the LLM" in unsure
        assert controller.current_state().step is WizardStep.LLM_CHOICE
        stub = _post(base, "/llm", {"choice": "local"})
        assert "Local LLM setup" in stub
        assert "not wired yet" in stub
        assert "Use Gemini instead" in stub
        assert controller.current_state().step is WizardStep.LOCAL_LLM_STUB

        setup = _post(base, "/local-llm", {"action": "gemini"})
        assert "Gemini API setup" in setup
        assert controller.current_state().step is WizardStep.SETUP
    finally:
        server.stop()


# Kept versions ------------------------------------------------------------
#
# These drive the controller with a fake renderer, so they need no FFmpeg:
# the fake replaces output.mp4 with a new file, as the real renderer does.


class _Facts:
    duration_s = 8.0

    def primary_video(self) -> None:
        return None


class _ReviseGemini:
    """Answers every result note with a lighter 2-5 s span."""

    def revise_upscale_params(self, **_kwargs: Any) -> dict[str, Any]:
        return {"start_s": 2, "end_s": 5, "scale": 2, "strength": 0.2}


@pytest.fixture
def fake_render(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    """Each render writes the next of ``renders`` (``render N`` by default)."""
    renders: list[bytes] = []

    def render(_source: Path, output: Path, *_args: Any, **_kwargs: Any) -> None:
        work = output.with_name("work.mp4")
        work.write_bytes(f"render {len(renders) + 1}".encode())
        renders.append(work.read_bytes())
        _commit()
        work.replace(output)

    monkeypatch.setattr("super_processor.wizard.probe_file", lambda _path: _Facts())
    monkeypatch.setattr("super_processor.wizard.render_plan", render)
    monkeypatch.setattr(
        "super_processor.wizard.video_stream_facts", lambda *a, **k: (8, "24")
    )
    monkeypatch.setattr("super_processor.wizard.seam_candidates", lambda *a, **k: [])
    monkeypatch.setattr(WizardController, "_check_seams", lambda *_args: None)
    return renders


def _job_on(tmp_path: Path, step: WizardStep) -> tuple[WizardController, str, Path]:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not really a video")
    gemini: Any = _ReviseGemini()
    controller = WizardController(
        tmp_path / "jobs", gemini=gemini, upscale_engine=FakeUpscaleEngine()
    )
    job_id = controller.store.create(source).job_id
    controller.store.transition(job_id, JobState.PROBED)
    job_dir = controller.store.job_dir(job_id)
    state = WizardState(step=step, job_id=job_id)
    save_job_wizard_state(job_dir, state)
    save_session_state(controller.jobs_dir, state)
    save_upscale_plan(job_dir, UpscalePlan(spans=(default_span(8.0),)))
    return controller, job_id, job_dir


def _render(controller: WizardController) -> None:
    assert controller.start_render()
    assert controller.wait_idle(5)
    assert controller.current_state().step is WizardStep.RESULT


def _kept(job_dir: Path) -> list[tuple[str, str, str | None, bytes]]:
    return [
        (v.name, v.kind, v.note, (job_dir / "outputs" / v.name).read_bytes())
        for v in list_versions(job_dir)
    ]


def test_first_render_and_two_revises_keep_three_versions(
    tmp_path: Path, fake_render: list[bytes]
) -> None:
    controller, job_id, job_dir = _job_on(tmp_path, WizardStep.RENDERING)
    controller.resume()
    _render(controller)
    for note in ("less detail from 2s to 5s", "even less there"):
        controller.handle_post("/result", {"note": [note]})
        assert controller.current_state().render_note == note
        _render(controller)
        assert controller.current_state().render_note is None

    assert _kept(job_dir) == [
        ("v001.mp4", "render", None, b"render 1"),
        ("v002.mp4", "render", "less detail from 2s to 5s", b"render 2"),
        ("v003.mp4", "render", "even less there", b"render 3"),
    ]
    view = controller.api_view()
    assert [v["url"] for v in view["versions"]] == [
        f"/jobs/{job_id}/outputs/v00{n}.mp4" for n in (1, 2, 3)
    ]
    assert [v["current"] for v in view["versions"]] == [False, False, True]
    assert view["versions"][0]["scale"] == 2
    assert view["error"] is None

    controller.handle_post("/result", {"mood": ["happy"]})
    view = controller.api_view()
    assert view["step"] == "done" and len(view["versions"]) == 3
    # Happy keeps nothing new.
    assert len(list_versions(job_dir)) == 3


def test_an_output_from_before_versions_is_kept_when_a_render_starts(
    tmp_path: Path, fake_render: list[bytes]
) -> None:
    controller, _job_id, job_dir = _job_on(tmp_path, WizardStep.RESULT)
    (job_dir / "output.mp4").write_bytes(b"legacy result")
    controller.handle_post("/result", {"note": ["less detail at 2s"]})
    _render(controller)

    assert _kept(job_dir) == [
        ("v001.mp4", "earlier", None, b"legacy result"),
        ("v002.mp4", "render", "less detail at 2s", b"render 1"),
    ]
    # Each version is listed with what it holds: the earlier output has the
    # plan from before the note, not the note's lighter span.
    earlier, revised = list_versions(job_dir)
    assert earlier.extra["scale"] == 2 and earlier.extra["strength"] == 0.5
    assert "0.20" not in earlier.extra["strength_summary"]
    assert revised.extra["strength"] == 0.2


def test_an_earlier_output_is_listed_at_the_scale_it_has(
    tmp_path: Path, fake_render: list[bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    def measured(_source: Path, output: Path) -> int | None:
        return 4 if output.read_bytes() == b"rendered at 4x" else None

    monkeypatch.setattr("super_processor.wizard.rendered_scale", measured)
    controller, _job_id, job_dir = _job_on(tmp_path, WizardStep.RESULT)
    (job_dir / "output.mp4").write_bytes(b"rendered at 4x")
    controller.handle_post("/result", {"note": ["less detail at 2s"]})
    _render(controller)

    assert [v.extra["scale"] for v in list_versions(job_dir)] == [4, 2]


def test_a_restart_moves_the_output_into_the_versions(
    tmp_path: Path, fake_render: list[bytes]
) -> None:
    controller, _job_id, job_dir = _job_on(tmp_path, WizardStep.RESULT)
    (job_dir / "output.mp4").write_bytes(b"result before the restart")
    controller.handle_post(
        "/result", {"note": ["the colours are all wrong"], "restart": ["1"]}
    )

    state = controller.current_state()
    assert state.step is WizardStep.SAMPLING
    assert state.render_note == "the colours are all wrong"
    assert state.notice is None
    assert not (job_dir / "output.mp4").exists()
    assert _kept(job_dir) == [
        ("v001.mp4", "earlier", None, b"result before the restart"),
    ]


def test_a_restart_copies_the_output_when_it_cannot_move_it(
    tmp_path: Path, fake_render: list[bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = archive_output

    def no_rename(job_dir: Path, **kwargs: Any) -> Any:
        if kwargs.get("move"):
            raise OSError("Permission denied")
        return real(job_dir, **kwargs)

    monkeypatch.setattr("super_processor.wizard.archive_output", no_rename)
    controller, _job_id, job_dir = _job_on(tmp_path, WizardStep.RESULT)
    (job_dir / "output.mp4").write_bytes(b"result before the restart")
    controller.handle_post("/result", {"note": ["start over"], "restart": ["1"]})

    state = controller.current_state()
    assert state.step is WizardStep.SAMPLING and state.notice is None
    assert not (job_dir / "output.mp4").exists()
    assert _kept(job_dir) == [
        ("v001.mp4", "earlier", None, b"result before the restart"),
    ]


def test_a_restart_that_cannot_keep_the_output_is_refused(
    tmp_path: Path, fake_render: list[bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    def full_disk(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr("super_processor.wizard.archive_output", full_disk)
    controller, _job_id, job_dir = _job_on(tmp_path, WizardStep.RESULT)
    (job_dir / "output.mp4").write_bytes(b"result before the restart")
    controller.handle_post("/result", {"note": ["start over"], "restart": ["1"]})

    state = controller.current_state()
    assert state.step is WizardStep.RESULT
    assert state.error is not None and "No space left" in state.error
    assert state.render_note is None
    # Nothing was deleted, and nothing was started.
    assert (job_dir / "output.mp4").read_bytes() == b"result before the restart"
    assert (job_dir / "upscale_plan.json").is_file()
    assert controller.api_view()["run"] is None


def test_a_restart_does_not_keep_an_output_twice(
    tmp_path: Path, fake_render: list[bytes]
) -> None:
    controller, _job_id, job_dir = _job_on(tmp_path, WizardStep.RENDERING)
    controller.resume()
    _render(controller)
    controller.handle_post("/result", {"note": ["start over"], "restart": ["1"]})

    assert not (job_dir / "output.mp4").exists()
    assert _kept(job_dir) == [("v001.mp4", "render", None, b"render 1")]


def test_a_version_that_cannot_be_kept_is_a_warning_only(
    tmp_path: Path, fake_render: list[bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    def full_disk(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr("super_processor.wizard.archive_output", full_disk)
    controller, _job_id, job_dir = _job_on(tmp_path, WizardStep.RENDERING)
    controller.resume()
    _render(controller)

    state = controller.current_state()
    assert state.error is not None
    assert "could not be kept as a version" in state.error
    assert "No space left" in state.error
    assert (job_dir / "output.mp4").read_bytes() == b"render 1"
    assert controller.api_view()["run"]["outcome"] == "done"


def test_going_back_to_the_segments_drops_the_render_note(
    tmp_path: Path, fake_render: list[bytes]
) -> None:
    controller, _job_id, job_dir = _job_on(tmp_path, WizardStep.RESULT)
    state = controller.current_state()
    state.render_note = "a note for a render that never ran"
    save_session_state(controller.jobs_dir, state)
    save_job_wizard_state(job_dir, state)
    save_segment_list(
        job_dir,
        SegmentList(segments=[LoopSegment(0.0, 8.0, label="all")], approved=True),
    )
    controller.handle_post("/goto", {"to": ["segments"]})

    state = controller.current_state()
    assert state.step is WizardStep.SEGMENTS
    assert state.render_note is None
