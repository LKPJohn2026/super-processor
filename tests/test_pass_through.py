"""Spans and previews that skip FlashVSR and are only resized."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from super_processor.gemini import GeminiClient
from super_processor.look import ShotLook
from super_processor.regions import Region
from super_processor.shot_plan import new_plan, render_previews
from super_processor.shots import Shot, ShotList
from super_processor.upscale import (
    FakeUpscaleEngine,
    UpscalePlan,
    UpscaleRequest,
    UpscaleSpan,
    apply_range_revise,
    load_upscale_plan,
    render_shots,
    save_upscale_plan,
    span_at,
    video_stream_facts,
)
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    save_job_wizard_state,
    save_session_state,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg required"
)


class _CountingEngine(FakeUpscaleEngine):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def upscale(self, request: UpscaleRequest) -> Path:
        self.calls += 1
        return super().upscale(request)


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
            f"testsrc2=size=160x90:rate=10:duration={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )
    return path


def _streams(video: Path) -> dict[str, tuple[int, int] | None]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    found: dict[str, tuple[int, int] | None] = {}
    for stream in json.loads(out)["streams"]:
        kind = stream["codec_type"]
        found[kind] = (stream["width"], stream["height"]) if kind == "video" else None
    return found


def _frame_hashes(video: Path) -> list[str]:
    out = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-map",
            "0:v",
            "-f",
            "framemd5",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [
        line.rsplit(",", 1)[-1].strip()
        for line in out.splitlines()
        if line[:1].isdigit()
    ]


def test_span_keeps_use_upscale() -> None:
    span = UpscaleSpan(0, 2, use_upscale=False)
    assert span.to_dict()["use_upscale"] is False
    assert UpscaleSpan.from_dict(span.to_dict()).use_upscale is False
    assert UpscaleSpan.from_dict({"start_s": 0, "end_s": 2}).use_upscale is True
    plan = UpscalePlan(spans=(span,))
    assert UpscalePlan.from_dict(plan.to_dict()).spans[0].use_upscale is False


class _RevisionTransport:
    def __init__(self, start: float, end: float, scale: int) -> None:
        self.payload = {
            "start_s": start,
            "end_s": end,
            "scale": scale,
            "strength": 0.2,
        }
        self.calls = 0
        self.prompts: list[str] = []

    def generate(
        self, *, model: str, api_key: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls += 1
        self.prompts.append(body["contents"][-1]["parts"][0]["text"])
        return {
            "candidates": [{"content": {"parts": [{"text": json.dumps(self.payload)}]}}]
        }


@needs_ffmpeg
@pytest.mark.parametrize(
    ("start", "end", "scale", "revision_calls"),
    [(1, 2, 2, 0), (1, 5, 2, 2), (0, 6, 2, 1), (1, 2, 4, 1), (1, 5, 4, 2)],
)
def test_result_note_preserves_each_shots_settings(
    tmp_path: Path, start: float, end: float, scale: int, revision_calls: int
) -> None:
    source = _clip(tmp_path / "in.mp4")
    engine = _CountingEngine()
    transport = _RevisionTransport(start, end, scale)
    controller = WizardController(
        tmp_path / "jobs",
        gemini=GeminiClient(api_key="fake", transport=transport),
        upscale_engine=engine,
    )
    job_id = controller.store.create(source).job_id
    job_dir = controller.store.job_dir(job_id)
    original = UpscalePlan(
        spans=(
            UpscaleSpan(
                0,
                3,
                use_upscale=False,
                look=ShotLook(brightness=0.02),
                regions=(Region("text", 0.1, 0.1, 0.4, 0.4),),
            ),
            UpscaleSpan(
                3,
                6,
                look=ShotLook(saturation=0.9),
                regions=(Region("faces", 0.5, 0.2, 0.8, 0.6),),
            ),
        )
    )
    save_upscale_plan(job_dir, original)
    state = WizardState(step=WizardStep.RENDERING, job_id=job_id)
    save_job_wizard_state(job_dir, state)
    save_session_state(controller.jobs_dir, state)
    controller.run_render()
    assert controller.current_state().step is WizardStep.RESULT
    assert engine.calls == 1

    controller.handle_post("/result", {"note": ["less invented detail"]})
    state = controller.current_state()
    assert state.step is WizardStep.RENDERING and state.error is None
    assert transport.calls == 1
    assert "scale=2, strength=0.50, applied from 3.0s to 6.0s." in transport.prompts[0]
    updated = load_upscale_plan(job_dir)
    assert updated is not None
    for span in updated.spans:
        under = span_at(original, (span.start_s + span.end_s) / 2)
        assert under is not None
        assert (span.use_upscale, span.look, span.regions) == (
            under.use_upscale,
            under.look,
            under.regions,
        )
        assert span.scale == scale
        overlaps = span.start_s < end and span.end_s > start
        assert span.strength == (0.2 if overlaps else under.strength)
    controller.run_render()
    state = controller.current_state()
    assert state.step is WizardStep.RESULT and state.error is None
    assert engine.calls == 1 + revision_calls
    output = job_dir / "output.mp4"
    assert video_stream_facts(output)[0] == 60
    assert _streams(output)["video"] == (160 * scale, 90 * scale)
    assert "audio" in _streams(output)
    saved = load_upscale_plan(job_dir)
    assert saved is not None and saved.pending is None
    assert saved.latest_revision == UpscaleSpan(start, end, scale=scale, strength=0.2)

    # Reopen after rendering: the full request must survive splitting and saving.
    controller = WizardController(
        controller.jobs_dir,
        gemini=GeminiClient(api_key="fake", transport=transport),
        upscale_engine=engine,
    )
    transport.payload["strength"] = 0.1
    controller.handle_post("/result", {"note": ["lower it again over the same range"]})
    state = controller.current_state()
    assert state.step is WizardStep.RENDERING and state.error is None
    assert transport.calls == 2
    assert (
        f"scale={scale}, strength=0.20, applied from {start:.1f}s to {end:.1f}s."
        in transport.prompts[-1]
    )
    followup = load_upscale_plan(job_dir)
    assert followup is not None
    assert followup.latest_revision == UpscaleSpan(
        start, end, scale=scale, strength=0.1
    )
    for span in followup.spans:
        under = span_at(original, (span.start_s + span.end_s) / 2)
        assert under is not None
        assert (span.use_upscale, span.look, span.regions) == (
            under.use_upscale,
            under.look,
            under.regions,
        )
        overlaps = span.start_s < end and span.end_s > start
        assert span.strength == (0.1 if overlaps else under.strength)
    controller.run_render()
    state = controller.current_state()
    assert state.step is WizardStep.RESULT and state.error is None
    saved = load_upscale_plan(job_dir)
    assert saved is not None and saved.pending is None
    assert saved.latest_revision == followup.latest_revision
    assert video_stream_facts(output)[0] == 60


@needs_ffmpeg
def test_mixed_spans_stitch_with_every_frame_and_audio(tmp_path: Path) -> None:
    source = _clip(tmp_path / "in.mp4")
    engine = _CountingEngine()
    output = tmp_path / "out" / "output.mp4"
    render_shots(
        source,
        output,
        (
            UpscaleSpan(0, 2, strength=0.5),
            UpscaleSpan(2, 4, strength=0.5, use_upscale=False),
            UpscaleSpan(4, 6, strength=0.5),
        ),
        engine=engine,
        duration_s=6,
    )
    assert engine.calls == 2
    assert video_stream_facts(output)[0] == 60
    streams = _streams(output)
    assert streams["video"] == (320, 180)
    assert "audio" in streams


@needs_ffmpeg
def test_pass_through_is_the_plain_base_of_the_blend(tmp_path: Path) -> None:
    source = _clip(tmp_path / "in.mp4", seconds=3)
    engine = _CountingEngine()
    plain = render_shots(
        source,
        tmp_path / "plain" / "output.mp4",
        (UpscaleSpan(0, 3, strength=0.7, use_upscale=False),),
        engine=engine,
        duration_s=3,
    )
    assert engine.calls == 0
    zero = render_shots(
        source,
        tmp_path / "zero" / "output.mp4",
        (UpscaleSpan(0, 3, strength=0.0),),
        engine=engine,
        duration_s=3,
    )
    assert engine.calls == 1
    assert _frame_hashes(plain) == _frame_hashes(zero)


@needs_ffmpeg
def test_range_revise_can_pass_through(tmp_path: Path) -> None:
    source = _clip(tmp_path / "in.mp4")
    engine = _CountingEngine()
    output = tmp_path / "out" / "output.mp4"
    render_shots(
        source, output, (UpscaleSpan(0, 6, strength=0.5),), engine=engine, duration_s=6
    )
    calls = engine.calls
    apply_range_revise(
        source,
        output,
        UpscaleSpan(2, 4, strength=0.5, use_upscale=False),
        output,
        engine=engine,
        duration_s=6,
    )
    assert engine.calls == calls
    assert video_stream_facts(output)[0] == 60


@needs_ffmpeg
def test_previews_skip_the_engine_for_pass_through_shots(tmp_path: Path) -> None:
    from dataclasses import replace

    source = _clip(tmp_path / "in.mp4")
    job_dir = tmp_path / "job"
    shots = ShotList(shots=[Shot(0, 3), Shot(3, 6)])
    plan = new_plan(shots, 2)
    plan.recipes[1] = replace(plan.recipes[1], use_upscale=False)
    engine = _CountingEngine()
    plan = render_previews(source, job_dir, shots, plan, [0, 1], engine=engine)
    assert engine.calls == 1
    for index in (0, 1):
        after = job_dir / "previews" / f"shot_{index:03d}_r1_after.mp4"
        assert _streams(after)["video"] == (320, 180)


class _HalfwayEngine(FakeUpscaleEngine):
    """Reports half of each clip done before finishing it."""

    def upscale(self, request: UpscaleRequest) -> Path:
        if request.on_progress is not None:
            request.on_progress(0.5)
        return super().upscale(request)


@needs_ffmpeg
def test_render_reports_progress_by_source_frames(tmp_path: Path) -> None:
    source = _clip(tmp_path / "in.mp4")
    seen: list[float] = []
    render_shots(
        source,
        tmp_path / "out" / "output.mp4",
        (
            UpscaleSpan(0, 3, strength=0.5),
            UpscaleSpan(3, 4.5, strength=0.5, use_upscale=False),
            UpscaleSpan(4.5, 6, strength=0.2),
        ),
        engine=_HalfwayEngine(),
        duration_s=6,
        on_progress=seen.append,
    )
    # Part one: halfway, then done; the pass-through part only reports done.
    assert seen == pytest.approx([0.25, 0.5, 0.75, 0.875, 1.0])
    single: list[float] = []
    render_shots(
        source,
        tmp_path / "one" / "output.mp4",
        (UpscaleSpan(0, 6, strength=0.5),),
        engine=_HalfwayEngine(),
        duration_s=6,
        on_progress=single.append,
    )
    assert single == [0.5]


@needs_ffmpeg
@pytest.mark.parametrize("audio_s", [4.7, 8.0])
def test_audio_of_another_length_keeps_every_frame(
    tmp_path: Path, audio_s: float
) -> None:
    source = tmp_path / "in.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=10:duration=6",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={audio_s}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    output = tmp_path / "out" / "output.mp4"
    render_shots(
        source,
        output,
        (UpscaleSpan(0, 3, strength=0.5), UpscaleSpan(3, 6, strength=0.2)),
        engine=FakeUpscaleEngine(),
        duration_s=6,
    )
    # -shortest used to cut the picture to 4.7 s of audio: 13 frames lost.
    assert video_stream_facts(output)[0] == 60
    assert "audio" in _streams(output)
    length = float(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    assert length == pytest.approx(6.0, abs=0.1)


@needs_ffmpeg
def test_a_look_does_not_shift_black_at_the_cut(tmp_path: Path) -> None:
    from super_processor.look import ShotLook

    source = tmp_path / "black.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:size=160x90:rate=10:duration=4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    output = tmp_path / "out" / "output.mp4"
    credits = ShotLook(contrast=1.1, brightness=0.02)
    render_shots(
        source,
        output,
        (
            UpscaleSpan(0, 2, strength=0.5),
            UpscaleSpan(2, 4, strength=0.0, use_upscale=False, look=credits),
        ),
        engine=FakeUpscaleEngine(),
        duration_s=4,
    )
    log = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(output),
            "-vf",
            "signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    levels = [float(v.split()[0]) for v in log.split("YAVG=")[1:]]
    assert len(levels) == 40
    # The trailer's credits dropped black from 16 to 7.5 here.
    assert max(levels) - min(levels) < 1.0
    assert min(levels) >= 15.5
