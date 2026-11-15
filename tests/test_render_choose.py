"""Tests for choosing a plan and rendering it."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from super_processor.cli import main
from super_processor.doctor import CheckResult
from super_processor.jobs import JobState, JobStore
from super_processor.render import RenderError, ensure_encoder, render_chosen_plan
from super_processor.segments import TimelineSegment, write_segments
from super_processor.treatments import treatments_for

JOB_ID = "abcd1234abcd1234"

pytestmark_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg not on PATH",
)


def _ready_job(tmp_path: Path) -> JobStore:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake-video")
    store = JobStore(tmp_path / "jobs")
    store.create(source, job_id=JOB_ID)
    store.transition(JOB_ID, JobState.PROBED)
    store.transition(JOB_ID, JobState.SPLIT_PROPOSED)
    store.transition(JOB_ID, JobState.SPLIT_ACCEPTED)
    write_segments(
        store.job_dir(JOB_ID),
        [
            TimelineSegment(0, 0.0, 5.0, "indoor", "noisy", 2.0, 0),
            TimelineSegment(1, 5.0, 10.0, "indoor", "low_contrast", 7.0, 1),
        ],
    )
    return store


def test_choose_stores_the_plan_and_prints_the_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ready_job(tmp_path)
    assert main(["--jobs-dir", str(store.root), "plans", JOB_ID]) == 0
    capsys.readouterr()

    def fake(
        job_dir: Path,
        source: Path,
        segments: list[TimelineSegment],
        treatments: list[object],
        **_kwargs: object,
    ) -> Path:
        assert source.name == "clip.mp4"
        assert len(segments) == len(treatments) == 2
        dest = job_dir / "output.mp4"
        dest.write_bytes(b"rendered")
        return dest

    monkeypatch.setattr("super_processor.cli.ensure_encoder", lambda _encoder: None)
    monkeypatch.setattr("super_processor.cli.render_chosen_plan", fake)
    code = main(["--jobs-dir", str(store.root), "plans", JOB_ID, "--choose", "1"])
    assert code == 0
    assert capsys.readouterr().out.strip().endswith("output.mp4")
    manifest = store.load(JOB_ID)
    assert manifest.state is JobState.COMPLETE
    assert manifest.notes["chosen_plan"] == 1


def test_missing_encoder_fails_before_ffmpeg(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ready_job(tmp_path)
    assert main(["--jobs-dir", str(store.root), "plans", JOB_ID]) == 0
    capsys.readouterr()

    def missing(*_args: object, **_kwargs: object) -> CheckResult:
        return CheckResult(
            name="encoder:h264_nvenc",
            ok=False,
            detail=(
                "h264_nvenc missing "
                "(install ffmpeg with this encoder or render with libx264)"
            ),
        )

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("ffmpeg ran")

    monkeypatch.setattr("super_processor.render.check_ffmpeg_capability", missing)
    monkeypatch.setattr("super_processor.render.subprocess.run", boom)
    code = main(
        [
            "--jobs-dir",
            str(store.root),
            "plans",
            JOB_ID,
            "--choose",
            "0",
            "--encoder",
            "h264_nvenc",
        ]
    )
    assert code == 1
    assert "h264_nvenc missing" in capsys.readouterr().out
    assert store.load(JOB_ID).state is JobState.PLANS_READY


def test_choose_refuses_an_index_outside_the_five_plans(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = _ready_job(tmp_path)
    assert main(["--jobs-dir", str(store.root), "plans", JOB_ID]) == 0
    capsys.readouterr()
    code = main(["--jobs-dir", str(store.root), "plans", JOB_ID, "--choose", "9"])
    assert code == 1
    assert "outside" in capsys.readouterr().out
    assert store.load(JOB_ID).state is JobState.PLANS_READY


def test_a_failed_render_marks_the_job_failed_with_the_ffmpeg_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ready_job(tmp_path)
    assert main(["--jobs-dir", str(store.root), "plans", JOB_ID]) == 0
    capsys.readouterr()

    def broken(*_args: object, **_kwargs: object) -> Path:
        raise RenderError("Conversion failed!")

    monkeypatch.setattr("super_processor.cli.ensure_encoder", lambda _encoder: None)
    monkeypatch.setattr("super_processor.cli.render_chosen_plan", broken)
    code = main(["--jobs-dir", str(store.root), "plans", JOB_ID, "--choose", "0"])
    assert code == 1
    assert "Conversion failed!" in capsys.readouterr().out
    # A job left in ENCODING would look like a render still running.
    manifest = store.load(JOB_ID)
    assert manifest.state is JobState.FAILED
    assert manifest.error == "Conversion failed!"
    assert manifest.notes["chosen_plan"] == 0


def test_an_encoder_without_a_choice_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = _ready_job(tmp_path)
    argv = ["--jobs-dir", str(store.root), "plans", JOB_ID, "--encoder", "libx264"]
    assert main(argv) == 1
    assert "pass --encoder with --choose" in capsys.readouterr().out
    # Nothing was planned: the refusal comes before any work.
    assert store.load(JOB_ID).state is JobState.SPLIT_ACCEPTED


def test_ensure_encoder_refuses_a_name_outside_the_h264_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def never(*_args: object, **_kwargs: object) -> CheckResult:
        raise AssertionError("ffmpeg was asked about a refused encoder")

    monkeypatch.setattr("super_processor.render.check_ffmpeg_capability", never)
    with pytest.raises(RenderError, match="unsupported encoder libx265"):
        ensure_encoder("libx265")


def _two_segments() -> list[TimelineSegment]:
    return [
        TimelineSegment(0, 0.0, 5.0, "indoor", "noisy", 2.0, 0),
        TimelineSegment(1, 5.0, 10.0, "indoor", "low_contrast", 7.0, 1),
    ]


def _encoder_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "super_processor.render.check_ffmpeg_capability",
        lambda *_a, **_k: CheckResult(name="encoder:libx264", ok=True, detail="ok"),
    )


def test_a_plan_that_does_not_match_the_segments_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _encoder_present(monkeypatch)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("ffmpeg ran")

    monkeypatch.setattr("super_processor.render.subprocess.run", boom)
    with pytest.raises(RenderError, match="assignment length"):
        render_chosen_plan(
            tmp_path / "job",
            tmp_path / "clip.mp4",
            _two_segments(),
            [treatments_for("noisy")[0]],
        )


def test_a_failing_segment_encode_stops_with_ffmpeg_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _encoder_present(monkeypatch)
    calls: list[list[str]] = []

    def failing(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=1, stderr=b"clip.mp4: Invalid data\n")

    monkeypatch.setattr("super_processor.render.subprocess.run", failing)
    with pytest.raises(RenderError, match="^clip.mp4: Invalid data$"):
        render_chosen_plan(
            tmp_path / "job",
            tmp_path / "clip.mp4",
            _two_segments(),
            [treatments_for("noisy")[0], treatments_for("low_contrast")[0]],
        )
    # The second segment and the concat never run after the first fails.
    assert len(calls) == 1


def test_ffmpeg_that_cannot_start_is_a_render_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _encoder_present(monkeypatch)

    def missing(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr("super_processor.render.subprocess.run", missing)
    with pytest.raises(RenderError, match="failed to start ffmpeg"):
        render_chosen_plan(
            tmp_path / "job",
            tmp_path / "clip.mp4",
            _two_segments()[:1],
            [treatments_for("noisy")[0]],
        )


@pytestmark_ffmpeg
def test_render_concatenates_two_tiny_segments(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:duration=10:size=160x90:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    segments = [
        TimelineSegment(0, 0.0, 5.0, "indoor", "noisy", 2.0, 0),
        TimelineSegment(1, 5.0, 10.0, "indoor", "low_contrast", 7.0, 1),
    ]
    treatments = [treatments_for("noisy")[0], treatments_for("low_contrast")[0]]
    output = render_chosen_plan(
        tmp_path / "job",
        source,
        segments,
        treatments,
        include_audio=False,
    )
    probed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert abs(float(probed.stdout.strip()) - 10.0) < 0.2
