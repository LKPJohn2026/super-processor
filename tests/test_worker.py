"""Tests for the FFmpeg worker and job run CLI."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from super_processor import cli
from super_processor.jobs import JobState, JobStore
from super_processor.probe import probe_file, write_media_facts
from super_processor.recipe import OpName, TargetMode, empty_recipe, write_recipe
from super_processor.templates import build_ffmpeg_plan
from super_processor.worker import FFmpegWorker, parse_progress_line


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


pytestmark = pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not on PATH")


def _make_clip(path: Path, *, duration: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration:g}:size=320x240:rate=10",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration:g}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def test_parse_progress_line() -> None:
    progress: dict[str, str] = {}
    parse_progress_line("out_time_ms=1234", progress)
    parse_progress_line("progress=continue", progress)
    parse_progress_line("ignored", progress)
    assert progress["out_time_ms"] == "1234"
    assert progress["progress"] == "continue"


def test_worker_preview_encode_atomic(tmp_path: Path) -> None:
    source = _make_clip(tmp_path / "src.mp4")
    recipe = empty_recipe("abcd1234abcd1234", str(source))
    for op in recipe.ops:
        if op.op is OpName.CONTRAST:
            op.enabled = True
            op.params = {"contrast": 1.1, "brightness": 0.0, "gamma": 1.0}
    recipe.target.mode = TargetMode.PREVIEW
    recipe.target.preview_window.duration_s = 0.5
    recipe.encode.preset = "ultrafast"
    recipe.encode.crf = 28

    output = tmp_path / "preview.mp4"
    plan = build_ffmpeg_plan(
        recipe,
        output_path=output,
        work_dir=tmp_path / "work",
        duration_s=1.0,
    )
    result = FFmpegWorker().run_plan(plan)
    assert result.ok is True
    assert output.is_file()
    assert not output.with_name(f"{output.stem}.partial{output.suffix}").exists()
    assert result.steps[0].returncode == 0


def test_worker_cancel(tmp_path: Path) -> None:
    source = _make_clip(tmp_path / "src.mp4", duration=8.0)
    recipe = empty_recipe("abcd1234abcd1234", str(source))
    recipe.target.mode = TargetMode.FINAL
    output = tmp_path / "out.mp4"
    plan = build_ffmpeg_plan(
        recipe,
        output_path=output,
        work_dir=tmp_path / "work",
        duration_s=8.0,
    )
    # Force a slower encode so cancel has time to land.
    encode = plan.steps[-1]
    crf_index = encode.argv.index("-crf")
    encode.argv[crf_index + 1] = "18"
    preset_index = encode.argv.index("-preset")
    encode.argv[preset_index + 1] = "slow"

    worker = FFmpegWorker()

    def _cancel_soon() -> None:
        time.sleep(0.2)
        worker.cancel()

    thread = threading.Thread(target=_cancel_soon)
    thread.start()
    result = worker.run_plan(plan)
    thread.join()
    assert result.cancelled is True or result.ok is False
    assert not output.is_file()


def test_job_run_dry_run_and_preview(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    jobs_dir = tmp_path / "jobs"
    source = _make_clip(tmp_path / "clip.mp4")
    store = JobStore(jobs_dir)
    manifest = store.create(source, job_id="abcd1234abcd1234")
    facts = probe_file(source)
    write_media_facts(store.job_dir(manifest.job_id), facts)
    store.transition(manifest.job_id, JobState.PROBED)

    recipe = empty_recipe(manifest.job_id, str(source.resolve()))
    for op in recipe.ops:
        if op.op is OpName.CONTRAST:
            op.enabled = True
            op.params = {"contrast": 1.05, "brightness": 0.0, "gamma": 1.0}
    recipe.target.preview_window.duration_s = 0.4
    recipe.encode.preset = "ultrafast"
    write_recipe(store.job_dir(manifest.job_id), recipe)

    assert (
        cli.main(
            [
                "--jobs-dir",
                str(jobs_dir),
                "job",
                "run",
                manifest.job_id,
                "--mode",
                "preview",
                "--dry-run",
            ]
        )
        == 0
    )
    dry = capsys.readouterr().out
    assert '"filter_graph"' in dry
    assert "-vf" in dry or "eq=contrast=" in dry

    assert (
        cli.main(
            [
                "--jobs-dir",
                str(jobs_dir),
                "job",
                "run",
                manifest.job_id,
                "--mode",
                "preview",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert '"ok": true' in out
    updated = store.load(manifest.job_id)
    assert updated.state is JobState.PREVIEWED
    assert (store.job_dir(manifest.job_id) / "preview.mp4").is_file()


def test_job_run_final_requires_approve(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    jobs_dir = tmp_path / "jobs"
    source = _make_clip(tmp_path / "clip.mp4")
    assert (
        cli.main(
            [
                "--jobs-dir",
                str(jobs_dir),
                "job",
                "create",
                str(source),
                "--job-id",
                "abcd1234abcd1234",
            ]
        )
        == 0
    )
    capsys.readouterr()
    store = JobStore(jobs_dir)
    write_media_facts(store.job_dir("abcd1234abcd1234"), probe_file(source))
    store.transition("abcd1234abcd1234", JobState.PROBED)
    write_recipe(
        store.job_dir("abcd1234abcd1234"),
        empty_recipe("abcd1234abcd1234", str(source.resolve())),
    )

    code = cli.main(
        [
            "--jobs-dir",
            str(jobs_dir),
            "job",
            "run",
            "abcd1234abcd1234",
            "--mode",
            "final",
        ]
    )
    assert code == 1
    assert "requires --approve" in capsys.readouterr().out


def test_job_run_final_approve(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    jobs_dir = tmp_path / "jobs"
    source = _make_clip(tmp_path / "clip.mp4", duration=0.5)
    store = JobStore(jobs_dir)
    manifest = store.create(source, job_id="abcd1234abcd1234")
    write_media_facts(store.job_dir(manifest.job_id), probe_file(source))
    store.transition(manifest.job_id, JobState.PROBED)
    recipe = empty_recipe(manifest.job_id, str(source.resolve()))
    recipe.encode.preset = "ultrafast"
    recipe.encode.crf = 28
    write_recipe(store.job_dir(manifest.job_id), recipe)

    assert (
        cli.main(
            [
                "--jobs-dir",
                str(jobs_dir),
                "job",
                "run",
                manifest.job_id,
                "--mode",
                "final",
                "--approve",
            ]
        )
        == 0
    )
    capsys.readouterr()
    updated = store.load(manifest.job_id)
    assert updated.state is JobState.COMPLETE
    assert (store.job_dir(manifest.job_id) / "output.mp4").is_file()


def test_job_validate_transitions_to_validated(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    jobs_dir = tmp_path / "jobs"
    source = _make_clip(tmp_path / "clip.mp4", duration=0.5)
    store = JobStore(jobs_dir)
    manifest = store.create(source, job_id="abcd1234abcd1234")
    write_media_facts(store.job_dir(manifest.job_id), probe_file(source))
    store.transition(manifest.job_id, JobState.PROBED)
    write_recipe(
        store.job_dir(manifest.job_id),
        empty_recipe(manifest.job_id, str(source.resolve())),
    )
    assert (
        cli.main(["--jobs-dir", str(jobs_dir), "job", "validate", manifest.job_id]) == 0
    )
    capsys.readouterr()
    assert store.load(manifest.job_id).state is JobState.VALIDATED
