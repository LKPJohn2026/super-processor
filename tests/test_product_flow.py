"""End-to-end CV CLI flow tests (diagnose → plan → preview gate helpers)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from super_processor import cli
from super_processor.jobs import JobState, JobStore


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


pytestmark = pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not on PATH")


def _clip(path: Path, duration: float = 0.8) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration:g}:size=320x240:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def test_models_and_show_commands(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["models"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "endpoints" in payload

    jobs_dir = tmp_path / "jobs"
    source = _clip(tmp_path / "clip.mp4")
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
    assert cli.main(["--jobs-dir", str(jobs_dir), "show", "abcd1234abcd1234"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["manifest"]["job_id"] == "abcd1234abcd1234"
    assert "artifacts" in shown


def test_diagnose_plan_preview_flow(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    jobs_dir = tmp_path / "jobs"
    source = _clip(tmp_path / "clip.mp4", duration=1.0)
    assert (
        cli.main(
            [
                "--jobs-dir",
                str(jobs_dir),
                "diagnose",
                str(source),
                "--job-id",
                "abcd1234abcd1234",
                "--max-size-mb",
                "80",
            ]
        )
        == 0
    )
    diagnosis = json.loads(capsys.readouterr().out)
    assert "suggestions" in diagnosis
    store = JobStore(jobs_dir)
    assert store.load("abcd1234abcd1234").state is JobState.DIAGNOSED

    assert (
        cli.main(
            [
                "--jobs-dir",
                str(jobs_dir),
                "--safe-mode",
                "plan",
                "abcd1234abcd1234",
                "--instruction",
                "less denoise",
            ]
        )
        == 0
    )
    planned = json.loads(capsys.readouterr().out)
    assert planned["validation"]["ok"] is True
    assert store.load("abcd1234abcd1234").state is JobState.VALIDATED

    assert (
        cli.main(
            [
                "--jobs-dir",
                str(jobs_dir),
                "preview",
                "abcd1234abcd1234",
                "--dry-run",
            ]
        )
        == 0
    )
    dry = json.loads(capsys.readouterr().out)
    assert "steps" in dry

    assert (
        cli.main(
            [
                "--jobs-dir",
                str(jobs_dir),
                "preview",
                "abcd1234abcd1234",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert store.load("abcd1234abcd1234").state is JobState.PREVIEWED
    assert (store.job_dir("abcd1234abcd1234") / "preview.mp4").is_file()
    assert (store.job_dir("abcd1234abcd1234") / "qa_report.json").is_file()

    assert (
        cli.main(
            [
                "--jobs-dir",
                str(jobs_dir),
                "apply",
                "abcd1234abcd1234",
                "--approve",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert store.load("abcd1234abcd1234").state is JobState.COMPLETE
    assert (store.job_dir("abcd1234abcd1234") / "output.mp4").is_file()
