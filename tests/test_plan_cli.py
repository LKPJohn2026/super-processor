"""Tests for `super-processor plans`."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_processor.cli import main
from super_processor.jobs import JobState, JobStore
from super_processor.segments import TimelineSegment, write_segments

JOB_ID = "abcd1234abcd1234"


def _accepted_job(tmp_path: Path) -> JobStore:
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
            TimelineSegment(0, 0.0, 60.0, "indoor", "noisy", 10.0, 0),
            TimelineSegment(1, 60.0, 120.0, "indoor", "low_contrast", 70.0, 1),
        ],
    )
    return store


def test_plans_prints_five_assignments(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = _accepted_job(tmp_path)
    code = main(["--jobs-dir", str(store.root), "plans", JOB_ID])
    output = capsys.readouterr().out
    assert code == 0
    assert store.load(JOB_ID).state is JobState.PLANS_READY
    lines = [line for line in output.splitlines() if line[:1].isdigit()]
    assert len(lines) == 5
    assert lines[0].startswith("0  2.0  ")
    assert "noisy." in lines[0]
    assert "low_contrast." in lines[0]

    again = main(["--jobs-dir", str(store.root), "plans", JOB_ID])
    assert again == 0
    assert store.load(JOB_ID).state is JobState.PLANS_READY


def test_plans_before_accept_is_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake-video")
    store = JobStore(tmp_path / "jobs")
    store.create(source, job_id=JOB_ID)
    store.transition(JOB_ID, JobState.PROBED)
    assert main(["--jobs-dir", str(store.root), "plans", JOB_ID]) == 1
    assert "accepted split" in capsys.readouterr().out
