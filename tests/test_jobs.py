"""Tests for the resumable job store."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_processor.jobs import (
    JOB_ID_PATTERN,
    InvalidTransitionError,
    JobError,
    JobNotFoundError,
    JobState,
    JobStore,
)


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"fake-video")
    return path


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "jobs")


def test_create_job_starts_imported(store: JobStore, source_file: Path) -> None:
    manifest = store.create(source_file)

    assert JOB_ID_PATTERN.fullmatch(manifest.job_id)
    assert manifest.state is JobState.IMPORTED
    assert Path(manifest.source_path) == source_file.resolve()
    assert store.manifest_path(manifest.job_id).is_file()


def test_create_rejects_missing_source(store: JobStore, tmp_path: Path) -> None:
    with pytest.raises(JobError, match="not a readable file"):
        store.create(tmp_path / "missing.mp4")


def test_load_and_list_jobs(store: JobStore, source_file: Path) -> None:
    first = store.create(source_file)
    second = store.create(source_file)

    loaded = store.load(first.job_id)
    assert loaded.job_id == first.job_id
    assert loaded.state is JobState.IMPORTED

    jobs = store.list_jobs()
    assert {job.job_id for job in jobs} == {first.job_id, second.job_id}


def test_load_missing_job(store: JobStore) -> None:
    with pytest.raises(JobNotFoundError, match="job not found"):
        store.load("deadbeefdeadbeef")


def test_legal_transition_to_probed(store: JobStore, source_file: Path) -> None:
    manifest = store.create(source_file)
    updated = store.transition(manifest.job_id, JobState.PROBED, notes={"probe": "ok"})

    assert updated.state is JobState.PROBED
    assert updated.notes["probe"] == "ok"
    assert store.load(manifest.job_id).state is JobState.PROBED


def test_illegal_transition_is_rejected(store: JobStore, source_file: Path) -> None:
    manifest = store.create(source_file)
    with pytest.raises(InvalidTransitionError, match="cannot move"):
        store.transition(manifest.job_id, JobState.ENCODING)


def test_fail_clears_on_recovery_path(store: JobStore, source_file: Path) -> None:
    manifest = store.create(source_file)
    failed = store.transition(
        manifest.job_id,
        JobState.FAILED,
        error="probe crashed",
    )
    assert failed.state is JobState.FAILED
    assert failed.error == "probe crashed"

    with pytest.raises(InvalidTransitionError):
        store.transition(manifest.job_id, JobState.PROBED)


def test_full_happy_path_to_complete(store: JobStore, source_file: Path) -> None:
    manifest = store.create(source_file)
    sequence = [
        JobState.PROBED,
        JobState.DIAGNOSED,
        JobState.PLANNED,
        JobState.VALIDATED,
        JobState.PREVIEWED,
        JobState.APPROVED,
        JobState.ENCODING,
        JobState.COMPLETE,
    ]
    for target in sequence:
        manifest = store.transition(manifest.job_id, target)
    assert manifest.state is JobState.COMPLETE
    with pytest.raises(InvalidTransitionError):
        store.transition(manifest.job_id, JobState.FAILED)
