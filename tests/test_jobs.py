"""Tests for the resumable job store."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from super_processor.jobs import (
    DEFAULT_JOBS_DIR_ENV,
    JOB_ID_PATTERN,
    InvalidTransitionError,
    JobError,
    JobNotFoundError,
    JobState,
    JobStore,
    default_jobs_root,
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


def test_a_failed_job_keeps_its_error_and_cannot_move_on(
    store: JobStore, source_file: Path
) -> None:
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


@pytest.mark.parametrize("job_id", ["../escape1", "ABCDEFGH", "short", "a" * 33])
def test_a_job_id_that_is_not_a_short_lowercase_token_is_refused(
    store: JobStore, source_file: Path, job_id: str
) -> None:
    # The id names a directory under the jobs root, so "../" must never pass.
    with pytest.raises(JobError, match="invalid job id"):
        store.create(source_file, job_id=job_id)
    with pytest.raises(JobError, match="invalid job id"):
        store.load(job_id)
    assert not (store.root.parent / "escape1").exists()


def test_creating_a_job_over_an_existing_one_is_refused(
    store: JobStore, source_file: Path
) -> None:
    first = store.create(source_file, job_id="abcd1234abcd1234")
    store.transition(first.job_id, JobState.PROBED)
    with pytest.raises(JobError, match="job already exists"):
        store.create(source_file, job_id="abcd1234abcd1234")
    # The earlier job's progress is not reset to imported.
    assert store.load(first.job_id).state is JobState.PROBED


def test_list_skips_a_corrupt_job_and_puts_the_newest_first(
    store: JobStore, source_file: Path
) -> None:
    older = store.create(source_file, job_id="aaaa1111aaaa1111")
    newer = store.create(source_file, job_id="bbbb2222bbbb2222")
    for job_id, stamp in ((older.job_id, "2026-01-01"), (newer.job_id, "2026-02-01")):
        path = store.manifest_path(job_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["updated_at"] = f"{stamp}T00:00:00+00:00"
        path.write_text(json.dumps(data), encoding="utf-8")
    broken = store.root / "cccc3333cccc3333"
    broken.mkdir()
    (broken / "manifest.json").write_text("{not json", encoding="utf-8")
    (store.root / "not-a-job.txt").write_text("x", encoding="utf-8")

    assert [job.job_id for job in store.list_jobs()] == [newer.job_id, older.job_id]


def test_a_failed_manifest_write_keeps_the_old_manifest(
    store: JobStore, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = store.create(source_file)
    before = store.manifest_path(manifest.job_id).read_text(encoding="utf-8")

    def full_disk(_fd: int) -> None:
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", full_disk)
        with pytest.raises(OSError, match="No space left"):
            store.transition(manifest.job_id, JobState.PROBED)
    job_dir = store.job_dir(manifest.job_id)
    assert store.manifest_path(manifest.job_id).read_text(encoding="utf-8") == before
    assert [path.name for path in job_dir.iterdir()] == ["manifest.json"]


def test_the_jobs_root_follows_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DEFAULT_JOBS_DIR_ENV, str(tmp_path / "custom"))
    assert default_jobs_root() == (tmp_path / "custom").resolve()
    monkeypatch.delenv(DEFAULT_JOBS_DIR_ENV)
    # Path.home() reads HOME on POSIX and USERPROFILE on Windows.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    assert default_jobs_root() == (tmp_path / "home/.super-processor/jobs").resolve()
