"""CLI coverage for job commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from super_processor import cli
from super_processor.jobs import JobState, JobStore
from super_processor.probe import media_facts_from_ffprobe, write_media_facts
from super_processor.recipe import load_recipe


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"fake-video")
    return path


def test_job_create_show_and_list(
    tmp_path: Path,
    source_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    jobs_dir = tmp_path / "jobs"
    assert (
        cli.main(
            [
                "--jobs-dir",
                str(jobs_dir),
                "job",
                "create",
                str(source_file),
                "--job-id",
                "abcd1234abcd1234",
            ]
        )
        == 0
    )
    created = json.loads(capsys.readouterr().out)
    assert created["job_id"] == "abcd1234abcd1234"
    assert created["state"] == "imported"
    assert created["source_path"] == str(source_file.resolve())

    assert (
        cli.main(["--jobs-dir", str(jobs_dir), "job", "show", "abcd1234abcd1234"]) == 0
    )
    shown = json.loads(capsys.readouterr().out)
    assert shown["manifest"]["job_id"] == "abcd1234abcd1234"
    # A fresh job has none of the later steps' files yet.
    assert shown["artifacts"]["recipe.json"] is False
    assert shown["artifacts"]["output.mp4"] is False

    assert cli.main(["--jobs-dir", str(jobs_dir), "job", "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [job["job_id"] for job in listed] == ["abcd1234abcd1234"]


def test_job_list_prints_an_empty_list_with_no_jobs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--jobs-dir", str(tmp_path / "jobs"), "job", "list"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_job_create_reports_missing_source(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = cli.main(
        [
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "job",
            "create",
            str(tmp_path / "nope.mp4"),
        ]
    )
    assert code == 1
    assert "error: source is not a readable file" in capsys.readouterr().out
    assert not (tmp_path / "jobs").exists() or not any((tmp_path / "jobs").iterdir())


def test_init_recipe_then_validate_moves_a_probed_job_to_validated(
    tmp_path: Path,
    source_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = JobStore(tmp_path / "jobs")
    job_id = store.create(source_file, job_id="abcd1234abcd1234").job_id
    write_media_facts(
        store.job_dir(job_id),
        media_facts_from_ffprobe(source_file, {"format": {"duration": "8"}}),
    )
    store.transition(job_id, JobState.PROBED)
    base = ["--jobs-dir", str(store.root), "job"]
    assert cli.main([*base, "init-recipe", job_id]) == 0
    recipe = json.loads(capsys.readouterr().out)
    assert recipe["job_id"] == job_id
    assert load_recipe(store.job_dir(job_id)).to_dict() == recipe
    assert cli.main([*base, "validate", job_id]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert store.load(job_id).state is JobState.VALIDATED


def test_apply_before_a_preview_is_refused(
    tmp_path: Path,
    source_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = JobStore(tmp_path / "jobs")
    job_id = store.create(source_file, job_id="abcd1234abcd1234").job_id
    code = cli.main(["--jobs-dir", str(store.root), "apply", job_id, "--approve"])
    assert code == 1
    assert "must be previewed before apply" in capsys.readouterr().out
    assert store.load(job_id).state is JobState.IMPORTED


def test_showing_a_job_that_does_not_exist_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["--jobs-dir", str(tmp_path / "jobs"), "show", "deadbeefdeadbeef"])
    assert code == 1
    assert "error: job not found: deadbeefdeadbeef" in capsys.readouterr().out
