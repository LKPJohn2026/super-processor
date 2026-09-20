"""CLI coverage for job commands."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_processor import cli


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
    created = capsys.readouterr().out
    assert '"job_id": "abcd1234abcd1234"' in created
    assert '"state": "imported"' in created

    assert (
        cli.main(["--jobs-dir", str(jobs_dir), "job", "show", "abcd1234abcd1234"]) == 0
    )
    shown = capsys.readouterr().out
    assert '"source_path"' in shown

    assert cli.main(["--jobs-dir", str(jobs_dir), "job", "list"]) == 0
    listed = capsys.readouterr().out
    assert "abcd1234abcd1234" in listed


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
    assert "error:" in capsys.readouterr().out
