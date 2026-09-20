"""Tests for the doctor diagnostics module."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_processor import cli, doctor


def test_which_and_check_tool_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("super_processor.doctor.shutil.which", lambda _: None)
    assert doctor.which("ffmpeg") is None
    result = doctor.check_tool("ffmpeg", "ffmpeg", "-version")
    assert result.ok is False
    assert "not found" in result.detail


def test_check_tool_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)

    monkeypatch.setattr("super_processor.doctor.shutil.which", lambda _: str(fake))

    class Completed:
        returncode = 0
        stdout = "ffmpeg version 7.0\n"
        stderr = ""

    monkeypatch.setattr(
        "super_processor.doctor.subprocess.run",
        lambda *args, **kwargs: Completed(),
    )
    result = doctor.check_tool("ffmpeg", "ffmpeg", "-version")
    assert result.ok is True
    assert "ffmpeg version 7.0" in result.detail
    assert result.path == str(fake.resolve())


def test_check_player_prefers_vlc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vlc = tmp_path / "vlc"
    vlc.write_text("#!/bin/sh\n", encoding="utf-8")
    vlc.chmod(0o755)

    def fake_which(command: str) -> str | None:
        return str(vlc) if command == "vlc" else None

    monkeypatch.setattr("super_processor.doctor.shutil.which", fake_which)

    class Completed:
        returncode = 0
        stdout = "VLC media player 3.0\n"
        stderr = ""

    monkeypatch.setattr(
        "super_processor.doctor.subprocess.run",
        lambda *args, **kwargs: Completed(),
    )
    result = doctor.check_player()
    assert result.ok is True
    assert result.detail.startswith("vlc:")


def test_check_jobs_dir(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    result = doctor.check_jobs_dir(root)
    assert result.ok is True
    assert root.exists()
    assert "GiB free" in result.detail


def test_collect_doctor_report_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("super_processor.doctor.shutil.which", lambda _: None)
    monkeypatch.delenv("SUPER_PROCESSOR_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    results = doctor.collect_doctor_report(tmp_path / "jobs")
    payload = doctor.doctor_report_as_dict(results)
    assert payload["ok"] is False
    assert any(item["name"] == "ffmpeg" for item in payload["checks"])
    text = doctor.format_doctor_text(results)
    assert "[FAIL] ffmpeg" in text
    assert "required tools: missing" in text


def test_ffmpeg_capability_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "super_processor.doctor._ffmpeg_listing",
        lambda kind: " V..... libx265\n",
    )
    result = doctor.check_ffmpeg_capability(
        "encoder:libx265",
        kind="encoders",
        token="libx265",
        required=True,
        hint="required",
    )
    assert result.ok is True
    assert "available" in result.detail


def test_ffmpeg_capability_missing_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "super_processor.doctor._ffmpeg_listing",
        lambda kind: "Filters:\n T. deshake\n",
    )
    result = doctor.check_ffmpeg_capability(
        "filter:stabilize",
        kind="filters",
        token="vidstabdetect",
        required=False,
        hint="falls back to deshake",
    )
    assert result.ok is True
    assert "missing" in result.detail


def test_ffmpeg_listing_without_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("super_processor.doctor.which", lambda _: None)
    assert doctor._ffmpeg_listing("filters") == ""


def test_ffmpeg_listing_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr("super_processor.doctor.which", lambda _: str(fake))

    class Completed:
        stdout = " V..... libx265\n"

    monkeypatch.setattr(
        "super_processor.doctor.subprocess.run",
        lambda *args, **kwargs: Completed(),
    )
    assert "libx265" in doctor._ffmpeg_listing("encoders")


def test_doctor_cli_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("super_processor.doctor.shutil.which", lambda _: None)
    code = cli.main(["--jobs-dir", str(tmp_path / "jobs"), "doctor", "--json"])
    assert code == 1
    payload = capsys.readouterr().out
    assert '"ok": false' in payload
    assert "ffmpeg" in payload
