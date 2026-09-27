"""Tests for selecting a hardware decoder while sampling."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_processor.cli import main
from super_processor.doctor import CheckResult
from super_processor.estimators import extract_gray_frame
from super_processor.jobs import JobState, JobStore

PIXELS = 160 * 90
JOB_ID = "abcd1234abcd1234"


def test_hardware_decoder_flags_precede_the_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, list[str]] = {}

    class Completed:
        returncode = 0
        stdout = bytes([8]) * PIXELS
        stderr = b""

    def run(cmd: list[str], **_kwargs: object) -> Completed:
        seen["cmd"] = cmd
        return Completed()

    monkeypatch.setattr("super_processor.estimators.subprocess.run", run)
    frame = extract_gray_frame(Path("clip.mp4"), at_s=1.0, decoder="hevc_cuvid")
    command = seen["cmd"]
    assert frame == bytes([8]) * PIXELS
    assert command.index("-hwaccel") < command.index("-i")
    assert "hevc_cuvid" in command


def test_missing_decoder_fails_before_sampling(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake-video")
    store = JobStore(tmp_path / "jobs")
    store.create(source, job_id=JOB_ID)
    store.transition(JOB_ID, JobState.PROBED)

    def missing(*_args: object, **_kwargs: object) -> CheckResult:
        return CheckResult(
            name="decoder:hevc_cuvid",
            ok=False,
            detail=(
                "hevc_cuvid missing "
                "(install ffmpeg with this decoder or sample in software)"
            ),
        )

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("sampler ran")

    monkeypatch.setattr("super_processor.estimators.check_ffmpeg_capability", missing)
    monkeypatch.setattr("super_processor.segments.sample_media", boom)
    code = main(
        ["--jobs-dir", str(store.root), "segment", JOB_ID, "--decode", "hevc_cuvid"]
    )
    assert code == 1
    assert "hevc_cuvid missing" in capsys.readouterr().out
