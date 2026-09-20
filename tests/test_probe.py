"""Tests for ffprobe media facts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from super_processor import cli
from super_processor.jobs import JobState, JobStore
from super_processor.probe import (
    ProbeError,
    detect_vfr,
    load_media_facts,
    media_facts_from_ffprobe,
    parse_frame_rate,
    probe_file,
    write_media_facts,
)

SAMPLE_PROBE = {
    "format": {
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "format_long_name": "QuickTime / MOV",
        "duration": "12.5",
        "size": "1048576",
        "bit_rate": "670000",
    },
    "streams": [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "pix_fmt": "yuv420p",
            "avg_frame_rate": "30/1",
            "r_frame_rate": "30/1",
            "nb_frames": "375",
            "color_primaries": "bt709",
            "color_transfer": "bt709",
            "color_space": "bt709",
            "tags": {"rotate": "0"},
        },
        {
            "index": 1,
            "codec_type": "audio",
            "codec_name": "aac",
            "sample_rate": "48000",
            "channels": 2,
            "channel_layout": "stereo",
        },
    ],
}


def test_parse_frame_rate_and_vfr() -> None:
    assert parse_frame_rate("30000/1001") == pytest.approx(29.970, rel=1e-3)
    assert detect_vfr("30/1", "30/1") is False
    assert detect_vfr("24/1", "30/1") is True
    assert detect_vfr("0/0", "30/1") is None


def test_media_facts_from_ffprobe_payload(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake")
    facts = media_facts_from_ffprobe(source, SAMPLE_PROBE)

    assert facts.has_video is True
    assert facts.has_audio is True
    assert facts.duration_s == 12.5
    assert facts.size_bytes == 1_048_576
    assert facts.is_vfr is False
    video = facts.primary_video()
    assert video is not None
    assert video.width == 1920
    assert video.height == 1080
    assert video.codec_name == "h264"


def test_write_and_load_media_facts(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake")
    facts = media_facts_from_ffprobe(source, SAMPLE_PROBE)
    job_dir = tmp_path / "job"
    write_media_facts(job_dir, facts)
    loaded = load_media_facts(job_dir)
    assert loaded.to_dict() == facts.to_dict()


def test_probe_file_uses_ffprobe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake")

    def fake_run(command: list[str], **kwargs: object) -> object:
        class Completed:
            returncode = 0
            stdout = json.dumps(SAMPLE_PROBE)
            stderr = ""

        assert command[0].endswith("ffprobe") or "ffprobe" in command[0]
        assert str(source) in command
        return Completed()

    monkeypatch.setattr(
        "super_processor.probe.shutil.which",
        lambda _: str(tmp_path / "ffprobe"),
    )
    monkeypatch.setattr("super_processor.probe.subprocess.run", fake_run)

    facts = probe_file(source)
    assert facts.format_name is not None
    assert "mp4" in facts.format_name


def test_probe_file_missing_ffprobe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake")
    monkeypatch.setattr("super_processor.probe.shutil.which", lambda _: None)
    with pytest.raises(ProbeError, match="ffprobe not found"):
        probe_file(source)


def test_job_probe_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake")
    jobs_dir = tmp_path / "jobs"
    store = JobStore(jobs_dir)
    manifest = store.create(source, job_id="abcd1234abcd1234")

    monkeypatch.setattr(
        "super_processor.cli.probe_file",
        lambda path: media_facts_from_ffprobe(path, SAMPLE_PROBE),
    )

    assert (
        cli.main(
            [
                "--jobs-dir",
                str(jobs_dir),
                "job",
                "probe",
                manifest.job_id,
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"has_video": true' in output
    assert '"codec_name": "h264"' in output

    updated = store.load(manifest.job_id)
    assert updated.state is JobState.PROBED
    assert updated.notes["has_video"] is True
    assert load_media_facts(store.job_dir(manifest.job_id)).duration_s == 12.5

    assert (
        cli.main(
            [
                "--jobs-dir",
                str(jobs_dir),
                "job",
                "probe",
                manifest.job_id,
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert store.load(manifest.job_id).state is JobState.PROBED
