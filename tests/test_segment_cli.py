"""Tests for `super-processor segment`."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_processor.cli import main
from super_processor.jobs import JobState, JobStore
from super_processor.probe import MediaFacts, write_media_facts
from super_processor.segments import (
    SAMPLE_HEIGHT,
    SAMPLE_WIDTH,
    FrameReader,
    SampleRow,
    write_samples,
)

PIXELS = SAMPLE_WIDTH * SAMPLE_HEIGHT
JOB_ID = "abcd1234abcd1234"


def _row(time_s: int, *, luma: float, upper: float) -> SampleRow:
    return SampleRow(
        time_s=float(time_s),
        luma_mean=luma,
        luma_p05=luma * 0.5,
        luma_p95=min(255.0, luma * 1.4),
        clip_low=0.0,
        clip_high=0.0,
        rb_cast=0.0,
        variance=20.0,
        motion=0.0,
        subject_x=0.5,
        upper_luma=upper,
    )


def _timeline() -> list[SampleRow]:
    dark = [_row(time_s, luma=30.0, upper=40.0) for time_s in range(60)]
    bright = [_row(time_s, luma=160.0, upper=200.0) for time_s in range(60, 130)]
    return dark + bright


def _reader(_source: Path) -> FrameReader:
    def read_frame(_time_s: float) -> tuple[bytes, tuple[float, float, float]]:
        return bytes([32]) * PIXELS, (10.0, 10.0, 10.0)

    return read_frame


def _probed_job(tmp_path: Path) -> JobStore:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake-video")
    store = JobStore(tmp_path / "jobs")
    store.create(source, job_id=JOB_ID)
    store.transition(JOB_ID, JobState.PROBED)
    return store


def test_segment_prints_split_and_marks_proposed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = _probed_job(tmp_path)
    write_samples(store.job_dir(JOB_ID), _timeline())
    monkeypatch.setattr("super_processor.cli.ffmpeg_frame_reader", _reader)

    code = main(["--jobs-dir", str(store.root), "segment", JOB_ID])
    output = capsys.readouterr().out

    assert code == 0
    assert "0  0-60s  indoor  low_light" in output
    assert "1  60-130s  outdoor" in output
    assert "segment_stills/seg_00.ppm" in output
    assert store.load(JOB_ID).state is JobState.SPLIT_PROPOSED

    again = main(["--jobs-dir", str(store.root), "segment", JOB_ID])
    assert again == 0
    assert store.load(JOB_ID).state is JobState.SPLIT_PROPOSED


def test_segment_samples_when_rows_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = _probed_job(tmp_path)
    source = Path(store.load(JOB_ID).source_path)
    write_media_facts(
        store.job_dir(JOB_ID),
        MediaFacts(
            schema_version=1,
            source_path=str(source),
            format_name="mp4",
            format_long_name="MP4",
            duration_s=130.0,
            size_bytes=1,
            bit_rate=1,
            streams=[],
            has_video=True,
            has_audio=False,
            is_vfr=False,
        ),
    )

    def fake_sample(_source: Path, duration_s: float) -> list[SampleRow]:
        assert duration_s == 130.0
        return _timeline()

    monkeypatch.setattr("super_processor.cli.sample_media", fake_sample)
    monkeypatch.setattr("super_processor.cli.ffmpeg_frame_reader", _reader)

    assert main(["--jobs-dir", str(store.root), "segment", JOB_ID]) == 0
    assert "0  0-60s" in capsys.readouterr().out
    assert store.load(JOB_ID).state is JobState.SPLIT_PROPOSED


def test_segment_note_resplits_and_stays_proposed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = _probed_job(tmp_path)
    write_samples(store.job_dir(JOB_ID), _timeline())
    monkeypatch.setattr("super_processor.cli.ffmpeg_frame_reader", _reader)

    assert main(["--jobs-dir", str(store.root), "segment", JOB_ID]) == 0
    capsys.readouterr()
    code = main(
        ["--jobs-dir", str(store.root), "segment", JOB_ID, "--note", "too few cuts"]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "0  0-60s" in output
    assert "1  60-95s" in output
    assert "2  95-130s" in output
    assert store.load(JOB_ID).state is JobState.SPLIT_PROPOSED


def test_segment_refuses_a_job_that_is_not_probed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake-video")
    store = JobStore(tmp_path / "jobs")
    store.create(source, job_id=JOB_ID)

    assert main(["--jobs-dir", str(store.root), "segment", JOB_ID]) == 1
    assert "must be probed" in capsys.readouterr().out
