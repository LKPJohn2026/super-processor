"""Tests for segments.json and one still per segment."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_processor.segments import (
    SAMPLE_HEIGHT,
    SAMPLE_WIDTH,
    SampleRow,
    SegmentError,
    load_segments,
    propose_segments,
    write_segment_review,
)

PIXELS = SAMPLE_WIDTH * SAMPLE_HEIGHT


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


def test_propose_cuts_on_the_strongest_change() -> None:
    segments = propose_segments(_timeline())
    assert [(segment.start_s, segment.end_s) for segment in segments] == [
        (0.0, 60.0),
        (60.0, 130.0),
    ]
    assert segments[0].context == "indoor"
    assert segments[0].problem == "low_light"
    assert segments[1].context == "outdoor"
    assert segments[0].look_group != segments[1].look_group


def test_review_writes_segments_and_stills(tmp_path: Path) -> None:
    def read_frame(time_s: float) -> tuple[bytes, tuple[float, float, float]]:
        return bytes([int(time_s) % 256]) * PIXELS, (10.0, 10.0, 10.0)

    segments = write_segment_review(tmp_path, _timeline(), read_frame)
    loaded = load_segments(tmp_path)
    assert loaded == segments
    for segment in loaded:
        still = tmp_path / segment.still_path
        assert still.is_file()
        assert still.read_bytes().startswith(b"P5\n")


def test_missing_segments_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SegmentError, match="not found"):
        load_segments(tmp_path)
