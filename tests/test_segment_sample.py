"""Tests for sampling feature rows from an injected frame reader."""

from __future__ import annotations

import pytest

from super_processor.segments import (
    SAMPLE_HEIGHT,
    SAMPLE_WIDTH,
    SegmentError,
    sample_from_reader,
)

PIXELS = SAMPLE_WIDTH * SAMPLE_HEIGHT


def test_reader_fills_one_row_per_second() -> None:
    def read_frame(time_s: float) -> tuple[bytes, tuple[float, float, float]]:
        value = 20 if time_s < 2 else 200
        return bytes([value]) * PIXELS, (40.0, 40.0, 180.0)

    rows = sample_from_reader(6.2, read_frame)
    assert [row.time_s for row in rows] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert rows[0].luma_mean == pytest.approx(20.0)
    assert rows[0].motion == 0.0
    assert rows[1].motion == 0.0
    assert rows[2].motion == pytest.approx(180.0)
    assert rows[0].rb_cast > 0.5


def test_overlong_video_is_refused_before_reading() -> None:
    def read_frame(_time_s: float) -> tuple[bytes, tuple[float, float, float]]:
        raise AssertionError("reader should not run")

    with pytest.raises(SegmentError, match="1800s"):
        sample_from_reader(1801.0, read_frame)


def test_short_frame_is_refused() -> None:
    def read_frame(_time_s: float) -> tuple[bytes, tuple[float, float, float]]:
        return b"\x00" * 10, (0.0, 0.0, 0.0)

    with pytest.raises(SegmentError, match="short frame"):
        sample_from_reader(5.0, read_frame)
