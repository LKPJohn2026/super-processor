"""Tests for samples.json round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_processor.segments import (
    SampleRow,
    SegmentError,
    load_samples,
    write_samples,
)


def test_samples_round_trip(tmp_path: Path) -> None:
    rows = [
        SampleRow(
            time_s=0.0,
            luma_mean=40.0,
            luma_p05=10.0,
            luma_p95=80.0,
            clip_low=0.01,
            clip_high=0.0,
            rb_cast=0.2,
            variance=90.0,
            motion=1.0,
            subject_x=0.4,
            upper_luma=70.0,
        ),
        SampleRow(
            time_s=1.0,
            luma_mean=140.0,
            luma_p05=80.0,
            luma_p95=200.0,
            clip_low=0.0,
            clip_high=0.02,
            rb_cast=-0.1,
            variance=30.0,
            motion=4.0,
            subject_x=0.6,
            upper_luma=180.0,
        ),
    ]
    write_samples(tmp_path, rows)
    loaded = load_samples(tmp_path)
    assert loaded == rows


def test_missing_samples_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SegmentError, match="not found"):
        load_samples(tmp_path)


def test_sample_list_must_be_objects(tmp_path: Path) -> None:
    path = tmp_path / "samples.json"
    path.write_text('{"samples": [1]}\n', encoding="utf-8")
    with pytest.raises(SegmentError, match="object"):
        load_samples(tmp_path)
