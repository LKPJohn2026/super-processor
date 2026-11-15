"""Tests for samples.json round-trip."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    path = write_samples(tmp_path, rows)
    loaded = load_samples(tmp_path)
    assert loaded == rows
    # Written through a temporary file that is renamed into place.
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


def test_missing_samples_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SegmentError, match="not found"):
        load_samples(tmp_path)


def test_sample_list_must_be_objects(tmp_path: Path) -> None:
    path = tmp_path / "samples.json"
    path.write_text('{"samples": [1]}\n', encoding="utf-8")
    with pytest.raises(SegmentError, match="object"):
        load_samples(tmp_path)


def _sample(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "time_s": 0.0,
        "luma_mean": 40.0,
        "luma_p05": 10.0,
        "luma_p95": 80.0,
        "clip_low": 0.0,
        "clip_high": 0.0,
        "rb_cast": 0.0,
        "variance": 90.0,
        "motion": 0.0,
        "subject_x": 0.5,
    }
    base.update(overrides)
    return base


def _write(tmp_path: Path, payload: object) -> None:
    (tmp_path / "samples.json").write_text(json.dumps(payload), encoding="utf-8")


def test_samples_saved_before_upper_luma_existed_still_load(tmp_path: Path) -> None:
    _write(tmp_path, {"samples": [_sample(time_s=3)]})
    (row,) = load_samples(tmp_path)
    assert row.upper_luma == 128.0
    assert row.time_s == 3.0 and isinstance(row.time_s, float)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "root must be an object"),
        ({"schema_version": 1}, "list is missing"),
        (
            {"samples": [{k: v for k, v in _sample().items() if k != "motion"}]},
            "missing motion",
        ),
        ({"samples": [_sample(variance=True)]}, "variance must be numeric"),
        ({"samples": [_sample(luma_mean="40")]}, "luma_mean must be numeric"),
    ],
)
def test_a_damaged_samples_file_is_refused(
    tmp_path: Path, payload: object, message: str
) -> None:
    _write(tmp_path, payload)
    with pytest.raises(SegmentError, match=message):
        load_samples(tmp_path)


def test_a_truncated_samples_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / "samples.json").write_text('{"samples": [', encoding="utf-8")
    with pytest.raises(SegmentError, match="samples file is not valid JSON"):
        load_samples(tmp_path)
