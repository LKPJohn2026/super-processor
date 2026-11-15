"""Tests for segments.json and one still per segment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from super_processor.segments import (
    SAMPLE_HEIGHT,
    SAMPLE_WIDTH,
    STILLS_DIR_NAME,
    SampleRow,
    SegmentError,
    load_segments,
    propose_segments,
    write_gray_still,
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


def test_review_writes_segments_and_the_key_frame_still_of_each(
    tmp_path: Path,
) -> None:
    # Each frame's gray level is its second, so a still shows which frame it
    # is; the trailing bytes past the frame must not reach the still.
    def read_frame(time_s: float) -> tuple[bytes, tuple[float, float, float]]:
        return bytes([int(time_s) % 256]) * PIXELS + b"tail", (10.0, 10.0, 10.0)

    # Second 17 is the darkest of the dark segment, so its still is not
    # simply the segment's first frame.
    rows = _timeline()
    rows[17] = _row(17, luma=10.0, upper=40.0)
    segments = write_segment_review(tmp_path, rows, read_frame)
    assert segments[0].keyframe_s == 17.0
    loaded = load_segments(tmp_path)
    assert loaded == segments
    header = f"P5\n{SAMPLE_WIDTH} {SAMPLE_HEIGHT}\n255\n".encode()
    for segment in loaded:
        assert segment.still_path == f"{STILLS_DIR_NAME}/seg_{segment.index:02d}.ppm"
        assert segment.start_s <= segment.keyframe_s < segment.end_s
        still = (tmp_path / segment.still_path).read_bytes()
        assert still == header + bytes([int(segment.keyframe_s)]) * PIXELS


def test_review_without_a_reader_writes_no_stills(tmp_path: Path) -> None:
    segments = write_segment_review(tmp_path, _timeline())
    assert [segment.still_path for segment in load_segments(tmp_path)] == ["", ""]
    assert len(segments) == 2
    assert not (tmp_path / STILLS_DIR_NAME).exists()


def test_a_short_still_frame_is_refused_before_anything_is_written(
    tmp_path: Path,
) -> None:
    path = tmp_path / STILLS_DIR_NAME / "seg_00.ppm"
    with pytest.raises(SegmentError, match="short still"):
        write_gray_still(path, b"\x00" * (PIXELS - 1))
    assert not path.exists()


def test_an_empty_sample_list_cannot_be_segmented() -> None:
    with pytest.raises(SegmentError, match="empty"):
        propose_segments([])


def test_missing_segments_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SegmentError, match="not found"):
        load_segments(tmp_path)


def _segment(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "index": 0,
        "start_s": 0.0,
        "end_s": 6.0,
        "context": "indoor",
        "problem": "noisy",
        "keyframe_s": 3.0,
        "look_group": 0,
        "still_path": "",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "root must be an object"),
        ({"schema_version": 1}, "list is missing"),
        ({"segments": ["seg"]}, "must be an object"),
        ({"segments": [_segment(start_s=True)]}, "start_s must be numeric"),
        ({"segments": [_segment(end_s="6")]}, "end_s must be numeric"),
        ({"segments": [_segment(context=3)]}, "context must be text"),
    ],
)
def test_a_damaged_segments_file_is_refused(
    tmp_path: Path, payload: object, message: str
) -> None:
    (tmp_path / "segments.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SegmentError, match=message):
        load_segments(tmp_path)


@pytest.mark.parametrize(
    "key", ["index", "start_s", "end_s", "keyframe_s", "look_group"]
)
def test_a_segment_missing_a_number_is_refused(tmp_path: Path, key: str) -> None:
    item = _segment()
    del item[key]
    (tmp_path / "segments.json").write_text(
        json.dumps({"segments": [item]}), encoding="utf-8"
    )
    with pytest.raises(SegmentError, match=f"segment missing {key}"):
        load_segments(tmp_path)


@pytest.mark.parametrize(
    ("key", "literal"),
    [
        ("index", "NaN"),
        ("look_group", "Infinity"),
        ("index", "1e400"),
        ("start_s", "NaN"),
    ],
)
def test_a_segment_number_that_is_not_finite_is_refused(
    tmp_path: Path, key: str, literal: str
) -> None:
    # json.loads takes NaN, Infinity and 1e400 (inf); int() of them raises.
    item = _segment()
    item[key] = 0
    text = json.dumps({"segments": [item]}).replace(
        f'"{key}": 0', f'"{key}": {literal}'
    )
    (tmp_path / "segments.json").write_text(text, encoding="utf-8")
    with pytest.raises(SegmentError, match=f"segment {key} must be numeric"):
        load_segments(tmp_path)


def test_a_truncated_segments_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / "segments.json").write_text('{"segments": [', encoding="utf-8")
    with pytest.raises(SegmentError, match="segments file is not valid JSON"):
        load_segments(tmp_path)
