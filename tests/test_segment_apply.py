"""Tests for applying a split note inside the duration bounds."""

from __future__ import annotations

import pytest

from super_processor.segments import (
    SampleRow,
    SegmentError,
    apply_split_note,
    parse_split_note,
    propose_segments,
)


def _row(time_s: int) -> SampleRow:
    return SampleRow(
        time_s=float(time_s),
        luma_mean=30.0,
        luma_p05=15.0,
        luma_p95=42.0,
        clip_low=0.0,
        clip_high=0.0,
        rb_cast=0.0,
        variance=20.0,
        motion=0.0,
        subject_x=0.5,
        upper_luma=40.0,
    )


def _rows(count: int) -> list[SampleRow]:
    return [_row(time_s) for time_s in range(count)]


def test_too_few_adds_a_cut_and_too_many_removes_it() -> None:
    rows = _rows(200)
    proposed = propose_segments(rows)
    assert [(item.start_s, item.end_s) for item in proposed] == [
        (0.0, 100.0),
        (100.0, 200.0),
    ]

    finer = apply_split_note(rows, proposed, parse_split_note("too few cuts"))
    assert [(item.start_s, item.end_s) for item in finer] == [
        (0.0, 50.0),
        (50.0, 100.0),
        (100.0, 200.0),
    ]

    coarser = apply_split_note(rows, finer, parse_split_note("too many cuts"))
    assert [(item.start_s, item.end_s) for item in coarser] == [
        (0.0, 100.0),
        (100.0, 200.0),
    ]
    again = apply_split_note(rows, coarser, parse_split_note("too many cuts"))
    assert [(item.start_s, item.end_s) for item in again] == [
        (0.0, 100.0),
        (100.0, 200.0),
    ]


def test_boundary_move_stays_inside_segment_limits() -> None:
    rows = _rows(200)
    proposed = propose_segments(rows)
    moved = apply_split_note(
        rows,
        proposed,
        parse_split_note("segment 1 starts 10 seconds later"),
    )
    assert [(item.start_s, item.end_s) for item in moved] == [
        (0.0, 110.0),
        (110.0, 200.0),
    ]

    clamped = apply_split_note(
        rows,
        proposed,
        parse_split_note("segment 1 starts 80 seconds later"),
    )
    assert [(item.start_s, item.end_s) for item in clamped] == [
        (0.0, 120.0),
        (120.0, 200.0),
    ]
    assert all(5.0 <= item.end_s - item.start_s <= 120.0 for item in clamped)


def test_relabel_overrides_context_without_moving_bounds() -> None:
    rows = _rows(200)
    proposed = propose_segments(rows)
    labeled = apply_split_note(
        rows, proposed, parse_split_note("segment 1 this is outdoor")
    )
    assert [(item.start_s, item.end_s) for item in labeled] == [
        (0.0, 100.0),
        (100.0, 200.0),
    ]
    assert labeled[0].context == "indoor"
    assert labeled[1].context == "outdoor"
    assert labeled[0].look_group != labeled[1].look_group


def test_move_without_an_interior_boundary_asks_for_a_rephrase() -> None:
    rows = _rows(200)
    proposed = propose_segments(rows)
    with pytest.raises(SegmentError, match="rephrase"):
        apply_split_note(rows, proposed, parse_split_note("segment 0 starts later"))
