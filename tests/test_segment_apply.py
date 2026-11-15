"""Tests for applying a split note inside the duration bounds."""

from __future__ import annotations

from dataclasses import replace

import pytest

from super_processor.segments import (
    SampleRow,
    SegmentError,
    SplitNote,
    TimelineSegment,
    apply_split_note,
    parse_split_note,
    propose_segments,
)


def _row(time_s: int, **overrides: float) -> SampleRow:
    values: dict[str, float] = {
        "luma_mean": 30.0,
        "luma_p05": 15.0,
        "luma_p95": 42.0,
        "clip_low": 0.0,
        "clip_high": 0.0,
        "rb_cast": 0.0,
        "variance": 20.0,
        "motion": 0.0,
        "subject_x": 0.5,
        "upper_luma": 40.0,
    }
    values.update(overrides)
    return SampleRow(time_s=float(time_s), **values)


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


def _spans(segments: list[TimelineSegment]) -> list[tuple[float, float]]:
    return [(item.start_s, item.end_s) for item in segments]


def _noisy_dark_warm() -> list[SampleRow]:
    """Three 100 s looks, each with its own problem: noisy, dark, then warm."""
    bright = {"luma_mean": 140.0, "luma_p05": 40.0, "luma_p95": 200.0}
    return [
        *(_row(t, variance=300.0, **bright) for t in range(100)),
        *(_row(t) for t in range(100, 200)),
        *(_row(t, rb_cast=0.9, **bright) for t in range(200, 300)),
    ]


def test_a_cue_moves_the_start_of_the_one_segment_with_that_problem() -> None:
    rows = _noisy_dark_warm()
    proposed = propose_segments(rows)
    assert [item.problem for item in proposed] == ["noisy", "low_light", "too_warm"]
    assert _spans(proposed) == [(0.0, 100.0), (100.0, 200.0), (200.0, 300.0)]

    # "the warm part" is segment 2, not the default first boundary.
    warm = apply_split_note(
        rows, proposed, parse_split_note("the warm part starts later")
    )
    assert _spans(warm) == [(0.0, 100.0), (100.0, 205.0), (205.0, 300.0)]

    # With no segment and no cue the note means the first interior boundary.
    plain = apply_split_note(rows, proposed, parse_split_note("starts 3 s earlier"))
    assert _spans(plain) == [(0.0, 97.0), (97.0, 200.0), (200.0, 300.0)]


def test_a_cue_that_matches_no_segment_or_several_asks_for_a_rephrase() -> None:
    rows = _noisy_dark_warm()
    proposed = propose_segments(rows)
    # No segment's problem is low_contrast ("bright").
    with pytest.raises(SegmentError, match="rephrase"):
        apply_split_note(
            rows, proposed, parse_split_note("the bright part starts later")
        )
    # Two warm segments, both with a start that could move: the cue cannot
    # say which boundary is meant.
    two_warm = [
        proposed[0],
        *(replace(item, problem="too_warm") for item in proposed[1:]),
    ]
    with pytest.raises(SegmentError, match="rephrase"):
        apply_split_note(rows, two_warm, parse_split_note("the warm part starts later"))


def test_relabelling_the_problem_moves_the_key_frame_to_that_problem() -> None:
    proposed = propose_segments(_rows(200))
    # One shaky second inside the first segment; the bounds stay as proposed.
    rows = [_row(t, motion=40.0 if t == 30 else 0.0) for t in range(200)]
    labeled = apply_split_note(
        rows, proposed, parse_split_note("segment 0 this is shaky")
    )
    assert _spans(labeled) == _spans(proposed)
    assert labeled[0].problem == "shaky"
    # The still must show the shake, not the darkest second of the segment.
    assert labeled[0].keyframe_s == 30.0
    assert labeled[1].problem == "low_light"


def test_a_relabel_past_the_last_segment_asks_for_a_rephrase() -> None:
    rows = _rows(200)
    proposed = propose_segments(rows)
    with pytest.raises(SegmentError, match="rephrase"):
        apply_split_note(rows, proposed, parse_split_note("segment 2 this is outdoor"))


def test_too_few_stops_at_the_most_segments_the_clip_allows() -> None:
    # Ten seconds hold at most two 5 s segments.
    rows = _rows(10)
    once = apply_split_note(rows, propose_segments(rows), SplitNote(intent="too_few"))
    assert _spans(once) == [(0.0, 5.0), (5.0, 10.0)]
    twice = apply_split_note(rows, once, SplitNote(intent="too_few"))
    assert _spans(twice) == [(0.0, 5.0), (5.0, 10.0)]


def test_a_note_without_samples_or_segments_or_a_known_intent_is_refused() -> None:
    rows = _rows(200)
    proposed = propose_segments(rows)
    with pytest.raises(SegmentError, match="rephrase"):
        apply_split_note([], proposed, SplitNote(intent="too_many"))
    with pytest.raises(SegmentError, match="rephrase"):
        apply_split_note(rows, [], SplitNote(intent="too_many"))
    with pytest.raises(SegmentError, match="rephrase"):
        apply_split_note(rows, proposed, SplitNote(intent="make_it_pop"))
