"""Tests for parsing a short split note into a known intent."""

from __future__ import annotations

import pytest

from super_processor.segments import SegmentError, SplitNote, parse_split_note


def test_too_many_and_too_few() -> None:
    assert parse_split_note("too many cuts") == SplitNote(intent="too_many")
    assert parse_split_note("Not enough cuts") == SplitNote(intent="too_few")


def test_boundary_move_defaults_to_five_seconds() -> None:
    note = parse_split_note("the warm part starts later")
    assert note == SplitNote(
        intent="move_boundary",
        delta_s=5.0,
        problem="too_warm",
    )


def test_boundary_move_reads_a_time_and_a_segment() -> None:
    note = parse_split_note("segment 2 starts 8 seconds earlier")
    assert note == SplitNote(
        intent="move_boundary",
        segment_index=2,
        delta_s=-8.0,
    )


def test_relabel_context_or_problem() -> None:
    assert parse_split_note("this is outdoor") == SplitNote(
        intent="relabel",
        context="outdoor",
    )
    assert parse_split_note("segment 1 this is shaky") == SplitNote(
        intent="relabel",
        segment_index=1,
        problem="shaky",
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "make it cinematic",
        "this is neon",
        "too many and the warm part starts later",
    ],
)
def test_unknown_or_mixed_notes_ask_for_a_rephrase(text: str) -> None:
    with pytest.raises(SegmentError, match="rephrase"):
        parse_split_note(text)
