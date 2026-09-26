"""Tests for the 5-second floor and the 15-segment cap."""

from __future__ import annotations

import pytest

from super_processor.segments import SegmentError, merge_short_and_cap


def test_short_tail_is_absorbed() -> None:
    pieces = merge_short_and_cap([(0.0, 40.0), (40.0, 43.0)])
    assert pieces == pytest.approx([(0.0, 43.0)])


def test_short_middle_joins_the_quieter_neighbor() -> None:
    scores = [0.0] * 90
    scores[39] = 8.0  # loud boundary at t=40
    scores[42] = 0.1  # quiet boundary at t=43
    pieces = merge_short_and_cap(
        [(0.0, 40.0), (40.0, 43.0), (43.0, 90.0)],
        scores=scores,
    )
    assert pieces == pytest.approx([(0.0, 40.0), (40.0, 90.0)])


def test_overlong_absorb_is_split_again() -> None:
    pieces = merge_short_and_cap([(0.0, 118.0), (118.0, 122.0)])
    assert pieces[0][0] == pytest.approx(0.0)
    assert pieces[-1][1] == pytest.approx(122.0)
    assert [end - start for start, end in pieces] == pytest.approx([61.0, 61.0])


def test_weakest_boundary_merges_first_down_to_fifteen() -> None:
    pieces_in = [(index * 60.0, (index + 1) * 60.0) for index in range(16)]
    scores = [5.0] * (16 * 60)
    scores[59] = 0.1  # weakest boundary at t=60, between the first two pieces
    pieces = merge_short_and_cap(pieces_in, scores=scores)
    assert len(pieces) == 15
    assert pieces[0] == pytest.approx((0.0, 120.0))
    assert pieces[-1][1] == pytest.approx(16 * 60.0)


def test_cap_refuses_a_merge_that_would_pass_two_minutes() -> None:
    pieces_in = [(index * 100.0, (index + 1) * 100.0) for index in range(16)]
    with pytest.raises(SegmentError, match="cannot merge"):
        merge_short_and_cap(pieces_in, scores=[1.0] * 1600)
