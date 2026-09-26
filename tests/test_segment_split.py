"""Tests for splitting ranges that exceed two minutes."""

from __future__ import annotations

import pytest

from super_processor.segments import SegmentError, split_oversized


def test_short_interval_is_unchanged() -> None:
    assert split_oversized([(0.0, 90.0)]) == [(0.0, 90.0)]


def test_flat_range_is_cut_into_equal_pieces() -> None:
    pieces = split_oversized([(0.0, 600.0)], scores=[0.0] * 599)
    assert len(pieces) == 5
    assert pieces[0][0] == pytest.approx(0.0)
    assert pieces[-1][1] == pytest.approx(600.0)
    lengths = [end - start for start, end in pieces]
    assert lengths == pytest.approx([120.0] * 5)


def test_flat_remainder_is_shared_equally() -> None:
    pieces = split_oversized([(0.0, 250.0)])
    assert len(pieces) == 3
    lengths = [end - start for start, end in pieces]
    assert lengths == pytest.approx([250.0 / 3.0] * 3)
    assert all(length <= 120.0 for length in lengths)


def test_strongest_interior_score_wins() -> None:
    scores = [0.0] * 199
    scores[49] = 4.0  # boundary at t=50
    scores[119] = 1.0
    pieces = split_oversized([(0.0, 200.0)], scores=scores)
    assert pieces[0] == pytest.approx((0.0, 50.0))
    assert all(end - start <= 120.0 for start, end in pieces)
    assert pieces[-1][1] == pytest.approx(200.0)


def test_scores_shorter_than_the_interval_fall_back_to_equal_pieces() -> None:
    pieces = split_oversized([(0.0, 240.0)], scores=[0.0, 0.0])
    assert [end - start for start, end in pieces] == pytest.approx([120.0, 120.0])


def test_reversed_interval_is_refused() -> None:
    with pytest.raises(SegmentError, match="before"):
        split_oversized([(10.0, 4.0)])
