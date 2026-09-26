"""Tests for legal segment counts."""

from __future__ import annotations

import pytest

from super_processor.segments import SegmentError, legal_segment_counts


def test_ten_minutes_allows_five_to_fifteen() -> None:
    assert legal_segment_counts(10 * 60) == (5, 15)


def test_twenty_minutes_allows_ten_to_fifteen() -> None:
    assert legal_segment_counts(20 * 60) == (10, 15)


def test_twenty_five_minutes_allows_thirteen_to_fifteen() -> None:
    assert legal_segment_counts(25 * 60) == (13, 15)


def test_thirty_minutes_is_fifteen_exact_slots() -> None:
    assert legal_segment_counts(30 * 60) == (15, 15)


def test_over_thirty_minutes_is_refused() -> None:
    with pytest.raises(SegmentError, match="exceeds"):
        legal_segment_counts(30 * 60 + 1)


def test_shorter_than_one_segment_is_refused() -> None:
    with pytest.raises(SegmentError, match="cannot be split"):
        legal_segment_counts(4.0)


def test_non_positive_duration_is_refused() -> None:
    with pytest.raises(SegmentError, match="positive"):
        legal_segment_counts(0.0)
