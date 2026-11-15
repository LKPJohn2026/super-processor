"""Tests for look groups across neighboring segments."""

from __future__ import annotations

from super_processor.segments import look_group_ids


def test_same_look_shares_a_group() -> None:
    labels = [
        ("outdoor", "too_warm"),
        ("outdoor", "too_warm"),
        ("indoor", "low_light"),
    ]
    assert look_group_ids(labels) == [0, 0, 1]


def test_a_different_problem_or_context_starts_a_new_group() -> None:
    labels = [
        ("outdoor", "too_warm"),
        ("outdoor", "low_light"),
        ("indoor", "low_light"),
        ("indoor", "low_light"),
    ]
    assert look_group_ids(labels) == [0, 1, 2, 2]


def test_empty_label_list() -> None:
    assert look_group_ids([]) == []


def test_a_look_that_comes_back_later_gets_a_new_group() -> None:
    # Groups are runs of neighbours: a returning look is graded on its own.
    labels = [("indoor", "noisy"), ("outdoor", "too_warm"), ("indoor", "noisy")]
    assert look_group_ids(labels) == [0, 1, 2]
