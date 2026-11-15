"""Tests for mapping a sentence onto the fault list."""

from __future__ import annotations

from super_processor.treatments import faults_from_sentence


def test_known_phrases_become_faults() -> None:
    assert faults_from_sentence("Still too dark") == ["too-dark"]
    assert faults_from_sentence("not sharp enough") == ["not-sharp"]
    assert faults_from_sentence("too much denoise at the end") == ["too-much-denoise"]


def test_an_unmatched_sentence_is_unused() -> None:
    assert faults_from_sentence("make it cinematic") == []
    assert faults_from_sentence("   ") == []


def test_a_sentence_can_name_several_faults_in_the_fixed_order() -> None:
    # Case and runs of whitespace do not hide a phrase.
    assert faults_from_sentence("TOO   warm and\nnot sharp") == [
        "not-sharp",
        "too-warm",
    ]
