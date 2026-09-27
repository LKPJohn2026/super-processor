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
