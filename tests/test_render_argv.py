"""Tests for segment encode and concat argument lists."""

from __future__ import annotations

from pathlib import Path

from super_processor.render import (
    build_concat_argv,
    build_segment_argv,
    write_concat_list,
)
from super_processor.segments import TimelineSegment
from super_processor.treatments import treatments_for


def _segment() -> TimelineSegment:
    return TimelineSegment(0, 10.0, 20.0, "indoor", "noisy", 12.0, 0)


def test_segment_argv_trims_every_stream_the_same_way(tmp_path: Path) -> None:
    treatment = treatments_for("noisy")[0]
    dest = tmp_path / "seg.mp4"
    argv = build_segment_argv(tmp_path / "clip.mp4", _segment(), treatment, dest)
    assert isinstance(argv, list)
    assert argv[argv.index("-ss") + 1] == "10.000"
    assert argv[argv.index("-t") + 1] == "10.000"
    assert "libx265" in argv
    assert "aac" in argv
    assert any(item.startswith("hqdn3d=") for item in argv)
    assert argv[-1] == str(dest)


def test_concat_list_quotes_paths_and_stays_in_order(tmp_path: Path) -> None:
    first = tmp_path / "seg 0.mp4"
    second = tmp_path / "seg 1.mp4"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    listing = write_concat_list(tmp_path / "list.txt", [first, second])
    text = listing.read_text(encoding="utf-8")
    assert text.index("seg 0.mp4") < text.index("seg 1.mp4")
    assert text.startswith("file '")
    argv = build_concat_argv(listing, tmp_path / "out.mp4")
    assert argv[argv.index("-f") + 1] == "concat"
    assert "-c" in argv
    assert "copy" in argv
