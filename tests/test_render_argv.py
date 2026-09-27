"""Tests for segment encode and concat argument lists."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_processor.encoders import EncoderError, encoder_rate_args
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
    assert argv[argv.index("-crf") + 1] == "24"
    assert "aac" in argv
    assert any(item.startswith("hqdn3d=") for item in argv)
    assert argv[-1] == str(dest)


def test_hardware_rate_flags_leave_trim_and_filters_in_place(tmp_path: Path) -> None:
    treatment = treatments_for("noisy")[0]
    argv = build_segment_argv(
        tmp_path / "clip.mp4",
        _segment(),
        treatment,
        tmp_path / "seg.mp4",
        encoder="hevc_nvenc",
    )
    assert argv[argv.index("-ss") + 1] == "10.000"
    assert argv[argv.index("-t") + 1] == "10.000"
    assert any(item.startswith("hqdn3d=") for item in argv)
    assert argv[argv.index("-rc") + 1] == "constqp"
    assert "libx265" not in argv


@pytest.mark.parametrize(
    ("encoder", "flag", "value"),
    [
        ("hevc_qsv", "-global_quality", "24"),
        ("hevc_amf", "-qp_i", "24"),
        ("hevc_videotoolbox", "-q:v", "76"),
    ],
)
def test_each_encoder_uses_its_own_quality_flag(
    encoder: str, flag: str, value: str
) -> None:
    argv = encoder_rate_args(encoder)
    assert argv[argv.index(flag) + 1] == value


def test_unknown_encoder_is_refused() -> None:
    with pytest.raises(EncoderError, match="unsupported"):
        encoder_rate_args("libx264")


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
