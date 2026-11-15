"""Tests for segment encode and concat argument lists."""

from __future__ import annotations

import os
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
    assert "libx264" in argv
    assert argv[argv.index("-crf") + 1] == "23"
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
        encoder="h264_nvenc",
    )
    assert argv[argv.index("-ss") + 1] == "10.000"
    assert argv[argv.index("-t") + 1] == "10.000"
    assert any(item.startswith("hqdn3d=") for item in argv)
    assert argv[argv.index("-rc") + 1] == "constqp"
    assert "libx264" not in argv


def test_a_segment_without_audio_drops_the_audio_stream(tmp_path: Path) -> None:
    argv = build_segment_argv(
        tmp_path / "clip.mp4",
        _segment(),
        treatments_for("noisy")[0],
        tmp_path / "seg.mp4",
        include_audio=False,
    )
    # A silent source has no stream for ``-c:a aac`` to encode.
    assert "-an" in argv
    assert "-c:a" not in argv


def test_software_and_nvenc_rate_flags_are_exact() -> None:
    # These argv lists are the encoder contract: quality, pixel format, codec.
    assert encoder_rate_args("libx264") == [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
    ]
    assert encoder_rate_args("h264_nvenc", crf=18) == [
        "-c:v",
        "h264_nvenc",
        "-rc",
        "constqp",
        "-qp",
        "18",
        "-pix_fmt",
        "yuv420p",
    ]
    amf = encoder_rate_args("h264_amf", crf=20)
    assert amf[amf.index("-qp_p") + 1] == "20"


@pytest.mark.parametrize(("crf", "quality"), [(0, "100"), (99, "1"), (150, "1")])
def test_videotoolbox_quality_stays_within_one_to_one_hundred(
    crf: int, quality: str
) -> None:
    argv = encoder_rate_args("h264_videotoolbox", crf=crf)
    assert argv[argv.index("-q:v") + 1] == quality


@pytest.mark.parametrize(
    ("encoder", "flag", "value"),
    [
        ("h264_qsv", "-global_quality", "23"),
        ("h264_amf", "-qp_i", "23"),
        ("h264_videotoolbox", "-q:v", "77"),
    ],
)
def test_each_encoder_uses_its_own_quality_flag(
    encoder: str, flag: str, value: str
) -> None:
    argv = encoder_rate_args(encoder)
    assert argv[argv.index(flag) + 1] == value


@pytest.mark.parametrize(
    "encoder", ["libx265", "hevc_nvenc", "libsvtav1", "av1_nvenc", "mpeg4"]
)
def test_only_h264_encoders_are_allowed(encoder: str) -> None:
    with pytest.raises(EncoderError, match="unsupported"):
        encoder_rate_args(encoder)


def test_concat_list_quotes_paths_and_stays_in_order(tmp_path: Path) -> None:
    first = tmp_path / "seg 0.mp4"
    second = tmp_path / "seg 1.mp4"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    listing = write_concat_list(tmp_path / "list.txt", [first, second])
    lines = listing.read_text(encoding="utf-8").splitlines()
    assert lines == [
        f"file '{first.resolve()}'",
        f"file '{second.resolve()}'",
    ]
    out = tmp_path / "out.mp4"
    argv = build_concat_argv(listing, out)
    assert argv[argv.index("-f") + 1] == "concat"
    # Absolute paths in the list need -safe 0; -c copy joins without re-encoding.
    assert argv[argv.index("-safe") + 1] == "0"
    assert argv[argv.index("-i") + 1] == str(listing)
    assert argv[argv.index("-c") + 1] == "copy"
    assert argv[-1] == str(out)


def test_concat_list_escapes_a_quote_in_a_path(tmp_path: Path) -> None:
    part = tmp_path / "it's here.mp4"
    part.write_bytes(b"a")
    listing = write_concat_list(tmp_path / "list.txt", [part])
    # The concat demuxer ends a quoted path at the first bare quote, so a
    # quote must close, escape and reopen or the join reads the wrong file.
    folder = part.resolve().parent
    expected = f"file '{folder}{os.sep}it'\\''s here.mp4'\n"
    assert listing.read_text(encoding="utf-8") == expected
