"""The source audio put onto an upscaled picture: copied when MP4 can carry
it, encoded to AAC otherwise."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from super_processor import upscale
from super_processor.upscale import mux_source_audio, video_stream_facts

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe required",
)


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def _picture(path: Path) -> Path:
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=6",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    )  # fmt: skip
    return path


def _source(path: Path, audio: str) -> Path:
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=80x45:rate=10:duration=6",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", audio, str(path),
    )  # fmt: skip
    return path


def _audio(video: Path) -> tuple[str, str]:
    """The audio codec, and an MD5 of the first 5 s of decoded samples."""
    codec = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:a:0", "-t", "5",
         "-f", "md5", "-"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip
    return codec, pcm


def _audio_args(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the ``-c:a`` of every FFmpeg run."""
    seen: list[str] = []
    real = subprocess.run

    def run(command: list[str], **kwargs: Any) -> Any:
        if "-c:a" in command:
            seen.append(command[command.index("-c:a") + 1])
        return real(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    return seen


@needs_ffmpeg
def test_aac_audio_is_copied_not_encoded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path / "in.mp4", "aac")
    video = _picture(tmp_path / "up.mp4")
    seen = _audio_args(monkeypatch)
    mux_source_audio(video, source)
    assert seen == ["copy"]
    # Encoding again would change the decoded samples; a copy keeps them.
    assert _audio(video) == _audio(source)
    assert video_stream_facts(video)[0] == 60


@needs_ffmpeg
def test_audio_mp4_cannot_carry_as_is_is_encoded_to_aac(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path / "in.mkv", "pcm_s16le")
    video = _picture(tmp_path / "up.mp4")
    seen = _audio_args(monkeypatch)
    mux_source_audio(video, source)
    assert seen == ["aac"]
    assert _audio(video)[0] == "aac"
    assert video_stream_facts(video)[0] == 60


def _spy_mux(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool, bool]]:
    """Record each mux attempt: codec, whether it gave a file, and whether
    a .mux.mp4 was left behind when it gave none."""
    real = upscale._mux_audio
    tried: list[tuple[str, bool, bool]] = []

    def spy(video: Path, source: Path, codec: str, *args: Any, **kwargs: Any) -> Any:
        mixed = real(video, source, codec, *args, **kwargs)
        left = mixed is None and video.with_suffix(".mux.mp4").exists()
        tried.append((codec, mixed is not None, left))
        return mixed

    monkeypatch.setattr(upscale, "_mux_audio", spy)
    return tried


@needs_ffmpeg
def test_a_copy_ffmpeg_refuses_falls_back_to_aac(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # MP4 cannot carry A-law; pretend it is copyable so FFmpeg is asked.
    source = _source(tmp_path / "in.mkv", "pcm_alaw")
    video = _picture(tmp_path / "up.mp4")
    monkeypatch.setattr(upscale, "_COPY_AUDIO", frozenset({"pcm_alaw"}))
    tried = _spy_mux(monkeypatch)
    mux_source_audio(video, source)
    assert tried == [("copy", False, False), ("aac", True, False)]
    assert _audio(video)[0] == "aac"
    assert not video.with_suffix(".mux.mp4").exists()


@needs_ffmpeg
def test_aac_that_changes_format_partway_is_encoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A TV recording: stereo 48 kHz, then 5.1 at 44.1 kHz. A copy muxes
    # without error but keeps only the first config, so the rest is garbled.
    halves = []
    for name, rate, layout, offset in (
        ("a", 48000, "stereo", "0"),
        ("b", 44100, "5.1", "3"),
    ):
        half = tmp_path / f"{name}.ts"
        _ffmpeg(
            "-f", "lavfi", "-i", "testsrc2=size=80x45:rate=10:duration=3",
            "-f", "lavfi", "-i",
            f"sine=frequency=440:duration=3:sample_rate={rate}",
            "-af", f"aformat=channel_layouts={layout}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-output_ts_offset", offset, str(half),
        )  # fmt: skip
        halves.append(half.read_bytes())
    source = tmp_path / "switch.ts"
    source.write_bytes(b"".join(halves))
    video = _picture(tmp_path / "up.mp4")
    tried = _spy_mux(monkeypatch)
    mux_source_audio(video, source)
    assert [codec for codec, _, _ in tried] == ["copy", "aac"]
    assert upscale._audio_decodes(video, ffmpeg_bin="ffmpeg")
    assert not video.with_suffix(".mux.mp4").exists()


def test_odd_bytes_from_ffprobe_keep_the_codec(tmp_path: Path) -> None:
    fake = tmp_path / "ffprobe"
    # stderr holds a path in UTF-8 (Đ) plus a byte no codec decodes.
    fake.write_bytes(
        b"#!/bin/sh\nprintf 'aac\\n'\nprintf '\\xc4\\x90\\xff: bad\\n' >&2\n"
    )
    fake.chmod(0o755)
    if shutil.which(str(fake)) is None:  # pragma: no cover - Windows
        pytest.skip("needs a shell script as ffprobe")
    source = tmp_path / "in.mkv"
    source.touch()
    assert upscale._audio_codec(source, ffprobe_bin=str(fake)) == "aac"


@needs_ffmpeg
def test_a_source_without_audio_leaves_the_picture_silent(tmp_path: Path) -> None:
    source = _picture(tmp_path / "in.mp4")
    video = _picture(tmp_path / "up.mp4")
    assert upscale._audio_codec(source, ffprobe_bin="ffprobe") is None
    mux_source_audio(video, source)
    streams = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout.split()  # fmt: skip
    assert streams == ["video"]


def test_no_ffprobe_gives_no_audio_codec(tmp_path: Path) -> None:
    source = tmp_path / "in.mp4"
    source.touch()
    missing = str(tmp_path / "no-ffprobe")
    assert upscale._audio_codec(source, ffprobe_bin=missing) is None
