"""Tests for Cython estimator primitives and look sampling."""

from __future__ import annotations

import shutil
import subprocess
from array import array
from pathlib import Path

import pytest

from super_processor import (
    histogram_u8,
    mean_luma,
    percentile_u8,
    sad_u8,
    variance_u8,
)
from super_processor.estimators import (
    LookEstimates,
    choose_sample_windows,
    estimate_look,
    load_estimates,
    write_estimates,
)
from super_processor.probe import MediaFacts, StreamFacts


def test_histogram_and_percentile() -> None:
    samples = bytes([0, 0, 64, 128, 255, 255])
    hist = array("Q", [0]) * 256
    histogram_u8(samples, memoryview(hist))
    assert hist[0] == 2
    assert hist[255] == 2
    assert percentile_u8(samples, 0.0) == 0.0
    assert percentile_u8(samples, 100.0) == 255.0
    assert percentile_u8(samples, 50.0) in {64.0, 128.0}


def test_variance_and_sad() -> None:
    left = bytes([10, 20, 30, 40])
    right = bytes([10, 25, 25, 50])
    assert variance_u8(left) == pytest.approx(125.0)
    assert sad_u8(left, right) == 20
    with pytest.raises(ValueError):
        sad_u8(left, bytes([1, 2]))


def test_choose_sample_windows() -> None:
    facts = MediaFacts(
        schema_version=1,
        source_path="/tmp/x.mp4",
        format_name="mp4",
        format_long_name="MP4",
        duration_s=30.0,
        size_bytes=1,
        bit_rate=1,
        streams=[],
        has_video=True,
        has_audio=False,
        is_vfr=False,
    )
    windows = choose_sample_windows(facts)
    assert len(windows) == 3
    assert windows[0].label == "head"
    assert windows[1].start_s < windows[2].start_s


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_estimate_look_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1.2:size=320x240:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    facts = MediaFacts(
        schema_version=1,
        source_path=str(source),
        format_name="mp4",
        format_long_name="MP4",
        duration_s=1.2,
        size_bytes=source.stat().st_size,
        bit_rate=1000,
        streams=[
            StreamFacts(
                index=0,
                codec_type="video",
                codec_name="h264",
                width=320,
                height=240,
            )
        ],
        has_video=True,
        has_audio=False,
        is_vfr=False,
    )
    estimates = estimate_look(source, facts)
    assert isinstance(estimates, LookEstimates)
    assert estimates.contrast.params
    path = write_estimates(tmp_path, estimates)
    assert path.is_file()
    loaded = load_estimates(tmp_path)
    assert loaded.source_path == estimates.source_path
    assert loaded.metrics
    # mean_luma still works on extracted-style buffers
    assert mean_luma(bytes([16, 64, 128, 235])) == pytest.approx(110.75)
