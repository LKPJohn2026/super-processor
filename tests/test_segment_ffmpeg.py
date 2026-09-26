"""Tests for one-hertz sampling through FFmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from super_processor.segments import sample_media

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def test_sample_media_reads_one_row_per_second(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:duration=6:size=160x90:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    rows = sample_media(source, 6.0)
    assert [row.time_s for row in rows] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert all(row.luma_mean > 0 for row in rows)
    assert rows[0].rb_cast > 0
