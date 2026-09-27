"""Tests for encoding one short graded preview."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from super_processor.preview import (
    build_preview_argv,
    encode_preview,
    preview_cache_key,
    preview_filters,
    preview_output_path,
)
from super_processor.segments import TimelineSegment
from super_processor.treatments import treatments_for

pytestmark_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def _segment() -> TimelineSegment:
    return TimelineSegment(0, 0.0, 6.0, "indoor", "noisy", 3.0, 0)


def test_preview_argv_grades_only_the_window(tmp_path: Path) -> None:
    treatment = treatments_for("noisy")[0]
    dest = tmp_path / "preview.mp4"
    argv = build_preview_argv(tmp_path / "clip.mp4", _segment(), treatment, dest)
    assert "-ss" in argv
    assert "1.500" in argv
    assert "-t" in argv
    assert "3.000" in argv
    assert any(item.startswith("hqdn3d=") for item in argv)
    assert argv[-1] == str(dest)


def test_matching_cache_key_skips_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    treatment = treatments_for("noisy")[0]
    segment = _segment()
    dest = preview_output_path(tmp_path, segment.index, treatment.treatment_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"cached")
    dest.with_suffix(".key").write_text(
        preview_cache_key(segment.index, treatment.treatment_id),
        encoding="utf-8",
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("ffmpeg should not run")

    monkeypatch.setattr(subprocess, "run", fail)
    clip = encode_preview(tmp_path, tmp_path / "clip.mp4", segment, treatment)
    assert clip.cached is True
    assert clip.format_line().startswith("in 1.5 out 4.5  ")


@pytestmark_ffmpeg
def test_encode_preview_writes_a_short_clip(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:duration=6:size=160x90:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    treatment = treatments_for("low_light")[0]
    clip = encode_preview(tmp_path, source, _segment(), treatment)
    assert clip.cached is False
    assert clip.path.is_file()
    assert clip.path.stat().st_size > 0
    assert any(item.startswith("eq=") for item in preview_filters(treatment))
