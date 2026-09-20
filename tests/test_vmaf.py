"""Optional VMAF regression helper (skips when libvmaf is unavailable)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def _ffmpeg_has_libvmaf() -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    completed = subprocess.run(
        [ffmpeg, "-hide_banner", "-filters"],
        check=False,
        capture_output=True,
        text=True,
    )
    return "libvmaf" in (completed.stdout or "")


@pytest.mark.skipif(not _ffmpeg_has_libvmaf(), reason="ffmpeg libvmaf not available")
def test_tiny_vmaf_fixture(tmp_path: Path) -> None:
    ref = tmp_path / "ref.mp4"
    dist = tmp_path / "dist.mp4"
    for path, noise in ((ref, None), (dist, "noise=alls=20:allf=t")):
        vf = "scale=160:120"
        if noise:
            vf = f"{vf},{noise}"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=0.5:size=160x120:rate=10",
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
    log = tmp_path / "vmaf.json"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(dist),
            "-i",
            str(ref),
            "-lavfi",
            f"libvmaf=log_path={log.as_posix()}:log_fmt=json",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert log.is_file()
    assert "vmaf" in log.read_text(encoding="utf-8").lower()
