"""Tests for encoding one short graded preview."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from super_processor.preview import (
    PreviewError,
    build_preview_argv,
    encode_preview,
    preview_cache_key,
    preview_filters,
    preview_output_path,
)
from super_processor.recipe import OpName
from super_processor.segments import TimelineSegment
from super_processor.treatments import Treatment, TreatmentStep, treatments_for

pytestmark_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def _segment() -> TimelineSegment:
    return TimelineSegment(0, 0.0, 6.0, "indoor", "noisy", 3.0, 0)


def test_preview_argv_grades_only_the_window(tmp_path: Path) -> None:
    treatment = treatments_for("noisy")[0]
    dest = tmp_path / "preview.mp4"
    argv = build_preview_argv(tmp_path / "clip.mp4", _segment(), treatment, dest)
    # Three seconds around the 3.0s key frame; -ss before -i seeks the input.
    assert argv[argv.index("-ss") + 1] == "1.500"
    assert argv.index("-ss") < argv.index("-i")
    assert argv[argv.index("-t") + 1] == "3.000"
    assert argv[argv.index("-vf") + 1] == "hqdn3d=2.1:1.575:1.575:1.18125"
    assert argv[-6:] == ["-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)]


def test_every_step_becomes_its_own_filter_in_order() -> None:
    treatment = Treatment(
        treatment_id="all.steps",
        problem="noisy",
        steps=(
            TreatmentStep(OpName.WHITE_BALANCE, (("temperature", 8000.0),)),
            TreatmentStep(
                OpName.CONTRAST,
                (("contrast", 1.2), ("brightness", 0.1), ("gamma", 1.05)),
            ),
            TreatmentStep(OpName.DENOISE, (("strength", 2.0),)),
            TreatmentStep(OpName.SHARPEN, (("luma_amount", 0.5), ("luma_size", 5.0))),
            TreatmentStep(OpName.STABILIZE, (("shakiness", 5.0),)),
            TreatmentStep(OpName.REFRAME_VERTICAL, (("subject_cx", 1.7),)),
            TreatmentStep(OpName.TRIM, (("start_s", 0.0), ("end_s", 6.0))),
            TreatmentStep(OpName.ENCODE_SIZE_CAP, (("max_size_mb", 10.0),)),
        ),
    )
    assert preview_filters(treatment) == [
        "colortemperature=temperature=8000:mix=1",
        "eq=contrast=1.2:brightness=0.1:gamma=1.05",
        # Denoise strength above 1 is held at the filter's ceiling.
        "hqdn3d=8:6:6:4.5",
        "unsharp=5:5:0.5",
        "deshake",
        # The subject is clamped into the frame.
        "crop=iw*0.9:ih:iw*1*0.1:0",
    ]


def test_a_treatment_without_filters_adds_no_filter_graph(tmp_path: Path) -> None:
    trim_only = Treatment(
        treatment_id="noisy.trim",
        problem="noisy",
        steps=(TreatmentStep(OpName.TRIM, (("start_s", 0.0), ("end_s", 6.0))),),
    )
    argv = build_preview_argv(
        tmp_path / "clip.mp4", _segment(), trim_only, tmp_path / "p.mp4"
    )
    assert "-vf" not in argv
    assert preview_output_path(tmp_path, 3, "a/b") == (
        tmp_path / "previews" / "seg_03_a_b.mp4"
    )


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


class _Completed:
    def __init__(self, returncode: int, stderr: bytes = b"") -> None:
        self.returncode = returncode
        self.stderr = stderr


def test_a_stale_cache_key_encodes_again_and_records_the_new_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    treatment = treatments_for("noisy")[0]
    segment = _segment()
    dest = preview_output_path(tmp_path, segment.index, treatment.treatment_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"old")
    marker = dest.with_suffix(".key")
    marker.write_text("0:noisy.medium", encoding="utf-8")
    runs: list[list[str]] = []

    def run(cmd: list[str], **_kwargs: object) -> _Completed:
        runs.append(cmd)
        Path(cmd[-1]).write_bytes(b"new")
        return _Completed(0)

    monkeypatch.setattr(subprocess, "run", run)
    clip = encode_preview(tmp_path, tmp_path / "clip.mp4", segment, treatment)
    assert clip.cached is False
    assert len(runs) == 1
    assert dest.read_bytes() == b"new"
    assert marker.read_text(encoding="utf-8") == "0:noisy.light"
    assert (clip.in_s, clip.out_s) == (1.5, 4.5)


@pytest.mark.parametrize(
    "outcome",
    [_Completed(1, b"Invalid data found"), OSError("no ffmpeg")],
    ids=["exit-code", "oserror"],
)
def test_a_failed_encode_raises_and_leaves_no_cache_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: _Completed | OSError,
) -> None:
    treatment = treatments_for("noisy")[0]
    segment = _segment()

    def run(*_args: object, **_kwargs: object) -> _Completed:
        if isinstance(outcome, OSError):
            raise outcome
        return outcome

    monkeypatch.setattr(subprocess, "run", run)
    expected = "Invalid data found" if isinstance(outcome, _Completed) else "no ffmpeg"
    with pytest.raises(PreviewError, match=expected):
        encode_preview(tmp_path, tmp_path / "clip.mp4", segment, treatment)
    dest = preview_output_path(tmp_path, segment.index, treatment.treatment_id)
    # Without a key, the next call tries again instead of serving nothing.
    assert not dest.with_suffix(".key").exists()


def test_a_zero_exit_without_an_output_file_is_a_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _Completed(0))
    with pytest.raises(PreviewError, match="ffmpeg preview failed"):
        encode_preview(
            tmp_path, tmp_path / "clip.mp4", _segment(), treatments_for("noisy")[0]
        )


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
    # The clip is the three-second window, not the whole six-second source.
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(clip.path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert float(probe.stdout) == pytest.approx(3.0, abs=0.15)
    # A second call with the same key is served from the cache.
    assert encode_preview(tmp_path, source, _segment(), treatment).cached is True
