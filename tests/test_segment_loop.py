"""Segment loop: dense still sampling and the wizard's segment list."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from super_processor.segment_loop import (
    END_MARGIN_S,
    MAX_STILLS,
    LoopSegment,
    SegmentList,
    SegmentLoopError,
    extract_segment_stills,
    load_segment_list,
    sample_times,
    save_segment_list,
    still_name,
)
from super_processor.segments import (
    SEGMENTS_FILE_NAME,
    STILLS_DIR_NAME,
    SegmentError,
    TimelineSegment,
    load_segments,
    write_segments,
)


def test_sample_times_one_a_second_with_both_ends() -> None:
    times = sample_times(10.0)
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(10.0 - END_MARGIN_S)
    assert len(times) == 11
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert max(gaps) <= 1.0 + 1e-6


def test_sample_times_caps_long_clips_with_even_stride() -> None:
    times = sample_times(30 * 60.0)
    assert len(times) == MAX_STILLS
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(30 * 60.0 - END_MARGIN_S)
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert max(gaps) - min(gaps) < 0.01


def test_sample_times_short_clip_keeps_first_and_last() -> None:
    assert sample_times(0.5) == [0.0, 0.4]
    assert sample_times(0.05) == [0.0]


def test_sample_times_are_sorted_unique_milliseconds() -> None:
    times = sample_times(7.3337, per_s=3.0)
    assert times == sorted(set(times))
    assert all(round(t, 3) == t for t in times)


@pytest.mark.parametrize("duration", [0.0, -1.0, math.nan, math.inf])
def test_sample_times_refuses_no_duration(duration: float) -> None:
    with pytest.raises(ValueError):
        sample_times(duration)


def test_sample_times_refuses_bad_policy() -> None:
    with pytest.raises(ValueError):
        sample_times(10.0, per_s=0.0)
    with pytest.raises(ValueError):
        sample_times(10.0, cap=1)


needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg required"
)


def _clip(path: Path, seconds: int = 3) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=320x180:rate=24:duration={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def test_still_name_carries_milliseconds() -> None:
    assert still_name(0.0) == "t_00000000.jpg"
    assert still_name(12.3456) == "t_00012346.jpg"


@needs_ffmpeg
def test_extract_segment_stills_writes_grid_under_job(tmp_path: Path) -> None:
    source = _clip(tmp_path / "in.mp4")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    times = sample_times(3.0)
    stills = extract_segment_stills(source, job_dir, times)
    assert [still.t_s for still in stills] == times
    for still in stills:
        assert still.path.startswith(f"{STILLS_DIR_NAME}/t_")
        path = job_dir / still.path
        assert path.is_file()
        assert path.resolve().is_relative_to(job_dir.resolve())


@needs_ffmpeg
def test_extract_segment_stills_replaces_old_grid_only(tmp_path: Path) -> None:
    source = _clip(tmp_path / "in.mp4")
    job_dir = tmp_path / "job"
    folder = job_dir / STILLS_DIR_NAME
    folder.mkdir(parents=True)
    (folder / "t_00099999.jpg").write_bytes(b"old")
    (folder / "seg_00.ppm").write_bytes(b"legacy")
    extract_segment_stills(source, job_dir, [0.0, 1.0])
    assert sorted(p.name for p in folder.iterdir()) == [
        "seg_00.ppm",
        "t_00000000.jpg",
        "t_00001000.jpg",
    ]


def test_extract_segment_stills_refuses_symlinked_folder(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / STILLS_DIR_NAME).symlink_to(outside, target_is_directory=True)
    with pytest.raises(SegmentLoopError, match="symbolic link"):
        extract_segment_stills(tmp_path / "in.mp4", job_dir, [0.0])


def test_extract_segment_stills_reports_ffmpeg_failure(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    with pytest.raises(SegmentLoopError, match="could not sample"):
        extract_segment_stills(
            tmp_path / "in.mp4",
            job_dir,
            [0.0],
            ffmpeg_bin=str(tmp_path / "no-ffmpeg"),
        )


def _two_segments() -> SegmentList:
    return SegmentList(
        segments=[
            LoopSegment(0.0, 4.0, label="kitchen, window light"),
            LoopSegment(4.0, 9.5, label="street at night", note="keep grain"),
        ]
    )


def test_segment_list_round_trips(tmp_path: Path) -> None:
    assert load_segment_list(tmp_path) is None
    saved = _two_segments()
    saved.approved = True
    path = save_segment_list(tmp_path, saved)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    assert data["kind"] == "wizard"
    loaded = load_segment_list(tmp_path)
    assert loaded == saved


@pytest.mark.parametrize(
    ("start", "end"), [(-1.0, 2.0), (3.0, 3.0), (4.0, 2.0), (0.0, math.inf)]
)
def test_loop_segment_refuses_bad_times(start: float, end: float) -> None:
    with pytest.raises(SegmentLoopError):
        LoopSegment(start, end)


@pytest.mark.parametrize(
    "raw",
    [
        {"kind": "wizard", "segments": "nope"},
        {"kind": "wizard", "segments": ["nope"]},
        {"kind": "wizard", "segments": [{"start_s": 1.0}]},
    ],
)
def test_segment_list_refuses_malformed_files(tmp_path: Path, raw: object) -> None:
    (tmp_path / SEGMENTS_FILE_NAME).write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SegmentLoopError):
        load_segment_list(tmp_path)


def test_segment_list_refuses_unreadable_json(tmp_path: Path) -> None:
    (tmp_path / SEGMENTS_FILE_NAME).write_text("{", encoding="utf-8")
    with pytest.raises(SegmentLoopError, match="cannot read"):
        load_segment_list(tmp_path)


def test_wizard_refuses_the_older_review_file(tmp_path: Path) -> None:
    write_segments(
        tmp_path,
        [
            TimelineSegment(
                index=0,
                start_s=0.0,
                end_s=6.0,
                context="indoor",
                problem="noisy",
                keyframe_s=3.0,
                look_group=0,
            )
        ],
    )
    with pytest.raises(SegmentLoopError, match="older segment review"):
        load_segment_list(tmp_path)


def test_older_review_refuses_the_wizard_file(tmp_path: Path) -> None:
    save_segment_list(tmp_path, _two_segments())
    with pytest.raises(SegmentError, match="segment loop"):
        load_segments(tmp_path)
    with pytest.raises(SegmentError, match="segment loop"):
        write_segments(tmp_path, [])
    assert load_segment_list(tmp_path) == _two_segments()


def test_older_review_overwrites_unreadable_file(tmp_path: Path) -> None:
    (tmp_path / SEGMENTS_FILE_NAME).write_text("{", encoding="utf-8")
    write_segments(tmp_path, [])
    assert load_segments(tmp_path) == []
