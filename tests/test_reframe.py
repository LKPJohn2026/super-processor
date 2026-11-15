"""Tests for vertical reframe path and size-cap assessment."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_processor.estimators import EstimatorError
from super_processor.probe import MediaFacts, StreamFacts
from super_processor.reframe import (
    ReframeError,
    assess_size_cap,
    compute_center_reframe,
    estimate_subject_x_frac,
    load_reframe_plan,
    plan_social_export,
    subject_x_from_gray,
    write_reframe_plan,
)
from super_processor.validator import MIN_BITRATE_KBPS


def test_center_reframe_landscape() -> None:
    path = compute_center_reframe(width=1920, height=1080, max_height=1920)
    assert path.output_width == 1080
    assert path.output_height == 1920
    # A full-height 9:16 window (1080 * 9 / 16 = 607.5, rounded to even),
    # centred: 960 - 304.
    assert (path.crop_x, path.crop_y, path.crop_w, path.crop_h) == (656, 0, 608, 1080)
    assert path.subject_strategy == "center"
    assert path.subject_cx == 0.5


def test_the_crop_never_leaves_the_frame() -> None:
    left = compute_center_reframe(width=1920, height=1080, subject_cx=0.0)
    right = compute_center_reframe(width=1920, height=1080, subject_cx=1.0)
    assert left.crop_x == 0
    assert right.crop_x + right.crop_w == 1920
    # Out-of-range inputs are clamped rather than refused.
    wild = compute_center_reframe(width=1920, height=1080, subject_cx=7.0, padding=3)
    assert wild.crop_x == right.crop_x
    assert wild.subject_cx == 1.0 and wild.padding == 0.25


def test_a_source_taller_than_9_16_is_cropped_top_and_bottom() -> None:
    path = compute_center_reframe(width=1080, height=2400, max_height=1280)
    assert (path.crop_x, path.crop_y, path.crop_w, path.crop_h) == (0, 240, 1080, 1920)
    assert (path.output_width, path.output_height) == (720, 1280)
    with pytest.raises(ReframeError, match="positive"):
        compute_center_reframe(width=0, height=1080)


def test_a_small_offset_still_counts_as_centred() -> None:
    def strategy(subject_cx: float) -> str:
        path = compute_center_reframe(width=1920, height=1080, subject_cx=subject_cx)
        return path.subject_strategy

    assert strategy(0.53) == "center"
    assert strategy(0.45) == "saliency"


def test_saliency_reframe_shifts_crop() -> None:
    left = compute_center_reframe(
        width=1920, height=1080, max_height=1920, subject_cx=0.25
    )
    right = compute_center_reframe(
        width=1920, height=1080, max_height=1920, subject_cx=0.75
    )
    assert left.subject_strategy == "saliency"
    assert right.subject_strategy == "saliency"
    assert left.crop_x < right.crop_x


def test_subject_x_from_gray_finds_bright_edge() -> None:
    width, height = 40, 20
    frame = bytearray(width * height)
    # Strong vertical edge on the right half.
    for y in range(height):
        for x in range(width):
            frame[y * width + x] = 220 if x > 28 else 20
    cx = subject_x_from_gray(bytes(frame), width, height)
    # All the edge energy is in the column pair at x=28/29.
    assert cx == pytest.approx(28 / 39, abs=0.01)
    mirrored = bytes(
        220 if x < width - 29 else 20 for _y in range(height) for x in range(width)
    )
    assert subject_x_from_gray(mirrored, width, height) < 0.45
    assert subject_x_from_gray(b"", 10, 10) == 0.5
    # A flat frame has no edges to follow.
    assert subject_x_from_gray(bytes([90]) * width * height, width, height) == 0.5


def test_size_cap_statuses() -> None:
    ok = assess_size_cap(max_size_mb=200.0, duration_s=60.0)
    assert ok.status == "ok"
    warn = assess_size_cap(max_size_mb=3.0, duration_s=60.0)
    assert warn.status == "warn"
    bad = assess_size_cap(max_size_mb=1.0, duration_s=120.0)
    assert bad.status == "infeasible"
    with pytest.raises(ReframeError):
        assess_size_cap(max_size_mb=10.0, duration_s=0.0)
    with pytest.raises(ReframeError):
        assess_size_cap(max_size_mb=-1.0, duration_s=60.0)


def test_size_cap_boundaries() -> None:
    # 1 MB is 8388.608 kbit. Over 8388.608 / floor seconds it is exactly the
    # floor: that is still deliverable, with a warning.
    at_floor = 8388.608 / MIN_BITRATE_KBPS
    assert assess_size_cap(max_size_mb=1.0, duration_s=at_floor).status == "warn"
    assert (
        assess_size_cap(max_size_mb=1.0, duration_s=at_floor * 1.001).status
        == "infeasible"
    )
    at_comfort = at_floor / 1.5
    assert assess_size_cap(max_size_mb=1.0, duration_s=at_comfort).status == "ok"
    assert (
        assess_size_cap(max_size_mb=1.0, duration_s=at_comfort * 1.001).status == "warn"
    )


def _facts(
    source: Path,
    *,
    duration_s: float | None = 90.0,
    streams: list[StreamFacts] | None = None,
    has_video: bool = True,
) -> MediaFacts:
    return MediaFacts(
        schema_version=1,
        source_path=str(source),
        format_name="mp4",
        format_long_name="MP4",
        duration_s=duration_s,
        size_bytes=1,
        bit_rate=1,
        streams=(
            [
                StreamFacts(
                    index=0,
                    codec_type="video",
                    codec_name="h264",
                    width=1280,
                    height=720,
                )
            ]
            if streams is None
            else streams
        ),
        has_video=has_video,
        has_audio=False,
        is_vfr=False,
    )


def test_plan_social_export_roundtrip(tmp_path: Path) -> None:
    facts = _facts(tmp_path / "clip.mp4")
    plan = plan_social_export(facts, max_height=1280, max_size_mb=40.0, padding=0.05)
    # The caller's limits reach the plan: diagnose and the CLI pass a
    # user-chosen height and size, and the default would be 1920 high.
    assert (plan.reframe.output_width, plan.reframe.output_height) == (720, 1280)
    assert plan.size_cap is not None and plan.size_cap.max_size_mb == 40.0
    assert plan.reframe.padding == 0.05
    # No file on disk: nothing to sample, so the crop is centred.
    assert plan.reframe.subject_strategy == "center"
    assert plan.size_cap is not None and plan.size_cap.status == "ok"
    write_reframe_plan(tmp_path, plan)
    loaded = load_reframe_plan(tmp_path)
    assert loaded.to_dict() == plan.to_dict()
    assert plan_social_export(facts).size_cap is None


def test_plan_social_export_follows_the_subject_in_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(
        "super_processor.reframe.estimate_subject_x_frac", lambda *_args: 0.8
    )
    plan = plan_social_export(_facts(source))
    assert plan.reframe.subject_strategy == "saliency"
    assert plan.reframe.subject_cx == 0.8

    def broken(*_args: object) -> float:
        raise RuntimeError("decoder crashed")

    # A failed saliency pass falls back to the centre instead of failing.
    monkeypatch.setattr("super_processor.reframe.estimate_subject_x_frac", broken)
    assert plan_social_export(_facts(source)).reframe.subject_cx == 0.5


def test_subject_estimate_averages_the_frames_it_could_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    width, height = 160, 90
    right_edge = bytes(
        220 if x > 120 else 20 for _y in range(height) for x in range(width)
    )
    calls = iter(range(10))

    def extract(*_args: object, **_kwargs: object) -> bytes:
        if next(calls) == 0:
            raise EstimatorError("bad frame")
        return right_edge

    monkeypatch.setattr("super_processor.estimators.extract_gray_frame", extract)
    facts = _facts(tmp_path / "clip.mp4")
    expected = subject_x_from_gray(right_edge, width, height)
    assert estimate_subject_x_frac(tmp_path / "clip.mp4", facts) == pytest.approx(
        expected
    )

    def always_fails(*_args: object, **_kwargs: object) -> bytes:
        raise EstimatorError("no frames")

    monkeypatch.setattr("super_processor.estimators.extract_gray_frame", always_fails)
    assert estimate_subject_x_frac(tmp_path / "clip.mp4", facts) == 0.5


def test_social_export_needs_a_sized_video_and_a_duration_for_a_cap(
    tmp_path: Path,
) -> None:
    source = tmp_path / "clip.mp4"
    with pytest.raises(ReframeError, match="requires a video stream"):
        plan_social_export(_facts(source, has_video=False))
    unsized = StreamFacts(index=0, codec_type="video", codec_name="h264")
    with pytest.raises(ReframeError, match="width/height"):
        plan_social_export(_facts(source, streams=[unsized]))
    with pytest.raises(ReframeError, match="requires media duration"):
        plan_social_export(_facts(source, duration_s=None), max_size_mb=10.0)


def test_missing_or_damaged_reframe_plan_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ReframeError, match="not found"):
        load_reframe_plan(tmp_path)
    (tmp_path / "reframe_plan.json").write_text('"plan"', encoding="utf-8")
    with pytest.raises(ReframeError, match="must be an object"):
        load_reframe_plan(tmp_path)
