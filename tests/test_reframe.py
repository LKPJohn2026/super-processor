"""Tests for vertical reframe path and size-cap assessment."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_processor.probe import MediaFacts, StreamFacts
from super_processor.reframe import (
    ReframeError,
    assess_size_cap,
    compute_center_reframe,
    load_reframe_plan,
    plan_social_export,
    write_reframe_plan,
)


def test_center_reframe_landscape() -> None:
    path = compute_center_reframe(width=1920, height=1080, max_height=1920)
    assert path.output_width == 1080
    assert path.output_height == 1920
    assert path.crop_w < path.input_width
    assert path.crop_h == 1080
    assert path.subject_strategy == "center"


def test_size_cap_statuses() -> None:
    ok = assess_size_cap(max_size_mb=200.0, duration_s=60.0)
    assert ok.status == "ok"
    warn = assess_size_cap(max_size_mb=3.0, duration_s=60.0)
    assert warn.status == "warn"
    bad = assess_size_cap(max_size_mb=1.0, duration_s=120.0)
    assert bad.status == "infeasible"
    with pytest.raises(ReframeError):
        assess_size_cap(max_size_mb=10.0, duration_s=0.0)


def test_plan_social_export_roundtrip(tmp_path: Path) -> None:
    facts = MediaFacts(
        schema_version=1,
        source_path=str(tmp_path / "clip.mp4"),
        format_name="mp4",
        format_long_name="MP4",
        duration_s=90.0,
        size_bytes=1,
        bit_rate=1,
        streams=[
            StreamFacts(
                index=0,
                codec_type="video",
                codec_name="h264",
                width=1280,
                height=720,
            )
        ],
        has_video=True,
        has_audio=False,
        is_vfr=False,
    )
    plan = plan_social_export(facts, max_height=1280, max_size_mb=40.0, padding=0.05)
    assert plan.reframe.padding == 0.05
    assert plan.size_cap is not None
    write_reframe_plan(tmp_path, plan)
    loaded = load_reframe_plan(tmp_path)
    assert loaded.reframe.output_height == 1280
    assert loaded.size_cap is not None
    assert loaded.size_cap.max_size_mb == 40.0
