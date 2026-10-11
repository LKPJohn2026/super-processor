"""Per-shot looks: bounds, filters, plan storage, and multi-shot rendering."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from super_processor.look import LookError, ShotLook
from super_processor.upscale import (
    FakeUpscaleEngine,
    UpscaleError,
    UpscalePlan,
    UpscaleSpan,
    blend_filter,
    default_span,
    is_delivery,
    load_upscale_plan,
    look_at,
    render_plan,
    render_shots,
    replace_overlapping,
    save_upscale_plan,
    shot_timeline,
)

BRIGHT = ShotLook(brightness=0.1)
CLEANED = ShotLook(deblock=1.0, denoise=1.0, grain=1.0)


def test_look_bounds_and_neutral_filters() -> None:
    neutral = ShotLook()
    assert neutral.is_neutral()
    assert neutral.clean_filter() == ""
    assert neutral.finish_filter() == ""
    for bad in (
        {"deblock": 1.5},
        {"brightness": -0.2},
        {"gamma": 2},
        {"grain": True},
        {"contrast": "1.1"},
    ):
        with pytest.raises(LookError):
            ShotLook(**bad)
    with pytest.raises(LookError, match="unknown"):
        ShotLook.from_dict({"sharpen": 1.0})
    with pytest.raises(LookError):
        ShotLook.from_dict(["denoise"])
    assert ShotLook.from_dict(None) == neutral


def test_look_filters_follow_the_fixed_order() -> None:
    look = ShotLook(deblock=0.5, denoise=0.5, contrast=1.1, grain=0.5)
    clean = look.clean_filter()
    assert clean.index("deblock=") < clean.index("hqdn3d=")
    finish = look.finish_filter()
    assert finish.index("eq=") < finish.index("noise=")
    assert "all_seed=1" in finish
    graph = blend_filter(320, 180, 0.5, look)
    # Clean on the plain-upscale base before scaling, finish after the mix.
    base = graph.split("[1:v]", 1)[1]
    assert base.index("deblock=") < base.index("scale=320:180")
    assert graph.index("blend=") < graph.index("eq=")
    assert graph.endswith("[v]")
    assert "[mix]" not in blend_filter(320, 180, 0.5)


def test_span_stores_its_look_and_old_plans_still_load(tmp_path: Path) -> None:
    span = UpscaleSpan(0, 4, look=BRIGHT)
    assert UpscaleSpan.from_dict(span.to_dict()) == span
    save_upscale_plan(tmp_path, UpscalePlan(spans=(span,)))
    loaded = load_upscale_plan(tmp_path)
    assert loaded is not None and loaded.spans[0].look == BRIGHT
    old = {"spans": [{"start_s": 0, "end_s": 4, "scale": 2, "strength": 0.5}]}
    (tmp_path / "upscale_plan.json").write_text(json.dumps(old), encoding="utf-8")
    loaded = load_upscale_plan(tmp_path)
    assert loaded is not None and loaded.spans[0].look.is_neutral()
    with pytest.raises(UpscaleError, match="invalid shot look"):
        UpscaleSpan.from_dict({"start_s": 0, "end_s": 4, "look": {"denoise": 9}})


def test_revise_keeps_each_shots_look() -> None:
    plan = UpscalePlan(
        spans=(
            UpscaleSpan(0, 10, look=BRIGHT),
            UpscaleSpan(10, 20, look=CLEANED),
        )
    )
    assert look_at(plan, 3) == BRIGHT
    assert look_at(plan, 10) == CLEANED
    assert look_at(plan, 25).is_neutral()
    updated = replace_overlapping(
        plan, UpscaleSpan(4, 6, strength=0.2, look=BRIGHT), 20
    )
    assert [(s.start_s, s.end_s, s.look) for s in updated.spans] == [
        (0.0, 4.0, BRIGHT),
        (6.0, 10.0, BRIGHT),
        (10.0, 20.0, CLEANED),
        (4.0, 6.0, BRIGHT),
    ]
    rescaled = replace_overlapping(plan, UpscaleSpan(4, 6, scale=4), 20)
    assert {s.scale for s in rescaled.spans} == {4}
    assert look_at(rescaled, 15) == CLEANED


def test_shot_timeline_checks_coverage() -> None:
    shots = (UpscaleSpan(5, 10), UpscaleSpan(0, 5))
    assert [s.start_s for s in shot_timeline(shots, 10)] == [0.0, 5.0]
    with pytest.raises(UpscaleError, match="gaps"):
        shot_timeline((UpscaleSpan(0, 4), UpscaleSpan(5, 10)), 10)
    with pytest.raises(UpscaleError, match="before the end"):
        shot_timeline((UpscaleSpan(0, 4),), 10)
    with pytest.raises(UpscaleError, match="same scale"):
        shot_timeline((UpscaleSpan(0, 5), UpscaleSpan(5, 10, scale=4)), 10)
    with pytest.raises(UpscaleError, match="no shots"):
        shot_timeline((), 10)


def _gray(path: Path, seconds: int, rate: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=gray:duration={seconds}:size=160x90:rate={rate}",
            "-f",
            "lavfi",
            "-i",
            f"sine=duration={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )


def _frame_lumas(path: Path) -> list[int]:
    """One average luma value per frame."""
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v",
            "-vf",
            "scale=1:1:flags=area,format=gray",
            "-f",
            "rawvideo",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    return list(raw)


def _has_audio(path: Path) -> bool:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return bool(out.strip())


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_each_shot_renders_with_its_own_look(tmp_path: Path) -> None:
    source = tmp_path / "in.mp4"
    _gray(source, 6, "30000/1001")  # 29.97 fps, 179 frames
    base = _frame_lumas(source)
    output = tmp_path / "output.mp4"
    # Edges that are not on frame times: 2.01s and 4.0s are frames 60 and 120.
    shots = (
        UpscaleSpan(0, 2.01),
        UpscaleSpan(2.01, 4.0, look=BRIGHT),
        UpscaleSpan(4.0, 6.0, look=CLEANED),
    )
    render_shots(source, output, shots, engine=FakeUpscaleEngine(), duration_s=6)
    lumas = _frame_lumas(output)
    assert len(lumas) == len(base)
    assert is_delivery(output)
    assert _has_audio(output)
    gray = base[0]
    assert all(abs(value - gray) <= 3 for value in lumas[:60])
    assert all(value - gray >= 15 for value in lumas[60:120])
    # Clean and grain move single pixels, not the average.
    assert all(abs(value - gray) <= 4 for value in lumas[120:])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_render_plan_splices_a_range_and_rerenders_on_a_new_scale(
    tmp_path: Path,
) -> None:
    source = tmp_path / "in.mp4"
    _gray(source, 4, "10")
    output = tmp_path / "output.mp4"
    engine = FakeUpscaleEngine()
    plan = UpscalePlan(spans=(default_span(4),))
    render_plan(source, output, plan, engine=engine, duration_s=4)
    assert len(_frame_lumas(output)) == 40
    plan = replace_overlapping(plan, UpscaleSpan(1, 2, look=BRIGHT), 4)
    render_plan(source, output, plan, engine=engine, duration_s=4)
    lumas = _frame_lumas(output)
    assert len(lumas) == 40
    assert lumas[15] - lumas[0] >= 15
    assert abs(lumas[35] - lumas[0]) <= 3
    plan = replace_overlapping(plan, UpscaleSpan(3, 4, scale=4), 4)
    assert plan.pending is not None and plan.pending.start_s == 0.0
    render_plan(source, output, plan, engine=engine, duration_s=4)
    lumas = _frame_lumas(output)
    assert len(lumas) == 40
    # The whole clip re-rendered at 4x, and the bright shot kept its look.
    assert lumas[15] - lumas[0] >= 15
