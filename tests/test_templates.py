"""Tests for FFmpeg template argv construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_processor.recipe import OpName, Recipe, TargetMode, empty_recipe
from super_processor.templates import TemplateError, build_ffmpeg_plan


def _recipe(source: Path) -> Recipe:
    recipe = empty_recipe("abcd1234abcd1234", str(source.resolve()))
    for op in recipe.ops:
        if op.op is OpName.CONTRAST:
            op.enabled = True
            op.params = {"contrast": 1.2, "brightness": 0.05, "gamma": 1.0}
        if op.op is OpName.WHITE_BALANCE:
            op.enabled = True
            op.params = {"temperature": 5600.0, "tint": 20.0}
        if op.op is OpName.DENOISE:
            op.enabled = True
            op.params = {"strength": 0.4}
    return recipe


def test_build_plan_includes_filters_and_no_shell(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake")
    recipe = _recipe(source)
    plan = build_ffmpeg_plan(
        recipe,
        output_path=tmp_path / "out.mp4",
        work_dir=tmp_path,
        duration_s=10.0,
        ffmpeg_bin="/usr/bin/ffmpeg",
    )
    assert plan.mode == TargetMode.PREVIEW.value
    assert len(plan.steps) == 1
    argv = plan.steps[0].argv
    assert argv[0] == "/usr/bin/ffmpeg"
    assert "-i" in argv
    assert str(source.resolve()) in argv
    assert "-vf" in argv
    assert "colortemperature=" in plan.filter_graph
    assert "eq=contrast=1.2" in plan.filter_graph
    assert "hqdn3d=" in plan.filter_graph
    assert "colorbalance=gm=" in plan.filter_graph
    # No shell metacharacters glued into a single string with paths.
    assert all(";" not in part for part in argv)
    assert "-crf" in argv


def test_stabilize_adds_detect_pass(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake")
    recipe = empty_recipe("abcd1234abcd1234", str(source.resolve()))
    for op in recipe.ops:
        if op.op is OpName.STABILIZE:
            op.enabled = True
            op.params = {"shakiness": 5, "smoothing": 12, "max_crop_pct": 8}
    plan = build_ffmpeg_plan(
        recipe,
        output_path=tmp_path / "out.mp4",
        work_dir=tmp_path,
        ffmpeg_bin="ffmpeg",
    )
    assert [step.name for step in plan.steps] == ["stabilize_detect", "encode"]
    assert "vidstabdetect=" in plan.steps[0].argv[plan.steps[0].argv.index("-vf") + 1]
    assert "vidstabtransform=" in plan.filter_graph


def test_reframe_and_size_cap_bitrate(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake")
    recipe = empty_recipe("abcd1234abcd1234", str(source.resolve()))
    recipe.target.mode = TargetMode.FINAL
    recipe.target.export.max_size_mb = 20.0
    recipe.target.export.max_height = 1280
    for op in recipe.ops:
        if op.op is OpName.REFRAME_VERTICAL:
            op.enabled = True
            op.params = {"padding": 0.05}
        if op.op is OpName.ENCODE_HEVC_SIZE_CAP:
            op.enabled = True
            op.params = {"max_size_mb": 20.0, "max_height": 1280}
    plan = build_ffmpeg_plan(
        recipe,
        output_path=tmp_path / "out.mp4",
        work_dir=tmp_path,
        duration_s=40.0,
        ffmpeg_bin="ffmpeg",
    )
    assert "pad=" in plan.filter_graph
    argv = plan.steps[0].argv
    assert "-b:v" in argv
    assert "-crf" not in argv


def test_missing_source_raises(tmp_path: Path) -> None:
    recipe = empty_recipe("abcd1234abcd1234", str(tmp_path / "missing.mp4"))
    with pytest.raises(TemplateError, match="not a readable file"):
        build_ffmpeg_plan(
            recipe,
            output_path=tmp_path / "out.mp4",
            work_dir=tmp_path,
            ffmpeg_bin="ffmpeg",
        )
