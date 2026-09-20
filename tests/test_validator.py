"""Tests for recipe policy validation."""

from __future__ import annotations

from pathlib import Path

from super_processor.probe import MediaFacts, StreamFacts, write_media_facts
from super_processor.recipe import OpName, empty_recipe, write_recipe
from super_processor.validator import (
    MIN_BITRATE_KBPS,
    validate_job_recipe,
    validate_recipe,
)


def _facts(source: Path, *, duration_s: float = 60.0) -> MediaFacts:
    return MediaFacts(
        schema_version=1,
        source_path=str(source.resolve()),
        format_name="mp4",
        format_long_name="MP4",
        duration_s=duration_s,
        size_bytes=1_000_000,
        bit_rate=1_000_000,
        streams=[
            StreamFacts(
                index=0,
                codec_type="video",
                codec_name="h264",
                width=1920,
                height=1080,
            )
        ],
        has_video=True,
        has_audio=False,
        is_vfr=False,
    )


def test_valid_recipe_passes(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    recipe = empty_recipe("abcd1234abcd1234", str(source.resolve()))
    for op in recipe.ops:
        if op.op is OpName.CONTRAST:
            op.enabled = True
            op.params = {"contrast": 1.1, "brightness": 0.0}
    result = validate_recipe(recipe, _facts(source))
    assert result.ok is True
    assert result.errors == []


def test_param_bounds_and_unknown_keys(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    recipe = empty_recipe("abcd1234abcd1234", str(source.resolve()))
    for op in recipe.ops:
        if op.op is OpName.DENOISE:
            op.enabled = True
            op.params = {"strength": 5.0, "magic": 1}
    result = validate_recipe(recipe, _facts(source))
    assert result.ok is False
    codes = {issue.code for issue in result.errors}
    assert "param_bounds" in codes
    assert "unknown_param" in codes


def test_op_order_rejected(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    recipe = empty_recipe("abcd1234abcd1234", str(source.resolve()))
    # Reverse enabled ops relative to canonical order.
    recipe.ops = list(reversed(recipe.ops))
    for op in recipe.ops:
        if op.op in {OpName.CONTRAST, OpName.DENOISE}:
            op.enabled = True
            op.params = (
                {"contrast": 1.0} if op.op is OpName.CONTRAST else {"strength": 0.2}
            )
    result = validate_recipe(recipe, _facts(source))
    assert any(issue.code == "op_order" for issue in result.errors)


def test_size_cap_infeasible(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    recipe = empty_recipe("abcd1234abcd1234", str(source.resolve()))
    recipe.target.export.max_size_mb = 1.0
    # 2 hours at 1 MB is far below the bitrate floor.
    facts = _facts(source, duration_s=7200.0)
    result = validate_recipe(recipe, facts)
    assert result.ok is False
    assert any(issue.code == "size_cap_infeasible" for issue in result.errors)
    # Sanity: floor constant remains intentional.
    assert MIN_BITRATE_KBPS == 300.0


def test_size_cap_acknowledged(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    recipe = empty_recipe("abcd1234abcd1234", str(source.resolve()))
    recipe.target.export.max_size_mb = 1.0
    for op in recipe.ops:
        if op.op is OpName.ENCODE_HEVC_SIZE_CAP:
            op.enabled = True
            op.params = {
                "max_size_mb": 1.0,
                "max_height": 1080.0,
                "acknowledge_size_risk": 1.0,
            }
    facts = _facts(source, duration_s=7200.0)
    result = validate_recipe(recipe, facts)
    assert result.ok is True
    assert any(issue.code == "size_cap_infeasible" for issue in result.warnings)


def test_validate_job_recipe_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    job_dir = tmp_path / "job"
    recipe = empty_recipe("abcd1234abcd1234", str(source.resolve()))
    write_recipe(job_dir, recipe)
    write_media_facts(job_dir, _facts(source))
    result = validate_job_recipe(job_dir)
    assert result.ok is True
