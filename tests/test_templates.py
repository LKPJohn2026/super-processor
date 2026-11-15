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


@pytest.mark.parametrize("has_vidstab", [True, False], ids=["vidstab", "deshake"])
def test_stabilize_uses_two_vidstab_passes_or_falls_back_to_deshake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, has_vidstab: bool
) -> None:
    # Pin the filter probe so both branches run whatever this ffmpeg has.
    monkeypatch.setattr(
        "super_processor.templates.ffmpeg_has_filter",
        lambda _name, **_kwargs: has_vidstab,
    )
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
    names = [step.name for step in plan.steps]
    if has_vidstab:
        assert names == ["stabilize_detect", "encode"]
        detect = plan.steps[0].argv
        assert detect[detect.index("-vf") + 1].startswith("vidstabdetect=shakiness=5:")
        # The detect pass only analyses; it writes no media file.
        assert detect[-3:] == ["-f", "null", "-"]
        assert "smoothing=12" in plan.filter_graph
        assert "zoom=8" in plan.filter_graph
    else:
        assert names == ["encode"]
        assert plan.filter_graph == "deshake"


def test_a_preview_encodes_only_the_preview_window(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake")
    recipe = empty_recipe("abcd1234abcd1234", str(source.resolve()))
    recipe.target.preview_window.start_s = 3.5
    recipe.target.preview_window.duration_s = 4.0
    recipe.encode.audio_action = "copy"
    plan = build_ffmpeg_plan(
        recipe,
        output_path=tmp_path / "out.mp4",
        work_dir=tmp_path,
        ffmpeg_bin="ffmpeg",
    )
    argv = plan.steps[0].argv
    # -ss/-t before -i seek the input, so only the window is decoded.
    assert argv.index("-ss") < argv.index("-i")
    assert argv[argv.index("-ss") + 1] == "3.5"
    assert argv[argv.index("-t") + 1] == "4"
    assert argv[argv.index("-c:a") + 1] == "copy"
    # The worker swaps the last element for a partial path; it must be the output.
    assert argv[-1] == str((tmp_path / "out.mp4").resolve())


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
            op.params = {
                "padding": 0.05,
                "crop_x": 100.0,
                "crop_y": 0.0,
                "crop_w": 720.0,
                "crop_h": 1280.0,
                "subject_cx": 0.3,
            }
        if op.op is OpName.ENCODE_SIZE_CAP:
            op.enabled = True
            op.params = {"max_size_mb": 20.0, "max_height": 1280}
    plan = build_ffmpeg_plan(
        recipe,
        output_path=tmp_path / "out.mp4",
        work_dir=tmp_path,
        duration_s=40.0,
        ffmpeg_bin="ffmpeg",
    )
    assert "crop=720:1280:100:0" in plan.filter_graph
    assert "pad=" in plan.filter_graph
    argv = plan.steps[0].argv
    # 20 MiB over 40 s, less 128 kbps kept for audio:
    # (20 * 1024 * 1024 * 8 - 128000 * 40) / 40 / 1000 = 4066 kbps.
    assert argv[argv.index("-b:v") + 1] == "4066k"
    assert argv[argv.index("-maxrate") + 1] == "4066k"
    assert argv[argv.index("-bufsize") + 1] == "8132k"
    assert "-crf" not in argv
    assert argv[argv.index("-c:a") + 1] == "aac"


def test_a_size_cap_without_a_duration_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake")
    recipe = empty_recipe("abcd1234abcd1234", str(source.resolve()))
    recipe.target.export.max_size_mb = 10.0
    # Without a duration there is no bitrate that keeps the file under the cap.
    with pytest.raises(TemplateError, match="positive media duration"):
        build_ffmpeg_plan(
            recipe,
            output_path=tmp_path / "out.mp4",
            work_dir=tmp_path,
            duration_s=None,
            ffmpeg_bin="ffmpeg",
        )


def test_missing_source_raises(tmp_path: Path) -> None:
    recipe = empty_recipe("abcd1234abcd1234", str(tmp_path / "missing.mp4"))
    with pytest.raises(TemplateError, match="not a readable file"):
        build_ffmpeg_plan(
            recipe,
            output_path=tmp_path / "out.mp4",
            work_dir=tmp_path,
            ffmpeg_bin="ffmpeg",
        )
