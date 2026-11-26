"""Tests for the upscale plan, the SPAN engine, and the range splice.

No test needs a GPU or torch: :func:`fake_span` stands a nearest-neighbour
FFmpeg 2x in for the network, so the pipeline around it (the decoder, the
threads, the delivery encode) runs for real.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import types
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from super_processor import dpir as dpir_module
from super_processor import span_sr, upscale
from super_processor.cancel import Cancelled
from super_processor.look import ShotLook
from super_processor.span_sr import (
    SPAN_FILE,
    SPAN_WEIGHTS_ENV,
    SpanError,
    batch_size,
    resolve_span_weights,
    span_weights_path,
)
from super_processor.upscale import (
    FakeUpscaleEngine,
    GpuOutOfMemory,
    SpanEngine,
    UpscaleError,
    UpscalePlan,
    UpscaleRequest,
    UpscaleSpan,
    _release_gpu,
    _upscale_failure,
    apply_range_revise,
    clamp_span,
    default_span,
    encode_delivery,
    is_delivery,
    load_upscale_plan,
    models_dir,
    planned_chunks,
    rendered_scale,
    replace_overlapping,
    save_upscale_plan,
    snap_to_keyframes,
    video_stream_facts,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg required"
)


def test_span_rejects_bad_scale_and_strength() -> None:
    with pytest.raises(UpscaleError):
        UpscaleSpan(0, 1, scale=3)
    with pytest.raises(UpscaleError):
        UpscaleSpan(0, 1, strength=1.2)
    with pytest.raises(UpscaleError):
        UpscaleRequest(Path("a.mp4"), Path("b.mp4"), scale=3)
    # 4x is gone: only a saved plan may still name it (see below).
    with pytest.raises(UpscaleError, match="scale must be 1 or 2"):
        UpscaleSpan(0, 1, scale=4)
    with pytest.raises(UpscaleError, match="scale must be 1 or 2"):
        UpscaleRequest(Path("a.mp4"), Path("b.mp4"), scale=4)
    with pytest.raises(UpscaleError):
        UpscaleSpan(0, 1, scale=True)
    assert UpscaleSpan(0, 1, scale=1).scale == 1


def test_a_plan_saved_at_4x_renders_at_2x(tmp_path: Path) -> None:
    old = UpscaleSpan(0, 6, scale=2, strength=0.3).to_dict() | {"scale": 4}
    assert UpscaleSpan.from_dict(old) == UpscaleSpan(0, 6, scale=2, strength=0.3)
    saved = {"spans": [old], "pending": old, "latest_revision": old}
    (tmp_path / upscale.UPSCALE_PLAN_FILE).write_text(
        json.dumps(saved), encoding="utf-8"
    )
    plan = load_upscale_plan(tmp_path)
    assert plan is not None
    assert plan.pending is not None and plan.latest_revision is not None
    assert {span.scale for span in (*plan.spans, plan.pending)} == {2}
    assert plan.latest_revision.scale == 2
    # Scale 1 and 2 load as saved; another number still fails.
    assert UpscaleSpan.from_dict(old | {"scale": 1}).scale == 1
    with pytest.raises(UpscaleError):
        UpscaleSpan.from_dict(old | {"scale": 3})


def test_a_saved_plan_loads_back_unchanged(tmp_path: Path) -> None:
    span = default_span(12)
    assert span == UpscaleSpan(0.0, 12.0, scale=2, strength=0.5)
    plan = UpscalePlan(
        spans=(
            UpscaleSpan(0, 4, look=ShotLook(denoise=0.4), use_upscale=False),
            UpscaleSpan(4, 12, scale=2, strength=0.8),
        ),
        pending=UpscaleSpan(4, 12, strength=0.8),
    )
    save_upscale_plan(tmp_path, plan)
    # Every field survives, the look and the pass-through flag included.
    assert load_upscale_plan(tmp_path) == plan


def test_long_clips_split_into_overlapping_eight_second_chunks() -> None:
    def edges(duration: float) -> list[tuple[float, float]]:
        return [(span.start_s, span.end_s) for span in planned_chunks(duration)]

    assert edges(4) == [(0.0, 4.0)]
    assert edges(8) == [(0.0, 8.0)]
    # 7.5 s steps, so each chunk shares half a second with the one before.
    assert edges(20) == [(0.0, 8.0), (7.5, 15.5), (15.0, 20.0)]
    # A last chunk would be under 0.05 s: the one before already reaches it.
    assert edges(15.55) == [(0.0, 8.0), (7.5, 15.5)]
    assert {span.strength for span in planned_chunks(20, strength=0.2)} == {0.2}
    with pytest.raises(UpscaleError, match="no duration"):
        planned_chunks(0)


def test_latest_revision_roundtrip_and_legacy_plans(tmp_path: Path) -> None:
    span = default_span(12)
    legacy = UpscalePlan.from_dict({"spans": [span.to_dict()]})
    assert legacy.latest_revision is None
    assert "latest_revision" not in legacy.to_dict()
    assert (
        UpscalePlan.from_dict({"spans": [span.to_dict()], "latest_revision": None})
        == legacy
    )
    revision = UpscaleSpan(2, 5, scale=1, strength=0.2)
    plan = UpscalePlan((span,), pending=span, latest_revision=revision)
    save_upscale_plan(tmp_path, replace(plan, pending=None))
    loaded = load_upscale_plan(tmp_path)
    assert loaded == replace(plan, pending=None)
    assert loaded is not None
    assert loaded.to_dict()["latest_revision"] == revision.to_dict()
    with pytest.raises(UpscaleError):
        UpscalePlan.from_dict({"spans": [span.to_dict()], "latest_revision": {}})


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("start,end,scale", [(1, 5, 2), (1, 2, 1), (0, 99, 1)])
def test_latest_revision_is_requested_patch_not_render_sentinel(
    preserve: bool, start: float, end: float, scale: int
) -> None:
    plan = UpscalePlan((UpscaleSpan(0, 3), UpscaleSpan(3, 6, strength=0.3)))
    patch = UpscaleSpan(start, end, scale=scale, strength=0.2)
    updated = replace_overlapping(plan, patch, 6, preserve_shot_settings=preserve)
    assert updated.latest_revision == clamp_span(patch, 6)
    assert UpscalePlan.from_dict(updated.to_dict()) == updated
    if scale == 1 or preserve:
        assert updated.pending is not None
        assert (updated.pending.start_s, updated.pending.end_s) == (0, 6)


def test_a_patch_over_the_whole_clip_replaces_every_shot() -> None:
    plan = UpscalePlan(spans=(UpscaleSpan(0, 5), UpscaleSpan(5, 12, strength=0.9)))
    # Past the end: clamped to the clip, so it covers all of it.
    full = replace_overlapping(plan, UpscaleSpan(0, 99, scale=1, strength=0.4), 12)
    assert full.spans == (UpscaleSpan(0, 12, scale=1, strength=0.4),)
    assert full.pending == full.spans[0]
    assert clamp_span(UpscaleSpan(0, 99), 12) == UpscaleSpan(0, 12)
    # A start past the end keeps a tenth of a second inside the clip.
    assert clamp_span(UpscaleSpan(50, 99), 12) == UpscaleSpan(11.9, 12)


def test_a_result_note_inside_one_shot_splices_that_shots_own_look() -> None:
    bright = ShotLook(brightness=0.05)
    plan = UpscalePlan(
        spans=(UpscaleSpan(0, 3, look=bright), UpscaleSpan(3, 6, strength=0.3))
    )
    updated = replace_overlapping(
        plan, UpscaleSpan(1, 2, strength=0.9), 6, preserve_shot_settings=True
    )
    # Only the strength changes; the range keeps the look of the shot under it,
    # so the splice is not rendered with the note's neutral look.
    assert updated.pending == UpscaleSpan(1, 2, strength=0.9, look=bright)
    assert sorted((s.start_s, s.end_s, s.strength, s.look) for s in updated.spans) == [
        (0.0, 1.0, 0.5, bright),
        (1.0, 2.0, 0.9, bright),
        (2.0, 3.0, 0.5, bright),
        (3.0, 6.0, 0.3, ShotLook()),
    ]


def test_a_result_note_across_shots_renders_every_shot_again() -> None:
    plan = UpscalePlan(spans=(UpscaleSpan(0, 3), UpscaleSpan(3, 6, strength=0.3)))
    updated = replace_overlapping(
        plan, UpscaleSpan(2, 4, strength=0.9), 6, preserve_shot_settings=True
    )
    # Two recipes cannot go through one range splice.
    assert updated.pending is not None
    assert (updated.pending.start_s, updated.pending.end_s) == (0.0, 6.0)
    assert [(s.start_s, s.end_s, s.strength) for s in updated.spans] == [
        (0.0, 2.0, 0.5),
        (4.0, 6.0, 0.3),
        (2.0, 3.0, 0.9),
        (3.0, 4.0, 0.9),
    ]


def test_a_result_note_outside_every_shot_is_refused() -> None:
    plan = UpscalePlan(spans=(UpscaleSpan(0, 3),))
    with pytest.raises(UpscaleError, match="does not overlap the plan"):
        replace_overlapping(plan, UpscaleSpan(4, 5), 6, preserve_shot_settings=True)


def test_release_gpu_moves_the_model_to_cpu() -> None:
    moved: list[str] = []

    class _Model:
        def to(self, device: str) -> _Model:
            moved.append(device)
            return self

    class _Stuck:
        def to(self, device: str) -> None:
            raise RuntimeError("CUDA error: device lost")

    _release_gpu(_Model())
    assert moved == ["cpu"]
    # A model that cannot move still lets the release finish.
    _release_gpu(_Stuck())


def test_bad_plan_and_span_shapes() -> None:
    with pytest.raises(UpscaleError):
        UpscaleSpan.from_dict(["nope"])
    with pytest.raises(UpscaleError):
        UpscaleSpan(1, 1)
    with pytest.raises(UpscaleError):
        UpscaleSpan(0, 1, source="output")
    with pytest.raises(UpscaleError):
        UpscalePlan.from_dict({"spans": []})
    with pytest.raises(UpscaleError):
        default_span(0)
    assert planned_chunks(4)[0].end_s == 4
    assert load_upscale_plan(Path("/tmp/does-not-exist-sp-plan")) is None


def _tiny(path: Path, seconds: int = 6) -> None:
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
            f"color=c=gray:duration={seconds}:size=160x90:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


@needs_ffmpeg
def test_a_revise_reports_the_keyframe_range_it_splices(tmp_path: Path) -> None:
    source = tmp_path / "in.mp4"
    _tiny(source, seconds=6)
    engine = FakeUpscaleEngine()
    output = tmp_path / "output.mp4"
    ranges: list[tuple[float, float]] = []

    def revise(span: UpscaleSpan) -> None:
        apply_range_revise(
            source,
            output,
            span,
            output,
            engine=engine,
            duration_s=6,
            on_revise_range=lambda start, end: ranges.append((start, end)),
        )

    # No output yet: a full render, which has no range to report.
    revise(UpscaleSpan(2.3, 3.6))
    assert ranges == []
    assert video_stream_facts(output) == (60, "10/1")
    revise(UpscaleSpan(2.3, 3.6, strength=0.2))
    # Widened out to the keyframes the delivery puts every second.
    assert ranges == [(2.0, 4.0)]
    # A range that runs to the end reports the clip's end, not None.
    revise(UpscaleSpan(4.5, 5.9, strength=0.2))
    assert ranges[-1] == (4.0, 6.0)
    assert video_stream_facts(output) == (60, "10/1")
    assert _frame_size_of(output) == (320, 180)
    assert is_delivery(output)


def test_scale_change_on_a_range_rerenders_the_whole_clip() -> None:
    plan = UpscalePlan(spans=(default_span(20),))
    updated = replace_overlapping(plan, UpscaleSpan(5, 10, scale=1, strength=0.4), 20)
    assert updated.pending == UpscaleSpan(0, 20, scale=1, strength=0.4)
    # Every shot takes the new scale; the others keep their own strength.
    assert [
        (span.start_s, span.end_s, span.scale, span.strength) for span in updated.spans
    ] == [
        (0.0, 5.0, 1, 0.5),
        (10.0, 20.0, 1, 0.5),
        (5.0, 10.0, 1, 0.4),
    ]


def test_same_scale_patch_trims_the_spans_it_overlaps() -> None:
    plan = UpscalePlan(spans=(default_span(20),))
    updated = replace_overlapping(plan, UpscaleSpan(5, 10, scale=2, strength=0.2), 20)
    assert [(span.start_s, span.end_s, span.strength) for span in updated.spans] == [
        (0.0, 5.0, 0.5),
        (10.0, 20.0, 0.5),
        (5.0, 10.0, 0.2),
    ]
    assert updated.pending == UpscaleSpan(5, 10, scale=2, strength=0.2)


def test_snap_to_keyframes_widens_outward() -> None:
    keys = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert snap_to_keyframes(keys, 1.4, 2.6, 5.0) == (1.0, 3.0)
    assert snap_to_keyframes(keys, 2.0, 3.0, 5.0) == (2.0, 3.0)
    # A key a hair after the asked time still counts (timestamps are rounded).
    assert snap_to_keyframes(keys, 1.99, 3.01, 5.0) == (2.0, 3.0)
    assert snap_to_keyframes(keys, 0.02, 4.5, 5.0) == (0.0, None)
    # A keyframe at the very end is the end: there is no tail to copy.
    assert snap_to_keyframes([*keys, 4.98], 1.5, 4.5, 5.0) == (1.0, None)
    # A first keyframe a few ms in is the start: there is no head to copy.
    assert snap_to_keyframes([0.04, 1.0, 2.0], 0.5, 1.5, 5.0) == (0.0, 2.0)
    # No keyframe info at all: the whole clip.
    assert snap_to_keyframes([], 1.5, 2.5, 5.0) == (0.0, None)


def _moving(path: Path, seconds: int = 12) -> None:
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
            f"testsrc2=size=160x90:rate=24:duration={seconds}",
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


def _frame_hashes(path: Path) -> list[str]:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v",
            "-f",
            "framemd5",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        line.rsplit(",", 1)[-1].strip()
        for line in completed.stdout.splitlines()
        if line and not line.startswith("#")
    ]


@needs_ffmpeg
def test_range_revise_leaves_untouched_frames_bit_identical(tmp_path: Path) -> None:
    source = tmp_path / "in.mp4"
    _moving(source)
    output = tmp_path / "output.mp4"
    engine = FakeUpscaleEngine()
    apply_range_revise(
        source, output, default_span(12), output, engine=engine, duration_s=12
    )
    assert is_delivery(output)
    before = _frame_hashes(output)
    assert len(before) == 12 * 24
    for patch in (UpscaleSpan(3.4, 5.6, strength=0.2), UpscaleSpan(8.2, 9.0)):
        apply_range_revise(source, output, patch, output, engine=engine, duration_s=12)
    after = _frame_hashes(output)
    assert len(after) == len(before)
    changed = [
        index for index, (a, b) in enumerate(zip(before, after, strict=True)) if a != b
    ]
    # Revises snap out to whole seconds: 3-6s and 8-9s are the only re-encodes.
    assert changed
    assert all(
        3 * 24 <= index < 6 * 24 or 8 * 24 <= index < 9 * 24 for index in changed
    )
    assert before[: 3 * 24] == after[: 3 * 24]
    assert before[9 * 24 :] == after[9 * 24 :]


@needs_ffmpeg
def test_range_revise_refuses_a_mismatched_size(tmp_path: Path) -> None:
    source = tmp_path / "in.mp4"
    _moving(source, seconds=6)
    output = tmp_path / "output.mp4"
    engine = FakeUpscaleEngine()
    apply_range_revise(
        source, output, default_span(6), output, engine=engine, duration_s=6
    )
    with pytest.raises(UpscaleError, match="whole clip"):
        apply_range_revise(
            source,
            output,
            UpscaleSpan(2, 4, scale=1),
            output,
            engine=engine,
            duration_s=6,
        )


@needs_ffmpeg
def test_range_revise_converts_an_older_output_once(tmp_path: Path) -> None:
    source = tmp_path / "in.mp4"
    _moving(source, seconds=6)
    output = tmp_path / "output.mp4"
    FakeUpscaleEngine().upscale(UpscaleRequest(source, output))
    assert not is_delivery(output)
    apply_range_revise(
        source,
        output,
        UpscaleSpan(2, 4),
        output,
        engine=FakeUpscaleEngine(),
        duration_s=6,
    )
    assert is_delivery(output)
    assert len(_frame_hashes(output)) == 6 * 24


class _TextureEngine:
    """Invents a brighter picture covered in a fine checkerboard.

    The brightness is a change to the coarse layer, which the blend always
    takes from the source; the checkerboard is fine detail, which comes
    through in proportion to strength.
    """

    def upscale(self, request: UpscaleRequest) -> Path:
        size = _frame_size_of(request.source)
        frames = len(_frame_hashes(request.source))
        rate = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "csv=p=0",
                str(request.source),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
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
                f"nullsrc=size={size[0] * request.scale}x"
                f"{size[1] * request.scale}:rate={rate}",
                "-vf",
                "geq=lum='188+50*(mod(floor(X/2)+floor(Y/2),2)*2-1)':cb=128:cr=128",
                "-frames:v",
                str(frames),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(request.output),
            ],
            check=True,
        )
        return request.output


def _frame_size_of(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    width, height = out.split(",")[:2]
    return int(width), int(height)


def _detail(path: Path) -> float:
    """Mean absolute difference between horizontal neighbours, first frame."""
    width, _height = _frame_size_of(path)
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-vf",
            "format=gray",
            "-f",
            "rawvideo",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    rows = [raw[i : i + width] for i in range(0, len(raw), width)]
    diffs = [abs(row[x + 1] - row[x]) for row in rows for x in range(width - 1)]
    return sum(diffs) / len(diffs)


def _mean_luma(path: Path) -> float:
    """Average luma over every frame, decoded with plain ``-i`` (no lavfi path
    escaping, so Windows drive paths work)."""
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            "scale=16:16:flags=area,format=gray",
            "-f",
            "rawvideo",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    assert raw, f"no frames decoded from {path}"
    return sum(raw) / len(raw)


@needs_ffmpeg
def test_strength_mixes_the_restored_picture_with_a_plain_upscale(
    tmp_path: Path,
) -> None:
    source = tmp_path / "gray.mp4"
    _tiny(source, seconds=4)  # flat gray, 10 fps
    detail: dict[float, float] = {}
    for strength in (0.0, 0.5, 1.0):
        output = tmp_path / f"out_{strength}.mp4"
        apply_range_revise(
            source,
            output,
            UpscaleSpan(0, 4, strength=strength),
            output,
            engine=_TextureEngine(),
            duration_s=4,
        )
        assert len(_frame_hashes(output)) == 40
        # The engine's brightening is a coarse change: it never reaches the output.
        assert abs(_mean_luma(output) - _mean_luma(source)) < 3
        detail[strength] = _detail(output)
    # Its fine texture does, in proportion to strength.
    assert detail[0.0] < 3
    assert detail[1.0] > 30
    assert 0.35 < detail[0.5] / detail[1.0] < 0.65


@needs_ffmpeg
def test_range_revise_blends_only_the_named_range(tmp_path: Path) -> None:
    source = tmp_path / "gray.mp4"
    _tiny(source, seconds=6)
    output = tmp_path / "output.mp4"
    engine = _TextureEngine()
    apply_range_revise(
        source,
        output,
        UpscaleSpan(0, 6, strength=1.0),
        output,
        engine=engine,
        duration_s=6,
    )
    before = _frame_hashes(output)
    apply_range_revise(
        source,
        output,
        UpscaleSpan(2, 4, strength=0.0),
        output,
        engine=engine,
        duration_s=6,
    )
    after = _frame_hashes(output)
    assert len(after) == len(before) == 60
    assert before[:20] == after[:20]
    assert before[40:] == after[40:]
    assert all(a != b for a, b in zip(before[20:40], after[20:40], strict=True))


class _ShortEngine(FakeUpscaleEngine):
    """Returns two frames fewer than it was given."""

    def upscale(self, request: UpscaleRequest) -> Path:
        super().upscale(request)
        frames, _rate = video_stream_facts(request.output)
        trimmed = request.output.with_name("short_" + request.output.name)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(request.output),
                "-frames:v",
                str(frames - 2),
                str(trimmed),
            ],
            check=True,
        )
        trimmed.replace(request.output)
        return request.output


@needs_ffmpeg
def test_splice_refuses_to_change_the_frame_count(tmp_path: Path) -> None:
    source = tmp_path / "in.mp4"
    _moving(source, seconds=6)
    output = tmp_path / "output.mp4"
    apply_range_revise(
        source,
        output,
        default_span(6),
        output,
        engine=FakeUpscaleEngine(),
        duration_s=6,
    )
    before = _frame_hashes(output)
    with pytest.raises(UpscaleError, match="previous output is left unchanged"):
        apply_range_revise(
            source,
            output,
            UpscaleSpan(2, 4),
            output,
            engine=_ShortEngine(),
            duration_s=6,
        )
    assert _frame_hashes(output) == before


def _render_into(
    entry: str, source: Path, output: Path, engine: upscale.UpscaleEngine
) -> Path:
    """Render ``source`` into ``output`` by one of the three commit points."""
    if entry == "full":
        return upscale.render_shots(
            source, output, (default_span(6),), engine=engine, duration_s=6
        )
    if entry == "shots":
        return upscale.render_shots(
            source,
            output,
            (UpscaleSpan(0, 3), UpscaleSpan(3, 6, strength=0.2)),
            engine=engine,
            duration_s=6,
        )
    return apply_range_revise(
        source, output, UpscaleSpan(2, 4), output, engine=engine, duration_s=6
    )


def _delivered(tmp_path: Path) -> tuple[Path, Path, bytes, bytes]:
    """A source, a finished delivery of it, and that delivery's bytes and marker."""
    source = tmp_path / "in.mp4"
    _moving(source, seconds=6)
    output = tmp_path / "out" / "output.mp4"
    output.parent.mkdir()
    apply_range_revise(
        source,
        output,
        default_span(6),
        output,
        engine=FakeUpscaleEngine(),
        duration_s=6,
    )
    assert is_delivery(output)
    marker = upscale.delivery_marker_path(output)
    return source, output, output.read_bytes(), marker.read_bytes()


@needs_ffmpeg
@pytest.mark.parametrize("entry", ["full", "shots", "revise"])
def test_a_failed_audio_mux_leaves_the_old_output_and_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    source, output, before, marker_before = _delivered(tmp_path)

    def broken_mux(video: Path, _source: Path, **_kwargs: Any) -> Path:
        # The audio goes onto the finished work file, never onto the output.
        assert video != output
        raise UpscaleError("mux failed")

    monkeypatch.setattr(upscale, "mux_source_audio", broken_mux)
    with pytest.raises(UpscaleError, match="mux failed"):
        _render_into(entry, source, output, FakeUpscaleEngine())
    assert output.read_bytes() == before
    assert upscale.delivery_marker_path(output).read_bytes() == marker_before


class _Control:
    """A render control that records what the output held at the commit."""

    def __init__(self, output: Path, *, refuse: bool = False) -> None:
        self.output = output
        self.refuse = refuse
        self.seen: list[bytes] = []

    def check(self) -> None:
        """Never asks to stop while FFmpeg runs; only the commit refuses."""

    def commit(self) -> None:
        self.seen.append(self.output.read_bytes())
        if self.refuse:
            raise UpscaleError("stopped")


@needs_ffmpeg
@pytest.mark.parametrize("entry", ["full", "shots", "revise"])
def test_the_output_is_replaced_once_after_the_commit(
    tmp_path: Path, entry: str
) -> None:
    source, output, before, marker_before = _delivered(tmp_path)
    refused = _Control(output, refuse=True)
    with upscale.render_control(refused), pytest.raises(UpscaleError, match="stop"):
        _render_into(entry, source, output, FakeUpscaleEngine())
    # Stopped at the commit: the old output and its marker were never touched.
    assert refused.seen == [before]
    assert output.read_bytes() == before
    assert upscale.delivery_marker_path(output).read_bytes() == marker_before

    control = _Control(output)
    with upscale.render_control(control):
        _render_into(entry, source, output, _TextureEngine())
    # One commit, while the output was still the old file; then the new one
    # is in place, complete, with its own marker.
    assert control.seen == [before]
    assert output.read_bytes() != before
    assert is_delivery(output)
    assert video_stream_facts(output)[0] == video_stream_facts(source)[0]
    # No control is left installed after the block.
    assert upscale._CONTROL.get() is None


class _FailingEngine(FakeUpscaleEngine):
    def upscale(self, request: UpscaleRequest) -> Path:
        raise UpscaleError("the GPU went away")


@needs_ffmpeg
def test_a_failed_full_cover_revise_keeps_the_output_a_delivery(
    tmp_path: Path,
) -> None:
    source, output, before, marker_before = _delivered(tmp_path)
    with pytest.raises(UpscaleError, match="GPU went away"):
        apply_range_revise(
            source,
            output,
            default_span(6),
            output,
            engine=_FailingEngine(),
            duration_s=6,
        )
    # The whole-file restore works in its own file, so the output's marker
    # is not removed before the new picture exists.
    assert is_delivery(output)
    assert output.read_bytes() == before
    assert upscale.delivery_marker_path(output).read_bytes() == marker_before


_FAKE_WEIGHTS = b"not really SPAN"


class FakeSpan:
    """What the stand-in GPU block was asked to do."""

    def __init__(self, weights: Path) -> None:
        self.weights = weights
        self.loads = 0
        self.batches: list[int] = []
        # Frames in each block handed to the GPU, and its first frame.
        self.frames: list[int] = []
        self.firsts: list[bytes] = []
        # What each clip's blend was prepared with.
        self.blends: list[dict[str, Any]] = []
        # DPIR: weights loaded, and (level, scale, SPAN model) per GPU call.
        self.dpir_loads = 0
        self.calls: list[tuple[int, int, str | None]] = []


def _yuv420(width: int, height: int) -> int:
    return upscale.yuv420_frame_bytes(width, height)


def fake_span(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_batch: int | None = None,
    drop: int = 0,
    fail: str = "",
) -> FakeSpan:
    """Stand in for the GPU block without torch or a GPU: a nearest-neighbour
    2x of the packed yuv420p frames, by FFmpeg.

    Batches over ``max_batch`` frames raise PyTorch's out-of-memory text,
    ``drop`` returns that many frames fewer, and ``fail`` raises instead.
    Everything around the GPU block (weights check, decoder, threads, the
    delivery encode) is the real code.
    """
    weights = tmp_path / "models" / "SPAN" / SPAN_FILE
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(_FAKE_WEIGHTS)
    monkeypatch.setenv("SUPER_PROCESSOR_MODELS", str(tmp_path / "models"))
    digest = hashlib.sha256(_FAKE_WEIGHTS).hexdigest()
    monkeypatch.setattr(span_sr, "SPAN_SHA256", digest)
    monkeypatch.setattr(upscale, "cuda_available", lambda: True)
    fake = FakeSpan(weights)

    def load_span(path: Path) -> str:
        assert path == weights
        fake.loads += 1
        return "model"

    def prepare_blend(width: int, height: int, **kwargs: Any) -> Any:
        fake.blends.append({"size": (width, height), **kwargs})
        return types.SimpleNamespace(width=width, height=height)

    def load_dpir(path: Path) -> str:
        fake.dpir_loads += 1
        return "dpir"

    def restore_block(
        model: str | None,
        blend: Any,
        frames: bytearray,
        *,
        batch: int,
        dpir: Any = None,
        dpir_level: int = 0,
        scale: int = 2,
    ) -> bytearray:
        assert model in ("model", None)
        assert (dpir == "dpir") == (dpir_level > 0)
        assert isinstance(frames, bytearray)
        scale = 2 if model is not None else scale
        fake.batches.append(batch)
        fake.calls.append((dpir_level, scale, model))
        if fail:
            raise RuntimeError(fail)
        if max_batch is not None and batch > max_batch:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        width, height = blend.width, blend.height
        count = len(frames) // _yuv420(width, height)
        fake.frames.append(count)
        fake.firsts.append(bytes(frames[: _yuv420(width, height)]))
        doubled = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-s",
                f"{width}x{height}",
                "-i",
                "-",
                "-vf",
                f"scale=iw*{scale}:ih*{scale}:flags=neighbor",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-",
            ],
            input=bytes(frames),
            check=True,
            capture_output=True,
        ).stdout
        keep = max(1, count - drop) * _yuv420(scale * width, scale * height)
        return bytearray(doubled[:keep])

    monkeypatch.setattr(dpir_module, "load_dpir", load_dpir)
    monkeypatch.setattr(span_sr, "load_span", load_span)
    monkeypatch.setattr(span_sr, "prepare_blend", prepare_blend)
    monkeypatch.setattr(span_sr, "restore_block", restore_block)
    return fake


# Frames whose fine detail spells their own index: band b (8 rows of the
# 64x48 source) is a 1-pixel checkerboard when bit b of the frame number is
# set and flat grey otherwise. The output's bands read back which upscaled
# frame landed where.
_BANDS = 6
_BAND_ROWS = 8


def _numbered(path: Path, frames: int = 40) -> None:
    bit = f"mod(floor(N/pow(2,floor(Y/{_BAND_ROWS}))),2)"
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
            f"nullsrc=s=64x48:r=10:d={frames / 10},"
            f"geq=lum='128+60*{bit}*(2*mod(X+Y,2)-1)':cb=128:cr=128",
            "-c:v",
            "libx264",
            "-qp",
            "0",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _frame_numbers(path: Path) -> list[int]:
    """The index each frame of a 2x :func:`_numbered` upscale spells."""
    width, height = _frame_size_of(path)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", "format=gray",
         "-f", "rawvideo", "-"],
        check=True,
        capture_output=True,
    ).stdout  # fmt: skip
    rows = height // _BANDS
    numbers = []
    for start in range(0, len(raw), width * height):
        frame = raw[start : start + width * height]
        number = 0
        for band in range(_BANDS):
            # Rows away from the band's edges, where the blur mixes bands.
            lines = range(band * rows + 4, (band + 1) * rows - 4)
            diffs = [
                abs(frame[y * width + x + 1] - frame[y * width + x])
                for y in lines
                for x in range(width - 1)
            ]
            if sum(diffs) / len(diffs) > 15:
                number |= 1 << band
        numbers.append(number)
    return numbers


def _started_processes(monkeypatch: pytest.MonkeyPatch) -> list[subprocess.Popen[Any]]:
    """Every process started from now on (the test's own FFmpeg runs too)."""
    started: list[subprocess.Popen[Any]] = []
    real = subprocess.Popen

    def tracked(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        process = real(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", tracked)
    return started


def _within(seconds: float, run: Callable[[], object]) -> BaseException | None:
    """What ``run`` raised, failing the test if it has not returned by then.

    A stuck stream would otherwise hang the whole test run.
    """
    raised: list[BaseException] = []

    def target() -> None:
        try:
            run()
        except BaseException as exc:  # noqa: BLE001 - handed to the test
            raised.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), f"still running after {seconds} s"
    return raised[0] if raised else None


def _swap_decoder(
    monkeypatch: pytest.MonkeyPatch, swap: Callable[[list[str]], list[str]]
) -> None:
    """Run ``swap(command)`` in place of SPAN's FFmpeg decoder."""
    real = subprocess.Popen

    def popen(command: list[str], *args: Any, **kwargs: Any) -> Any:
        if "-nostdin" in command:  # only the decoder reads no stdin
            command = swap(command)
        return real(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", popen)


@needs_ffmpeg
def test_span_writes_every_frame_at_twice_the_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fake_span(tmp_path, monkeypatch)
    source = tmp_path / "in.mp4"
    _tiny(source, seconds=4)
    output = SpanEngine().upscale(
        UpscaleRequest(source, tmp_path / "out" / "clip.mp4", strength=0.2)
    )
    assert video_stream_facts(output) == (40, "10/1")
    assert _frame_size_of(output) == (320, 180)
    assert fake.loads == 1
    # A 4 s clip is one block, upscaled in one call.
    assert fake.batches == [batch_size(160, 90)]
    assert fake.frames == [40]
    # The blend is prepared once for the clip, at 3 output pixels of detail.
    assert fake.blends == [
        {"size": (160, 90), "strength": 0.2, "regions": (), "sigma": 3.0}
    ]
    # Streamed: nothing but the delivery file and its marker is left behind.
    assert sorted(p.name for p in output.parent.iterdir()) == [
        "clip.mp4",
        "clip.mp4.delivery.json",
    ]


@needs_ffmpeg
def test_many_blocks_keep_the_frame_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fake_span(tmp_path, monkeypatch)
    # Seven 64x48 frames per block: 40 frames are six blocks.
    monkeypatch.setattr(upscale, "STREAM_BLOCK_BYTES", _yuv420(64, 48) * 7 + 5)
    assert upscale.block_frames(64, 48) == 7
    source = tmp_path / "numbered.mp4"
    _numbered(source)
    seen: list[float] = []
    output = SpanEngine().upscale(
        UpscaleRequest(
            source, tmp_path / "out" / "clip.mp4", strength=1.0, on_progress=seen.append
        )
    )
    assert fake.frames == [7, 7, 7, 7, 7, 5]
    assert fake.loads == 1
    assert video_stream_facts(output) == (40, "10/1")
    assert _frame_size_of(output) == (128, 96)
    assert _frame_numbers(output) == list(range(40))
    # Progress is frames written over the source's: once per block, to 1.
    assert seen == pytest.approx([7 / 40, 14 / 40, 21 / 40, 28 / 40, 35 / 40, 1.0])
    assert all(a < b for a, b in zip(seen, seen[1:], strict=False))


@needs_ffmpeg
def test_the_decoder_applies_the_clean_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fake_span(tmp_path, monkeypatch)
    source = tmp_path / "in.mp4"
    _moving(source, seconds=1)
    look = ShotLook(deblock=0.5, denoise=0.5)
    output = tmp_path / "out" / "clip.mp4"
    SpanEngine().upscale(UpscaleRequest(source, output, look=look))
    assert video_stream_facts(output)[0] == 24

    def first_frame(*vf: str) -> bytes:
        return subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(source), *vf, "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
            check=True,
            capture_output=True,
        ).stdout  # fmt: skip

    cleaned = first_frame("-vf", look.clean_filter())
    assert cleaned != first_frame()
    # SPAN was handed the cleaned frames, not the source's.
    assert fake.firsts[0] == cleaned
    # And the delivery marker records the look the encode finished with.
    marker = json.loads(upscale.delivery_marker_path(output).read_text("utf-8"))
    assert marker["look"] == look.to_dict()


@needs_ffmpeg
def test_previews_share_one_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fake_span(tmp_path, monkeypatch)
    first = tmp_path / "a.mp4"
    second = tmp_path / "b.mp4"
    long = tmp_path / "long.mp4"
    _tiny(first, seconds=3)
    _tiny(second, seconds=2)
    # Longer than the 8 s previews once had to be: streaming has no limit.
    _tiny(long, seconds=12)
    engine = SpanEngine()
    assert engine.upscale_many([]) == []
    outputs = engine.upscale_many(
        [
            UpscaleRequest(first, tmp_path / "out" / "a.mp4"),
            UpscaleRequest(
                second, tmp_path / "out" / "b.mp4", look=ShotLook(denoise=0.5)
            ),
            UpscaleRequest(long, tmp_path / "out" / "c.mp4"),
        ]
    )
    assert [video_stream_facts(o)[0] for o in outputs] == [30, 20, 120]
    assert all(is_delivery(o) for o in outputs)
    assert fake.loads == 1
    assert fake.frames == [30, 20, 120]


@needs_ffmpeg
def test_span_output_is_a_delivery_the_caller_only_renames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_span(tmp_path, monkeypatch)
    source = tmp_path / "in.mp4"
    _moving(source, seconds=2)
    look = ShotLook(deblock=0.5)
    raw = tmp_path / "out" / "restored.mp4"
    SpanEngine().upscale(UpscaleRequest(source, raw, strength=0.3, look=look))
    marker = json.loads(upscale.delivery_marker_path(raw).read_text("utf-8"))
    # libx264 here; h264_nvenc when the suite runs against NVENC.
    expected = upscale._delivery_marker(upscale.delivery_encoder("ffmpeg", (320, 180)))
    assert marker["version"] == 3
    assert {key: marker[key] for key in expected} == expected
    assert marker["blend"]["strength"] == 0.3
    assert is_delivery(raw)
    before = _frame_hashes(raw)
    final = tmp_path / "final.mp4"
    assert (
        encode_delivery(raw, final, blend_with=source, strength=0.3, look=look) == final
    )
    assert not raw.exists()
    assert not upscale.delivery_marker_path(raw).exists()
    assert is_delivery(final)
    # The same file, moved: an encode would have changed the frames' bits.
    assert _frame_hashes(final) == before


def test_a_marker_from_an_older_version_is_not_a_delivery(tmp_path: Path) -> None:
    video = tmp_path / "out.mp4"
    video.write_bytes(b"picture")
    marker = upscale.delivery_marker_path(video)
    current = upscale._delivery_marker(upscale.delivery_encoder())
    marker.write_text(json.dumps(current), encoding="utf-8")
    assert is_delivery(video)
    # Version 1 used the "medium" preset and wrote no preset key.
    old = {key: value for key, value in current.items() if key != "preset"}
    marker.write_text(json.dumps(old | {"version": 1}), encoding="utf-8")
    assert not is_delivery(video)
    # Version 2 had every key version 3 has, so only the version tells.
    marker.write_text(json.dumps(current | {"version": 2}), encoding="utf-8")
    assert not is_delivery(video)
    marker.write_text("[]", encoding="utf-8")
    assert not is_delivery(video)
    marker.write_text("{broken", encoding="utf-8")
    assert not is_delivery(video)
    # A marker whose video is gone marks nothing.
    marker.write_text(json.dumps(current), encoding="utf-8")
    video.unlink()
    assert not is_delivery(video)


@needs_ffmpeg
def test_a_revise_re_encodes_a_version_1_output(tmp_path: Path) -> None:
    source = tmp_path / "in.mp4"
    _moving(source, seconds=6)
    output = tmp_path / "output.mp4"
    engine = FakeUpscaleEngine()
    apply_range_revise(
        source, output, default_span(6), output, engine=engine, duration_s=6
    )
    marker = upscale.delivery_marker_path(output)
    old = json.loads(marker.read_text("utf-8"))
    del old["preset"]
    marker.write_text(json.dumps(old | {"version": 1}), encoding="utf-8")
    assert not is_delivery(output)
    apply_range_revise(
        source, output, UpscaleSpan(2, 4), output, engine=engine, duration_s=6
    )
    assert is_delivery(output)
    expected = upscale._delivery_marker(upscale.delivery_encoder("ffmpeg", (320, 180)))
    assert json.loads(marker.read_text("utf-8"))["preset"] == expected["preset"]
    assert len(_frame_hashes(output)) == 6 * 24


def test_weights_are_found_and_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = span_weights_path(tmp_path)
    assert path == tmp_path / "SPAN" / SPAN_FILE
    with pytest.raises(SpanError, match="SPAN weights were not found") as missing:
        resolve_span_weights(tmp_path)
    assert span_sr.SPAN_URL in str(missing.value)
    path.parent.mkdir()
    path.write_bytes(b"truncated")
    with pytest.raises(SpanError, match="not the expected SPAN weights"):
        resolve_span_weights(tmp_path)
    monkeypatch.setattr(
        span_sr, "SPAN_SHA256", hashlib.sha256(b"truncated").hexdigest()
    )
    assert resolve_span_weights(tmp_path) == path
    # The setting names the file itself, wherever it is.
    mine = tmp_path / "mine.pth"
    mine.write_bytes(b"truncated")
    monkeypatch.setenv(SPAN_WEIGHTS_ENV, str(mine))
    assert span_weights_path(tmp_path / "unused") == mine
    assert resolve_span_weights(tmp_path / "unused") == mine
    assert issubclass(SpanError, UpscaleError)


def test_engine_needs_cuda_weights_and_2x(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This unit test isolates readiness; media timing has its own preflight.
    monkeypatch.setattr(upscale, "require_supported_timing", lambda _p: None)
    monkeypatch.setenv("SUPER_PROCESSOR_MODELS", str(tmp_path / "models"))
    request = UpscaleRequest(tmp_path / "a.mp4", tmp_path / "b.mp4")
    monkeypatch.setattr(upscale, "cuda_available", lambda: False)
    with pytest.raises(UpscaleError, match="SPAN needs an NVIDIA GPU with CUDA"):
        SpanEngine().upscale(request)
    with pytest.raises(UpscaleError, match="CUDA"):
        SpanEngine().upscale_many([request])
    monkeypatch.setattr(upscale, "cuda_available", lambda: True)
    with pytest.raises(SpanError, match="weights were not found"):
        SpanEngine().upscale(request)
    with pytest.raises(UpscaleError, match="only upscales 2x, not 1x"):
        SpanEngine().upscale(replace(request, scale=1))


@needs_ffmpeg
def test_load_errors_name_span(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_span(tmp_path, monkeypatch)
    source = tmp_path / "in.mp4"
    _tiny(source, seconds=2)
    request = UpscaleRequest(source, tmp_path / "out" / "clip.mp4")

    def broken(_weights: Path) -> None:
        raise RuntimeError("boom\nwith a long traceback")

    monkeypatch.setattr(span_sr, "load_span", broken)
    for run in (
        lambda: SpanEngine().upscale(request),
        lambda: SpanEngine().upscale_many([request]),
    ):
        with pytest.raises(UpscaleError, match=r"^SPAN failed: boom$"):
            run()

    def no_room(_weights: Path) -> None:
        raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")

    monkeypatch.setattr(span_sr, "load_span", no_room)
    with pytest.raises(GpuOutOfMemory, match="Close other programs that use the GPU"):
        SpanEngine().upscale(request)


@needs_ffmpeg
def test_run_errors_name_span_and_are_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fake_span(tmp_path, monkeypatch, fail="bad shape")
    source = tmp_path / "in.mp4"
    _tiny(source, seconds=2)
    output = tmp_path / "out" / "clip.mp4"
    request = UpscaleRequest(source, output)
    started = _started_processes(monkeypatch)
    for run in (
        lambda: SpanEngine().upscale(request),
        lambda: SpanEngine().upscale_many([request]),
    ):
        fake.batches.clear()
        with pytest.raises(UpscaleError, match="SPAN failed: bad shape") as failed:
            run()
        # Not out of memory, so no smaller batch is tried.
        assert type(failed.value) is UpscaleError
        assert len(fake.batches) == 1
    assert not output.exists()
    assert not list(output.parent.glob("*.encoding.mp4"))
    assert all(process.poll() is not None for process in started)


def test_system_memory_is_not_a_gpu_problem() -> None:
    for exc in (
        MemoryError(),
        RuntimeError(
            "DefaultCPUAllocator: can't allocate memory: you tried to allocate 6 GB"
        ),
    ):
        failure = _upscale_failure(exc)
        assert type(failure) is UpscaleError
        assert "the upscale ran out of system memory" in str(failure)
    kept = GpuOutOfMemory("already said")
    assert _upscale_failure(kept) is kept
    plain = UpscaleError("no frames")
    assert _upscale_failure(plain) is plain


@needs_ffmpeg
def test_a_few_missing_frames_repeat_the_last_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_span(tmp_path, monkeypatch, drop=3)
    source = tmp_path / "numbered.mp4"
    _numbered(source)
    output = SpanEngine().upscale(
        UpscaleRequest(source, tmp_path / "out" / "clip.mp4", strength=1.0)
    )
    assert video_stream_facts(output) == (40, "10/1")
    # 37 frames came back; the last of them fills the three that did not.
    assert _frame_numbers(output) == [*range(37), 36, 36, 36]


@needs_ffmpeg
def test_extra_frames_are_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_span(tmp_path, monkeypatch)
    source = tmp_path / "numbered.mp4"
    _numbered(source)
    real = upscale.video_stream_facts

    def two_fewer(video: Path, **kwargs: Any) -> tuple[int, str]:
        frames, rate = real(video, **kwargs)
        return (frames - 2, rate) if video == source else (frames, rate)

    # The decoder then yields two frames more than the source should hold.
    monkeypatch.setattr(upscale, "video_stream_facts", two_fewer)
    output = SpanEngine().upscale(
        UpscaleRequest(source, tmp_path / "out" / "clip.mp4", strength=1.0)
    )
    assert real(output) == (38, "10/1")
    assert _frame_numbers(output) == list(range(38))


@needs_ffmpeg
def test_span_refuses_a_badly_misaligned_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_span(tmp_path, monkeypatch, drop=20)
    source = tmp_path / "in.mp4"
    _tiny(source, seconds=4)
    output = tmp_path / "out" / "clip.mp4"
    started = _started_processes(monkeypatch)
    # The shortfall is found after the last block: it once hung the stream.
    failed = _within(60, lambda: SpanEngine().upscale(UpscaleRequest(source, output)))
    assert isinstance(failed, UpscaleError)
    assert re.search(
        "SPAN read 20 frames of a 40-frame clip .*misaligned picture", str(failed)
    )
    assert not output.exists()
    assert not list(output.parent.glob("*.encoding.mp4"))
    assert all(process.poll() is not None for process in started)


@needs_ffmpeg
def test_span_refuses_far_too_many_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fake_span(tmp_path, monkeypatch)
    monkeypatch.setattr(upscale, "STREAM_BLOCK_BYTES", _yuv420(64, 48) * 4)
    source = tmp_path / "numbered.mp4"
    _numbered(source)
    real = upscale.video_stream_facts

    def half(video: Path, **kwargs: Any) -> tuple[int, str]:
        frames, rate = real(video, **kwargs)
        return (frames // 2, rate) if video == source else (frames, rate)

    # 40 frames decoded for a clip said to hold 20: more than 8 over is wrong.
    monkeypatch.setattr(upscale, "video_stream_facts", half)
    output = tmp_path / "out" / "clip.mp4"
    failed = _within(60, lambda: SpanEngine().upscale(UpscaleRequest(source, output)))
    assert isinstance(failed, UpscaleError)
    assert "SPAN read over 28 frames of a 20-frame clip" in str(failed)
    assert not output.exists()
    # Refused as soon as the count went over, not after the whole clip.
    assert sum(fake.frames) < 40


@needs_ffmpeg
def test_a_gpu_error_partway_stops_the_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_span(tmp_path, monkeypatch)
    monkeypatch.setattr(upscale, "STREAM_BLOCK_BYTES", _yuv420(64, 48) * 4)
    real = span_sr.restore_block
    calls: list[int] = []

    def third_fails(
        model: str, blend: Any, frames: Any, *, batch: int, **kwargs: Any
    ) -> Any:
        calls.append(batch)
        if len(calls) == 3:
            raise RuntimeError("illegal memory access")
        return real(model, blend, frames, batch=batch, **kwargs)

    monkeypatch.setattr(span_sr, "restore_block", third_fails)
    source = tmp_path / "numbered.mp4"
    _numbered(source)
    output = tmp_path / "out" / "clip.mp4"
    started = _started_processes(monkeypatch)
    # Two blocks were written already, so the writer is mid-clip: it once hung.
    failed = _within(60, lambda: SpanEngine().upscale(UpscaleRequest(source, output)))
    assert isinstance(failed, UpscaleError)
    assert str(failed) == "SPAN failed: illegal memory access"
    assert len(calls) == 3
    assert list(output.parent.iterdir()) == []
    assert all(process.poll() is not None for process in started)


@needs_ffmpeg
@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in launcher is sh")
def test_a_failed_stream_ends_ffmpeg_behind_a_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chocolatey's ffmpeg.exe on Windows is a shim that starts the real one.
    Killing the shim left the real encoder running with its output open
    (WinError 32 on the delete). Here ``ffmpeg`` is a launcher that starts the
    real one as a child, and the stream must still end both."""
    real = shutil.which("ffmpeg")
    assert real is not None
    children = tmp_path / "children.txt"
    shim = tmp_path / "bin" / "ffmpeg"
    shim.parent.mkdir()
    # An explicit <&0 keeps the pipe as the child's stdin; sh would otherwise
    # give a background job /dev/null.
    shim.write_text(
        f'#!/bin/sh\n"{real}" "$@" <&0 &\necho "$! $*" >> "{children}"\nwait $!\n'
    )
    shim.chmod(0o755)
    fake_span(tmp_path, monkeypatch)
    monkeypatch.setattr(upscale, "STREAM_BLOCK_BYTES", _yuv420(64, 48) * 4)
    source = tmp_path / "numbered.mp4"
    _numbered(source)
    monkeypatch.setenv("PATH", f"{shim.parent}{os.pathsep}{os.environ['PATH']}")
    restore = span_sr.restore_block
    calls: list[int] = []

    def third_fails(
        model: str, blend: Any, frames: Any, *, batch: int, **kwargs: Any
    ) -> Any:
        calls.append(batch)
        if len(calls) == 3:
            raise RuntimeError("illegal memory access")
        return restore(model, blend, frames, batch=batch, **kwargs)

    monkeypatch.setattr(span_sr, "restore_block", third_fails)
    # Windows refuses to delete a file a live process has open: note any
    # FFmpeg behind the launcher still running when the output is deleted.
    held_open: list[int] = []
    unlink = Path.unlink

    def watched_unlink(path: Path, missing_ok: bool = False) -> None:
        if path.name.endswith(".encoding.mp4") and children.exists():
            # Only the FFmpeg writing this file holds it open.
            for line in children.read_text().splitlines():
                pid, _, args = line.partition(" ")
                if path.name in args and _alive(int(pid)):
                    held_open.append(int(pid))
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", watched_unlink)
    output = tmp_path / "out" / "clip.mp4"
    failed = _within(60, lambda: SpanEngine().upscale(UpscaleRequest(source, output)))
    assert str(failed) == "SPAN failed: illegal memory access"
    assert list(output.parent.iterdir()) == []
    # The decoder and the encoder both ran behind the launcher.
    started = children.read_text()
    assert "yuv420p -" in started and "clip.encoding.mp4" in started
    assert held_open == []


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@needs_ffmpeg
def test_an_encoder_failure_reports_ffmpeg_and_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_span(tmp_path, monkeypatch)
    source = tmp_path / "in.mp4"
    _tiny(source, seconds=4)
    output = tmp_path / "out" / "clip.mp4"
    real = upscale._delivery_codec
    monkeypatch.setattr(
        upscale,
        "_delivery_codec",
        lambda *args: ["-vf", "no_such_filter", *real(*args)],
    )
    started = _started_processes(monkeypatch)
    with pytest.raises(
        UpscaleError, match="encoding SPAN frames failed: .*no_such_filter"
    ):
        SpanEngine().upscale(UpscaleRequest(source, output))
    assert list(output.parent.iterdir()) == []
    assert all(process.poll() is not None for process in started)


@needs_ffmpeg
def test_decoder_failures_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_span(tmp_path, monkeypatch)
    source = tmp_path / "in.mp4"
    _tiny(source, seconds=4)
    output = tmp_path / "out" / "clip.mp4"
    request = UpscaleRequest(source, output)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    # Not a video at all: the timing preflight or the size probe refuses it.
    with pytest.raises(UpscaleError):
        SpanEngine().upscale(UpscaleRequest(bad, output))
    with pytest.raises(UpscaleError, match="frame size of bad.mp4"):
        upscale._stream_span(upscale._Loaded("model"), UpscaleRequest(bad, output))

    def unreadable(command: list[str]) -> list[str]:
        at = command.index("-i") + 1
        return [*command[:at], str(tmp_path / "gone.mp4"), *command[at + 1 :]]

    # A decoder that fails to open its input names why, not the missing frames.
    _swap_decoder(monkeypatch, unreadable)
    with pytest.raises(
        UpscaleError, match="decoding in.mp4 for SPAN failed: .*No such file"
    ):
        SpanEngine().upscale(request)
    assert not output.exists()

    def says_nothing(_command: list[str]) -> list[str]:
        return [sys.executable, "-c", "pass"]

    # A decoder that succeeds with no frames gives nothing to splice.
    _swap_decoder(monkeypatch, says_nothing)
    with pytest.raises(UpscaleError, match="SPAN read 0 frames of a 40-frame"):
        SpanEngine().upscale(request)
    assert not output.exists()

    def dies_at_the_end(_command: list[str]) -> list[str]:
        script = (
            "import sys; sys.stdout.buffer.write(bytes(160 * 90 * 3 // 2 * 40)); "
            "sys.stdout.flush(); sys.stderr.write('corrupt packet'); sys.exit(1)"
        )
        return [sys.executable, "-c", script]

    # Every frame read, but the decoder failed: its own error is reported,
    # and the encode it fed is never published as a finished delivery.
    _swap_decoder(monkeypatch, dies_at_the_end)
    with pytest.raises(UpscaleError, match="decoding in.mp4 for SPAN failed: corrupt"):
        SpanEngine().upscale(request)
    assert list(output.parent.iterdir()) == []

    def quits(_command: list[str]) -> list[str]:
        script = (
            "import os, sys; sys.stdout.buffer.write(bytes(160 * 90 * 3 // 2 * 38)); "
            "sys.stdout.flush(); os._exit(3)"
        )
        return [sys.executable, "-c", script]

    # Two frames short (within the slack, so it would be padded) and failed
    # with nothing on stderr: still refused, by its exit status.
    _swap_decoder(monkeypatch, quits)
    with pytest.raises(UpscaleError, match="for SPAN failed: ffmpeg exited with 3"):
        SpanEngine().upscale(request)
    assert list(output.parent.iterdir()) == []

    def missing(_command: list[str]) -> list[str]:
        return [str(tmp_path / "no-ffmpeg-here")]

    _swap_decoder(monkeypatch, missing)
    with pytest.raises(UpscaleError, match="failed to start ffmpeg"):
        SpanEngine().upscale(request)


def test_batches_by_size() -> None:
    # About four megapixels of input per GPU call, 1 to 8 frames.
    assert [batch_size(*size) for size in ((640, 480), (1440, 1080), (3840, 2160))] == [
        8,
        3,
        1,
    ]


def test_a_piped_encode_needs_a_base_and_a_source(tmp_path: Path) -> None:
    with pytest.raises(UpscaleError, match="always blended"):
        encode_delivery(
            None, tmp_path / "out.mp4", frames_in=(2, 2, "10/1", lambda _pipe: None)
        )
    with pytest.raises(UpscaleError, match="needs a source or frames_in"):
        encode_delivery(None, tmp_path / "out.mp4")


def test_gpu_helpers_with_a_stand_in_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: calls.append("empty"),
        ipc_collect=lambda: calls.append("ipc"),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    assert upscale.cuda_available()
    upscale._empty_cuda_cache()
    assert calls == ["empty", "ipc"]
    monkeypatch.setitem(sys.modules, "torch", None)
    assert not upscale.cuda_available()


def test_models_dir_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPER_PROCESSOR_MODELS", str(tmp_path))
    assert models_dir() == tmp_path
    monkeypatch.delenv("SUPER_PROCESSOR_MODELS")
    assert models_dir() == Path.cwd() / ".dogfood" / "models"


@needs_ffmpeg
def test_rendered_scale_reads_the_sizes(tmp_path: Path) -> None:
    source = tmp_path / "in.mp4"
    _tiny(source, seconds=1)
    doubled = FakeUpscaleEngine().upscale(
        UpscaleRequest(source, tmp_path / "two.mp4", scale=2)
    )
    assert rendered_scale(source, doubled) == 2
    assert rendered_scale(source, source) == 1
    assert rendered_scale(source, tmp_path / "missing.mp4") is None
    odd = tmp_path / "odd.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-vf", "scale=320:90",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(odd)],
        check=True,
    )  # fmt: skip
    assert rendered_scale(source, odd) is None


@needs_ffmpeg
def test_joined_shots_with_a_frame_missing_leave_the_output_alone(
    tmp_path: Path,
) -> None:
    source = tmp_path / "in.mp4"
    _moving(source, seconds=6)
    output = tmp_path / "out" / "output.mp4"
    output.parent.mkdir()
    output.write_bytes(b"previous render")
    # Two shots that do not coalesce, each two frames short from the engine.
    with pytest.raises(
        UpscaleError,
        match="joined shots have 140 frames instead of 144; the previous output",
    ):
        upscale.render_shots(
            source,
            output,
            (UpscaleSpan(0, 3), UpscaleSpan(3, 6, strength=0.2)),
            engine=_ShortEngine(),
            duration_s=6,
        )
    assert output.read_bytes() == b"previous render"


def _script(path: Path, body: str) -> str:
    """A stand-in tool: a shell script that ignores its arguments."""
    path.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    path.chmod(0o755)
    return str(path)


needs_sh = pytest.mark.skipif(
    sys.platform == "win32", reason="the stand-in tools are sh scripts"
)


@needs_ffmpeg
def test_probes_of_a_file_that_is_not_a_video_raise_upscale_errors(
    tmp_path: Path,
) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    with pytest.raises(UpscaleError, match="could not count the frames of .*junk"):
        video_stream_facts(junk)
    with pytest.raises(UpscaleError, match=r"\S"):
        upscale.keyframe_times(junk)
    missing = str(tmp_path / "no-ffprobe")
    with pytest.raises(UpscaleError, match="failed to start ffprobe"):
        video_stream_facts(junk, ffprobe_bin=missing)
    with pytest.raises(UpscaleError, match="failed to start ffprobe"):
        upscale.keyframe_times(junk, ffprobe_bin=missing)


@needs_sh
def test_keyframe_times_skip_lines_that_are_not_times(tmp_path: Path) -> None:
    # ffprobe prints N/A for a frame without a timestamp, and a trailing comma
    # with some builds; duplicates come from a stream listed twice.
    lines = "1.500000,\\nN/A\\n\\n2.000000\\n0.000000\\nbad\\n2.0\\n"
    probe = _script(tmp_path / "ffprobe", f"printf '{lines}'")
    assert upscale.keyframe_times(tmp_path / "any.mp4", ffprobe_bin=probe) == [
        0.0,
        1.5,
        2.0,
    ]


@needs_sh
@pytest.mark.parametrize("watched", [False, True])
def test_a_failed_ffmpeg_run_reports_its_own_error(
    tmp_path: Path, watched: bool
) -> None:
    seen: list[float] = []

    def run(command: list[str]) -> None:
        if watched:
            upscale._run_ffmpeg(command, on_progress=seen.append, total_frames=10)
        else:
            upscale._run_ffmpeg(command)

    broken = _script(
        tmp_path / "ffmpeg", "echo frame=5; echo 'Invalid data found' >&2; exit 1"
    )
    with pytest.raises(UpscaleError, match="^Invalid data found$"):
        run([broken, "-i", "x"])
    # Progress is read from FFmpeg's -progress lines while it runs.
    assert seen == ([0.5] if watched else [])
    with pytest.raises(UpscaleError, match="^ffmpeg failed$"):
        run([_script(tmp_path / "quiet", "exit 3")])
    with pytest.raises(UpscaleError, match="failed to start ffmpeg"):
        run([str(tmp_path / "no-ffmpeg")])


def _piped(width: int = 64, height: int = 48, frames: int = 3) -> upscale.FramesIn:
    def feed(pipe: Any) -> None:
        pipe.write(bytes(upscale.yuv420_frame_bytes(width, height) * frames))

    return (width, height, "10/1", feed)


@needs_sh
def test_an_encoder_that_fails_after_its_input_publishes_nothing(
    tmp_path: Path,
) -> None:
    # It reads every frame, then fails while finishing the file.
    ffmpeg = _script(
        tmp_path / "ffmpeg",
        # A shell loop, not cat, drains stdin: CI's no-ffmpeg run has no PATH.
        "while read -r _; do :; done; echo 'moov atom failed' >&2; exit 1",
    )
    dest = tmp_path / "out" / "clip.mp4"
    dest.parent.mkdir()
    dest.write_bytes(b"previous")
    marker = upscale.delivery_marker_path(dest)
    marker.write_text("{}", encoding="utf-8")
    with pytest.raises(
        UpscaleError, match="^encoding SPAN frames failed: moov atom failed$"
    ):
        encode_delivery(
            None, dest, ffmpeg_bin=ffmpeg, blend_with=dest, frames_in=_piped()
        )
    assert dest.read_bytes() == b"previous"
    # The old marker is gone, so nothing splices into a half-replaced file.
    assert not marker.exists()
    assert not list(dest.parent.glob("*.encoding.mp4"))
    with pytest.raises(UpscaleError, match="failed to start ffmpeg"):
        encode_delivery(
            None,
            dest,
            ffmpeg_bin=str(tmp_path / "no-ffmpeg"),
            blend_with=dest,
            frames_in=_piped(),
        )


@needs_ffmpeg
def test_a_piped_encode_is_a_delivery_with_every_frame(tmp_path: Path) -> None:
    dest = tmp_path / "clip.mp4"
    look = ShotLook(grain=0.2)
    encode_delivery(
        None,
        dest,
        blend_with=tmp_path / "base.mp4",
        strength=0.4,
        look=look,
        frames_in=_piped(frames=12),
    )
    assert video_stream_facts(dest) == (12, "10/1")
    assert is_delivery(dest)
    marker = json.loads(upscale.delivery_marker_path(dest).read_text("utf-8"))
    assert marker["blend"] == upscale._blend_record(0.4, ())
    assert marker["look"] == look.to_dict()


class _StopAfter:
    """A render control that says stop once ``event`` is set."""

    def __init__(self, event: threading.Event | None = None) -> None:
        self.event = event or threading.Event()
        self.checks = 0

    def check(self) -> None:
        self.checks += 1
        if self.event.is_set():
            raise Cancelled

    def commit(self) -> None:
        self.check()


def _stop(_share: float) -> None:
    """An ``on_progress`` that stops the run, as the wizard's does."""
    raise Cancelled


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


@needs_sh
@pytest.mark.parametrize("how", ["control", "control-watched", "progress"])
def test_a_stop_ends_a_slow_ffmpeg_at_once(tmp_path: Path, how: str) -> None:
    pid_file = tmp_path / "pid"
    # Shell builtins only: the release runs the tests with no PATH.
    slow = _script(
        tmp_path / "ffmpeg", f"echo $$ > {pid_file}; echo frame=1; while :; do :; done"
    )
    stop = threading.Event()
    control = _StopAfter(stop)

    def run() -> None:
        with upscale.render_control(control):
            if how == "control":
                upscale._run_ffmpeg([slow])
            else:
                progress = _stop if how == "progress" else (lambda _share: None)
                upscale._run_ffmpeg([slow], on_progress=progress, total_frames=10)

    timer = threading.Timer(0.3, stop.set)
    timer.start()
    started = time.monotonic()
    try:
        raised = _within(10, run)
    finally:
        timer.cancel()
    assert isinstance(raised, Cancelled)
    assert time.monotonic() - started < 2.0
    # FFmpeg itself is gone, not left encoding in the background.
    assert _gone(int(pid_file.read_text(encoding="utf-8")))
    if how != "progress":
        assert control.checks >= 1


@needs_sh
def test_without_a_stop_a_controlled_ffmpeg_runs_to_its_end(tmp_path: Path) -> None:
    go = tmp_path / "go"
    script = _script(
        tmp_path / "ffmpeg", f"while [ ! -e {go} ]; do :; done; echo frame=10"
    )
    seen: list[float] = []
    control = _StopAfter()
    timer = threading.Timer(0.3, go.touch)
    timer.start()
    with upscale.render_control(control):
        upscale._run_ffmpeg([script], on_progress=seen.append, total_frames=10)
    assert seen == [1.0]
    # It was asked while FFmpeg ran, and said go on.
    assert control.checks >= 1


@needs_ffmpeg
def test_a_stopped_delivery_encode_leaves_no_encoding_file(tmp_path: Path) -> None:
    source = tmp_path / "in.mp4"
    _tiny(source, seconds=2)
    out = tmp_path / "out"
    with pytest.raises(Cancelled):
        upscale.plain_delivery(
            UpscaleRequest(source, source, on_progress=_stop), out / "plain.mp4"
        )
    with pytest.raises(Cancelled):
        encode_delivery(source, out / "blend.mp4", blend_with=source, on_progress=_stop)
    assert sorted(p.name for p in out.iterdir()) == []


@needs_ffmpeg
def test_a_stopped_span_render_unloads_and_frees_the_gpu_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fake_span(tmp_path, monkeypatch)
    monkeypatch.setattr(upscale, "STREAM_BLOCK_BYTES", _yuv420(160, 90) * 5)
    unloads: list[object] = []
    real_unload = upscale._unload

    def unload(loaded: Any) -> None:
        unloads.append(loaded.model)
        real_unload(loaded)

    monkeypatch.setattr(upscale, "_unload", unload)
    started = _started_processes(monkeypatch)
    source = tmp_path / "in.mp4"
    _tiny(source, seconds=4)
    output = tmp_path / "out" / "clip.mp4"
    raised = _within(
        30,
        lambda: SpanEngine().upscale(UpscaleRequest(source, output, on_progress=_stop)),
    )
    # A stop stays a stop: it is not turned into an UpscaleError.
    assert isinstance(raised, Cancelled)
    assert unloads == ["model"] and fake.loads == 1
    assert not upscale._GPU_LOCK.locked()
    assert all(process.poll() is not None for process in started)
    assert not output.parent.exists() or sorted(output.parent.iterdir()) == []

    # Between clips, the installed control is asked before each one.
    stop = threading.Event()
    stop.set()
    with upscale.render_control(_StopAfter(stop)), pytest.raises(Cancelled):
        SpanEngine().upscale_many([UpscaleRequest(source, output)])
    assert len(unloads) == 2
    assert not upscale._GPU_LOCK.locked()


@needs_sh
def test_a_stopped_piped_encode_stays_a_stop_when_ffmpeg_printed_errors(
    tmp_path: Path,
) -> None:
    noisy = _script(
        tmp_path / "ffmpeg",
        "echo 'Conversion failed!' >&2; while read -r _line; do :; done",
    )
    dest = tmp_path / "clip.mp4"

    def feed_then(exc: BaseException) -> Callable[[Any], None]:
        def feed(pipe: Any) -> None:
            pipe.write(b"x" * 64)
            raise exc

        return feed

    with pytest.raises(Cancelled):
        upscale._encode_piped(
            (16, 16, "10/1", feed_then(Cancelled())),
            dest,
            ShotLook(),
            noisy,
            upscale._blend_record(1.0, ()),
        )
    # Any other failure still reports the encoder's own error.
    with pytest.raises(UpscaleError, match="Conversion failed!"):
        upscale._encode_piped(
            (16, 16, "10/1", feed_then(RuntimeError("broken pipe"))),
            dest,
            ShotLook(),
            noisy,
            upscale._blend_record(1.0, ()),
        )
    assert not dest.exists()
    assert not list(tmp_path.glob("*.encoding.mp4"))
