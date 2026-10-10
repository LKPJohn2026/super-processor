"""Tests for the FlashVSR upscale plan, fake engine, and range splice."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from super_processor.upscale import (
    FakeUpscaleEngine,
    FlashVsrEngine,
    UpscaleError,
    UpscalePlan,
    UpscaleRequest,
    UpscaleSpan,
    apply_range_revise,
    clamp_span,
    default_span,
    flashvsr_knobs,
    is_delivery,
    load_upscale_plan,
    planned_chunks,
    replace_overlapping,
    resolve_flashvsr_weights,
    save_upscale_plan,
    snap_to_keyframes,
)


def test_span_rejects_bad_scale_and_strength() -> None:
    with pytest.raises(UpscaleError):
        UpscaleSpan(0, 1, scale=3)
    with pytest.raises(UpscaleError):
        UpscaleSpan(0, 1, strength=1.2)
    with pytest.raises(UpscaleError):
        UpscaleRequest(Path("a.mp4"), Path("b.mp4"), scale=1)


def test_plan_roundtrip_and_chunks(tmp_path: Path) -> None:
    span = default_span(12)
    assert span.scale == 2
    assert span.strength == 0.5
    plan = UpscalePlan(spans=(span,))
    save_upscale_plan(tmp_path, plan)
    loaded = load_upscale_plan(tmp_path)
    assert loaded is not None
    assert loaded.spans[0].end_s == 12
    chunks = planned_chunks(20)
    assert len(chunks) >= 2
    assert chunks[0].end_s - chunks[0].start_s <= 8.05
    assert chunks[-1].end_s == 20
    sparse, local = flashvsr_knobs(0.2)
    assert sparse == 2.0
    assert local == 11
    assert flashvsr_knobs(0.9) == (1.5, 9)


def test_replace_overlapping_keeps_scale_and_range() -> None:
    plan = UpscalePlan(spans=(default_span(12),))
    updated = replace_overlapping(plan, UpscaleSpan(2, 5, scale=2, strength=0.2), 12)
    assert updated.pending is not None
    assert updated.pending.strength == 0.2
    full = replace_overlapping(plan, UpscaleSpan(0, 12, scale=4, strength=0.4), 12)
    assert full.spans[0].scale == 4
    clamped = clamp_span(UpscaleSpan(0, 99, scale=2, strength=0.5), 12)
    assert clamped.end_s == 12


def test_flashvsr_runner_writes_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "FlashVSR"
    wan = home / "examples" / "WanVSR"
    weights = wan / "FlashVSR-v1.1"
    weights.mkdir(parents=True)
    for name in (
        "diffusion_pytorch_model_streaming_dmd.safetensors",
        "LQ_proj_in.ckpt",
        "TCDecoder.ckpt",
    ):
        (weights / name).write_bytes(b"w")
    infer = wan / "infer_flashvsr_v1.1_tiny.py"
    infer.write_text(
        "class _Cuda:\n"
        "    @staticmethod\n"
        "    def empty_cache():\n"
        "        return None\n"
        "class _Torch:\n"
        "    bfloat16 = 'bf16'\n"
        "    cuda = _Cuda()\n"
        "torch = _Torch()\n"
        "def init_pipeline():\n"
        "    def pipe(**kwargs):\n"
        "        return kwargs['LQ_video']\n"
        "    return pipe\n"
        "def prepare_input_tensor(path, scale, dtype, device):\n"
        "    return ('lq', 8, 8, 1, 10)\n"
        "def tensor2video(video):\n"
        "    return [video]\n"
        "def save_video(pictures, save_path, fps, quality):\n"
        "    from pathlib import Path\n"
        "    Path(save_path).write_bytes(b'ok')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("super_processor.upscale.cuda_available", lambda: True)
    monkeypatch.setattr("super_processor.upscale.flashvsr_home", lambda: home)
    source = tmp_path / "in.mp4"
    source.write_bytes(b"src")
    output = tmp_path / "out" / "clip.mp4"
    FlashVsrEngine().upscale(UpscaleRequest(source, output, scale=2, strength=0.2))
    assert output.read_bytes() == b"ok"


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


def test_missing_weights_and_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(UpscaleError, match="weights"):
        resolve_flashvsr_weights(tmp_path)
    monkeypatch.setattr("super_processor.upscale.cuda_available", lambda: False)
    engine = FlashVsrEngine()
    with pytest.raises(UpscaleError, match="CUDA"):
        engine.upscale(UpscaleRequest(tmp_path / "a.mp4", tmp_path / "b.mp4"))


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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_fake_engine_range_splice(tmp_path: Path) -> None:
    source = tmp_path / "in.mp4"
    _tiny(source, seconds=6)
    engine = FakeUpscaleEngine()
    output = tmp_path / "output.mp4"
    span = default_span(6)
    apply_range_revise(
        source,
        output,
        span,
        output,
        engine=engine,
        duration_s=6,
    )
    assert output.is_file()
    revised = UpscaleSpan(2, 4, scale=2, strength=0.2)
    apply_range_revise(
        source,
        output,
        revised,
        output,
        engine=engine,
        duration_s=6,
    )
    assert output.is_file()
    assert output.stat().st_size > 0


def _flashvsr_tree(tmp_path: Path, script: str) -> tuple[Path, Path]:
    home = tmp_path / "FlashVSR"
    wan = home / "examples" / "WanVSR"
    weights = wan / "FlashVSR-v1.1"
    weights.mkdir(parents=True)
    for name in (
        "diffusion_pytorch_model_streaming_dmd.safetensors",
        "LQ_proj_in.ckpt",
        "TCDecoder.ckpt",
    ):
        (weights / name).write_bytes(b"w")
    infer = wan / "infer_flashvsr_v1.1_tiny.py"
    infer.write_text(script, encoding="utf-8")
    return home, infer


_FAKE_INFER = """
class _Cuda:
    @staticmethod
    def empty_cache():
        return None

class _Torch:
    bfloat16 = "bf16"
    cuda = _Cuda()

torch = _Torch()

def init_pipeline():
    def pipe(**kwargs):
        return kwargs["LQ_video"]
    return pipe

def prepare_input_tensor(path, scale, dtype, device):
    return ("lq", 8, 8, 1, 10)

def tensor2video(video):
    return [video]

def save_video(pictures, save_path, fps, quality):
    import subprocess
    from pathlib import Path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=gray:duration=3:size=32x18:rate=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", save_path,
        ],
        check=True,
    )
"""


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_flashvsr_chunks_long_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, _script = _flashvsr_tree(tmp_path, _FAKE_INFER)
    monkeypatch.setattr("super_processor.upscale.cuda_available", lambda: True)
    monkeypatch.setattr("super_processor.upscale.flashvsr_home", lambda: home)
    source = tmp_path / "long.mp4"
    _tiny(source, seconds=12)
    output = tmp_path / "out" / "clip.mp4"
    FlashVsrEngine().upscale(UpscaleRequest(source, output, scale=2, strength=0.2))
    assert output.is_file()
    assert output.stat().st_size > 0
    assert str(home) not in sys.path


def test_chunked_flashvsr_wraps_loader_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, _script = _flashvsr_tree(tmp_path, "raise RuntimeError('boom')\n")
    monkeypatch.setattr("super_processor.upscale.cuda_available", lambda: True)
    monkeypatch.setattr("super_processor.upscale.flashvsr_home", lambda: home)

    class _Facts:
        duration_s = 20.0

    monkeypatch.setattr("super_processor.probe.probe_file", lambda _path: _Facts())
    source = tmp_path / "in.mp4"
    source.write_bytes(b"src")
    with pytest.raises(UpscaleError, match="FlashVSR failed"):
        FlashVsrEngine().upscale(
            UpscaleRequest(source, tmp_path / "out.mp4", scale=2, strength=0.5)
        )


def test_scale_change_on_a_range_rerenders_the_whole_clip() -> None:
    plan = UpscalePlan(spans=(default_span(20),))
    updated = replace_overlapping(plan, UpscaleSpan(5, 10, scale=4, strength=0.4), 20)
    assert updated.pending == UpscaleSpan(0, 20, scale=4, strength=0.4)
    assert updated.spans == (updated.pending,)


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
    assert snap_to_keyframes(keys, 0.02, 4.5, 5.0) == (0.0, None)


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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
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
            UpscaleSpan(2, 4, scale=4),
            output,
            engine=engine,
            duration_s=6,
        )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
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
