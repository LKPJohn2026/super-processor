"""Tests for the FlashVSR upscale plan, fake engine, and range splice."""

from __future__ import annotations

import shutil
import subprocess
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
    load_upscale_plan,
    planned_chunks,
    replace_overlapping,
    resolve_flashvsr_weights,
    save_upscale_plan,
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
    updated = replace_overlapping(
        plan, UpscaleSpan(2, 5, scale=2, strength=0.2), 12
    )
    assert updated.pending is not None
    assert updated.pending.strength == 0.2
    full = replace_overlapping(
        plan, UpscaleSpan(0, 12, scale=4, strength=0.4), 12
    )
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
        "class _Torch:\n"
        "    bfloat16 = 'bf16'\n"
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
    FlashVsrEngine().upscale(
        UpscaleRequest(source, output, scale=2, strength=0.2)
    )
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
