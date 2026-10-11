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
    is_delivery,
    load_upscale_plan,
    planned_chunks,
    replace_overlapping,
    resolve_flashvsr_weights,
    save_upscale_plan,
    snap_to_keyframes,
    video_stream_facts,
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


def test_replace_overlapping_keeps_scale_and_range() -> None:
    plan = UpscalePlan(spans=(default_span(12),))
    updated = replace_overlapping(plan, UpscaleSpan(2, 5, scale=2, strength=0.2), 12)
    assert updated.pending is not None
    assert updated.pending.strength == 0.2
    full = replace_overlapping(plan, UpscaleSpan(0, 12, scale=4, strength=0.4), 12)
    assert full.spans[0].scale == 4
    clamped = clamp_span(UpscaleSpan(0, 99, scale=2, strength=0.5), 12)
    assert clamped.end_s == 12


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_flashvsr_runner_writes_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, _script = _flashvsr_tree(tmp_path, _FAKE_INFER)
    monkeypatch.setattr("super_processor.upscale.cuda_available", lambda: True)
    monkeypatch.setattr("super_processor.upscale.flashvsr_home", lambda: home)
    source = tmp_path / "in.mp4"
    _tiny(source, seconds=4)
    output = tmp_path / "out" / "clip.mp4"
    FlashVsrEngine().upscale(UpscaleRequest(source, output, scale=2, strength=0.2))
    assert video_stream_facts(output) == (40, "10/1")


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
import os
import subprocess
from pathlib import Path


class _Cuda:
    @staticmethod
    def empty_cache():
        return None


class _Torch:
    bfloat16 = "bf16"
    cuda = _Cuda()


torch = _Torch()
_SOURCE = {}


def init_pipeline():
    def pipe(**kwargs):
        return kwargs["LQ_video"]
    return pipe


def prepare_input_tensor(path, scale, dtype, device):
    _SOURCE["path"] = path
    _SOURCE["scale"] = int(scale)
    return ("lq", 8, 8, 1, 10)


def tensor2video(video):
    return [video]


def save_video(pictures, save_path, fps, quality):
    # Upscale the real input, then drop FAKE_DROP trailing frames and write
    # at FAKE_RATE, the way a block-padded model or a rounded fps would.
    drop = int(os.environ.get("FAKE_DROP", "0"))
    rate = os.environ.get("FAKE_RATE", "10")
    count = int(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0",
         _SOURCE["path"]],
        check=True, capture_output=True, text=True,
    ).stdout.strip())
    scale = _SOURCE["scale"]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", _SOURCE["path"],
         "-vf", f"scale=iw*{scale}:ih*{scale},setpts=N/({rate})/TB",
         "-r", rate, "-frames:v", str(max(1, count - drop)),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", save_path],
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
    assert video_stream_facts(output)[0] == 120
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
    # Every shot takes the new scale; the others keep their own strength.
    assert [
        (span.start_s, span.end_s, span.scale, span.strength) for span in updated.spans
    ] == [
        (0.0, 5.0, 4, 0.5),
        (10.0, 20.0, 4, 0.5),
        (5.0, 10.0, 4, 0.4),
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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
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
        # FlashVSR's brightening is a coarse change: it never reaches the output.
        assert abs(_mean_luma(output) - _mean_luma(source)) < 3
        detail[strength] = _detail(output)
    # Its fine texture does, in proportion to strength.
    assert detail[0.0] < 3
    assert detail[1.0] > 30
    assert 0.35 < detail[0.5] / detail[1.0] < 0.65


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
@pytest.mark.parametrize(("drop", "rate"), [(3, "10"), (0, "12"), (5, "12")])
def test_flashvsr_output_is_conformed_to_the_source_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drop: int,
    rate: str,
) -> None:
    home, _script = _flashvsr_tree(tmp_path, _FAKE_INFER)
    monkeypatch.setattr("super_processor.upscale.cuda_available", lambda: True)
    monkeypatch.setattr("super_processor.upscale.flashvsr_home", lambda: home)
    monkeypatch.setenv("FAKE_DROP", str(drop))
    monkeypatch.setenv("FAKE_RATE", rate)
    for seconds, frames in ((6, 60), (12, 120)):  # one pass, then chunked
        source = tmp_path / f"in_{seconds}.mp4"
        _tiny(source, seconds=seconds)
        output = tmp_path / f"out_{seconds}" / "clip.mp4"
        FlashVsrEngine().upscale(UpscaleRequest(source, output, scale=2))
        assert video_stream_facts(output) == (frames, "10/1")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_flashvsr_refuses_a_badly_misaligned_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, _script = _flashvsr_tree(tmp_path, _FAKE_INFER)
    monkeypatch.setattr("super_processor.upscale.cuda_available", lambda: True)
    monkeypatch.setattr("super_processor.upscale.flashvsr_home", lambda: home)
    monkeypatch.setenv("FAKE_DROP", "20")
    source = tmp_path / "in.mp4"
    _tiny(source, seconds=6)
    with pytest.raises(UpscaleError, match="40 frames for a 60-frame clip"):
        FlashVsrEngine().upscale(
            UpscaleRequest(source, tmp_path / "out" / "clip.mp4", scale=2)
        )


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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
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
