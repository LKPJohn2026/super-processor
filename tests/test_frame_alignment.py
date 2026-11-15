"""Every output frame is the source frame at the same position.

The clips here number their frames: each frame is a flat grey whose level
encodes its index, so a part that starts a frame early, or a tail that
freezes, shows up as a wrong number instead of a slightly lower PSNR.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from test_upscale import fake_span

from super_processor import upscale
from super_processor.span_sr import batch_size
from super_processor.upscale import (
    FakeUpscaleEngine,
    GpuOutOfMemory,
    SpanEngine,
    UpscaleError,
    UpscaleRequest,
    UpscaleSpan,
    _upscale_failure,
    block_frames,
    render_shots,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg required"
)
FPS = 24
STEP = 5  # grey levels per frame; numbers repeat every 40 frames
FRAME_BYTES = 96 * 64 * 3 // 2  # one yuv420p frame of the numbered clips


def _small_blocks(monkeypatch: pytest.MonkeyPatch, frames: int) -> list[int]:
    """Hand SPAN ``frames`` frames per block; the frames each block held."""
    monkeypatch.setattr(upscale, "STREAM_BLOCK_BYTES", FRAME_BYTES * frames)
    restore = upscale._restore_span
    counts: list[int] = []

    def counted(loaded: Any, prepared: Any) -> Any:
        counts.append(len(prepared.payload) // FRAME_BYTES)
        return restore(loaded, prepared)

    monkeypatch.setattr(upscale, "_restore_span", counted)
    return counts


def _numbered(path: Path, frames: int) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=gray:size=96x64:rate={FPS}:duration={frames / FPS + 1:.3f}",
            "-vf",
            f"geq=lum='20+mod(N*{STEP}\\,200)':cb=128:cr=128",
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-bf",
            "2",
            # One tick per frame, like the trailer: seek times then round to
            # whole frames, which is where a half-frame cut tipped over.
            "-video_track_timescale",
            str(FPS),
            str(path),
        ],
        check=True,
    )
    return path


def _numbers(path: Path) -> list[int]:
    log = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-vf",
            "signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [round((float(v.split()[0]) - 20) / STEP) for v in log.split("YAVG=")[1:]]


def _segment_errors(want: list[int], got: list[int], cuts: list[int]) -> list[str]:
    edges = [0, *cuts, len(want)]
    errors = []
    for n, (first, stop) in enumerate(zip(edges, edges[1:], strict=False), 1):
        wrong = [i for i in range(first, stop) if got[i] != want[i]]
        if wrong:
            errors.append(
                f"segment {n} (frames {first}-{stop - 1}): {len(wrong)} wrong"
            )
    return errors


@needs_ffmpeg
def test_every_segment_lines_up_with_the_source(tmp_path: Path) -> None:
    # The trailer's cuts that came out a frame early: 141.5/24 written to six
    # decimals rounds down to frame 141.
    cuts = [142, 307, 448]
    source = _numbered(tmp_path / "in.mp4", 520)
    want = _numbers(source)
    assert want[:3] == [0, 1, 2] and len(want) == 520
    edges = [0, *cuts, 520]
    looks = [(0.5, True), (0.0, False), (0.3, True), (0.6, False)]
    spans = tuple(
        UpscaleSpan(first / FPS, stop / FPS, strength=strength, use_upscale=use)
        for (first, stop), (strength, use) in zip(
            zip(edges, edges[1:], strict=False), looks, strict=True
        )
    )
    output = render_shots(
        source,
        tmp_path / "out" / "output.mp4",
        spans,
        engine=FakeUpscaleEngine(),
        duration_s=520 / FPS,
    )
    got = _numbers(output)
    assert len(got) == 520
    assert _segment_errors(want, got, cuts) == []


@needs_ffmpeg
@pytest.mark.parametrize("frames", [48, 49, 95])
def test_span_returns_every_frame_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frames: int
) -> None:
    # Seven frames per block: 49 ends on a block edge, 48 and 95 part way in.
    fake_span(tmp_path, monkeypatch)
    counts = _small_blocks(monkeypatch, 7)
    source = _numbered(tmp_path / "in.mp4", frames)
    output = SpanEngine().upscale(
        UpscaleRequest(source, tmp_path / "out" / "clip.mp4", scale=2)
    )
    assert _numbers(output) == _numbers(source)
    assert sum(counts) == frames and len(counts) == -(-frames // 7)
    assert set(counts[:-1]) == {7}


@needs_ffmpeg
def test_out_of_memory_retries_with_smaller_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Like a render next to a game: 8 frames per call do not fit, 2 do.
    fake = fake_span(tmp_path, monkeypatch, max_batch=2)
    counts = _small_blocks(monkeypatch, 20)
    source = _numbered(tmp_path / "in.mp4", 130)
    seen: list[float] = []
    output = SpanEngine().upscale(
        UpscaleRequest(
            source, tmp_path / "out" / "clip.mp4", scale=2, on_progress=seen.append
        )
    )
    assert _numbers(output) == _numbers(source)
    assert len(counts) == 7
    assert seen and seen[-1] == pytest.approx(1.0)
    assert seen == sorted(seen)
    first = batch_size(96, 64)
    assert first == 8
    # Halved until it fits in the first block, then kept for every later one.
    assert fake.batches == [8, 4, 2] + [2] * 6
    assert fake.loads == 1


@needs_ffmpeg
def test_out_of_memory_one_frame_at_a_time_says_what_to_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fake_span(tmp_path, monkeypatch, max_batch=0)
    _small_blocks(monkeypatch, 7)
    source = _numbered(tmp_path / "in.mp4", 60)
    with pytest.raises(GpuOutOfMemory, match="Close other programs that use the GPU"):
        SpanEngine().upscale(UpscaleRequest(source, tmp_path / "out.mp4", scale=2))
    # The first block fails; the stream stops there, with nothing half written.
    assert fake.batches == [8, 4, 2, 1]
    assert not (tmp_path / "out.mp4").exists()
    assert not (tmp_path / "out.encoding.mp4").exists()
    fake.batches.clear()
    with pytest.raises(GpuOutOfMemory, match="even one frame at a time"):
        SpanEngine().upscale_many(
            [UpscaleRequest(source, tmp_path / "out" / "a.mp4", scale=2)]
        )
    assert fake.batches == [8, 4, 2, 1]


@needs_ffmpeg
def test_preview_batch_shrinks_its_batch_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fake_span(tmp_path, monkeypatch, max_batch=3)
    counts = _small_blocks(monkeypatch, 16)
    first = _numbered(tmp_path / "a.mp4", 48)
    second = _numbered(tmp_path / "b.mp4", 20)
    outputs = SpanEngine().upscale_many(
        [
            UpscaleRequest(first, tmp_path / "out" / "a.mp4", scale=2),
            UpscaleRequest(second, tmp_path / "out" / "b.mp4", scale=2),
        ]
    )
    assert [_numbers(o) for o in outputs] == [_numbers(first), _numbers(second)]
    assert counts == [16, 16, 16, 16, 4]
    # The later blocks, and the second clip, start at the batch the first found.
    assert fake.batches == [8, 4, 2, 2, 2, 2, 2]
    assert fake.loads == 1


def test_a_failure_lets_go_of_what_the_call_held() -> None:
    import gc
    import weakref

    class Tensor:
        """Stands in for the frames and pipeline a failed call has in its locals."""

    def infer() -> None:
        held = Tensor()
        alive.append(weakref.ref(held))
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.60 GiB")

    alive: list[weakref.ref[Tensor]] = []
    try:
        infer()
    except RuntimeError as exc:
        failure = _upscale_failure(exc)
    gc.collect()
    assert alive[0]() is None
    assert isinstance(failure, GpuOutOfMemory)
    assert failure.__cause__ is None and failure.__traceback__ is None
    other = _upscale_failure(ValueError("bad shape"))
    assert type(other) is UpscaleError
    assert str(other) == "SPAN failed: bad shape"


def test_blocks_hold_whole_frames_of_a_fixed_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 128 MiB of yuv420p input per hand-off, whatever the frame size.
    assert block_frames(1440, 1080) == 57
    assert block_frames(640, 480) == 291
    monkeypatch.setattr(upscale, "STREAM_BLOCK_BYTES", FRAME_BYTES * 3 + 1)
    assert block_frames(96, 64) == 3
    # A frame larger than a block still goes one frame at a time.
    assert block_frames(96 * 4, 64) == 1


def test_releasing_the_gpu_returns_freed_heap(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes
    import ctypes.util
    import sys

    import super_processor.upscale as up

    trims: list[int] = []

    class Libc:
        def malloc_trim(self, pad: int) -> int:
            trims.append(pad)
            return 1

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: "libc.so.6")
    monkeypatch.setattr(ctypes, "CDLL", lambda name: Libc())
    up._release_gpu(None)
    assert trims == [0]


def test_no_libc_trim_is_fine(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes
    import ctypes.util
    import sys

    import super_processor.upscale as up

    tried: list[str] = []

    def missing(name: str) -> object:
        tried.append(name)
        raise OSError("no libc here")

    # On Linux, so the trim is really attempted (and fails) on every runner.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: "libc.so.6")
    monkeypatch.setattr(ctypes, "CDLL", missing)
    up._release_gpu(None)
    assert tried == ["libc.so.6"]
    # Elsewhere there is nothing to trim, and nothing is loaded.
    monkeypatch.setattr(sys, "platform", "darwin")
    up._release_gpu(None)
    assert tried == ["libc.so.6"]
