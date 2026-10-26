"""Every output frame is the source frame at the same position.

The clips here number their frames: each frame is a flat grey whose level
encodes its index, so a part that starts a frame early, or a tail that
freezes, shows up as a wrong number instead of a slightly lower PSNR.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from super_processor.upscale import (
    FLASHVSR_MIN_CHUNK_FRAMES,
    FakeUpscaleEngine,
    FlashVsrEngine,
    GpuOutOfMemory,
    UpscaleError,
    UpscaleRequest,
    UpscaleSpan,
    _flashvsr_failure,
    chunk_frames,
    flashvsr_extra_frames,
    render_shots,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg required"
)
FPS = 24
STEP = 5  # grey levels per frame; numbers repeat every 40 frames


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


def test_extra_frames_make_flashvsr_return_every_frame() -> None:
    for frames in range(1, 300):
        extra = flashvsr_extra_frames(frames)
        assert 0 <= extra < 8
        padded = frames + extra
        # FlashVSR's loader: four copies on, largest 8n + 1, four off.
        returned = ((padded + 4 - 1) // 8) * 8 + 1 - 4
        assert returned == padded >= frames


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


# Like FlashVSR's loader: return only the largest 8n + 1 of (frames + 4),
# less four, at twice the size.
_EIGHT_N_INFER = """
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
    import os

    count = int(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
        check=True, capture_output=True, text=True,
    ).stdout.strip())
    limit = int(os.environ.get("FAKE_MAX_FRAMES", "100000"))
    if count > limit:
        # What PyTorch raises when the working memory does not fit.
        raise RuntimeError(f"CUDA out of memory. Tried to allocate 2 GiB ({count})")
    _SOURCE.update(path=path, scale=int(scale), keep=((count + 3) // 8) * 8 + 1 - 4)
    return ("lq", 8, 8, 1, 24)


def tensor2video(video):
    return [video]


def save_video(pictures, save_path, fps, quality):
    scale = _SOURCE["scale"]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", _SOURCE["path"], "-vf", f"scale=iw*{scale}:ih*{scale}",
         "-frames:v", str(_SOURCE["keep"]),
         "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", save_path],
        check=True,
    )
"""


@needs_ffmpeg
@pytest.mark.parametrize("frames", [48, 94, 95])
def test_flashvsr_tail_is_real_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frames: int
) -> None:
    from test_upscale import _flashvsr_tree

    home, _script = _flashvsr_tree(tmp_path, _EIGHT_N_INFER)
    monkeypatch.setattr("super_processor.upscale.cuda_available", lambda: True)
    monkeypatch.setattr("super_processor.upscale.flashvsr_home", lambda: home)
    source = _numbered(tmp_path / "in.mp4", frames)
    output = FlashVsrEngine().upscale(
        UpscaleRequest(source, tmp_path / "out" / "clip.mp4", scale=2)
    )
    # Without the extra frames, 48 came back as 45 and the last three froze.
    assert _numbers(output) == _numbers(source)


def _fake_flashvsr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from test_upscale import _flashvsr_tree

    home, _script = _flashvsr_tree(tmp_path, _EIGHT_N_INFER)
    monkeypatch.setattr("super_processor.upscale.cuda_available", lambda: True)
    monkeypatch.setattr("super_processor.upscale.flashvsr_home", lambda: home)


@needs_ffmpeg
def test_out_of_memory_retries_in_shorter_pieces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_flashvsr(tmp_path, monkeypatch)
    # Like the trailer next to a game: 24 frames do not fit, 20 do.
    monkeypatch.setenv("FAKE_MAX_FRAMES", "23")
    source = _numbered(tmp_path / "in.mp4", 130)
    seen: list[float] = []
    output = FlashVsrEngine().upscale(
        UpscaleRequest(
            source, tmp_path / "out" / "clip.mp4", scale=2, on_progress=seen.append
        )
    )
    assert _numbers(output) == _numbers(source)
    assert seen and seen[-1] == pytest.approx(1.0)


@needs_ffmpeg
def test_out_of_memory_in_the_shortest_piece_says_what_to_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_flashvsr(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_MAX_FRAMES", "5")
    source = _numbered(tmp_path / "in.mp4", 60)
    with pytest.raises(GpuOutOfMemory, match="Close other programs that use the GPU"):
        FlashVsrEngine().upscale(UpscaleRequest(source, tmp_path / "out.mp4", scale=2))


@needs_ffmpeg
def test_preview_batch_falls_back_to_pieces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_flashvsr(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_MAX_FRAMES", "30")
    first = _numbered(tmp_path / "a.mp4", 48)
    second = _numbered(tmp_path / "b.mp4", 20)
    outputs = FlashVsrEngine().upscale_many(
        [
            UpscaleRequest(first, tmp_path / "out" / "a.mp4", scale=2),
            UpscaleRequest(second, tmp_path / "out" / "b.mp4", scale=2),
        ]
    )
    assert [_numbers(o) for o in outputs] == [_numbers(first), _numbers(second)]


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
        failure = _flashvsr_failure(exc)
    gc.collect()
    assert alive[0]() is None
    assert isinstance(failure, GpuOutOfMemory)
    assert failure.__cause__ is None and failure.__traceback__ is None
    other = _flashvsr_failure(ValueError("bad shape"))
    assert type(other) is UpscaleError
    assert str(other) == "FlashVSR failed: bad shape"


def test_chunks_are_whole_frames_that_meet() -> None:
    assert chunk_frames(100, 200, 12) == [(0, 100)]
    # 8 s at 25 fps with 0.5 s overlap: no half frames anywhere.
    ranges = chunk_frames(500, 200, 12)
    # The last piece is a full 200, reaching back past the usual overlap.
    assert ranges == [(0, 200), (188, 388), (300, 500)]
    # A remainder shorter than FlashVSR accepts never happens.
    short = chunk_frames(130, 17, 4)
    assert all(stop - first == 17 for first, stop in short)
    assert short[0][0] == 0 and short[-1][1] == 130
    for before, after in zip(short, short[1:], strict=False):
        assert after[0] < before[1]
    assert FLASHVSR_MIN_CHUNK_FRAMES >= 17
    with pytest.raises(UpscaleError):
        chunk_frames(0, 200, 12)


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

    import super_processor.upscale as up

    def missing(name: str) -> object:
        raise OSError("no libc here")

    monkeypatch.setattr(ctypes, "CDLL", missing)
    up._release_gpu(None)
