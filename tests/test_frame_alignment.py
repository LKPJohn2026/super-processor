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
    FakeUpscaleEngine,
    FlashVsrEngine,
    UpscaleRequest,
    UpscaleSpan,
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
    count = int(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
        check=True, capture_output=True, text=True,
    ).stdout.strip())
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
