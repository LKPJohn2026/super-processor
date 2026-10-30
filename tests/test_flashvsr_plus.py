"""Tests for running FlashVSR+ in place of v1.1, against a fake checkout."""

from __future__ import annotations

import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from super_processor.upscale import (
    FLASHVSR_BLOCK,
    FlashVsrEngine,
    GpuOutOfMemory,
    UpscaleError,
    UpscaleRequest,
    flashvsr_model,
    resolve_flashvsr_plus,
    video_stream_facts,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg required"
)

# Upscales the real input with ffmpeg to the requested size and hands the
# frames back shaped like FlashVSR+'s result: video[0] is [C, F, H, W], and
# video[0][:, i] is one frame for inference.output.tensor_to_uint8_frame.
_FAKE_INFERENCE = """
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

CALLS = []


@dataclass
class InferenceRequest:
    input_path: str | None = None
    height: int | None = None
    width: int | None = None


class _Frame:
    def __init__(self, data):
        self._data = data

    def tobytes(self):
        return self._data


class _Clip:
    def __init__(self, frames, height, width):
        self._frames = frames
        self.shape = (3, len(frames), height, width)

    def __getitem__(self, key):
        return _Frame(self._frames[key[1]])


class _Result:
    def __init__(self, video):
        self.video = video


class _Pipe:
    def __init__(self, config):
        self.config = config

    def infer(self, request):
        if os.environ.get("FAKE_PLUS_OOM"):
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.19 GiB")
        CALLS.append((request.height, request.width))
        log = os.environ.get("FAKE_PLUS_LOG")
        if log:
            with open(log, "a") as handle:
                handle.write(json.dumps([request.height, request.width,
                                         self.config]) + "\\n")
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", request.input_path, "-vf",
             f"scale={request.width}:{request.height}", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-"],
            check=True, capture_output=True,
        ).stdout
        size = request.width * request.height * 3
        frames = [raw[i:i + size] for i in range(0, len(raw), size)]
        return _Result([_Clip(frames, request.height, request.width)])


def stream_config_for_mode(mode, seed, color_correction):
    return {"mode": mode, "seed": seed, "color_correction": color_correction}


def build_inference_pipeline(config, paths):
    return _Pipe(config)
"""


def _plus_tree(tmp_path: Path, inference: str = _FAKE_INFERENCE) -> Path:
    home = tmp_path / "FlashVSR_Plus"
    files = {
        "flashvsr_plus/__init__.py": (
            "from inference import (InferenceRequest, build_inference_pipeline,"
            " stream_config_for_mode)\n"
        ),
        "inference/__init__.py": inference,
        "inference/presets.py": (
            "def model_paths_for_mode(root, mode):\n    return root\n"
        ),
        "inference/output.py": "def tensor_to_uint8_frame(frame):\n    return frame\n",
        "models/__init__.py": "",
        "model_io/__init__.py": "",
        "prompt_tensor/posi_prompt.pth": "p",
        "ckpts/TCDecoder.ckpt": "w",
        "ckpts/turbo/dit.safetensors": "w",
        "ckpts/turbo/lq_proj_in.safetensors": "w",
    }
    for name, text in files.items():
        path = home / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return home


@pytest.fixture
def plus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = _plus_tree(tmp_path)
    monkeypatch.setenv("SUPER_PROCESSOR_FLASHVSR", "plus")
    monkeypatch.setenv("FLASHVSR_PLUS_HOME", str(home))
    monkeypatch.setattr("super_processor.upscale.cuda_available", lambda: True)
    # The loader imports torch first for FlashVSR+'s attention extension.
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    return home


def _testsrc(path: Path, seconds: float, size: str = "160x90") -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={size}:rate=10:duration={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _size(video: Path) -> str:
    return subprocess.run(
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
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _ssim(output: Path, source: Path, width: int, height: int) -> float:
    log = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(output),
            "-i",
            str(source),
            "-lavfi",
            f"[0:v]scale={width}:{height}:flags=area[a];[a][1:v]ssim",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stderr
    return float(log.rsplit("All:", 1)[1].split()[0])


def test_the_setting_picks_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPER_PROCESSOR_FLASHVSR", raising=False)
    assert flashvsr_model() == "v1.1"
    monkeypatch.setenv("SUPER_PROCESSOR_FLASHVSR", " Plus ")
    assert flashvsr_model() == "plus"
    monkeypatch.setenv("SUPER_PROCESSOR_FLASHVSR", "v2")
    with pytest.raises(UpscaleError, match="must be one of v1.1, plus"):
        flashvsr_model()


def test_missing_plus_files_are_named(tmp_path: Path) -> None:
    home = _plus_tree(tmp_path)
    (home / "ckpts" / "turbo" / "dit.safetensors").unlink()
    with pytest.raises(UpscaleError, match=r"ckpts/turbo/dit\.safetensors"):
        resolve_flashvsr_plus(home)
    assert resolve_flashvsr_plus(_plus_tree(tmp_path / "whole")) is not None


def test_a_bad_setting_fails_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUPER_PROCESSOR_FLASHVSR", "plus")
    monkeypatch.setenv("FLASHVSR_PLUS_HOME", str(tmp_path / "missing"))
    monkeypatch.setattr("super_processor.upscale.cuda_available", lambda: True)
    monkeypatch.setattr(
        "super_processor.upscale.require_supported_timing", lambda _p: None
    )
    with pytest.raises(UpscaleError, match="FlashVSR\\+ files were not found"):
        FlashVsrEngine().upscale(UpscaleRequest(tmp_path / "a.mp4", tmp_path / "b.mp4"))


@needs_ffmpeg
def test_plus_keeps_every_pixel_and_frame(
    plus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_PLUS_LOG", str(log))
    source = tmp_path / "in.mp4"
    _testsrc(source, seconds=2)
    output = tmp_path / "out" / "clip.mp4"
    FlashVsrEngine().upscale(UpscaleRequest(source, output, scale=2))
    assert _size(output) == "320,180"
    assert video_stream_facts(output) == (20, "10/1")
    # A crop in the wrong place would shift the picture against its source.
    assert _ssim(output, source, 160, 90) > 0.9
    import json

    height, width, config = json.loads(log.read_text().splitlines()[0])
    assert height % FLASHVSR_BLOCK == 0 and width % FLASHVSR_BLOCK == 0
    assert config == {"mode": "turbo", "seed": 0, "color_correction": True}


@needs_ffmpeg
def test_plus_leaves_no_modules_or_path_behind(plus: Path, tmp_path: Path) -> None:
    source = tmp_path / "in.mp4"
    _testsrc(source, seconds=1)
    for _round in range(2):  # a second load imports afresh
        FlashVsrEngine().upscale(
            UpscaleRequest(source, tmp_path / "out" / "clip.mp4", scale=2)
        )
        assert str(plus) not in sys.path
        assert not {"flashvsr_plus", "inference", "models", "model_io"} & set(
            sys.modules
        )


@needs_ffmpeg
def test_plus_refuses_a_clashing_module(
    plus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = types.ModuleType("models")
    elsewhere.__file__ = str(tmp_path / "other" / "models" / "__init__.py")
    monkeypatch.setitem(sys.modules, "models", elsewhere)
    source = tmp_path / "in.mp4"
    _testsrc(source, seconds=1)
    with pytest.raises(UpscaleError, match="a different 'models' module"):
        FlashVsrEngine().upscale(
            UpscaleRequest(source, tmp_path / "out" / "clip.mp4", scale=2)
        )
    assert sys.modules["models"] is elsewhere


@needs_ffmpeg
def test_plus_chunks_long_clips_and_batches_previews(
    plus: Path, tmp_path: Path
) -> None:
    long = tmp_path / "long.mp4"
    _testsrc(long, seconds=12)
    output = tmp_path / "out" / "long.mp4"
    FlashVsrEngine().upscale(UpscaleRequest(long, output, scale=2, strength=0.5))
    assert video_stream_facts(output)[0] == 120
    shorts = []
    for index in range(2):
        clip = tmp_path / f"short_{index}.mp4"
        _testsrc(clip, seconds=2, size="128x72")
        shorts.append(
            UpscaleRequest(clip, tmp_path / "previews" / f"{index}.mp4", scale=4)
        )
    for restored in FlashVsrEngine().upscale_many(shorts):
        assert _size(restored) == "512,288"
        assert video_stream_facts(restored) == (20, "10/1")


@needs_ffmpeg
def test_plus_out_of_memory_takes_the_usual_fallback(
    plus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_PLUS_OOM", "1")
    asked: list[float] = []

    def _child(request: UpscaleRequest, *, chunk_s: float) -> Path:
        asked.append(chunk_s)
        raise GpuOutOfMemory("short of GPU memory and system RAM")

    monkeypatch.setattr(
        "super_processor.managed_memory.upscale_in_managed_memory", _child
    )
    source = tmp_path / "in.mp4"
    _testsrc(source, seconds=4)
    with pytest.raises(GpuOutOfMemory, match="system RAM"):
        FlashVsrEngine().upscale(
            UpscaleRequest(source, tmp_path / "out" / "clip.mp4", scale=2)
        )
    # Halved down to the shortest piece, then handed to the system-RAM child.
    assert asked == [pytest.approx(1.7)]


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError(
            "[enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: "
            "can't allocate memory: you tried to allocate 19110297600 bytes. "
            "Error code 12 (Cannot allocate memory)"
        ),
        type("_ArrayMemoryError", (MemoryError,), {})("Unable to allocate 19 GiB"),
    ],
)
def test_host_memory_refusals_count_as_out_of_memory(error: Exception) -> None:
    from super_processor.upscale import _flashvsr_failure

    assert isinstance(_flashvsr_failure(error), GpuOutOfMemory)
