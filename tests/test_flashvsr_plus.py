"""Tests for running FlashVSR+ in place of v1.1, against a fake checkout."""

from __future__ import annotations

import shutil
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from super_processor import upscale
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
    monkeypatch.setenv("SUPER_PROCESSOR_FLASHVSR", "AUTO")
    assert flashvsr_model() == "auto"
    monkeypatch.setenv("SUPER_PROCESSOR_FLASHVSR", "v2")
    with pytest.raises(UpscaleError, match="must be one of v1.1, plus"):
        flashvsr_model()


def _v11_tree(tmp_path: Path) -> Path:
    home = tmp_path / "FlashVSR"
    wan = home / "examples" / "WanVSR"
    for name in (
        "infer_flashvsr_v1.1_tiny.py",
        "FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors",
        "FlashVSR-v1.1/LQ_proj_in.ckpt",
        "FlashVSR-v1.1/TCDecoder.ckpt",
    ):
        (wan / name).parent.mkdir(parents=True, exist_ok=True)
        (wan / name).write_text("w", encoding="utf-8")
    return home


@pytest.mark.parametrize(
    ("installed", "scale", "runs"),
    [
        ({"v1.1", "plus"}, 2, "v1.1"),
        ({"v1.1", "plus"}, 4, "plus"),
        ({"plus"}, 2, "plus"),
        ({"v1.1"}, 4, "v1.1"),
    ],
)
def test_auto_picks_the_model_by_scale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed: set[str],
    scale: int,
    runs: str,
) -> None:
    monkeypatch.setenv("SUPER_PROCESSOR_FLASHVSR", "auto")
    monkeypatch.setattr("super_processor.upscale.cuda_available", lambda: True)
    v11 = _v11_tree(tmp_path) if "v1.1" in installed else tmp_path / "no-v11"
    plus = _plus_tree(tmp_path) if "plus" in installed else tmp_path / "no-plus"
    monkeypatch.setenv("FLASHVSR_HOME", str(v11))
    monkeypatch.setenv("FLASHVSR_PLUS_HOME", str(plus))
    backend = upscale._flashvsr_ready(scale)
    assert isinstance(backend, upscale._Plus if runs == "plus" else upscale._V11)


def test_auto_with_neither_says_auto_found_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUPER_PROCESSOR_FLASHVSR", "auto")
    monkeypatch.setattr("super_processor.upscale.cuda_available", lambda: True)
    monkeypatch.setenv("FLASHVSR_HOME", str(tmp_path / "no-v11"))
    monkeypatch.setenv("FLASHVSR_PLUS_HOME", str(tmp_path / "no-plus"))
    for scale in (2, 4):
        with pytest.raises(UpscaleError, match="=auto found no FlashVSR checkout"):
            upscale._flashvsr_ready(scale)


def test_auto_names_what_an_incomplete_checkout_lacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUPER_PROCESSOR_FLASHVSR", "auto")
    monkeypatch.setattr("super_processor.upscale.cuda_available", lambda: True)
    plus = _plus_tree(tmp_path)
    (plus / "ckpts" / "turbo" / "dit.safetensors").unlink()
    monkeypatch.setenv("FLASHVSR_HOME", str(_v11_tree(tmp_path)))
    monkeypatch.setenv("FLASHVSR_PLUS_HOME", str(plus))
    # Not passed over for the slower v1.1: the missing weight is named.
    with pytest.raises(UpscaleError, match=r"ckpts/turbo/dit\.safetensors"):
        upscale._flashvsr_ready(4)
    assert isinstance(upscale._flashvsr_ready(2), upscale._V11)


@dataclass(frozen=True)
class _PlainV11:
    """Stands in for v1.1: a lanczos resize, with each clip noted in ``log``."""

    log: Path

    def load(self) -> upscale._Loaded:
        with self.log.open("a") as handle:
            handle.write("v1.1 load\n")

        def run(
            _module: object, _pipe: object, source: Path, output: Path, *, scale: int
        ) -> None:
            with self.log.open("a") as handle:
                handle.write(f"v1.1 {source.name} {scale}\n")
            upscale.plain_upscale(UpscaleRequest(source, output, scale=scale))

        return upscale._Loaded(None, None, run, lambda: None)


@needs_ffmpeg
def test_auto_batches_each_model_once_and_keeps_the_order(
    plus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUPER_PROCESSOR_FLASHVSR", "auto")
    calls = tmp_path / "v11.log"
    # Auto looks for the checkout folder before calling _v11_ready, so the
    # folder must exist here rather than depend on the machine's models.
    (tmp_path / "FlashVSR").mkdir()
    monkeypatch.setenv("FLASHVSR_HOME", str(tmp_path / "FlashVSR"))
    monkeypatch.setattr(upscale, "_v11_ready", lambda: _PlainV11(calls))
    loads: list[str] = []
    real_load = upscale._Plus.load

    def counted_load(self: upscale._Plus) -> upscale._Loaded:
        loads.append("plus")
        return real_load(self)

    monkeypatch.setattr(upscale._Plus, "load", counted_load)
    requests = []
    for index, scale in enumerate((4, 2, 4, 2)):
        clip = tmp_path / f"clip_{index}.mp4"
        _testsrc(clip, seconds=1, size="128x72")
        requests.append(
            UpscaleRequest(clip, tmp_path / "out" / f"{index}.mp4", scale=scale)
        )
    outputs = FlashVsrEngine().upscale_many(requests)
    assert outputs == [r.output.resolve() for r in requests]
    assert [_size(o) for o in outputs] == ["512,288", "256,144", "512,288", "256,144"]
    assert loads == ["plus"]
    assert calls.read_text().split("\n")[:-1] == [
        "v1.1 load",
        "v1.1 clip_1.mp4 2",
        "v1.1 clip_3.mp4 2",
    ]


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
