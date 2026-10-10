"""Guardrails and fakes for the capped restore → RTX VSR pipeline."""

from __future__ import annotations

import builtins
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from super_processor.upscale import (
    HostChwFrame,
    NvencEncoder,
    NvvfxVsr,
    SeedVR2Restorer,
    UpscaleError,
    UpscaleParams,
    _read_exact,
    build_nvenc_command,
    clone_dlpack_output,
    decode_rgb_chunks,
    enforce_note_direction,
    host_chw_to_cuda,
    note_blocks_detail_increase,
    note_requests_sharpness,
    open_video_super_res,
    output_size,
    resolve_seedvr2_weights,
    rgb24_to_chw,
    run_two_pass,
)


def test_defaults_and_strength_cap() -> None:
    params = UpscaleParams()
    assert params.restore_strength == 0.15
    assert params.scale == 2
    assert params.vsr_quality == "MEDIUM"
    with pytest.raises(UpscaleError, match="0.35"):
        UpscaleParams(restore_strength=0.36)
    with pytest.raises(UpscaleError, match="0.35"):
        UpscaleParams.from_dict(
            {"restore_strength": 0.5, "scale": 2, "vsr_quality": "HIGH"}
        )
    assert UpscaleParams(restore_strength=0).restore_strength == 0.0
    assert UpscaleParams(restore_strength=0.35).restore_strength == 0.35


def test_ultra_scale_and_forbidden_fields_rejected() -> None:
    with pytest.raises(UpscaleError, match="ULTRA"):
        UpscaleParams(vsr_quality="ULTRA")
    with pytest.raises(UpscaleError, match="scale"):
        UpscaleParams(scale=1)
    with pytest.raises(UpscaleError, match="prompt"):
        UpscaleParams.from_dict(
            {
                "restore_strength": 0.15,
                "scale": 2,
                "vsr_quality": "MEDIUM",
                "prompt": "add a hat",
            }
        )
    with pytest.raises(UpscaleError, match="model"):
        UpscaleParams.from_dict(
            {
                "restore_strength": 0.15,
                "scale": 2,
                "vsr_quality": "LOW",
                "model_name": "SeedVR2-7B",
            }
        )
    with pytest.raises(UpscaleError, match="FFmpeg"):
        UpscaleParams.from_dict(
            {
                "restore_strength": 0.15,
                "scale": 2,
                "vsr_quality": "LOW",
                "argv": ["ffmpeg", "-vf", "unsharp"],
            }
        )
    with pytest.raises(UpscaleError, match="unknown"):
        UpscaleParams.from_dict(
            {
                "restore_strength": 0.15,
                "scale": 2,
                "vsr_quality": "LOW",
                "seed": 1,
            }
        )


def test_artificial_note_cannot_increase_knobs() -> None:
    prior = UpscaleParams(restore_strength=0.15, scale=2, vsr_quality="MEDIUM")
    for note in (
        "too artificial",
        "looks plastic",
        "over-sharpened",
        "not natural",
    ):
        assert note_blocks_detail_increase(note)
        with pytest.raises(UpscaleError, match="cannot increase"):
            enforce_note_direction(
                prior,
                UpscaleParams(restore_strength=0.2, scale=2, vsr_quality="MEDIUM"),
                note,
            )
        with pytest.raises(UpscaleError, match="VSR quality"):
            enforce_note_direction(
                prior,
                UpscaleParams(restore_strength=0.15, scale=2, vsr_quality="HIGH"),
                note,
            )
    held = enforce_note_direction(prior, prior, "over sharpened and not natural")
    assert held == prior
    lowered = enforce_note_direction(
        prior,
        UpscaleParams(restore_strength=0.05, scale=4, vsr_quality="LOW"),
        "plastic",
    )
    assert lowered.restore_strength == 0.05
    assert lowered.vsr_quality == "LOW"
    assert lowered.scale == 4


def test_soft_note_can_raise_only_inside_the_cap() -> None:
    prior = UpscaleParams()
    assert note_requests_sharpness("too soft, more sharpness")
    raised = enforce_note_direction(
        prior,
        UpscaleParams(restore_strength=0.35, scale=4, vsr_quality="HIGH"),
        "too soft",
    )
    assert raised.restore_strength == 0.35
    assert raised.vsr_quality == "HIGH"
    with pytest.raises(UpscaleError, match="cap"):
        UpscaleParams.from_dict(
            {"restore_strength": 0.9, "scale": 2, "vsr_quality": "HIGH"}
        )


def test_output_size_is_multiple_of_8() -> None:
    width, height = output_size(160, 90, 2)
    assert width % 8 == 0
    assert height % 8 == 0
    assert width == 320
    assert height == 176


def test_rgb_chw_is_unit_interval() -> None:
    frame = bytes([0, 128, 255])
    chw = rgb24_to_chw(frame, 1, 1)
    import struct

    red, green, blue = struct.unpack("<fff", chw.data)
    assert red == 0.0
    assert green == pytest.approx(128 / 255)
    assert blue == 1.0


def test_clone_dlpack_copies_bytes_and_tensor() -> None:
    class _Once:
        def __init__(self) -> None:
            self.calls = 0

        def __dlpack__(self) -> bytes:
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("buffer reused")
            return b"rgb"

    assert clone_dlpack_output(_Once()) == b"rgb"

    class _Array:
        def tobytes(self) -> bytes:
            return b"cloned"

    class _Tensor:
        def clone(self) -> _Tensor:
            return self

        def cpu(self) -> _Tensor:
            return self

        def numpy(self) -> _Array:
            return _Array()

    assert clone_dlpack_output(_Tensor()) == b"cloned"
    with pytest.raises(UpscaleError, match="DLPack"):
        clone_dlpack_output(object())


def test_missing_seedvr2_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SEEDVR2_3B_WEIGHTS", raising=False)
    with pytest.raises(UpscaleError, match="SeedVR2-3B"):
        SeedVR2Restorer(tmp_path / "missing.safetensors")
    weights = tmp_path / "seedvr2-3b.safetensors"
    weights.write_bytes(b"not-a-real-weight")
    restorer = SeedVR2Restorer(weights)
    with pytest.raises(UpscaleError, match="restore runtime"):
        restorer.restore([b"\x00" * 12], width=2, height=2, strength=0.15)


def test_missing_nvvfx_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import super_processor.upscale as upscale

    real_import = upscale.importlib.import_module

    def _blocked(name: str, package: str | None = None) -> Any:
        if name == "nvvfx" or name.startswith("nvvfx."):
            raise ImportError("no nvvfx")
        return real_import(name, package)

    monkeypatch.setattr(upscale.importlib, "import_module", _blocked)
    with pytest.raises(UpscaleError, match="nvvfx"):
        open_video_super_res(scale=2, quality="MEDIUM")


def test_nvvfx_without_video_super_res(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "nvvfx", types_module())
    with pytest.raises(UpscaleError, match="VideoSuperRes"):
        open_video_super_res(scale=2, quality="LOW")


def types_module() -> Any:
    import types

    return types.ModuleType("nvvfx")


def test_host_chw_to_cuda_and_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = rgb24_to_chw(bytes([255, 0, 0, 0, 255, 0]), 2, 1)

    class _Tensor:
        def reshape(self, *_shape: int) -> _Tensor:
            return self

        def contiguous(self) -> _Tensor:
            return self

        def cuda(self) -> str:
            return "cuda-chw"

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class _Torch:
        float32 = "float32"
        cuda = _Cuda()

        @staticmethod
        def frombuffer(data: bytearray, dtype: str) -> _Tensor:
            assert dtype == "float32"
            assert len(data) == len(frame.data)
            return _Tensor()

    assert host_chw_to_cuda(frame, torch_module=_Torch()) == "cuda-chw"

    class _NoGpu(_Torch):
        class cuda:  # noqa: N801
            @staticmethod
            def is_available() -> bool:
                return False

    with pytest.raises(UpscaleError, match="RTX GPU"):
        host_chw_to_cuda(frame, torch_module=_NoGpu())

    real_import = builtins.__import__

    def _blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(UpscaleError, match="PyTorch"):
        host_chw_to_cuda(frame)


class _FakeRestore:
    def __init__(self) -> None:
        self.calls = 0

    def restore(
        self,
        frames: list[bytes],
        *,
        width: int,
        height: int,
        strength: float,
    ) -> list[bytes]:
        self.calls += 1
        assert strength <= 0.35
        assert width == 16 and height == 16
        return [bytes([(pixel + 1) % 256 for pixel in frame]) for frame in frames]


class _FakeVsr:
    def __init__(self) -> None:
        self.seen: list[HostChwFrame] = []

    def run(
        self,
        frame: HostChwFrame,
        *,
        scale: int,
        quality: str,
        out_width: int,
        out_height: int,
    ) -> Any:
        assert quality in {"LOW", "MEDIUM", "HIGH"}
        assert scale in {2, 3, 4}
        self.seen.append(frame)
        payload = bytes([9]) * (out_width * out_height * 3)

        class _Buf:
            def __dlpack__(self) -> bytes:
                return payload

        return _Buf()


class _CaptureEncoder:
    def __init__(self) -> None:
        self.frames: list[bytes] = []
        self.closed = False

    def write_frame(self, frame: bytes) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.closed = True


def _tiny_clip(path: Path) -> None:
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
            "color=c=gray:duration=0.2:size=16x16:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_two_pass_fakes_skip_real_backends(tmp_path: Path) -> None:
    source = tmp_path / "in.mp4"
    _tiny_clip(source)
    output = tmp_path / "out.mp4"
    restorer = _FakeRestore()
    vsr = _FakeVsr()
    encoder = _CaptureEncoder()
    run_two_pass(
        source,
        output,
        UpscaleParams(restore_strength=0.15, scale=2, vsr_quality="MEDIUM"),
        width=16,
        height=16,
        fps=10,
        restorer=restorer,
        vsr=vsr,
        chunk_frames=1,
        encoder=encoder,
    )
    assert restorer.calls >= 1
    assert vsr.seen
    assert encoder.closed
    assert encoder.frames
    assert len(encoder.frames[0]) == 32 * 32 * 3
    assert all(len(chw.data) == 3 * 16 * 16 * 4 for chw in vsr.seen)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_zero_strength_skips_seedvr2(tmp_path: Path) -> None:
    source = tmp_path / "in.mp4"
    _tiny_clip(source)
    restorer = _FakeRestore()
    encoder = _CaptureEncoder()
    run_two_pass(
        source,
        tmp_path / "out.mp4",
        UpscaleParams(restore_strength=0, scale=2, vsr_quality="LOW"),
        width=16,
        height=16,
        fps=10,
        restorer=restorer,
        vsr=_FakeVsr(),
        encoder=encoder,
    )
    assert restorer.calls == 0
    assert encoder.frames


def test_nvenc_command_uses_hevc_nvenc(tmp_path: Path) -> None:
    command = build_nvenc_command(
        ffmpeg_bin="ffmpeg",
        width=32,
        height=16,
        fps=10,
        output=tmp_path / "out.mp4",
    )
    assert "hevc_nvenc" in command
    assert "unsharp" not in command


def test_nvenc_encoder_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Stdin:
        def write(self, data: bytes) -> None:
            del data

        def close(self) -> None:
            return None

    class _Proc:
        def __init__(self) -> None:
            self.stdin = _Stdin()
            self.stderr = None
            self.returncode = 1

        def wait(self) -> int:
            return 1

    monkeypatch.setattr(
        "super_processor.upscale.subprocess.Popen",
        lambda *args, **kwargs: _Proc(),
    )
    encoder = NvencEncoder(
        tmp_path / "out.mp4",
        ffmpeg_bin="ffmpeg",
        width=8,
        height=8,
        fps=10,
    )
    encoder.write_frame(bytes(8 * 8 * 3))
    with pytest.raises(UpscaleError, match="hevc_nvenc"):
        encoder.close()


def test_read_exact_and_restore_size_errors() -> None:
    class _Stream:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def read(self, size: int) -> bytes:
            piece = self._data[:size]
            self._data = self._data[size:]
            return piece

    assert _read_exact(_Stream(b"abcd"), 4) == b"abcd"
    assert _read_exact(_Stream(b""), 4) is None
    with pytest.raises(UpscaleError, match="mid-frame"):
        _read_exact(_Stream(b"ab"), 4)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_restore_must_keep_source_size(tmp_path: Path) -> None:
    source = tmp_path / "in.mp4"
    _tiny_clip(source)

    class _Bad:
        def restore(
            self,
            frames: list[bytes],
            *,
            width: int,
            height: int,
            strength: float,
        ) -> list[bytes]:
            del width, height, strength
            return [b"nope" for _ in frames]

    with pytest.raises(UpscaleError, match="source frame size"):
        run_two_pass(
            source,
            tmp_path / "out.mp4",
            UpscaleParams(restore_strength=0.1),
            width=16,
            height=16,
            fps=10,
            restorer=_Bad(),
            vsr=_FakeVsr(),
            encoder=_CaptureEncoder(),
        )


def test_param_edges_and_pipeline_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert UpscaleParams.from_dict(UpscaleParams()) == UpscaleParams()
    with pytest.raises(UpscaleError, match="JSON"):
        UpscaleParams.from_dict(["nope"])
    with pytest.raises(UpscaleError, match="missing"):
        UpscaleParams.from_dict({"scale": 2})
    with pytest.raises(UpscaleError, match="scale"):
        UpscaleParams(scale=True)
    with pytest.raises(UpscaleError, match="vsr_quality"):
        UpscaleParams(vsr_quality="nope")
    with pytest.raises(UpscaleError, match="number"):
        UpscaleParams(restore_strength=True)
    held = enforce_note_direction(
        UpscaleParams(),
        UpscaleParams(scale=3),
        "a bit warmer",
    )
    assert held.scale == 3
    with pytest.raises(UpscaleError, match="chunk_frames"):
        list(
            decode_rgb_chunks(
                tmp_path / "x.mp4",
                ffmpeg_bin="ffmpeg",
                width=8,
                height=8,
                chunk_frames=0,
            )
        )
    with pytest.raises(UpscaleError, match="frame rate"):
        run_two_pass(
            tmp_path / "a.mp4",
            tmp_path / "b.mp4",
            UpscaleParams(restore_strength=0),
            width=16,
            height=16,
            fps=0,
            vsr=_FakeVsr(),
            encoder=_CaptureEncoder(),
        )

    def _empty(*_args: object, **_kwargs: object) -> Any:
        return iter(())

    monkeypatch.setattr("super_processor.upscale.decode_rgb_chunks", _empty)
    with pytest.raises(UpscaleError, match="no video"):
        run_two_pass(
            tmp_path / "a.mp4",
            tmp_path / "b.mp4",
            UpscaleParams(restore_strength=0),
            width=16,
            height=16,
            fps=10,
            vsr=_FakeVsr(),
            encoder=_CaptureEncoder(),
        )

    frame = b"\x00" * (16 * 16 * 3)

    def _one(*_args: object, **_kwargs: object) -> Any:
        yield [frame]

    monkeypatch.setattr("super_processor.upscale.decode_rgb_chunks", _one)

    class _Short:
        def restore(
            self,
            frames: list[bytes],
            *,
            width: int,
            height: int,
            strength: float,
        ) -> list[bytes]:
            del frames, width, height, strength
            return []

    with pytest.raises(UpscaleError, match="frame count"):
        run_two_pass(
            tmp_path / "a.mp4",
            tmp_path / "b.mp4",
            UpscaleParams(restore_strength=0.1),
            width=16,
            height=16,
            fps=10,
            restorer=_Short(),
            vsr=_FakeVsr(),
            encoder=_CaptureEncoder(),
        )

    class _TinyVsr:
        def run(self, frame: HostChwFrame, **_kwargs: object) -> Any:
            del frame

            class _Buf:
                def __dlpack__(self) -> bytes:
                    return b"x"

            return _Buf()

    with pytest.raises(UpscaleError, match="aligned output"):
        run_two_pass(
            tmp_path / "a.mp4",
            tmp_path / "b.mp4",
            UpscaleParams(restore_strength=0),
            width=16,
            height=16,
            fps=10,
            vsr=_TinyVsr(),
            encoder=_CaptureEncoder(),
        )


def test_nvvfx_adapter_clones_and_reports_startup_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    class _Effect:
        def run(self, frame: object) -> Any:
            del frame

            class _Out:
                def __dlpack__(self) -> bytes:
                    return b"ab"

            return _Out()

    module = types.ModuleType("nvvfx")

    def _factory(*, scale: int, quality: str) -> _Effect:
        assert scale == 2
        assert quality == "MEDIUM"
        return _Effect()

    module.VideoSuperRes = _factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nvvfx", module)
    monkeypatch.setattr(
        "super_processor.upscale.host_chw_to_cuda",
        lambda frame: frame,
    )
    backend = NvvfxVsr(scale=2, quality="MEDIUM")
    cloned = backend.run(
        rgb24_to_chw(b"\x00\x00\x00", 1, 1),
        scale=2,
        quality="MEDIUM",
        out_width=8,
        out_height=8,
    )
    assert clone_dlpack_output(cloned) == b"ab"

    def _explode(*, scale: int, quality: str) -> Any:
        del scale, quality
        raise RuntimeError("no device")

    module.VideoSuperRes = _explode  # type: ignore[attr-defined]
    with pytest.raises(UpscaleError, match="Could not start"):
        open_video_super_res(scale=2, quality="HIGH")


def test_weights_env_and_encoder_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"x")
    monkeypatch.setenv("SEEDVR2_3B_WEIGHTS", str(weights))
    assert resolve_seedvr2_weights() == weights
    monkeypatch.setenv("SEEDVR2_3B_WEIGHTS", "")
    with pytest.raises(UpscaleError, match="SEEDVR2_3B_WEIGHTS"):
        resolve_seedvr2_weights()

    class _Stdin:
        def write(self, data: bytes) -> None:
            del data
            raise BrokenPipeError

        def close(self) -> None:
            return None

    class _Proc:
        def __init__(self) -> None:
            self.stdin: _Stdin | None = _Stdin()
            self.stderr: Any = io_bytes(b"nvenc down")

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        "super_processor.upscale.subprocess.Popen",
        lambda *_args, **_kwargs: _Proc(),
    )
    encoder = NvencEncoder(
        tmp_path / "out.mp4",
        ffmpeg_bin="ffmpeg",
        width=8,
        height=8,
        fps=10,
    )
    with pytest.raises(UpscaleError, match="encode frame size"):
        encoder.write_frame(b"short")
    with pytest.raises(UpscaleError, match="hevc_nvenc"):
        encoder.write_frame(bytes(8 * 8 * 3))
    assert encoder._proc.stdin is not None
    encoder._proc.stdin = None
    with pytest.raises(UpscaleError, match="not accepting"):
        encoder.write_frame(bytes(8 * 8 * 3))


def io_bytes(payload: bytes) -> Any:
    import io

    return io.BytesIO(payload)


def test_decode_chunks_reads_frames(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg required")
    source = tmp_path / "in.mp4"
    _tiny_clip(source)
    chunks = list(
        decode_rgb_chunks(
            source,
            ffmpeg_bin="ffmpeg",
            width=16,
            height=16,
            chunk_frames=8,
        )
    )
    assert chunks
    assert len(chunks[0][0]) == 16 * 16 * 3
