"""The H.264 delivery encoder (NVENC or libx264) and SPAN's decoder (NVDEC
or FFmpeg's own).

The conftest fixes libx264 and software decoding for every test; these put
the real choice back where they need it. The tests that use NVENC or NVDEC
themselves skip unless FFmpeg can open them here, so CI runs the rest.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest
from test_upscale import (
    _frame_hashes,
    _frame_size_of,
    _moving,
    _swap_decoder,
    _tiny,
    fake_span,
)

from super_processor import upscale
from super_processor.look import ShotLook
from super_processor.upscale import (
    NVENC,
    X264,
    FakeUpscaleEngine,
    SpanEngine,
    UpscaleRequest,
    UpscaleSpan,
    apply_range_revise,
    default_span,
    delivery_encoder,
    encode_delivery,
    is_delivery,
    video_stream_facts,
)
from super_processor.upscale import cuda_decode_works as real_cuda_decode_works
from super_processor.upscale import nvenc_works as real_nvenc_works

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg required"
)


def _has_nvenc() -> bool:
    return shutil.which("ffmpeg") is not None and upscale._nvenc_probe("ffmpeg")


def _has_nvdec() -> bool:
    return shutil.which("ffmpeg") is not None and upscale._cuda_decode_probe("ffmpeg")


def test_the_encoder_follows_the_probe_and_the_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(upscale, "nvenc_works", lambda ffmpeg_bin="ffmpeg": False)
    assert delivery_encoder() == X264
    assert delivery_encoder("ffmpeg", (2880, 2160)) == X264
    asked: list[str] = []

    def works(ffmpeg_bin: str = "ffmpeg") -> bool:
        asked.append(ffmpeg_bin)
        return True

    monkeypatch.setattr(upscale, "nvenc_works", works)
    assert delivery_encoder("/opt/ffmpeg", (2880, 2160)) == NVENC
    assert asked == ["/opt/ffmpeg"]
    assert delivery_encoder() == NVENC
    # NVENC refuses frames this small or this large: those stay on x264.
    # Each is one pixel past a limit in one direction only.
    for size in ((144, 360), (640, 48), (4097, 2160), (3840, 4097), (128, 96)):
        assert delivery_encoder("ffmpeg", size) == X264
    for size in ((145, 49), (4096, 4096), (145, 4096), (4096, 49)):
        assert delivery_encoder("ffmpeg", size) == NVENC


def _fake_run(
    monkeypatch: pytest.MonkeyPatch, outcome: int | BaseException
) -> list[list[str]]:
    ran: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        ran.append(command)
        if isinstance(outcome, BaseException):
            raise outcome
        return subprocess.CompletedProcess(command, outcome, b"", b"")

    monkeypatch.setattr("super_processor.upscale.subprocess.run", run)
    return ran


@pytest.mark.parametrize(
    ("probe", "works", "flag"),
    [
        (upscale._nvenc_probe, real_nvenc_works, NVENC),
        (upscale._cuda_decode_probe, real_cuda_decode_works, "cuda"),
    ],
)
def test_probes_encode_one_frame_once_per_ffmpeg(
    monkeypatch: pytest.MonkeyPatch, probe: Any, works: Any, flag: str
) -> None:
    probe.cache_clear()
    ran = _fake_run(monkeypatch, 0)
    assert works("/x/ffmpeg") and works("/x/ffmpeg")
    assert len(ran) == 1
    assert ran[0][0] == "/x/ffmpeg" and flag in ran[0]
    assert "color=c=black:s=256x144:d=0.04" in ran[0]
    # Another FFmpeg is probed on its own.
    assert works("/y/ffmpeg") and len(ran) == 2
    for failure in (
        1,
        OSError("no such file"),
        subprocess.TimeoutExpired("ffmpeg", 60),
    ):
        probe.cache_clear()
        _fake_run(monkeypatch, failure)
        assert not works("/x/ffmpeg")
    probe.cache_clear()


def test_probes_pipe_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A timed-out probe behind Chocolatey's shim would otherwise wait on
    pipes the real ffmpeg still holds."""
    seen: dict[str, Any] = {}

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("super_processor.upscale.subprocess.run", run)
    assert upscale._succeeds(["ffmpeg"])
    assert "capture_output" not in seen
    for stream in ("stdin", "stdout", "stderr"):
        assert seen[stream] is subprocess.DEVNULL
    assert seen["timeout"] == 60


@pytest.mark.parametrize("probe", [upscale._nvenc_probe, upscale._cuda_decode_probe])
def test_a_failed_probe_is_asked_again_and_a_working_one_is_kept(
    monkeypatch: pytest.MonkeyPatch, probe: Any
) -> None:
    probe.cache_clear()
    clock = [1000.0]
    monkeypatch.setattr("super_processor.upscale.time.monotonic", lambda: clock[0])
    outcomes = [1, 1, 0]
    ran: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        ran.append(command)
        return subprocess.CompletedProcess(command, outcomes.pop(0))

    monkeypatch.setattr("super_processor.upscale.subprocess.run", run)
    assert not probe("ffmpeg")
    clock[0] += upscale._PROBE_RETRY_S - 1
    assert not probe("ffmpeg") and len(ran) == 1
    clock[0] += 1
    assert not probe("ffmpeg") and len(ran) == 2
    clock[0] += upscale._PROBE_RETRY_S
    assert probe("ffmpeg") and len(ran) == 3
    clock[0] += 10 * upscale._PROBE_RETRY_S
    assert probe("ffmpeg") and len(ran) == 3
    probe.cache_clear()


def test_one_output_sees_one_encoder_while_the_probe_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = [False, True, True]
    asked: list[str] = []

    def works(ffmpeg_bin: str = "ffmpeg") -> bool:
        asked.append(ffmpeg_bin)
        return answers[min(len(asked), len(answers)) - 1]

    monkeypatch.setattr(upscale, "nvenc_works", works)
    with upscale.one_encoder():
        assert delivery_encoder("ffmpeg", (640, 360)) == X264
        with upscale.one_encoder():
            assert delivery_encoder("ffmpeg", (640, 360)) == X264
        assert delivery_encoder("ffmpeg") == X264
        # Each FFmpeg is decided on its own.
        assert delivery_encoder("/opt/ffmpeg") == NVENC
    assert asked == ["ffmpeg", "/opt/ffmpeg"]
    # Outside, every call asks.
    assert delivery_encoder("ffmpeg") == NVENC
    assert len(asked) == 3


def test_another_thread_decides_its_own_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A preview on one wizard thread must not pin the encoder a render on
    # another thread sees: the pin belongs to the thread that set it.
    answers = iter([False, True])
    monkeypatch.setattr(
        upscale, "nvenc_works", lambda ffmpeg_bin="ffmpeg": next(answers)
    )
    seen: list[str] = []
    with upscale.one_encoder():
        assert delivery_encoder() == X264
        other = threading.Thread(target=lambda: seen.append(delivery_encoder()))
        other.start()
        other.join()
    assert seen == [NVENC]


@needs_ffmpeg
def test_a_render_decides_its_encoder_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []

    def works(ffmpeg_bin: str = "ffmpeg") -> bool:
        asked.append(ffmpeg_bin)
        return False

    monkeypatch.setattr(upscale, "nvenc_works", works)
    source = tmp_path / "in.mp4"
    _tiny(source, seconds=2)
    output = upscale.render_shots(
        source,
        tmp_path / "out" / "output.mp4",
        (UpscaleSpan(0, 1, use_upscale=False), UpscaleSpan(1, 2, strength=0.4)),
        engine=FakeUpscaleEngine(),
        duration_s=2.0,
    )
    assert asked == ["ffmpeg"]
    assert is_delivery(output)


@needs_ffmpeg
def test_a_missing_ffmpeg_has_no_nvenc_or_nvdec(tmp_path: Path) -> None:
    missing = str(tmp_path / "no-ffmpeg-here")
    assert not upscale._nvenc_probe(missing)
    assert not upscale._cuda_decode_probe(missing)


def test_each_encoder_has_its_own_settings() -> None:
    keys = [
        "-pix_fmt",
        "yuv420p",
        "-force_key_frames",
        "expr:gte(t,n_forced*1)",
        "-forced-idr",
        "1",
        "-bf",
        "0",
        "-an",
    ]
    assert upscale._delivery_codec(NVENC) == [
        "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
        "-cq", "17", "-b:v", "0", "-profile:v", "high", *keys,
    ]  # fmt: skip
    assert upscale._delivery_codec(X264) == [
        "-c:v", "libx264", "-preset", "faster", "-crf", "16", *keys,
    ]  # fmt: skip
    nvenc = upscale._delivery_marker(NVENC)
    x264 = upscale._delivery_marker(X264)
    assert nvenc["codec"] == NVENC and nvenc["cq"] == 17 and "crf" not in nvenc
    assert x264["codec"] == X264 and x264["crf"] == 16 and "cq" not in x264
    assert nvenc["version"] == x264["version"] == 3


@pytest.mark.parametrize(
    ("size", "rate", "level"),
    [
        # Small or slow pictures still get 4.1, not the level they need.
        ((320, 180), "10", "4.1"),
        ((1280, 960), "30000/1001", "4.1"),
        ((1920, 1088), "30", "4.1"),
        ((1920, 1080), "60", "4.2"),
        ((1920, 1080), "120", "5.1"),
        ((2560, 1920), "30", "5.0"),
        ((2880, 2160), "25", "5.1"),
        ((3840, 2160), "60", "5.2"),
        ((4096, 4096), "30", "6.0"),
        ((4096, 4096), "240", "6.2"),
        ((4096, 4096), "480", None),
        ((320, 180), "0/0", None),
        ((320, 180), "0", None),
        ((320, 180), "fast", None),
    ],
)
def test_nvenc_gets_a_level_that_does_not_cap_its_bitrate(
    size: tuple[int, int], rate: str, level: str | None
) -> None:
    assert upscale.h264_level(size, rate) == level


def test_only_nvenc_is_given_a_level() -> None:
    nvenc = upscale._delivery_codec(NVENC, (320, 180), "10")
    assert nvenc[nvenc.index("-level") + 1] == "4.1"
    assert "-level" not in upscale._delivery_codec(NVENC, (320, 180), None)
    assert "-level" not in upscale._delivery_codec(NVENC, (320, 180), "0/0")
    assert upscale._delivery_codec(X264, (320, 180), "10") == upscale._delivery_codec(
        X264
    )


@needs_ffmpeg
def test_a_delivery_from_the_other_encoder_is_encoded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(upscale, "nvenc_works", lambda ffmpeg_bin="ffmpeg": False)
    source = tmp_path / "in.mp4"
    _moving(source, seconds=6)
    output = tmp_path / "output.mp4"
    engine = FakeUpscaleEngine()
    apply_range_revise(
        source, output, default_span(6), output, engine=engine, duration_s=6
    )
    marker = upscale.delivery_marker_path(output)
    assert json.loads(marker.read_text("utf-8"))["codec"] == X264
    assert is_delivery(output)
    # This machine now encodes with NVENC (at 320x180 it would): an x264
    # output is not one a new part may be spliced into.
    monkeypatch.setattr(upscale, "nvenc_works", lambda ffmpeg_bin="ffmpeg": True)
    assert not is_delivery(output)
    monkeypatch.setattr(upscale, "nvenc_works", lambda ffmpeg_bin="ffmpeg": False)
    # And the other way round: an NVENC output on a machine without it.
    data = json.loads(marker.read_text("utf-8"))
    del data["crf"]
    marker.write_text(json.dumps(data | {"codec": NVENC, "preset": "p5", "cq": 17}))
    assert not is_delivery(output)
    apply_range_revise(
        source, output, UpscaleSpan(2, 4), output, engine=engine, duration_s=6
    )
    assert is_delivery(output)
    assert json.loads(marker.read_text("utf-8"))["codec"] == X264
    assert len(_frame_hashes(output)) == 6 * 24


def _ffv1(path: Path, width: int, height: int, frames: int = 20) -> Path:
    """A clip NVDEC cannot decode (FFV1, 4:4:4), at any size, even an odd one
    (4:2:0 would round it to even)."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"testsrc2=s={width + 1}x{height + 1}:r=10,format=yuv444p,"
         f"crop={width}:{height}:0:0", "-frames:v", str(frames),
         "-c:v", "ffv1", "-pix_fmt", "yuv444p", str(path)],
        check=True,
    )  # fmt: skip
    return path


def _decoded(path: Path, *vf: str) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), *vf, "-f", "rawvideo",
         "-pix_fmt", "yuv420p", "-"],
        check=True,
        capture_output=True,
    ).stdout  # fmt: skip


@needs_ffmpeg
def test_odd_sizes_stream_in_ffmpegs_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fake_span(tmp_path, monkeypatch)
    source = _ffv1(tmp_path / "odd.mkv", 65, 47)
    output = SpanEngine().upscale(UpscaleRequest(source, tmp_path / "out" / "c.mp4"))
    assert video_stream_facts(output)[0] == 20
    assert _frame_size_of(output) == (130, 94)
    # 65x47 has 33x24 chroma planes: each block holds whole frames.
    assert upscale.yuv420_frame_bytes(65, 47) == 65 * 47 + 2 * 33 * 24
    assert fake.frames == [20]
    assert fake.firsts[0] == _decoded(source)[: upscale.yuv420_frame_bytes(65, 47)]


def _without_nvdec(command: list[str]) -> list[str]:
    """``command`` with ``-hwaccel cuda`` taken out: CI has no CUDA device."""
    if "-hwaccel" not in command:
        return command
    at = command.index("-hwaccel")
    return [*command[:at], *command[at + 2 :]]


def _asks_for_nvdec(command: list[str]) -> bool:
    """True when ``command`` decodes its input on NVDEC."""
    if "-hwaccel" not in command:
        return False
    at = command.index("-hwaccel")
    # FFmpeg applies -hwaccel to the input that follows it.
    return command[at + 1] == "cuda" and at < command.index("-i")


@needs_ffmpeg
@pytest.mark.parametrize("nvdec", [True, False])
def test_spans_decoder_uses_nvdec_only_when_it_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvdec: bool
) -> None:
    fake_span(tmp_path, monkeypatch)
    source = tmp_path / "in.mp4"
    _tiny(source, seconds=2)
    commands: list[list[str]] = []

    def software(command: list[str]) -> list[str]:
        commands.append(command)
        return _without_nvdec(command)

    monkeypatch.setattr(upscale, "cuda_decode_works", lambda ffmpeg_bin="ffmpeg": nvdec)
    _swap_decoder(monkeypatch, software)
    SpanEngine().upscale(UpscaleRequest(source, tmp_path / "out" / "clip.mp4"))
    [command] = commands
    assert _asks_for_nvdec(command) is nvdec
    assert command[-4:] == ["rawvideo", "-pix_fmt", "yuv420p", "-"]


@needs_ffmpeg
@pytest.mark.parametrize("nvdec", [True, False])
def test_the_plain_resize_uses_nvdec_only_when_it_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvdec: bool
) -> None:
    source = tmp_path / "in.mp4"
    _tiny(source, seconds=1)
    monkeypatch.setattr(upscale, "cuda_decode_works", lambda ffmpeg_bin="ffmpeg": nvdec)
    real = upscale._run_ffmpeg
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> None:
        commands.append(command)
        real(_without_nvdec(command), **kwargs)

    monkeypatch.setattr(upscale, "_run_ffmpeg", run)
    dest = tmp_path / "out" / "plain.mp4"
    output = upscale.plain_delivery(UpscaleRequest(source, dest), dest)
    [command] = commands
    assert _asks_for_nvdec(command) is nvdec
    assert _frame_size_of(output) == (320, 180)
    assert is_delivery(output)


@pytest.mark.gpu
@pytest.mark.skipif(not _has_nvdec(), reason="needs FFmpeg with CUDA decoding")
@pytest.mark.parametrize("codec", ["h264", "ffv1"])
def test_nvdec_decodes_with_the_clean_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, codec: str
) -> None:
    # FFV1 is not a codec NVDEC has: FFmpeg decodes it on the CPU instead.
    fake = fake_span(tmp_path, monkeypatch)
    monkeypatch.setattr(upscale, "cuda_decode_works", real_cuda_decode_works)
    if codec == "ffv1":
        source = _ffv1(tmp_path / "in.mkv", 320, 180, frames=24)
    else:
        source = tmp_path / "in.mp4"
        _moving(source, seconds=1)
    started: list[list[str]] = []

    def seen(command: list[str]) -> list[str]:
        started.append(command)
        return command

    _swap_decoder(monkeypatch, seen)
    look = ShotLook(deblock=0.5, denoise=0.5)
    output = SpanEngine().upscale(
        UpscaleRequest(source, tmp_path / "out" / "clip.mp4", look=look)
    )
    assert "-hwaccel" in started[0]
    assert video_stream_facts(output)[0] == 24
    cleaned = _decoded(source, "-vf", look.clean_filter())
    assert cleaned != _decoded(source)
    # The frames SPAN got are the cleaned ones, bit for bit.
    assert fake.firsts[0] == cleaned[: len(fake.firsts[0])]


def _stream_info(path: Path) -> dict[str, Any]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,profile,has_b_frames,pix_fmt,nb_read_frames",
         "-count_frames", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout  # fmt: skip
    stream: dict[str, Any] = json.loads(out)["streams"][0]
    return stream


@pytest.mark.gpu
@pytest.mark.skipif(not _has_nvenc(), reason="needs FFmpeg with a working NVENC")
def test_nvenc_deliveries_keep_the_splice_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(upscale, "nvenc_works", real_nvenc_works)
    source = tmp_path / "in.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=s=320x180:r=24:d=3.5", "-c:v", "libx264", "-pix_fmt",
         "yuv420p", str(source)],
        check=True,
    )  # fmt: skip
    output = encode_delivery(source, tmp_path / "out.mp4")
    marker = json.loads(upscale.delivery_marker_path(output).read_text("utf-8"))
    assert marker["codec"] == NVENC and marker["cq"] == 17
    assert is_delivery(output)
    info = _stream_info(output)
    assert info["codec_name"] == "h264" and info["profile"] == "High"
    assert info["has_b_frames"] == 0 and info["pix_fmt"] == "yuv420p"
    assert int(info["nb_read_frames"]) == 84
    keys = upscale.keyframe_times(output)
    assert [round(t, 3) for t in keys] == [0.0, 1.0, 2.0, 3.0]
    # The streamed SPAN path encodes with it too.
    fake_span(tmp_path, monkeypatch)
    clip = tmp_path / "clip.mp4"
    _moving(clip, seconds=2)
    spanned = SpanEngine().upscale(UpscaleRequest(clip, tmp_path / "s" / "up.mp4"))
    marker = json.loads(upscale.delivery_marker_path(spanned).read_text("utf-8"))
    assert marker["codec"] == NVENC
    info = _stream_info(spanned)
    assert info["profile"] == "High" and int(info["nb_read_frames"]) == 48
    assert [round(t, 3) for t in upscale.keyframe_times(spanned)] == [0.0, 1.0]


@pytest.mark.gpu
@pytest.mark.skipif(not _has_nvenc(), reason="needs FFmpeg with a working NVENC")
def test_nvenc_keeps_moving_grain_on_a_small_picture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Left to pick its own level (1.1 here), NVENC held this to 192 kbit/s
    # and 26.9 dB, and the shimmer check no longer saw the grain crawl.
    monkeypatch.setattr(upscale, "nvenc_works", real_nvenc_works)
    grain = tmp_path / "grain.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=s=320x180:r=10:d=3", "-vf", "noise=alls=20:allf=t",
         "-c:v", "ffv1", str(grain)],
        check=True,
    )  # fmt: skip
    output = encode_delivery(grain, tmp_path / "out.mp4")
    assert (
        json.loads(upscale.delivery_marker_path(output).read_text("utf-8"))["codec"]
        == NVENC
    )
    level = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=level", "-of", "csv=p=0", str(output)],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    assert int(level) == 41
    psnr = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(output), "-i", str(grain),
         "-lavfi", "psnr", "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    ).stderr  # fmt: skip
    assert float(psnr.split("average:")[1].split()[0]) > 34
